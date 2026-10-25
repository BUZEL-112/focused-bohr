"""Unit tests for the spec 04 Dolos audit (SVD metrics, logit lens, patching, reference-free, CLI)."""

from __future__ import annotations

import json
import sys

import numpy as np
import pytest

from src.spec04_dolos_audit.cli.run_spec04_dolos_audit import main as cli_main
from src.spec04_dolos_audit.repositories.checkpoint_repository import (
    INSTALL_HINT,
    MissingDependencyError,
    SyntheticModelConfig,
    generate_synthetic_checkpoints,
    import_optional,
    module_tensor_name,
    require_real_mode_dependencies,
)
from src.spec04_dolos_audit.services.logit_lens_service import VocabularyLogitLensService, excess_kurtosis
from src.spec04_dolos_audit.services.reference_free_service import (
    ReferenceFreeSpectralService,
    marchenko_pastur_outliers,
    second_differences,
)
from src.spec04_dolos_audit.services.subspace_patching_service import (
    SubspacePatchingService,
    make_synthetic_behavior_probe,
    single_token_forward,
)
from src.spec04_dolos_audit.services.weight_svd_service import (
    WeightDeltaSvdService,
    randomized_svd,
    stable_rank,
)

CONFIG = SyntheticModelConfig()
PLANTED_MODULES = {"gate_proj", "up_proj", "down_proj"}


@pytest.fixture(scope="module")
def synthetic():
    base, dolos, planted = generate_synthetic_checkpoints(CONFIG, seed=0)
    service = WeightDeltaSvdService(top_k=16)
    metrics, factors = service.scan(dolos, base)
    return base, dolos, planted, service, metrics, factors


# --------------------------------------------------------------------------- #
# Stable rank / SVD math on known matrices
# --------------------------------------------------------------------------- #
def test_stable_rank_of_rank_one_matrix_is_one() -> None:
    rng = np.random.default_rng(0)
    matrix = np.outer(rng.normal(size=30), rng.normal(size=20))
    metrics, _ = WeightDeltaSvdService(top_k=4).matrix_metrics(matrix)
    assert metrics.stable_rank == pytest.approx(1.0, abs=1e-9)
    assert metrics.top1_energy_ratio == pytest.approx(1.0, abs=1e-9)


def test_stable_rank_of_identity_equals_dimension() -> None:
    metrics, _ = WeightDeltaSvdService(top_k=3).matrix_metrics(np.eye(10))
    assert metrics.frobenius_norm == pytest.approx(np.sqrt(10))
    assert metrics.spectral_norm == pytest.approx(1.0)
    assert metrics.stable_rank == pytest.approx(10.0)
    assert metrics.top1_energy_ratio == pytest.approx(0.1)
    assert metrics.topk_energy_ratio == pytest.approx(0.3)


def test_stable_rank_of_diagonal_matrix() -> None:
    metrics, _ = WeightDeltaSvdService(top_k=2).matrix_metrics(np.diag([3.0, 4.0]))
    assert metrics.spectral_norm == pytest.approx(4.0)
    assert metrics.stable_rank == pytest.approx(25.0 / 16.0)
    assert stable_rank(0.0, 0.0) == float("inf")


def test_randomized_svd_matches_exact_top_singular_values() -> None:
    rng = np.random.default_rng(1)
    left, _ = np.linalg.qr(rng.normal(size=(600, 200)))
    right, _ = np.linalg.qr(rng.normal(size=(200, 200)))
    spectrum = 10.0 * 0.9 ** np.arange(200)  # decaying, like a real weight spectrum
    matrix = (left * spectrum) @ right.T
    exact = np.linalg.svd(matrix, compute_uv=False)[:5]
    approx = randomized_svd(matrix, 5, seed=3).s
    np.testing.assert_allclose(approx, exact, rtol=1e-6)
    np.testing.assert_allclose(exact, spectrum[:5], rtol=1e-9)


# --------------------------------------------------------------------------- #
# Stage 1: recovery of the planted rank-1 delta
# --------------------------------------------------------------------------- #
def test_scan_ranks_planted_modules_top3(synthetic) -> None:
    _base, _dolos, planted, service, metrics, _factors = synthetic
    top3 = service.rank_by_concentration(metrics)[:3]
    assert {m.layer_idx for m in top3} == {planted.layer_idx}
    assert {m.module_name for m in top3} == PLANTED_MODULES
    assert all(m.stable_rank < 1.2 for m in top3)  # rank-1 dominated
    outliers = service.concentration_outliers(metrics, z_threshold=8.0)
    assert {m.key for m in outliers} == {(planted.layer_idx, m) for m in PLANTED_MODULES}


def test_scan_recovers_planted_singular_direction(synthetic) -> None:
    base, _dolos, planted, _service, _metrics, factors = synthetic
    layer = planted.layer_idx
    svd = factors[(layer, "down_proj")]
    assert abs(svd.vt[0, planted.neuron_idx]) > 0.99  # right vector = planted neuron basis vector
    write_dir = base.final_norm() * base.unembedding()[planted.target_token_id]
    cosine = abs(svd.u[:, 0] @ write_dir) / np.linalg.norm(write_dir)
    assert cosine > 0.99
    gate = factors[(layer, "gate_proj")]
    assert abs(gate.u[planted.neuron_idx, 0]) > 0.99
    assert gate.s[0] == pytest.approx(planted.strength, rel=0.05)


def test_drift_only_modules_are_high_rank(synthetic) -> None:
    _base, _dolos, planted, _service, metrics, _factors = synthetic
    others = [m for m in metrics if m.layer_idx != planted.layer_idx]
    assert max(m.top1_energy_ratio for m in others) < 0.2
    assert min(m.stable_rank for m in others) > 5.0


def test_logit_lens_returns_planted_trigger_and_target(synthetic) -> None:
    _base, dolos, planted, _service, _metrics, factors = synthetic
    lens = VocabularyLogitLensService(top_n=5)
    layer = planted.layer_idx
    for module in ("gate_proj", "up_proj"):
        projection = lens.project(
            factors[(layer, module)], layer, module, dolos.embedding(), dolos.unembedding(),
            final_norm=dolos.final_norm(), input_norm=dolos.input_norm(layer, module), singular_indices=[0],
        )[0]
        assert projection.side == "input"
        assert projection.top_promoted_ids[0] == planted.trigger_token_id
    projection = lens.project(
        factors[(layer, "down_proj")], layer, "down_proj", dolos.embedding(), dolos.unembedding(),
        final_norm=dolos.final_norm(), singular_indices=[0],
    )[0]
    assert projection.side == "output"
    assert projection.top_promoted_ids[0] == planted.target_token_id
    assert projection.top_promoted_tokens[0][0] == f"<tok_{planted.target_token_id}>"


def test_logit_lens_sign_canonicalization_is_flip_invariant() -> None:
    rng = np.random.default_rng(2)
    vocab = rng.normal(size=(50, 8))
    direction = vocab[7] / np.linalg.norm(vocab[7])
    lens = VocabularyLogitLensService()
    np.testing.assert_allclose(lens.vocabulary_scores(direction, vocab), lens.vocabulary_scores(-direction, vocab))
    assert int(np.argmax(lens.vocabulary_scores(-direction, vocab))) == 7


def test_logit_lens_rejects_unknown_module_and_bad_dim() -> None:
    lens = VocabularyLogitLensService()
    with pytest.raises(ValueError):
        lens.side_for("lm_head")
    with pytest.raises(ValueError):
        lens.vocabulary_scores(np.ones(3), np.ones((10, 4)))


# --------------------------------------------------------------------------- #
# Stage 2: subspace patching
# --------------------------------------------------------------------------- #
def test_cure_removes_and_implant_restores_planted_delta(synthetic) -> None:
    base, dolos, planted, _service, _metrics, factors = synthetic
    modules = [(planted.layer_idx, m) for m in sorted(PLANTED_MODULES)]
    patcher = SubspacePatchingService()
    cured = patcher.cure_overrides(dolos, factors, modules, k=1)
    implanted = patcher.implant_overrides(base, factors, modules, k=1)
    drift_floor = CONFIG.drift_std * np.sqrt(np.prod(CONFIG.module_shape("down_proj")))
    for layer, module in modules:
        name = module_tensor_name(layer, module)
        delta_norm = np.linalg.norm(dolos.get_tensor(name) - base.get_tensor(name))
        cured_residual = np.linalg.norm(cured[name] - base.get_tensor(name))
        assert cured_residual < 0.3 * delta_norm  # planted spike gone, only drift remains
        assert cured_residual == pytest.approx(drift_floor, rel=0.25)
        implant_delta = np.linalg.norm(implanted[name] - base.get_tensor(name))
        assert implant_delta == pytest.approx(planted.strength, rel=0.05)  # spike restored
        assert np.linalg.norm(implanted[name] - dolos.get_tensor(name)) < 0.3 * delta_norm


def test_intervention_curve_flips_behavior_at_rank_one(synthetic) -> None:
    base, dolos, planted, _service, _metrics, factors = synthetic
    modules = [(planted.layer_idx, m) for m in sorted(PLANTED_MODULES)]
    controls = [t for t in range(CONFIG.vocab_size) if t not in (planted.trigger_token_id, planted.target_token_id)]
    probe = make_synthetic_behavior_probe([planted.trigger_token_id], planted.target_token_id, controls, CONFIG.head_dim)
    assert probe(dolos) == (1.0, 0.0)
    assert probe(base) == (0.0, 0.0)
    results = SubspacePatchingService().intervention_curve(dolos, base, factors, modules, ranks=(0, 1, 4), behavior_probe=probe)
    by_key = {(r.mode, r.rank_k): r for r in results}
    assert by_key[("cure", 0)].behavior_score == 1.0 and by_key[("cure", 1)].behavior_score == 0.0
    assert by_key[("implant", 0)].behavior_score == 0.0 and by_key[("implant", 1)].behavior_score == 1.0
    assert by_key[("implant", 1)].control_false_positive_rate == 0.0
    assert by_key[("cure", 0)].residual_delta_fraction == pytest.approx(1.0)
    assert by_key[("cure", 1)].residual_delta_fraction < 0.3
    assert by_key[("cure", 1)].removed_energy_fraction > 0.9


def test_single_token_forward_shapes() -> None:
    base, _dolos, _planted = generate_synthetic_checkpoints(CONFIG, seed=4)
    logits, residuals = single_token_forward(base, [1, 2, 3], CONFIG.head_dim)
    assert logits.shape == (3, CONFIG.vocab_size)
    assert residuals.shape == (CONFIG.num_layers + 1, 3, CONFIG.d_model)


# --------------------------------------------------------------------------- #
# Stage 3: reference-free audit
# --------------------------------------------------------------------------- #
def test_reference_free_flags_planted_layer_without_base(synthetic) -> None:
    _base, dolos, planted, _service, _metrics, _factors = synthetic
    service = ReferenceFreeSpectralService()
    anomalies = service.audit_weights(dolos)  # only the Dolos checkpoint is passed
    layer_scores = service.layer_scores(anomalies)
    assert max(layer_scores, key=layer_scores.get) == planted.layer_idx
    flagged = {a.key for a in anomalies if a.flagged}
    assert {(planted.layer_idx, m) for m in PLANTED_MODULES} <= flagged


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_reference_free_negative_control_flags_nothing(seed: int) -> None:
    _base, drift_only, planted = generate_synthetic_checkpoints(CONFIG, plant=False, seed=seed)
    assert planted is None
    service = ReferenceFreeSpectralService()
    anomalies = service.audit_weights(drift_only)
    assert service.flagged_layers(anomalies) == []
    assert max(a.anomaly_score for a in anomalies) < service.z_threshold / 2


def test_activation_profile_recovers_trigger_at_planted_layer(synthetic) -> None:
    _base, dolos, planted, _service, _metrics, _factors = synthetic
    _, residuals = single_token_forward(dolos, range(CONFIG.vocab_size), CONFIG.head_dim)
    profile = ReferenceFreeSpectralService.activation_profile(residuals, dolos.unembedding(), dolos.final_norm())
    assert int(np.argmax(profile["update_kurtosis"])) == planted.layer_idx
    assert int(profile["kl_surge_argmax"][planted.layer_idx]) == planted.trigger_token_id


def test_reference_free_helpers() -> None:
    np.testing.assert_allclose(second_differences(np.array([1.0, 1.0, 5.0, 1.0, 1.0])), [0.0, 2.0, 4.0, 2.0, 0.0])
    rng = np.random.default_rng(5)
    noise = rng.normal(size=(400, 200))
    assert marchenko_pastur_outliers(np.linalg.svd(noise, compute_uv=False), noise.shape) == 0
    left, right = rng.normal(size=400), rng.normal(size=200)
    spike = np.outer(left / np.linalg.norm(left), right / np.linalg.norm(right))
    spiked = noise + 3.0 * (np.sqrt(400) + np.sqrt(200)) * spike  # sigma well above the MP edge
    assert marchenko_pastur_outliers(np.linalg.svd(spiked, compute_uv=False), spiked.shape) >= 1
    assert abs(excess_kurtosis(rng.normal(size=200_000))) < 0.05
    assert excess_kurtosis(np.ones(10)) == 0.0


# --------------------------------------------------------------------------- #
# Lazy optional dependencies + CLI
# --------------------------------------------------------------------------- #
def test_missing_optional_dependency_raises_install_hint(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "safetensors", None)  # forces ImportError on import
    with pytest.raises(MissingDependencyError, match=r"pip install -e \.\[spec04\]"):
        import_optional("safetensors")
    with pytest.raises(MissingDependencyError, match="safetensors"):
        require_real_mode_dependencies(("safetensors",))
    assert "pip install -e .[spec04]" in INSTALL_HINT


def test_cli_real_mode_without_dependencies_exits_nonzero(monkeypatch, tmp_path, capsys) -> None:
    for name in ("torch", "safetensors", "huggingface_hub", "transformers"):
        monkeypatch.setitem(sys.modules, name, None)
    assert cli_main(["--stage", "1", "--output-dir", str(tmp_path / "real")]) != 0
    assert "pip install -e .[spec04]" in capsys.readouterr().err
    assert not (tmp_path / "real").exists()


def test_cli_synthetic_all_stages_recovers_planted_circuit(tmp_path) -> None:
    assert cli_main(["--synthetic", "--stage", "all", "--output-dir", str(tmp_path)]) == 0
    summary = json.loads((tmp_path / "synthetic_summary.json").read_text(encoding="utf-8"))
    assert summary["mode"] == "synthetic" and "SYNTHETIC" in summary["disclaimer"]
    assert all(summary["recovery_checks"].values()), summary["recovery_checks"]
    for stem in (
        "synthetic_fig1_delta_heatmap.png",
        "synthetic_fig2_patching_curve.png",
        "synthetic_fig3_reference_free_layers.png",
        "synthetic_stage1_logit_lens_tokens.csv",
        "synthetic_stage3_reference_free_anomaly.csv",
    ):
        assert (tmp_path / stem).is_file()
