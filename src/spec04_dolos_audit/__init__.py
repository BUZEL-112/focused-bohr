"""Spec 04: Weight-delta SVD and reference-free spectral audit of Dolos-1.5B.

Implements spec/04_dolos_weight_svd_and_reference_free_audit.md (Track 2 red-teaming,
collusion_hackathon/redteaming/) following spec/plan/plan_04_dolos_weight_svd.md:
    - Stage 1: Delta W = W_Dolos - W_Qwen per (layer, module); Frobenius / spectral norm,
      stable rank, top-1 energy ratio; weight-space logit lens (u_i -> W_U, v_i -> W_E)
    - Stage 2: surgical subspace patching (cure: Dolos - top-k; implant: Qwen + top-k)
    - Stage 3: reference-free audit on Dolos weights alone (cross-layer second-difference
      of sigma_1 / median(sigma), Marchenko-Pastur outlier count, power-law tail exponent,
      vocabulary-projection kurtosis, activation kurtosis)

All math runs on numpy arrays. torch / safetensors / transformers / huggingface_hub are
lazy imports used only for loading the real checkpoints (``pip install -e .[spec04]``).
A synthetic planted-backdoor checkpoint makes the whole pipeline demonstrable offline.
"""
