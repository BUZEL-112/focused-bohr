"""Stage 2B: Conditional Normalized Compression Distance (NCD) on interleaved
residual-and-message streams (zlib / lzma Kolmogorov-complexity approximation)."""

from __future__ import annotations

import lzma
import zlib
from collections.abc import Sequence


class CompressionDistanceService:
    """Computes NCD(x, y) = (C(xy) - min(C(x), C(y))) / max(C(x), C(y)) and CCS = 1 - NCD.

    Byte encodings interleave round-by-round symbols (z_{i,1}, z_{j,1}, z_{i,2}, ...) so
    that a shared multi-round pattern (3-round bid rotation, steganographic phrasing)
    compresses jointly. Message text can be folded into each round's payload, which lets
    the detector catch textual steganography with the same machinery.
    """

    def __init__(self, compressor: str = "zlib") -> None:
        if compressor not in ("zlib", "lzma"):
            raise ValueError("compressor must be 'zlib' or 'lzma'")
        self._compressor = compressor

    # ------------------------------------------------------------------ #
    # Byte encoding
    # ------------------------------------------------------------------ #
    def encode_symbol_stream(self, stream: Sequence[int], interleave: Sequence[int] | None = None) -> bytes:
        """Symbols -> bytes, optionally interleaved round-by-round with another stream.

        Each symbol is one byte (alphabets up to 256 are supported; spec 01 uses K = 5).
        """
        if interleave is None:
            return bytes(int(s) & 0xFF for s in stream)
        if len(interleave) != len(stream):
            raise ValueError("interleaved streams must have equal length")
        payload: list[int] = []
        for a, b in zip(stream, interleave):
            payload.extend((int(a) & 0xFF, int(b) & 0xFF))
        return bytes(payload)

    def encode_message_stream(self, messages: Sequence[str]) -> bytes:
        """Messages -> UTF-8 bytes with a per-round separator (empty messages kept as NUL)."""
        chunks = []
        for message in messages:
            text = message if message else "\x00"
            chunks.append(text.encode("utf-8"))
            chunks.append(b"\x01")  # round separator
        return b"".join(chunks)

    # ------------------------------------------------------------------ #
    # Compression
    # ------------------------------------------------------------------ #
    def compressed_size(self, payload: bytes) -> int:
        if self._compressor == "zlib":
            return len(zlib.compress(payload, 9))
        return len(lzma.compress(payload))

    def ncd(self, stream_a: bytes, stream_b: bytes) -> float:
        """Raw NCD of two byte payloads (can leave [0, 1] for degenerate tiny inputs)."""
        size_a, size_b = self.compressed_size(stream_a), self.compressed_size(stream_b)
        joint = self.compressed_size(stream_a + stream_b)
        denominator = max(size_a, size_b)
        if denominator == 0:
            return 0.0
        return (joint - min(size_a, size_b)) / denominator

    def compute_coupling_score(
        self,
        stream_a: Sequence[int],
        stream_b: Sequence[int],
        msgs_a: Sequence[str] | None = None,
        msgs_b: Sequence[str] | None = None,
    ) -> float:
        """Compression Coupling Score CCS(i, j) = 1 - NCD in [0, 1] (higher = more coupled)."""
        bytes_a = self.encode_symbol_stream(stream_a)
        bytes_b = self.encode_symbol_stream(stream_b)
        if msgs_a is not None and msgs_b is not None:
            bytes_a = bytes_a + b"\x02" + self.encode_message_stream(msgs_a)
            bytes_b = bytes_b + b"\x02" + self.encode_message_stream(msgs_b)
        ncd = self.ncd(bytes_a, bytes_b)
        return max(0.0, min(1.0, 1.0 - ncd))
