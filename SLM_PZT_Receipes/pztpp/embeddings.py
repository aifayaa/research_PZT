from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np


def encode_texts(texts: Iterable[str], *, model_name: str, cache_dir: Path | None = None) -> Tuple[np.ndarray, dict]:
    texts = list(texts)
    if model_name != "hash":
        try:
            from sentence_transformers import SentenceTransformer

            model = SentenceTransformer(model_name, cache_folder=str(cache_dir) if cache_dir else None)
            arr = model.encode(texts, convert_to_numpy=True, show_progress_bar=False, normalize_embeddings=True)
            return arr.astype(np.float32), {
                "backend": "sentence_transformers",
                "model_name": model_name,
                "dim": int(arr.shape[1]) if arr.ndim == 2 else 0,
                "dtype": "float16",
            }
        except Exception as exc:
            fallback_reason = str(exc)
        else:
            fallback_reason = ""
    else:
        fallback_reason = "explicit hash model"

    arr = _hash_embeddings(texts, dim=384)
    return arr, {
        "backend": "hash_fallback",
        "model_name": model_name,
        "dim": int(arr.shape[1]),
        "dtype": "float16",
        "fallback_reason": fallback_reason,
    }


def _hash_embeddings(texts: list[str], dim: int) -> np.ndarray:
    out = np.zeros((len(texts), dim), dtype=np.float32)
    for i, text in enumerate(texts):
        tokens = [tok for tok in text.lower().replace("\n", " ").split() if tok]
        if not tokens:
            tokens = ["empty"]
        for token in tokens:
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            bucket = int.from_bytes(digest[:4], "little") % dim
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            out[i, bucket] += sign
        norm = np.linalg.norm(out[i])
        if norm > 0:
            out[i] /= norm
    return out


def l2_normalize_rows(arr: np.ndarray) -> np.ndarray:
    arr = arr.astype(np.float32, copy=False)
    norm = np.linalg.norm(arr, axis=1, keepdims=True) + 1e-12
    return arr / norm
