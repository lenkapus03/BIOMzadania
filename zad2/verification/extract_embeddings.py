"""
verification/extract_embeddings.py - Extract ArcFace embeddings for normalized faces.

Usage:
    embeddings = get_embeddings("Kate_Winslet_0")  # (N, 512) float32
    sim = cosine_similarity(emb_a, emb_b)          # float in [-1, 1]
"""

from pathlib import Path

import numpy as np

NORMALIZED_DIR = Path(__file__).parent.parent / "data/videos-K-O-normalized"

_model = None


def _get_model():
    """Load ArcFace recognition model (downloaded automatically on first run).
    Loaded directly via model_zoo — skips face detection since faces are
    already normalized to 112x112."""
    global _model
    if _model is None:
        import insightface
        _model = insightface.model_zoo.get_model("buffalo_l", providers=["CPUExecutionProvider"])
        _model.prepare(ctx_id=-1)  # ctx_id=-1 = CPU
    return _model


def get_embeddings(video_name: str) -> np.ndarray | None:
    """Return (N, 512) L2-normalized ArcFace embeddings for all frames of a video."""
    npz_path = NORMALIZED_DIR / f"{video_name}.npz"
    if not npz_path.exists():
        print(f"[WARN] Normalized file not found: {npz_path}")
        return None

    frames = np.load(npz_path)["normalized_faces"]  # (N, 112, 112, 3) uint8 BGR
    model  = _get_model()
    embeddings = []

    for frame in frames:
        emb  = model.get_feat(frame).flatten().astype(np.float32)
        norm = np.linalg.norm(emb)
        # L2 normalization: divide by vector length so ||emb|| = 1
        # required by ArcFace: paper normalizes features before computing
        # cosine distance (Deng et al. 2019)
        if norm > 1e-6:
            emb = emb / norm
        embeddings.append(emb)

    return np.stack(embeddings).astype(np.float32)  # (N, 512)


def cosine_similarity(emb_a: np.ndarray, emb_b: np.ndarray) -> float:
    """Cosine similarity between two L2-normalized embeddings.

    sim(a, b) = (a . b) / (||a|| * ||b||)
    Since both vectors are unit length, ||a|| = ||b|| = 1, so:
    sim(a, b) = a . b

    Returns float in [-1, 1]. Higher = more similar.
    Recommended metric for ArcFace (Deng et al. 2019).
    """
    return float(np.dot(emb_a, emb_b))