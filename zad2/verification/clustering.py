"""
verification/clustering.py - Cluster face embeddings using K-Means + PCA visualization.

Pipeline:
  1. For each video in videos-K-O-normalized/, compute mean embedding (re-normalized).
  2. K-Means clustering on the (n_videos, 512) embedding matrix.
  3. PCA 2D scatter plot coloured by cluster.
  4. For each cluster save a grid of one representative frame per video.
"""

from pathlib import Path
import math

import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from zad2.verification.extract_embeddings import get_embeddings, NORMALIZED_DIR

N_CLUSTERS  = 4
RANDOM_SEED = 42
GRID_COLS   = 6          # columns in per-cluster photo grid
MAX_PHOTOS  = 36         # max photos shown per cluster

OUT_DIR = Path(__file__).parent.parent / "data" / "clusters"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_all_embeddings() -> tuple[list[str], np.ndarray, list[np.ndarray]]:
    """
    Returns:
        names      - list of video stems (length n_videos)
        mean_embs  - (n_videos, 512) mean L2-normalised embedding per video
        rep_frames - list of one representative BGR frame per video
    """
    npz_files = sorted(NORMALIZED_DIR.glob("*.npz"))
    names, mean_embs, rep_frames = [], [], []

    for i, npz in enumerate(npz_files, 1):
        video_name = npz.stem
        print(f"  [{i:3d}/{len(npz_files)}] {video_name}", end="\r")

        # get_embeddings returns (N, 512) L2-normalised embeddings for each frame
        embs = get_embeddings(video_name)
        if embs is None or len(embs) == 0:
            continue

        # Average N embeddings → one vector representing the whole video.
        # Smooths out random variations caused by expression, pose, or lighting.
        mean_emb = embs.mean(axis=0)

        # Re-normalisation
        norm = np.linalg.norm(mean_emb)
        if norm > 1e-6:
            mean_emb = mean_emb / norm

        # Middle frame used as a visual representative in cluster grids
        frames = np.load(npz)["normalized_faces"]  # (N, 112, 112, 3) BGR uint8
        rep_frame = frames[len(frames) // 2]

        names.append(video_name)
        mean_embs.append(mean_emb)
        rep_frames.append(rep_frame)

    print()
    return names, np.stack(mean_embs).astype(np.float32), rep_frames



def cluster(embs: np.ndarray, n_clusters: int) -> np.ndarray:
    # n_init=10 runs K-Means from 10 different random initialisations and keeps
    # the best result (lowest inertia), making the output more stable
    km = KMeans(n_clusters=n_clusters, random_state=RANDOM_SEED, n_init=10)
    return km.fit_predict(embs)  # returns cluster label (0..n_clusters-1) per video


def plot_pca(embs: np.ndarray, labels: np.ndarray, names: list[str], out_path: Path, n_clusters: int = N_CLUSTERS):
    # Reduce 512D embeddings to 3D for visualisation only — clustering was done
    # in the full 512D space, PCA is used purely for the interactive scatter plot
    pca  = PCA(n_components=3, random_state=RANDOM_SEED)
    proj = pca.fit_transform(embs)   # (n_videos, 3)
    var  = pca.explained_variance_ratio_

    cmap = plt.cm.get_cmap("tab10", n_clusters)

    traces = []
    for c in range(n_clusters):
        mask     = labels == c
        # Convert matplotlib colour to plotly-compatible hex string
        r, g, b, _ = cmap(c)
        colour   = f"rgb({int(r*255)},{int(g*255)},{int(b*255)})"
        traces.append(go.Scatter3d(
            x=proj[mask, 0],
            y=proj[mask, 1],
            z=proj[mask, 2],
            mode="markers",
            name=f"Cluster {c}  (n={mask.sum()})",
            text=[names[i] for i in range(len(names)) if labels[i] == c],
            hovertemplate="%{text}<extra>Cluster " + str(c) + "</extra>",
            marker=dict(size=4, color=colour, opacity=0.8),
        ))

    fig = go.Figure(data=traces)
    fig.update_layout(
        title=f"PCA 3D projekcia ArcFace embeddingov — K-Means (K={n_clusters})",
        scene=dict(
            xaxis_title=f"PC1 ({var[0]*100:.1f} %)",
            yaxis_title=f"PC2 ({var[1]*100:.1f} %)",
            zaxis_title=f"PC3 ({var[2]*100:.1f} %)",
        ),
        legend=dict(itemsizing="constant"),
        margin=dict(l=0, r=0, b=0, t=40),
    )

    # Save as interactive HTML — open in any browser to rotate/zoom
    html_path = out_path.with_suffix(".html")
    fig.write_html(str(html_path))
    print(f"PCA 3D scatter saved -> {html_path}")


def plot_cluster_grid(cluster_id: int, video_names: list[str],
                      frames: list[np.ndarray], out_path: Path):
    """One frame per video; each person shown once (videos already unique)."""
    n    = min(len(frames), MAX_PHOTOS)
    cols = min(GRID_COLS, n)
    rows = math.ceil(n / cols)

    fig, axes = plt.subplots(rows, cols,
                             figsize=(cols * 1.6, rows * 1.8 + 0.6))
    # reshape to 1D so we can index axes the same way regardless of grid shape
    axes = np.array(axes).reshape(-1)

    # hide all axes first; only the used ones will get an image
    for ax in axes:
        ax.axis("off")

    for i, (name, frame) in enumerate(zip(video_names[:n], frames[:n])):
        ax = axes[i]
        ax.imshow(frame[:, :, ::-1])  # BGR → RGB for matplotlib
        ax.set_title(name, fontsize=5, pad=2)
        ax.axis("off")

    fig.suptitle(f"Cluster {cluster_id}  ({len(video_names)} videí)",
                 fontsize=11, y=1.01)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()


def run_for_k(k: int, names: list[str], embs: np.ndarray, rep_frames: list[np.ndarray]):
    """Run clustering, PCA, and grid generation for a single value of K."""
    out_dir = OUT_DIR.parent / f"clusters_{k}"
    out_dir.mkdir(parents=True, exist_ok=True)

    labels = cluster(embs, k)

    # Short summary per cluster
    print(f"\n  {'Cluster':>8}  {'Videos':>7}  {'Sample names'}")
    print(f"  {'-'*65}")
    for c in range(k):
        members = [names[i] for i in range(len(names)) if labels[i] == c]
        print(f"  {c:>8}  {len(members):>7}  {', '.join(members[:4])}{'...' if len(members) > 4 else ''}")

    # Interactive 3D PCA scatter
    plot_pca(embs, labels, names, out_dir / "pca_clusters.png", n_clusters=k)

    # Photo grid per cluster
    for c in range(k):
        indices  = [i for i, lbl in enumerate(labels) if lbl == c]
        c_names  = [names[i]      for i in indices]
        c_frames = [rep_frames[i] for i in indices]
        plot_cluster_grid(c, c_names, c_frames, out_dir / f"cluster_{c:02d}.png")
        print(f"    cluster_{c:02d}.png  ({len(c_names)} videos)")

    print(f"  Saved -> {out_dir}")


def run():
    # Step 1: load embeddings once — reused for every K
    print("=== Loading embeddings ===")
    names, embs, rep_frames = load_all_embeddings()
    print(f"Loaded {len(names)} videos.\n")

    # Step 2: run the full pipeline for K = 4 .. 10
    for k in range(4, 16):
        print(f"\n{'='*50}")
        print(f"=== K = {k} ===")
        print(f"{'='*50}")
        run_for_k(k, names, embs, rep_frames)

    print("\nDone. Results in:", OUT_DIR.parent)


if __name__ == "__main__":
    run()