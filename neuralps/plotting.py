"""Labeled raw compatibility and signed supervised evidence; no probability map."""
import numpy as np

def plot_result(result):
    import matplotlib.pyplot as plt
    rows = result["pretrained_map"]
    supervised = result.get("supervised")
    fig, axes = plt.subplots(2 if supervised else 1, 1, figsize=(min(22, max(10, len(rows)*.28)), 6 if supervised else 3.8), squeeze=False)
    ax = axes[0, 0]
    values = [[r["raw"].get(mode, np.nan) if r["raw"] else np.nan for r in rows] for mode in ("local", "full")]
    cmap = plt.get_cmap("RdYlBu").copy()
    cmap.set_bad("#d1d5db")
    im = ax.imshow(values, aspect="auto", vmin=-1, vmax=1, cmap=cmap)
    ax.set_yticks([0, 1], ["Local", "Full context"])
    ax.set_xticks(range(len(rows)), [f"{r['object_index']} {r['label']}" for r in rows], rotation=70, ha="right", fontsize=7)
    ax.set_title("Pretrained map · single-object masking · raw agreement")
    fig.colorbar(im, ax=ax, label="Reconstruction agreement")
    if supervised:
        ax = axes[1, 0]
        v = [r["logit_contribution"] for r in supervised["objects"]]
        ax.bar(range(len(v)), v, color=["#156c80" if s >= 0 else "#bd603c" for s in v])
        ax.axhline(0, color="#777", lw=.7)
        ax.set_xlabel("Object index")
        ax.set_ylabel("Logit contribution")
        ax.set_title(f"Supervised activity score {supervised['activity_score']:.3f} · signed evidence, uncalibrated")
    fig.tight_layout()
    return fig
