import sys
from pathlib import Path

import numpy as np

from utils.metrics import result_metrics


def summarize(root, result_subdir="."):
    root = Path(root)
    batches = []
    seen = set()

    for fold in range(5):
        with np.load(root / f"fold_{fold}" / result_subdir / "results.npz") as data:
            batch = {
                k: data[k]
                for k in ("preds", "trues", "sites", "years", "clusters")
            }

        sites = set(batch["sites"].tolist())
        if not sites or seen & sites:
            raise ValueError(f"fold {fold} has no sites or shares sites with another fold")
        seen.update(sites)

        pred, true = batch["preds"], batch["trues"]
        if pred.shape != true.shape or pred.ndim != 3:
            raise ValueError(f"fold {fold} has invalid prediction or target dimensions")
        if any(len(v) != len(pred) for v in batch.values()):
            raise ValueError(f"fold {fold} has inconsistent sample-field lengths")
        if not np.isfinite(pred).all() or not np.isfinite(true).all():
            raise ValueError(f"fold {fold} contains nonfinite predictions or targets")

        batches.append(batch)

    merged = {
        k: np.concatenate([b[k] for b in batches], axis=0)
        for k in batches[0]
    }
    pred, true = merged["preds"], merged["trues"]

    scores = result_metrics(pred, true)
    text = '\n'.join(f'{key}: {value:.4f}' for key, value in scores.items())
    print(text)
    (root / "global_metrics.txt").write_text(text + "\n", encoding="utf-8")
    np.savez(root / "global_results.npz", **merged, **scores)


if __name__ == "__main__":
    summarize(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else ".")
