"""Step 2.2: describe how a trajectory changed between two data roots.

  python traj_diff.py --old OLD_ROOT --new NEW_ROOT --sim 3656.h5 --out OUT_DIR
"""
import argparse, glob, json, os

import numpy as np
import pyarrow as pa
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIELDS = ["u", "v", "vo"]


def load_traj(root, sim):
    for path in sorted(glob.glob(os.path.join(root, "cylinder/hf_dataset/real/data-*.arrow"))):
        with pa.memory_map(path) as src:
            tab = pa.ipc.open_stream(src).read_all()
        ids = tab.column("sim_id").to_pylist()
        if sim not in ids:
            continue
        i = ids.index(sim)
        g = lambda k: tab.column(k)[i].as_py()
        shape = (g("shape_t"), g("shape_h"), g("shape_w"))
        out = {f: np.frombuffer(g(f), np.float32).reshape(shape).astype(np.float64) for f in FIELDS}
        for k, shp in (("x", (g("x_shape_h"), g("x_shape_w"))), ("y", (g("y_shape_h"), g("y_shape_w"))), ("t", (g("t_shape"),))):
            raw = g(k)
            dt = np.float64 if len(raw) == 8 * int(np.prod(shp)) else np.float32
            out[k] = np.frombuffer(raw, dt).reshape(shp).astype(np.float64)
        out["shard"] = os.path.basename(path)
        return out
    raise KeyError(sim)


def fit(new, old):
    A = np.stack([old.ravel(), np.ones(old.size)], 1)
    (a, b), *_ = np.linalg.lstsq(A, new.ravel(), rcond=None)
    r = new.ravel() - A @ np.array([a, b])
    return float(a), float(b), float(np.linalg.norm(r) / np.linalg.norm(new))


def corr(a, b):
    a, b = a.ravel() - a.mean(), b.ravel() - b.mean()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))


def best_lag(new, old, max_lag):
    """Lag L maximizing mean framewise correlation between new[t] and old[t+L]."""
    T = new.shape[0]
    n = (new - new.mean((1, 2), keepdims=True)).reshape(T, -1)
    o = (old - old.mean((1, 2), keepdims=True)).reshape(T, -1)
    n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-30
    o /= np.linalg.norm(o, axis=1, keepdims=True) + 1e-30
    scores = {}
    for L in range(-max_lag, max_lag + 1):
        if L >= 0:
            s = (n[: T - L] * o[L:]).sum(1).mean()
        else:
            s = (n[-L:] * o[: T + L]).sum(1).mean()
        scores[L] = float(s)
    L = max(scores, key=scores.get)
    return L, scores[L], scores[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--old", required=True)
    ap.add_argument("--new", required=True)
    ap.add_argument("--sim", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-lag", type=int, default=200)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    O, N = load_traj(args.old, args.sim), load_traj(args.new, args.sim)
    tag = args.sim.replace(".h5", "")
    rep = {"sim_id": args.sim, "shard_old": O["shard"], "shard_new": N["shard"], "fields": {}, "coords": {}}

    for k in ("x", "y", "t"):
        rep["coords"][k] = {"identical": bool(np.array_equal(O[k], N[k])),
                            "max_abs_diff": float(np.abs(O[k] - N[k]).max()) if O[k].shape == N[k].shape else None,
                            "old_range": [float(O[k].min()), float(O[k].max())],
                            "new_range": [float(N[k].min()), float(N[k].max())]}

    for f in FIELDS:
        o, n = O[f], N[f]
        d = n - o
        per_frame = np.linalg.norm(d.reshape(len(d), -1), axis=1) / (np.linalg.norm(o.reshape(len(o), -1), axis=1) + 1e-30)
        L, sL, s0 = best_lag(n, o, args.max_lag)
        a, b, resid = fit(n, o)
        transforms = {
            "identity": corr(n, o), "neg": corr(n, -o),
            "flip_h": corr(n, o[:, ::-1, :]), "flip_w": corr(n, o[:, :, ::-1]),
            "flip_hw": corr(n, o[:, ::-1, ::-1]),
            "neg_flip_h": corr(n, -o[:, ::-1, :]), "neg_flip_w": corr(n, -o[:, :, ::-1]),
        }
        cross = {g: corr(n, O[g]) for g in FIELDS if g != f}
        rep["fields"][f] = {
            "rel_l2_total": float(np.linalg.norm(d) / np.linalg.norm(o)),
            "rel_l2_per_frame_min_median_max": [float(per_frame.min()), float(np.median(per_frame)), float(per_frame.max())],
            "best_lag_frames": L, "corr_at_best_lag": sL, "corr_at_lag0": s0,
            "linear_fit_new_eq_a_old_plus_b": {"a": a, "b": b, "rel_residual": resid},
            "corr_under_transforms": transforms,
            "corr_new_vs_old_other_fields": cross,
            "old_mean_std": [float(o.mean()), float(o.std())], "new_mean_std": [float(n.mean()), float(n.std())],
        }

        fig, ax = plt.subplots(2, 3, figsize=(15, 6))
        for j, (arr, name) in enumerate([(o.mean(0), "old (time mean)"), (n.mean(0), "new (time mean)"), ((n - o).mean(0), "new-old (time mean)")]):
            im = ax[0, j].imshow(arr, origin="lower", cmap="RdBu_r" if j == 2 else "viridis")
            ax[0, j].set_title(f"{f}: {name}"); fig.colorbar(im, ax=ax[0, j], shrink=0.8)
        ax[1, 0].plot(per_frame); ax[1, 0].set_title(f"{f}: per-frame rel L2 of (new-old)"); ax[1, 0].set_xlabel("frame")
        ax[1, 1].plot(o.mean((1, 2)), label="old"); ax[1, 1].plot(n.mean((1, 2)), label="new")
        ax[1, 1].set_title(f"{f}: spatial mean vs frame"); ax[1, 1].legend()
        sel = np.random.default_rng(0).choice(o.size, 20000, replace=False)
        ax[1, 2].scatter(o.ravel()[sel], n.ravel()[sel], s=1, alpha=0.3)
        ax[1, 2].set_xlabel("old"); ax[1, 2].set_ylabel("new"); ax[1, 2].set_title(f"{f}: new vs old (fit a={a:.3g}, b={b:.3g})")
        fig.tight_layout(); fig.savefig(os.path.join(args.out, f"{tag}_{f}.png"), dpi=110); plt.close(fig)

    with open(os.path.join(args.out, f"{tag}_diff.json"), "w") as fh:
        json.dump(rep, fh, indent=1)
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
