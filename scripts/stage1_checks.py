"""Step 2.3: real-data sanity checks (coordinates, sign convention, divergence floor), per trajectory.

Checks, for every real trajectory:
  * coordinates: which array axis x / y vary along, grid spacing, uniformity, value range (units)
  * sign convention: fit the stored vorticity `vo` against curl(u, v) = dv/dx - du/dy computed on the
    stored x / y grid; a ~ +1 means the stored v, y and vo follow the same right-handed convention
  * divergence floor: ||du/dx + dv/dy|| / sqrt(||du/dx||^2 + ||dv/dy||^2)  (0 = divergence-free;
    ~1 = uncorrelated gradients). 2D PIV of a 3D wake is never exactly 0, so this is a floor to compare.

  python stage1_checks.py --root DATA_ROOT --out OUT.csv [--frame-stride 10]
"""
import argparse, csv, glob, os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pyarrow as pa


def arr(raw, shape):
    dt = np.float64 if len(raw) == 8 * int(np.prod(shape)) else np.float32
    return np.frombuffer(raw, dt).reshape(shape).astype(np.float64)


def check(u, v, vo, x, y):
    r = {}
    # axis along which each coordinate varies
    gx = [np.abs(np.diff(x, axis=a)).mean() for a in (0, 1)]
    gy = [np.abs(np.diff(y, axis=a)).mean() for a in (0, 1)]
    ax_x, ax_y = int(np.argmax(gx)), int(np.argmax(gy))
    dx = np.diff(x, axis=ax_x); dy = np.diff(y, axis=ax_y)
    r.update(x_axis=ax_x, y_axis=ax_y, dx_mean=dx.mean(), dx_rel_spread=dx.std() / abs(dx.mean()),
             dy_mean=dy.mean(), dy_rel_spread=dy.std() / abs(dy.mean()),
             x_min=x.min(), x_max=x.max(), y_min=y.min(), y_max=y.max(),
             x_const_along_other=float(np.abs(np.diff(x, axis=1 - ax_x)).max()),
             y_const_along_other=float(np.abs(np.diff(y, axis=1 - ax_y)).max()))
    # 1D coordinate vectors (arrays are (T, H, W); spatial axes are 1 and 2)
    xv = x.mean(axis=1 - ax_x); yv = y.mean(axis=1 - ax_y)
    dudx = np.gradient(u, xv, axis=1 + ax_x); dvdy = np.gradient(v, yv, axis=1 + ax_y)
    dvdx = np.gradient(v, xv, axis=1 + ax_x); dudy = np.gradient(u, yv, axis=1 + ax_y)
    curl = dvdx - dudy
    # exclude a 2-cell border where one-sided differences dominate
    s = (slice(None), slice(2, -2), slice(2, -2))
    c, w = curl[s].ravel(), vo[s].ravel()
    a = float(c @ w / (c @ c + 1e-30))
    r.update(vo_fit_a=a, vo_corr=float(np.corrcoef(c, w)[0, 1]),
             vo_rel_resid=float(np.linalg.norm(w - a * c) / (np.linalg.norm(w) + 1e-30)))
    div = (dudx + dvdy)[s]
    r["div_ratio"] = float(np.linalg.norm(div) / np.sqrt(np.linalg.norm(dudx[s]) ** 2 + np.linalg.norm(dvdy[s]) ** 2))
    r["div_rms"] = float(np.sqrt((div ** 2).mean()))
    # alternative conventions: v stored with the opposite sign to y  ->  div' = du/dx - dv/dy, curl' = -dv/dx - du/dy
    div_alt = (dudx - dvdy)[s]
    r["div_ratio_vflip"] = float(np.linalg.norm(div_alt) / np.sqrt(np.linalg.norm(dudx[s]) ** 2 + np.linalg.norm(dvdy[s]) ** 2))
    c_alt = (-dvdx - dudy)[s].ravel()
    r["vo_corr_vflip"] = float(np.corrcoef(c_alt, w)[0, 1])
    # correlation of du/dx with dv/dy: -1 for an exactly divergence-free 2D field
    r["corr_dudx_dvdy"] = float(np.corrcoef(dudx[s].ravel(), dvdy[s].ravel())[0, 1])
    r["u_mean"], r["v_mean"] = float(u.mean()), float(v.mean())
    return r


def shard(args):
    path, stride = args
    with pa.memory_map(path) as src:
        tab = pa.ipc.open_stream(src).read_all()
    rows = []
    for i in range(tab.num_rows):
        g = lambda k: tab.column(k)[i].as_py()
        shp = (g("shape_t"), g("shape_h"), g("shape_w"))
        u, v, vo = (arr(g(k), shp)[::stride] for k in ("u", "v", "vo"))
        x = arr(g("x"), (g("x_shape_h"), g("x_shape_w")))
        y = arr(g("y"), (g("y_shape_h"), g("y_shape_w")))
        t = arr(g("t"), (g("t_shape"),))
        r = {"sim_id": g("sim_id"), "shard": os.path.basename(path),
             "dt_mean": float(np.diff(t).mean()), "dt_rel_spread": float(np.diff(t).std() / np.diff(t).mean())}
        r.update(check(u, v, vo, x, y))
        rows.append(r)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--frame-stride", type=int, default=10)
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args()
    shards = sorted(glob.glob(os.path.join(args.root, "cylinder/hf_dataset/real/data-*.arrow")))
    with ProcessPoolExecutor(args.workers) as ex:
        rows = [r for rs in ex.map(shard, [(s, args.frame_stride) for s in shards]) for r in rs]
    rows.sort(key=lambda r: r["sim_id"])
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, list(rows[0]))
        w.writeheader(); w.writerows(rows)
    print(f"{len(rows)} trajectories -> {args.out}")


if __name__ == "__main__":
    main()
