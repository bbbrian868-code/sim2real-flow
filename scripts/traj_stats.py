"""Step 2.1: per-trajectory integrity and channel statistics for cylinder real (or numerical) Arrow shards.

  python traj_stats.py --root DATA_ROOT --type real --out stats.csv [--workers 16]
"""
import argparse, csv, glob, os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pyarrow as pa

FIELDS = {"real": ["u", "v", "vo"], "numerical": ["u", "v", "p"]}


def shard_stats(args):
    path, fields = args
    rows = []
    with pa.memory_map(path) as src:
        tab = pa.ipc.open_stream(src).read_all()
    for i in range(tab.num_rows):
        shape = tuple(int(tab.column(k)[i].as_py()) for k in ("shape_t", "shape_h", "shape_w"))
        row = {"shard": os.path.basename(path), "sim_id": tab.column("sim_id")[i].as_py(),
               "shape_t": shape[0], "shape_h": shape[1], "shape_w": shape[2]}
        for f in fields:
            if f not in tab.column_names:
                continue
            a = np.frombuffer(tab.column(f)[i].as_py(), dtype=np.float32).reshape(shape).astype(np.float64)
            fin = np.isfinite(a)
            row[f"{f}_nan"] = int(np.isnan(a).sum())
            row[f"{f}_inf"] = int(np.isinf(a).sum())
            b = a[fin]
            row[f"{f}_mean"], row[f"{f}_std"] = float(b.mean()), float(b.std())
            row[f"{f}_min"], row[f"{f}_max"] = float(b.min()), float(b.max())
        rows.append(row)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--type", default="real", choices=["real", "numerical"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()

    shards = sorted(glob.glob(os.path.join(args.root, "cylinder/hf_dataset", args.type, "data-*.arrow")))
    with ProcessPoolExecutor(args.workers) as ex:
        rows = [r for rs in ex.map(shard_stats, [(s, FIELDS[args.type]) for s in shards]) for r in rs]
    rows.sort(key=lambda r: r["sim_id"])
    keys = list(dict.fromkeys(k for r in rows for k in r))
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, keys)
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} trajectories from {len(shards)} shards -> {args.out}")


if __name__ == "__main__":
    main()
