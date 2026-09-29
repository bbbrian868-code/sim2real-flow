"""Build a derived, subsampled trajectory cache for fast FM data loading.

For every trajectory of <root>/cylinder/hf_dataset/<type>, writes <out>/<type>/<sim_id>.npy with shape
(T, H/sub_s, W/sub_s, C) float32, channel-last, fields (u, v[, p]) exactly as the official loader slices them
(`field[:, ::sub_s, ::sub_s]`), so any window [t:t+horizon] is one contiguous read.
sub_s follows the official CylinderHFDataset defaults: real 1, numerical 2.

  python build_cache.py --root DATA_ROOT --out CACHE_DIR --type numerical [--workers 8]
"""
import argparse, glob, json, os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pyarrow as pa

SUB = {"real": 1, "numerical": 2}
FIELDS = {"real": ("u", "v"), "numerical": ("u", "v", "p")}


def one_shard(args):
    path, typ, out = args
    with pa.memory_map(path) as src:
        tab = pa.ipc.open_stream(src).read_all()
    done = []
    for i in range(tab.num_rows):
        sim = tab.column("sim_id")[i].as_py()
        shape = tuple(int(tab.column(k)[i].as_py()) for k in ("shape_t", "shape_h", "shape_w"))
        s = SUB[typ]
        arrs = [np.frombuffer(tab.column(f)[i].as_buffer(), np.float32).reshape(shape)[:, ::s, ::s] for f in FIELDS[typ]]
        data = np.ascontiguousarray(np.stack(arrs, axis=-1))
        dst = os.path.join(out, typ, sim.replace(".h5", ".npy"))
        tmp = dst + ".tmp.npy"
        np.save(tmp, data)
        os.replace(tmp, dst)
        done.append((sim, list(data.shape)))
    return done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--type", required=True, choices=["real", "numerical"])
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    os.makedirs(os.path.join(a.out, a.type), exist_ok=True)
    shards = sorted(glob.glob(os.path.join(a.root, "cylinder/hf_dataset", a.type, "data-*.arrow")))
    with ProcessPoolExecutor(a.workers) as ex:
        done = [d for ds in ex.map(one_shard, [(s, a.type, a.out) for s in shards]) for d in ds]
    manifest = {"source_root": os.path.realpath(a.root), "type": a.type, "sub_s": SUB[a.type],
                "fields": FIELDS[a.type], "trajectories": dict(done)}
    version = json.load(open(os.path.join(a.root, "version.json")))["data_version"]
    manifest["dataset_version"] = version
    json.dump(manifest, open(os.path.join(a.out, a.type, "manifest.json"), "w"), indent=1)
    print(f"{len(done)} trajectories -> {a.out}/{a.type} (dataset_version {version})")


if __name__ == "__main__":
    main()
