"""Step 1.3: compare index/params JSON between two data roots and quantify val/test overlap.

Usage: python split_analysis.py --old ROOT_OLD --new ROOT_NEW --out OUT_JSON [--horizon 40]
"""
import argparse, collections, hashlib, json, os

SPLITS = ["train", "val", "test"]
TYPES = ["real", "numerical"]
PARAMS = ["remain_params", "in_dist_test_params", "out_dist_test_params"]


def load(p):
    with open(p) as f:
        return json.load(f)


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def split_summary(idx):
    by_sim = collections.defaultdict(list)
    for e in idx:
        by_sim[e["sim_id"]].append(e["time_id"])
    tids = sorted(e["time_id"] for e in idx)
    strides = collections.Counter()
    for ts in by_sim.values():
        ts = sorted(ts)
        strides.update(b - a for a, b in zip(ts, ts[1:]))
    return {
        "n_samples": len(idx),
        "n_sim": len(by_sim),
        "sim_ids": sorted(by_sim),
        "time_id_min": tids[0] if tids else None,
        "time_id_max": tids[-1] if tids else None,
        "stride_hist_top5": strides.most_common(5),
        "samples_per_sim_min_max": [min(map(len, by_sim.values())), max(map(len, by_sim.values()))] if by_sim else None,
    }, by_sim


def window_overlap(a_by_sim, b_by_sim, horizon):
    """For every b sample, does its [t, t+horizon) frame window intersect any a window of the same sim?"""
    shared = sorted(set(a_by_sim) & set(b_by_sim))
    n_b_overlap, frames_b, frames_shared = 0, 0, 0
    for s in shared:
        a_frames = set()
        for t in a_by_sim[s]:
            a_frames.update(range(t, t + horizon))
        b_frames = set()
        for t in b_by_sim[s]:
            w = set(range(t, t + horizon))
            b_frames |= w
            if w & a_frames:
                n_b_overlap += 1
        frames_b += len(b_frames)
        frames_shared += len(b_frames & a_frames)
    n_b = sum(len(v) for v in b_by_sim.values())
    return {
        "shared_sim_ids": shared,
        "n_shared_sim": len(shared),
        "b_samples_with_window_overlap": n_b_overlap,
        "b_samples_total": n_b,
        "frac_b_samples_overlapping": n_b_overlap / n_b if n_b else None,
        "b_unique_frames_in_shared_sims": frames_b,
        "b_frames_also_in_a": frames_shared,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--old", required=True)
    ap.add_argument("--new", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--horizon", type=int, default=40, help="T_in + T_out at N_ar=1")
    args = ap.parse_args()

    res = {"horizon": args.horizon, "index": {}, "params": {}, "overlap": {}}
    for typ in TYPES:
        by = {}
        for sp in SPLITS:
            rel = f"cylinder/hf_dataset/{sp}_index_{typ}.json"
            po, pn = os.path.join(args.old, rel), os.path.join(args.new, rel)
            so, bo = split_summary(load(po))
            sn, bn = split_summary(load(pn))
            by[sp] = bn
            res["index"][f"{sp}_{typ}"] = {
                "sha256_old": sha(po), "sha256_new": sha(pn), "identical_bytes": sha(po) == sha(pn),
                "old": {k: v for k, v in so.items() if k != "sim_ids"},
                "new": {k: v for k, v in sn.items() if k != "sim_ids"},
                "sim_ids_added": sorted(set(sn["sim_ids"]) - set(so["sim_ids"])),
                "sim_ids_removed": sorted(set(so["sim_ids"]) - set(sn["sim_ids"])),
                "sim_ids_new": sn["sim_ids"],
            }
        for a, b in [("val", "test"), ("train", "val"), ("train", "test")]:
            res["overlap"][f"{typ}:{a}_vs_{b}"] = window_overlap(by[a], by[b], args.horizon)
        for pr in PARAMS:
            rel = f"cylinder/{pr}_{typ}.json"
            po, pn = os.path.join(args.old, rel), os.path.join(args.new, rel)
            o, n = load(po), load(pn)
            res["params"][f"{pr}_{typ}"] = {
                "identical_bytes": sha(po) == sha(pn),
                "n_old": len(o), "n_new": len(n),
                "keys_added": sorted(set(n) - set(o)), "keys_removed": sorted(set(o) - set(n)),
                "keys_new": sorted(n),
            }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(res, f, indent=1)

    for k, v in res["index"].items():
        print(f"{k:16s} identical={v['identical_bytes']} n={v['old']['n_samples']}->{v['new']['n_samples']} "
              f"n_sim={v['new']['n_sim']} t=[{v['new']['time_id_min']},{v['new']['time_id_max']}] "
              f"stride={v['new']['stride_hist_top5'][:2]} per_sim={v['new']['samples_per_sim_min_max']}")
    for k, v in res["params"].items():
        print(f"{k:32s} identical={v['identical_bytes']} n={v['n_old']}->{v['n_new']}")
    for k, v in res["overlap"].items():
        print(f"{k:24s} shared_sim={v['n_shared_sim']} b_overlap={v['b_samples_with_window_overlap']}/{v['b_samples_total']} "
              f"frames_shared={v['b_frames_also_in_a']}/{v['b_unique_frames_in_shared_sims']}")


if __name__ == "__main__":
    main()
