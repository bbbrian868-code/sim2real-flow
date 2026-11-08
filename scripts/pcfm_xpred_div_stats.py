"""pcfm_xpred Step 2: div_rel statistics of the real TRAIN and VAL targets (test is not read), used to choose the target
value tau of the per-frame div_rel constraint (C2). GPU job.

  python scripts/pcfm_xpred_div_stats.py --omega OMEGA.npz --out OUT.json

div_rel per frame follows D32 exactly (scripts/eval_xpred_width.div_rel, float64, Ω and grid spacing of each trajectory
from the npz of scripts/xpred_width_omega.py). It is computed once for every frame of every real trajectory (CLFT arrow
fields; Step 0 / S6a: loader frames == arrow frames) and then indexed by the official split indices (N_ar = 1):
target frames of sample (sim, time_id) = time_id + 20 .. time_id + 39, input frames = time_id .. time_id + 19.

Reported:
  distribution  per sample-frame (as the evaluation counts them), per output-frame position 1..20, per trajectory
  tau options   how far each candidate tau is from the true per-frame div_rel of the VAL targets (mean / median absolute
                difference, and the share of frames within 0.01 / 0.02); options built from TRAIN only, or from TRAIN+VAL
                (then the VAL score is in-sample, marked), or from the sample's own input frames (available at inference)
"""
import argparse, json, os, sys

import numpy as np
import torch

CODE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLFT = "/work/b314513067/clft_work"
DATA = "/work/b314513067/pi-lfm/data_v2.0.1"
sys.path.insert(0, CODE); sys.path.insert(0, CLFT)
from scripts.eval_xpred_width import div_rel  # noqa: E402
from clft.data import CylinderArrow  # noqa: E402

T_IN = T_OUT = 20


def q(v):
    v = np.asarray(v)
    return {"n": int(v.size), "median": float(np.median(v)), "q25": float(np.quantile(v, .25)), "q75": float(np.quantile(v, .75)),
            "mean": float(v.mean()), "std": float(v.std()), "min": float(v.min()), "max": float(v.max())}


def score(pred, true):
    d = np.abs(pred - true)
    return {"mean_abs": float(d.mean()), "median_abs": float(np.median(d)), "frac_within_0.01": float((d <= .01).mean()),
            "frac_within_0.02": float((d <= .02).mean())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--omega", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    dev = "cuda:0"
    z = np.load(a.omega)
    geo = {str(s): (torch.from_numpy(z["omega"][i]).to(dev), float(z["dx"][i]), float(z["dy"][i])) for i, s in enumerate(z["sim_id"])}
    idx = {s: json.load(open(os.path.join(DATA, "cylinder", "hf_dataset", f"{s}_index_real.json"))) for s in ("train", "val")}
    sims = sorted({e["sim_id"] for s in idx for e in idx[s]})
    arrow = CylinderArrow("real")
    frame_div = {}
    for sid in sims:
        f = arrow.fields(sid)
        u = torch.from_numpy(np.ascontiguousarray(f["u"])).to(dev).double()
        v = torch.from_numpy(np.ascontiguousarray(f["v"])).to(dev).double()
        om, dx, dy = geo[sid]
        frame_div[sid] = torch.cat([div_rel(u[i:i + 500], v[i:i + 500], dx, dy, om) for i in range(0, u.shape[0], 500)]).cpu().numpy()
        print(sid, len(frame_div[sid]), float(np.median(frame_div[sid])), flush=True)

    def windows(split):
        tgt = np.stack([frame_div[e["sim_id"]][e["time_id"] + T_IN: e["time_id"] + T_IN + T_OUT] for e in idx[split]])
        inp = np.stack([frame_div[e["sim_id"]][e["time_id"]: e["time_id"] + T_IN] for e in idx[split]])
        return tgt, inp, np.array([e["sim_id"] for e in idx[split]])
    W = {s: windows(s) for s in ("train", "val")}
    out = {"definition": "D32 (scripts/eval_xpred_width.div_rel, float64)", "omega": a.omega,
           "n_samples": {s: int(len(idx[s])) for s in idx}, "n_traj": {s: int(len(set(W[s][2]))) for s in W}}

    dist = {}
    for name, splits in (("train", ("train",)), ("val", ("val",)), ("train+val", ("train", "val"))):
        tgt = np.concatenate([W[s][0] for s in splits]); sid = np.concatenate([W[s][2] for s in splits])
        per_traj = [float(np.median(tgt[sid == s])) for s in np.unique(sid)]
        dist[name] = {"per_frame": q(tgt.ravel()), "per_traj_median": q(per_traj),
                      "median_of_traj_medians": float(np.median(per_traj)),
                      "per_position_median": [float(np.median(tgt[:, k])) for k in range(T_OUT)],
                      "within_sample_std_median": float(np.median(tgt.std(1))),
                      "within_traj_std_median": float(np.median([tgt[sid == s].std() for s in np.unique(sid)]))}
    out["distribution"] = dist
    out["test_target_median_reference_only"] = {"value": 0.6812856194069074, "source": "dit_xpred_width S6 JSON (divergence.target.median_frames); not used for any choice"}

    # tau options, scored on VAL targets
    trT, trI, trS = W["train"]; vaT, vaI, vaS = W["val"]
    tvT = np.concatenate([trT, vaT]); tvS = np.concatenate([trS, vaS])
    opts = {}
    g_tr = float(np.median(trT)); g_tv = float(np.median(tvT))
    opts["const_train_median"] = {"tau": g_tr, "fit": "train", **score(np.full_like(vaT, g_tr), vaT)}
    opts["const_trainval_median"] = {"tau": g_tv, "fit": "train+val (val score in-sample)", **score(np.full_like(vaT, g_tv), vaT)}
    pos = np.median(trT, 0)
    opts["per_position_train_median"] = {"tau": pos.tolist(), "fit": "train", **score(np.broadcast_to(pos, vaT.shape), vaT)}
    traj_tr = {s: float(np.median(trT[trS == s])) for s in np.unique(trS)}
    pt = np.array([traj_tr.get(s, g_tr) for s in vaS])[:, None]
    opts["per_traj_train_median"] = {"fit": "train; val trajectories absent from train use the train median",
                                     "n_val_samples_fallback": int(sum(s not in traj_tr for s in vaS)),
                                     **score(np.broadcast_to(pt, vaT.shape), vaT)}
    m_in = np.median(vaI, 1, keepdims=True)
    opts["own_input_median"] = {"fit": "none (median of the sample's 20 input frames, known at inference)",
                                **score(np.broadcast_to(m_in, vaT.shape), vaT)}
    opts["own_input_last"] = {"fit": "none (last input frame)", **score(np.broadcast_to(vaI[:, -1:], vaT.shape), vaT)}
    opts["oracle_own_target_median"] = {"fit": "uses the target itself: lower bound for any per-sample constant, NOT usable",
                                        **score(np.broadcast_to(np.median(vaT, 1, keepdims=True), vaT.shape), vaT)}
    out["tau_options_scored_on_val"] = opts
    out["corr_val_input_median_vs_target_median"] = float(np.corrcoef(np.median(vaI, 1), np.median(vaT, 1))[0, 1])
    out["corr_train_input_median_vs_target_median"] = float(np.corrcoef(np.median(trI, 1), np.median(trT, 1))[0, 1])
    json.dump(out, open(a.out, "w"), indent=1)
    print(json.dumps({k: v for k, v in out.items() if k != "distribution"}, indent=1))


if __name__ == "__main__":
    main()
