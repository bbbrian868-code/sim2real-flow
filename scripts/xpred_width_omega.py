"""dit_xpred_width Step 6 (D32): Ω and grid spacing of every real test trajectory, taken from CLFT Phase 1.

Ω comes from clft_work/scripts/run_E.geometry, used read-only: Phase D maps and per-trajectory records, >= 4 cells
from the edge, dilated obstacle and PIV-invalid cells excluded; dx, dy are signed (from the coordinate arrays).
Two checks are printed and stored:
  * frames: the official test dataset target frames (u, v) equal the CLFT arrow fields at time_id + 20 ..
    (same orientation, time and channel order)
  * definition: eval_xpred_width.div_rel equals clft.fields.divergence_stats(..., v_sign=1)["div_rel"] (float64)

  python scripts/xpred_width_omega.py --out OMEGA.npz
"""
import argparse, json, os, subprocess, sys

import numpy as np
import torch

CLFT = "/work/b314513067/clft_work"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, CLFT)
sys.path.insert(0, os.path.join(CLFT, "scripts"))

from pilfm.official_eval import Env  # noqa: E402
from scripts.eval_xpred_width import div_rel  # noqa: E402
import run_E  # noqa: E402
from clft.data import CylinderArrow  # noqa: E402
from clft.fields import divergence_stats  # noqa: E402

T_IN = 20


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--data-root", default="/work/b314513067/pi-lfm/data_v2.0.1")
    ap.add_argument("--n-check", type=int, default=8, help="test samples used for the two checks")
    a = ap.parse_args()
    env = Env(a.data_root, need_test=True)
    idx = env.test._indices
    sids = sorted({e["sim_id"] for e in idx})
    recs = {(r["split"], r["sim_id"]): r for r in json.load(open(os.path.join(run_E.DRES, "per_traj.json")))}
    geo = {s: run_E.geometry("real", s, recs, 1) for s in sids}
    om = np.stack([geo[s]["omega"] for s in sids])
    dx = np.array([geo[s]["dx"] for s in sids]); dy = np.array([geo[s]["dy"] for s in sids])

    arrow = CylinderArrow("real")
    rng = np.random.default_rng(0)
    frame_diff, def_diff = 0.0, 0.0
    for i in rng.choice(len(idx), a.n_check, replace=False):
        e = idx[int(i)]
        _, y = env.test[int(i)]
        f = arrow.fields(e["sim_id"]); u, v = f["u"], f["v"]
        t0 = e["time_id"] + T_IN
        ref = np.stack([u[t0:t0 + y.shape[0]], v[t0:t0 + y.shape[0]]], -1)
        frame_diff = max(frame_diff, float(np.abs(y[..., :2].numpy() - ref).max()))
        g = geo[e["sim_id"]]
        uu, vv = y[..., 0].double().numpy(), y[..., 1].double().numpy()
        want = divergence_stats(uu, vv, g["dx"], g["dy"], g["omega"], v_sign=1.0)["div_rel"]
        got = div_rel(torch.from_numpy(uu), torch.from_numpy(vv), g["dx"], g["dy"], torch.from_numpy(g["omega"]))
        def_diff = max(def_diff, float(np.abs(got.numpy() - want).max()))
    clft_commit = subprocess.run(["git", "-C", CLFT, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    checks = {"n_check": a.n_check, "max_abs_frame_diff": frame_diff, "max_abs_div_rel_diff": def_diff,
              "clft_commit": clft_commit, "source": "clft_work/scripts/run_E.geometry('real', sid, per_traj, 1)",
              "omega_npx_median": int(np.median(om.reshape(len(sids), -1).sum(1)))}
    np.savez_compressed(a.out, sim_id=np.array(sids), omega=om, dx=dx, dy=dy, checks=json.dumps(checks))
    print(json.dumps({**checks, "n_traj": len(sids), "dx_mean": float(dx.mean()), "dy_mean": float(dy.mean())}, indent=1))


if __name__ == "__main__":
    main()
