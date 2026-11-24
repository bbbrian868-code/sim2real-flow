"""leray_xpred (docs/leray_xpred, DECISIONS L6, L7, L16, L17): per-trajectory geometry for the Leray projection.

From clft_work/scripts/run_E.geometry (read-only), the same source as the D32 Ω file of scripts/xpred_width_omega.py:
  omega         Ω (D32): >= 4 cells from the edge, not in the dilated obstacle, not PIV-invalid
  omega_strict  Ω minus the large-scale artefact zones (real only; = Ω for sim)
  obstacle      dilated obstacle that Ω excludes: dilate(fill mask, 2) | disk(centre, D/2 + 2h)
  body          cylinder body = geometric disk of radius D/2 at the Phase D centre (L17)
  omega_wall    >= 4 cells from the edge, not body, not PIV-invalid (= Ω plus the fluid cells of the dilated band, L17)
  invalid       PIV-invalid cells (sim: none)
  dx, dy        signed grid spacing in metres (from the coordinate arrays)
  s_raw         scale of dv/dy in the divergence with the stored v: real +1, sim -1 (stored sim v has the opposite
                sign, Phase 1 D-D03; L16)
  r             real: per-trajectory CLFT beta/alpha (vo_consistency_real.json, row[2] / row[1], the value of the v/r
                metric variant of scripts/realdata_checks.py, L6); sim: NaN
  s_vr          real: 1 / r (D_s, L6); sim: NaN (D_s is not used on sim, L5)
real: the 92 real trajectories (test and val use the same ones), 64x128. sim: the sim test trajectories at the official
runtime grid [::2, ::2] (64x128). Checks stored in the npz: real Ω bitwise equal to the D32 npz; omega_wall equals
Ω | (obstacle & ~body & ~invalid & inside the border); obstacle covers the body.

  python scripts/leray_xpred_geometry.py --split real --out GEOM_real.npz
  python scripts/leray_xpred_geometry.py --split sim --out GEOM_sim-test.npz
"""
import argparse, json, os, subprocess, sys

import numpy as np

CLFT = "/work/b314513067/clft_work"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, CLFT)
sys.path.insert(0, os.path.join(CLFT, "scripts"))

import run_E  # noqa: E402
from clft.data import D_CYL, re_of  # noqa: E402
from clft.fields import disk  # noqa: E402

HF = "/work/b314513067/clft_data/cylinder/hf_dataset"
VO = os.path.join(CLFT, "results/2026-10-22_clft-phase1_cylinder-v2.0.1/D_data-validation/vo_consistency_real.json")
OMEGA_D32 = "/work/b314513067/pi-lfm/results/fm/2026-10-26_dit-xpred-width/S6_eval-test/omega_real-test_clft-phase1.npz"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, choices=["real", "sim"])
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    recs = {(r["split"], r["sim_id"]): r for r in json.load(open(os.path.join(run_E.DRES, "per_traj.json")))}
    if a.split == "real":
        split, sub = "real", 1
        sids = sorted({e["sim_id"] for e in json.load(open(os.path.join(HF, "test_index_real.json")))}, key=re_of)
        vo = {r[0]: r[2] / r[1] for r in json.load(open(VO))}
        z = np.load(OMEGA_D32)
        ref = {str(s): z["omega"][i] for i, s in enumerate(z["sim_id"])}
    else:
        split, sub = "numerical", 2
        sids = sorted({e["sim_id"] for e in json.load(open(os.path.join(HF, "test_index_numerical.json")))}, key=re_of)
    keys = ("omega", "omega_strict", "obstacle", "body", "omega_wall", "invalid")
    arr = {k: [] for k in keys}
    dx, dy, r = [], [], []
    checks = {"omega_eq_D32": [], "omega_wall_def": [], "obstacle_covers_body": []}
    for sid in sids:
        g = run_E.geometry(split, sid, recs, sub)
        H, W = g["omega"].shape
        border = np.zeros((H, W), bool)
        border[run_E.BORDER:H - run_E.BORDER, run_E.BORDER:W - run_E.BORDER] = True
        inv = np.asarray(g["invalid"], bool)
        body = disk((H, W), g["x"], g["y"], g["cx"], g["cy"], D_CYL / 2)
        omw = border & ~body & ~inv
        m = dict(omega=g["omega"], omega_strict=g["omega_strict"], obstacle=g["obstacle"], body=body, omega_wall=omw,
                 invalid=inv)
        for k in keys:
            arr[k].append(np.asarray(m[k], bool))
        dx.append(g["dx"]); dy.append(g["dy"])
        r.append(vo[sid] if a.split == "real" else np.nan)
        checks["omega_wall_def"].append(bool(np.array_equal(omw, g["omega"] | (g["obstacle"] & ~body & ~inv & border))))
        checks["obstacle_covers_body"].append(bool(not np.any(body & ~g["obstacle"])))
        if a.split == "real":
            checks["omega_eq_D32"].append(bool(np.array_equal(g["omega"], ref[sid])))
    r = np.array(r, np.float64)
    out = {k: np.stack(v) for k, v in arr.items()}
    summ = {k: all(v) for k, v in checks.items() if v}
    summ.update(split=a.split, n_traj=len(sids), source="clft_work/scripts/run_E.geometry(%r, sid, per_traj, %d)" % (split, sub),
                clft_commit=subprocess.run(["git", "-C", CLFT, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip(),
                body="disk(centre, D/2) (L17)", s_raw=1 if a.split == "real" else -1,
                counts={k: [int(out[k].reshape(len(sids), -1).sum(1).min()), int(out[k].reshape(len(sids), -1).sum(1).max())]
                        for k in keys})
    if a.split == "real":
        summ["r"] = {"source": VO, "min": float(r.min()), "median": float(np.median(r)), "max": float(r.max())}
    np.savez_compressed(a.out, sim_id=np.array(sids), dx=np.array(dx), dy=np.array(dy), r=r,
                        s_raw=np.full(len(sids), 1.0 if a.split == "real" else -1.0),
                        s_vr=(1.0 / r) if a.split == "real" else np.full(len(sids), np.nan),
                        checks=json.dumps(summ), **out)
    print(json.dumps(summ, indent=1))
    if not all(v for k, v in summ.items() if isinstance(v, bool)):
        sys.exit("geometry check failed")


if __name__ == "__main__":
    main()
