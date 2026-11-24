"""leray_xpred Step 5 summary (N_ar = 10, K = 1, real test, 4672 samples): one CSV row per model x group, two figures
per model.

Groups: A0 (Euler), A2' / A3' (C2 per step / at the end, implied retreat), the 8 Leray groups and P-raw.
Sources: Step 5 runs (S5_eval-nar10/eval) for every Leray group, P-raw and the R6num A0 / A2' / A3'; R6 A0 / A2' / A3' from
the L29 reruns (S5_eval-nar10/rerun, with the region metrics) when present, otherwise the existing R6 runs for A0 (dit_xpred_width S6, run without --omega: no div_rel, no region metrics) and A2' / A3' (pcfm_xpred S8: div_rel,
no region metrics, no v/r div_rel) (L13). Missing quantities are left empty.
Per row: 200-frame Rel L2 (official, and the L8 region variants when present), per-round Rel L2 (official, excl_body),
per-round div_rel median (D32 and v/r, prediction and truth), constraint residual per round (Leray: max |D_f u| and own
div_rel recomputed by the evaluation; C2: ||h||), CG / Newton diagnostics.
Figures (summary/figures/): per model, per-round Rel L2 (official, excl_body) and per-round div_rel median (D32, v/r),
groups in three panels (reference and C2 / Leray at the end / Leray every step).
  python scripts/summarize_leray_xpred_nar10.py --s5 S5_DIR
"""
import argparse, csv, json, os

import numpy as np

X = "/work/b314513067/pi-lfm/results/fm"
NAMES = {"R6": "R6_xpred_w512", "R6num": "R6num_xpred_w512"}
LERAY = [(sc, w, when) for sc in ("raw", "s") for w in ("free", "wall") for when in ("final", "step")]
GROUPS = ["A0", "A2'", "A3'"] + [f"L-{sc}-{w}-{when}" for sc, w, when in LERAY] + ["P-raw"]
C = {"blue": "#2a78d6", "orange": "#eb6834", "aqua": "#1baf7a", "yellow": "#eda100", "magenta": "#e87ba4",
     "green": "#008300", "violet": "#6250d6", "red": "#e34948"}
INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
VARIANT_COLOR = {"raw-free": C["blue"], "raw-wall": C["orange"], "s-free": C["aqua"], "s-wall": C["violet"]}
GC = {"A0": C["blue"], "A2'": C["yellow"], "A3'": C["magenta"], "P-raw": C["green"]}


def tag(g):
    if g in GROUPS[:3]:
        return {"A0": "euler", "A2'": "pcfm-implied", "A3'": "final_only-implied"}[g]
    if g == "P-raw":
        return "final_only-implied-partial-raw"
    _, sc, w, when = g.split("-")
    return f"{'pcfm' if when == 'step' else 'final_only'}-implied-leray-{'raw' if sc == 'raw' else 'vr'}-{w}"


def path(s5, m, g):
    n, t = NAMES[m], tag(g)
    p = os.path.join(s5, "eval", f"{n}_nar10_{t}20_K1.json")
    rr = os.path.join(s5, "rerun", f"{n}_nar10_{t}20_K1.json")  # L29: reruns with the region metrics, when present
    if m == "R6" and g in ("A0", "A2'", "A3'") and os.path.exists(rr):
        p = rr
    elif m == "R6" and g == "A0":
        p = f"{X}/2026-10-26_dit-xpred-width/S6_eval-test/eval/{n}_nar10_euler20_K1.json"
    elif m == "R6" and g in ("A2'", "A3'"):
        p = f"{X}/2026-11-08_pcfm-xpred/S8_implied-noise-rollout10-test/eval/{n}_nar10_{t}20_K1.json"
    return p


def get(d, *ks):
    for k in ks:
        if not isinstance(d, dict) or k not in d:
            return None
        d = d[k]
    return d


def style(ax):
    ax.set_facecolor("#fcfcfb")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(AXIS)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.grid(color=GRID, lw=0.6)
    ax.set_axisbelow(True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--s5", required=True)
    a = ap.parse_args()
    out_dir = os.path.join(a.s5, "summary"); fig_dir = os.path.join(out_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    rows, R, missing = [], {}, []
    for m in NAMES:
        for g in GROUPS:
            p = path(a.s5, m, g)
            if not os.path.exists(p):
                missing.append(p); continue
            r = json.load(open(p)); R[(m, g)] = r
            row = {"model": m, "group": g, "sampler_tag": tag(g), "eval_file": os.path.relpath(p, X),
                   "n_samples": r["n_samples"], "eval_seconds": r["seconds"]}
            for k, v in r["single"].items():
                row[f"single_{k}"] = v
            row["crps"] = r["crps"]
            for reg in ("all", "excl_body", "excl_obstacle", "strict"):
                for suf in ("", "_vr"):
                    row[f"relL2_{reg}{suf}_single"] = get(r, "region_metrics", reg + suf, "single", "rel_l2")
            pr = {"relL2_official": get(r, "per_round", "single", "rel_l2"),
                  "relL2_excl_body": get(r, "region_metrics", "excl_body", "single", "per_round_rel_l2"),
                  "div_rel_d32_median": get(r, "divergence", "single", "per_round_median"),
                  "div_rel_d32_median_truth": get(r, "divergence", "target", "per_round_median"),
                  "div_rel_vr_median": get(r, "divergence_vr", "single", "per_round_median"),
                  "div_rel_vr_median_truth": get(r, "divergence_vr", "target", "per_round_median"),
                  "leray_max_abs_Du_median": get(r, "leray_residual", "single", "per_round_max_abs_median"),
                  "leray_div_rel_median": get(r, "leray_residual", "single", "per_round_div_rel_median"),
                  "c2_h_norm_median": get(r, "constraint_residual", "single", "per_round_norm_median")}
            for k, v in pr.items():
                for i in range(10):
                    row[f"r{i + 1}_{k}"] = None if v is None else v[i]
            row["div_rel_d32_median_all_frames"] = get(r, "divergence", "single", "median_frames")
            row["div_rel_vr_median_all_frames"] = get(r, "divergence_vr", "single", "median_frames")
            sd = r.get("sampler_diagnostics") or {}
            row["cg_iters_final_median"] = get(sd, "leray_final", "cg_iters", "median")
            row["cg_iters_final_max"] = get(sd, "leray_final", "cg_iters", "max")
            row["cg_not_converged_frames_final"] = get(sd, "leray_final", "not_converged_frames")
            row["cg_nan_frames_final"] = get(sd, "leray_final", "nan_frames")
            row["cg_iters_per_step_max"] = (max(sd["leray_per_step"]["iters"]["max"]) if "leray_per_step" in sd else None)
            row["cg_not_converged_frames_per_step"] = get(sd, "leray_per_step", "not_converged_frames")
            row["partial_alpha_median"] = get(sd, "partial_final", "alpha", "median")
            row["partial_unreachable_frames"] = get(sd, "partial_final", "unreachable_frames")
            row["newton_iters_max"] = get(sd, "newton_iters", "max")
            row["newton_not_converged"] = sd.get("not_converged") if "newton_iters" in sd else None
            row["newton_nan"] = sd.get("nan") if "newton_iters" in sd else None
            rows.append(row)
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(os.path.join(out_dir, "leray_xpred_nar10.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, keys); w.writeheader(); w.writerows(rows)
    json.dump({"rows": rows, "missing": missing}, open(os.path.join(out_dir, "leray_xpred_nar10.json"), "w"), indent=1)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 8, "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "text.color": INK})
    by = {(r["model"], r["group"]): r for r in rows}
    panels = [("reference and C2", ["A0", "A2'", "A3'", "P-raw"]),
              ("Leray, at the end", [f"L-{v}-final" for v in VARIANT_COLOR]),
              ("Leray, every step", [f"L-{v}-step" for v in VARIANT_COLOR])]
    rd = np.arange(1, 11)

    def series(m, g, key):
        r = by.get((m, g))
        if r is None:
            return None
        v = [r[f"r{i}_{key}"] for i in rd]
        return None if any(x is None for x in v) else np.array(v, float)

    for m in NAMES:
        up = "  (sim-trained, upper bound)" if m == "R6num" else ""
        for fname, specs, ylog in (
                ("per-round-relL2", [("relL2_official", "Rel L2, official (all cells)"),
                                     ("relL2_excl_body", "Rel L2, excl. cylinder disk")], False),
                ("per-round-div-rel", [("div_rel_d32_median", "div_rel median, v as stored (D32)"),
                                       ("div_rel_vr_median", "div_rel median, v / r")], True)):
            fig, axs = plt.subplots(2, 3, figsize=(12, 7), sharex=True, sharey="row")
            for i, (key, ylab) in enumerate(specs):
                for j, (ttl, gs) in enumerate(panels):
                    ax = axs[i, j]; style(ax)
                    if ylog:
                        t = next((series(m, g, key + "_truth") for g in GROUPS if series(m, g, key + "_truth") is not None), None)
                        if t is not None:
                            ax.plot(rd, t, color=INK, lw=2, label="truth")
                    if j and series(m, "A0", key) is not None:
                        ax.plot(rd, series(m, "A0", key), color=MUTED, lw=1.5, label="A0")
                    for g in gs:
                        v = series(m, g, key)
                        if v is None:
                            continue
                        col = GC.get(g) or VARIANT_COLOR["-".join(g.split("-")[1:3])]
                        ax.plot(rd, v, color=col, lw=2, marker="o", ms=3, label=g)
                    if ylog:
                        ax.set_yscale("log")
                    ax.set_title(ttl, loc="left", fontsize=9)
                    if j == 0:
                        ax.set_ylabel(ylab)
                    if i == 1:
                        ax.set_xlabel("round (20 frames each)")
                    ax.set_xticks(rd)
                    ax.legend(frameon=False, fontsize=7)
            fig.suptitle(f"{m} — real test, N_ar = 10, K = 1{up}", x=0.01, ha="left", color=INK)
            fig.tight_layout(); fig.savefig(os.path.join(fig_dir, f"fig_{fname}_{m}.png"), dpi=150)
            plt.close(fig)
    print(json.dumps({"rows": len(rows), "missing": missing}, indent=1))


if __name__ == "__main__":
    main()
