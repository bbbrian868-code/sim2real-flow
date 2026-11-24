"""leray_xpred Step 4 summary (N_ar = 1, K = 8, real test): one CSV row per model x group, and four figures.

Groups: A0 (Euler), A2' / A3' (C2 per step / at the end, implied retreat), the 8 Leray groups and P-raw. A0 / A2' / A3'
of R6 and A0 of R6num are the L22 / L26 reruns (identical to the existing results, with the region metrics); A2' / A3'
of R6num are new in Step 4.
Timing: bench (first 64 real test inputs, batch 1 and 64), all groups re-timed in S4_eval-nar1/bench (L28); fallbacks
if a file is missing: A0 from dit_xpred_width S6 bench (Euler, random inputs), R6 A2' / A3' from pcfm_xpred S7 bench.
Figures (summary/figures/):
  1 per model: Rel L2 (official = all cells, and excl_body), single sample and K = 8 mean, with the truth-oracle cost of
    the matching Leray variant (Step 2) as a reference tick; div_rel median (D32 and v/r) with the truth's value
  2 constraint residual per sample (log): Leray groups max |D_f u| recomputed by the evaluation; C2 groups ||h||
  3 radial energy spectrum (first test batch, single sample) by group, with the truth
  4 Rel L2 (K = 8 mean, official) vs inference time per frame (batch 1)
  python scripts/summarize_leray_xpred_nar1.py --s4 S4_DIR --oracle S2_DIR/oracle_real-test.json
"""
import argparse, csv, json, os

import numpy as np

X = "/work/b314513067/pi-lfm/results/fm"
NAMES = {"R6": "R6_xpred_w512", "R6num": "R6num_xpred_w512"}
LERAY = [(sc, w, when) for sc in ("raw", "s") for w in ("free", "wall") for when in ("final", "step")]
GROUPS = ["A0", "A2'", "A3'"] + [f"L-{sc}-{w}-{when}" for sc, w, when in LERAY] + ["P-raw"]
ORACLE = {f"L-{sc}-{w}-{when}": f"D_{sc}-{w}" for sc, w, when in LERAY}
# categorical slots in fixed order (dataviz reference palette, light)
C = {"blue": "#2a78d6", "orange": "#eb6834", "aqua": "#1baf7a", "yellow": "#eda100", "magenta": "#e87ba4",
     "green": "#008300", "violet": "#6250d6", "red": "#e34948"}
INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
VARIANT_COLOR = {"raw-free": C["blue"], "raw-wall": C["orange"], "s-free": C["aqua"], "s-wall": C["violet"]}


def tag(g):
    if g == "A0":
        return "euler"
    if g == "A2'":
        return "pcfm-implied"
    if g == "A3'":
        return "final_only-implied"
    if g == "P-raw":
        return "final_only-implied-partial-raw"
    _, sc, w, when = g.split("-")
    return f"{'pcfm' if when == 'step' else 'final_only'}-implied-leray-{'raw' if sc == 'raw' else 'vr'}-{w}"


def paths(s4, m, g):
    n, t = NAMES[m], tag(g)
    sub = "rerun" if (g == "A0" or (m == "R6" and g in ("A2'", "A3'"))) else "eval"
    ev = os.path.join(s4, sub, f"{n}_nar1_{t}20_K8.json")
    b = os.path.join(s4, "bench", f"bench_{n}_{t}.json")
    if g == "A0" and not os.path.exists(b):  # L28: re-timed in S4 bench when present
        b = f"{X}/2026-10-26_dit-xpred-width/S6_eval-test/bench/bench_R6_xpred_w512.json"
    elif not os.path.exists(b) and m == "R6" and g in ("A2'", "A3'"):
        b = f"{X}/2026-11-08_pcfm-xpred/S7_implied-noise-retreat/bench/bench_{n}_{t}.json"
    return ev, b


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
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--s4", required=True)
    ap.add_argument("--oracle", required=True)
    a = ap.parse_args()
    out_dir = os.path.join(a.s4, "summary"); fig_dir = os.path.join(out_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    orc = json.load(open(a.oracle))["variants"]
    rows, R, Z, missing = [], {}, {}, []
    for m in NAMES:
        for g in GROUPS:
            ev, bp = paths(a.s4, m, g)
            if not os.path.exists(ev):
                missing.append(ev); continue
            r = json.load(open(ev)); R[(m, g)] = r
            npz = ev[:-5] + "_per_sample.npz"
            Z[(m, g)] = dict(np.load(npz)) if os.path.exists(npz) else {}
            b = json.load(open(bp)) if os.path.exists(bp) else {}
            row = {"model": m, "group": g, "sampler_tag": tag(g), "eval_file": os.path.relpath(ev, X),
                   "ckpt_step": r["ckpt_step"], "n_samples": r["n_samples"], "eval_seconds": r["seconds"]}
            for part in ("single", "kmean"):
                for k, v in r[part].items():
                    row[f"{part}_{k}"] = v
            row["crps"] = r["crps"]; row["spread_std_mean"] = get(r, "spread", "std_mean")
            for reg in ("all", "excl_body", "excl_obstacle", "strict"):
                for suf in ("", "_vr"):
                    for part in ("single", "kmean"):
                        row[f"relL2_{reg}{suf}_{part}"] = get(r, "region_metrics", reg + suf, part, "rel_l2")
            for part in ("single", "kmean", "target"):
                row[f"div_rel_d32_median_{part}"] = get(r, "divergence", part, "median_frames")
                row[f"div_rel_vr_median_{part}"] = get(r, "divergence_vr", part, "median_frames")
            lr = r.get("leray_residual")
            for part in ("single", "kmean"):
                for q in ("median", "q99", "max"):
                    row[f"leray_max_abs_Du_{q}_{part}"] = get(lr, part, "max_abs", q)
                row[f"leray_div_rel_median_{part}"] = get(lr, part, "div_rel", "median")
                row[f"c2_h_norm_median_{part}"] = get(r, "constraint_residual", part, "norm", "median")
                row[f"c2_h_norm_q99_{part}"] = get(r, "constraint_residual", part, "norm", "q99")
            sd = r.get("sampler_diagnostics") or {}
            row["cg_iters_final_median"] = get(sd, "leray_final", "cg_iters", "median")
            row["cg_iters_final_max"] = get(sd, "leray_final", "cg_iters", "max")
            row["cg_not_converged_frames_final"] = get(sd, "leray_final", "not_converged_frames")
            row["cg_nan_frames_final"] = get(sd, "leray_final", "nan_frames")
            row["cg_not_converged_frames_per_step"] = get(sd, "leray_per_step", "not_converged_frames")
            row["cg_iters_per_step_max"] = (max(sd["leray_per_step"]["iters"]["max"]) if "leray_per_step" in sd else None)
            row["partial_alpha_median"] = get(sd, "partial_final", "alpha", "median")
            row["partial_unreachable_frames"] = get(sd, "partial_final", "unreachable_frames")
            row["newton_iters_max"] = get(sd, "newton_iters", "max")
            row["newton_not_converged"] = sd.get("not_converged") if "newton_iters" in sd else None
            row["newton_nan"] = sd.get("nan") if "newton_iters" in sd else None
            row["bench_file"] = os.path.relpath(bp, X) if b else None
            row["bench_b1_ms_per_frame"] = b.get("call_b1_ms_per_frame")
            row["bench_b64_ms_per_sample_frame"] = (b["call_b64_ms_per_sample"] / 20) if b.get("call_b64_ms_per_sample") else None
            row["bench_b64_peak_mem_GiB"] = b.get("call_b64_peak_mem_GiB")
            o = orc.get(ORACLE.get(g, ""), {}).get("rel_l2_vs_truth", {})
            row["truth_oracle_relL2_all"] = o.get("all"); row["truth_oracle_relL2_excl_body"] = o.get("excl_body")
            rows.append(row)
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(os.path.join(out_dir, "leray_xpred_nar1.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, keys); w.writeheader(); w.writerows(rows)
    json.dump({"rows": rows, "missing": missing}, open(os.path.join(out_dir, "leray_xpred_nar1.json"), "w"), indent=1)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 8, "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "text.color": INK})
    by = {(r["model"], r["group"]): r for r in rows}

    # 1: Rel L2 and div_rel per group, per model
    for m in NAMES:
        gs = [g for g in GROUPS if (m, g) in by]
        x = np.arange(len(gs))
        fig, axs = plt.subplots(3, 1, figsize=(9, 8.5), sharex=True)
        for ax, part, ttl in ((axs[0], "single", "Rel L2, single sample"), (axs[1], "kmean", "Rel L2, K = 8 mean")):
            style(ax)
            for j, (reg, col, lab) in enumerate((("all", C["blue"], "official (all cells)"),
                                                 ("excl_body", C["orange"], "excl. cylinder disk"))):
                v = [by[(m, g)][f"relL2_{reg}_{part}"] for g in gs]
                v = [np.nan if t is None else t for t in v]
                ax.bar(x + (j - 0.5) * 0.36, v, 0.34, color=col, label=lab, edgecolor="#fcfcfb", lw=1)
            for i, g in enumerate(gs):
                o = by[(m, g)]["truth_oracle_relL2_all"]
                if o is not None:
                    ax.plot([i - 0.4, i + 0.4], [o, o], color=INK, lw=1.5)
            ax.plot([], [], color=INK, lw=1.5, label="truth projected (Step 2 oracle, official)")
            ax.set_title(ttl, loc="left", color=INK, fontsize=9); ax.set_ylabel("Rel L2")
            ax.legend(frameon=False, fontsize=7, loc="upper left")
        ax = axs[2]; style(ax)
        for j, (k, col, lab) in enumerate((("d32", C["blue"], "div_rel, v as stored (D32)"),
                                           ("vr", C["aqua"], "div_rel, v / r"))):
            v = [by[(m, g)][f"div_rel_{k}_median_single"] for g in gs]
            ax.bar(x + (j - 0.5) * 0.36, [np.nan if t is None else t for t in v], 0.34, color=col, label=lab,
                   edgecolor="#fcfcfb", lw=1)
            t = by[(m, gs[0])][f"div_rel_{k}_median_target"]
            if t is not None:
                ax.axhline(t, color=col, lw=1, ls="--")
        ax.plot([], [], color=MUTED, lw=1, ls="--", label="truth (dashed, same colour)")
        ax.set_yscale("log"); ax.set_ylabel("div_rel median (single)")
        ax.set_title("div_rel on Ω, single sample, recomputed by the evaluation", loc="left", color=INK, fontsize=9)
        ax.legend(frameon=False, fontsize=7, loc="lower left")
        ax.set_xticks(x); ax.set_xticklabels(gs, rotation=35, ha="right")
        fig.suptitle(f"{m} — real test, N_ar = 1, K = 8" + ("  (sim-trained, upper bound)" if m == "R6num" else ""),
                     x=0.01, ha="left", color=INK)
        fig.tight_layout(); fig.savefig(os.path.join(fig_dir, f"fig1_relL2-and-div-rel_by-group_{m}.png"), dpi=150)
        plt.close(fig)

    # 2: constraint residual per sample
    fig, axs = plt.subplots(1, 3, figsize=(12, 4), gridspec_kw={"width_ratios": [4, 4, 1.6]})
    for ax, m in zip(axs[:2], NAMES):
        style(ax)
        gs = [g for g in GROUPS if g.startswith("L-") and "leray_max_abs_single" in Z.get((m, g), {})]
        data = [Z[(m, g)]["leray_max_abs_single"] for g in gs]
        if data:
            bp = ax.boxplot(data, whis=(1, 99), showfliers=False, patch_artist=True, widths=0.5)
            for p, g in zip(bp["boxes"], gs):
                p.set_facecolor(VARIANT_COLOR["-".join(g.split("-")[1:3])]); p.set_edgecolor(INK2)
            for k in ("medians",):
                for l in bp[k]:
                    l.set_color(INK)
            ax.set_xticks(range(1, len(gs) + 1)); ax.set_xticklabels(gs, rotation=35, ha="right")
        ax.set_yscale("log"); ax.set_ylabel("max |D_f u| per sample (1/s)")
        ax.set_title(f"{m}: Leray residual after the evaluation round trip", loc="left", fontsize=9)
    ax = axs[2]; style(ax)
    labs, data = [], []
    for m in NAMES:
        for g in ("A2'", "A3'"):
            h = Z.get((m, g), {}).get("h_norm_single")
            if h is not None:
                labs.append(f"{m} {g}"); data.append(h)
    if data:
        bp = ax.boxplot(data, whis=(1, 99), showfliers=False, patch_artist=True, widths=0.5)
        for p in bp["boxes"]:
            p.set_facecolor(C["yellow"]); p.set_edgecolor(INK2)
        for l in bp["medians"]:
            l.set_color(INK)
        ax.set_xticks(range(1, len(labs) + 1)); ax.set_xticklabels(labs, rotation=35, ha="right")
    ax.set_yscale("log"); ax.set_ylabel("C2 ||h|| per sample"); ax.set_title("C2 residual", loc="left", fontsize=9)
    fig.text(0.01, 0.01, "boxes: quartiles, whiskers: 1st-99th percentile, line: median", color=MUTED, fontsize=7)
    fig.tight_layout(rect=(0, 0.03, 1, 1)); fig.savefig(os.path.join(fig_dir, "fig2_constraint-residual_per-sample.png"), dpi=150)
    plt.close(fig)

    # 3: radial energy spectrum (first test batch), small multiples
    panels = [("reference and C2", ["A0", "A2'", "A3'", "P-raw"]),
              ("Leray, at the end", [f"L-{v}-final" for v in VARIANT_COLOR]),
              ("Leray, every step", [f"L-{v}-step" for v in VARIANT_COLOR])]
    gc = {"A0": C["blue"], "A2'": C["yellow"], "A3'": C["magenta"], "P-raw": C["green"]}
    fig, axs = plt.subplots(2, 3, figsize=(12, 7), sharex=True, sharey=True)
    for i, m in enumerate(NAMES):
        for j, (ttl, gs) in enumerate(panels):
            ax = axs[i, j]; style(ax)
            base = R.get((m, "A0")) or next((R[k] for k in R if k[0] == m), None)
            if base is None:
                continue
            k = np.array(base["spectrum"]["k_rad_per_m"])
            ax.plot(k, base["spectrum"]["target"], color=INK, lw=2, label="truth")
            if j:
                ax.plot(k, base["spectrum"]["single"], color=MUTED, lw=1.5, label="A0")
            for g in gs:
                if (m, g) not in R:
                    continue
                col = gc.get(g) or VARIANT_COLOR["-".join(g.split("-")[1:3])]
                ax.plot(k, R[(m, g)]["spectrum"]["single"], color=col, lw=1.5, label=g)
            ax.set_xscale("log"); ax.set_yscale("log")
            ax.set_title(f"{m}: {ttl}", loc="left", fontsize=9); ax.legend(frameon=False, fontsize=7)
            if i == 1:
                ax.set_xlabel("k (rad/m)")
            if j == 0:
                ax.set_ylabel("radial energy spectrum (single sample)")
    fig.tight_layout(); fig.savefig(os.path.join(fig_dir, "fig3_energy-spectrum_first-test-batch.png"), dpi=150)
    plt.close(fig)

    # 4: accuracy vs time per frame
    fam = lambda g: ("A0" if g == "A0" else "C2" if g in ("A2'", "A3'") else "partial" if g == "P-raw"
                     else "Leray, at the end" if g.endswith("final") else "Leray, every step")
    fc = {"A0": C["blue"], "C2": C["yellow"], "partial": C["green"], "Leray, at the end": C["orange"],
          "Leray, every step": C["violet"]}
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, m in zip(axs, NAMES):
        style(ax); ax.grid(axis="x", color=GRID, lw=0.6)
        seen = set()
        for g in GROUPS:
            r = by.get((m, g))
            if r is None or r["bench_b1_ms_per_frame"] is None:
                continue
            f = fam(g)
            ax.scatter(r["bench_b1_ms_per_frame"], r["kmean_rel_l2_error"], s=40, color=fc[f], edgecolor="#fcfcfb",
                       lw=1.5, zorder=3, label=None if f in seen else f)
            seen.add(f)
            ax.annotate(g, (r["bench_b1_ms_per_frame"], r["kmean_rel_l2_error"]),
                        xytext={"A2'": (-6, -11), "A3'": (4, -11)}.get(g, (4, 3)),
                        textcoords="offset points", fontsize=6.5, color=INK2)
        ax.set_xscale("log"); ax.set_xlabel("inference time per frame, batch 1 (ms, H100)")
        ax.set_ylabel("Rel L2, K = 8 mean (official)")
        ax.set_title(m + ("  (sim-trained, upper bound)" if m == "R6num" else ""), loc="left", fontsize=9)
        ax.legend(frameon=False, fontsize=7)
    fig.tight_layout(); fig.savefig(os.path.join(fig_dir, "fig4_relL2-vs-time-per-frame.png"), dpi=150)
    plt.close(fig)
    print(json.dumps({"rows": len(rows), "missing": missing}, indent=1))


if __name__ == "__main__":
    main()
