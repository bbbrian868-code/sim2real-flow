"""Step 6.4: Phase 6 FM result tables (+ U-Net baseline and persistence references), K decomposition, N sweep,
Update Ratio (D-026 convention) and val curves.

  python scripts/summarize_phase6.py --out /work/b314513067/pi-lfm/results/fm/phase6_summary
"""
import argparse, csv, glob, json, math, os, re, statistics

FM = "/work/b314513067/pi-lfm/results/fm"
V201 = "/work/b314513067/pi-lfm/results/v2.0.1"
METRICS = [("rmse", "RMSE"), ("mae", "MAE"), ("rel_l2_error", "RelL2"), ("r2", "R2"), ("f_error", "fRMSE"),
           ("low_f_error", "low"), ("mid_f_error", "mid"), ("high_f_error", "high"), ("freq_error", "FE"),
           ("ke_error", "KE")]
NAME = re.compile(r"(unet|dit)_([SML](?:v2)?)(_condnoise)?_(numerical|real|finetune)_s(\d)b?$")
# D-028: U-Net-S (base_ch 40 < 60 per-pixel target dims) cannot carry y_t; loss stuck at ~0.4. Replaced by Sv2.
INVALID = {("unet", "S")}


def parse(run):
    m = NAME.match(os.path.basename(run.rstrip("/")))
    bb, size, cn, setting, seed = m.groups()
    return {"backbone": bb, "size": size, "variant": "condnoise" if cn else "base", "setting": setting, "seed": int(seed)}


def sim_noise_in_target(variant, setting):
    # Step 0.2: official multiplicative gaussian noise (scale 0.1) hits input AND target, numerical data only.
    if setting == "real":
        return "n/a (no sim data)"
    return "no (cond only)" if variant == "condnoise" else "yes"


def fmt(xs):
    if len(xs) == 1:
        return f"{xs[0]:.4g}"
    return f"{statistics.mean(xs):.4g} ± {statistics.stdev(xs):.2g}"


def val_curve(run):
    return [(r["step"], r["val"]["rmse"]) for r in map(json.loads, open(f"{run}/log.jsonl")) if "val" in r]


def load_tests(pattern):
    out = []
    for f in sorted(glob.glob(pattern)):
        d = json.load(open(f))
        out.append({**parse(d["run"]), "N": d["N"], "K": d["K"], "m": d["metrics"], "run": d["run"],
                    "version": d["meta"]["dataset_version"], "file": f})
    return out


def update_ratios():
    """Pairs every FM finetune run with the real run of the same backbone/size/seed. Real data carry no sim noise,
    so the condnoise finetune is paired with the base real run."""
    runs = {}
    for d in glob.glob(f"{FM}/phase6/*/"):
        if not os.path.exists(f"{d}/best.pt"):
            continue
        p = parse(d)
        runs[(p["backbone"], p["size"], p["variant"], p["setting"], p["seed"])] = d.rstrip("/")
    rows = []
    for (bb, size, var, setting, seed), frun in sorted(runs.items()):
        if setting != "finetune" or (bb, size, "base", "real", seed) not in runs:
            continue
        rrun = runs[(bb, size, "base", "real", seed)]
        rc, fc = val_curve(rrun), val_curve(frun)
        n2, r0 = min(rc, key=lambda x: x[1])
        hit = [it for it, v in fc if v <= r0]
        n1 = hit[0] if hit else None
        rows.append({"backbone": bb, "size": size, "variant": var, "seed": seed, "rmse0_val": f"{r0:.6f}", "N2": n2,
                     "real_best_is_last": n2 == rc[-1][0], "N1": n1 or "",
                     "update_ratio": f"{n1 / n2:.4f}" if n1 else "not reached",
                     "finetune_best_val": f"{min(v for _, v in fc):.6f}", "finetune_last_step": fc[-1][0],
                     "real_run": rrun, "finetune_run": frun})
    return rows, runs


def write_csv(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, list(rows[0]))
        w.writeheader(); w.writerows(rows)


def md_table(header, rows):
    return "\n".join(["| " + " | ".join(header) + " |", "|" + "---|" * len(header)] +
                     ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    tests = load_tests(f"{FM}/phase6_test/*.json")
    ur, runs = update_ratios()
    ur_by = {}
    for r in ur:
        ur_by.setdefault((r["backbone"], r["size"], r["variant"]), []).append(r["update_ratio"])

    # ---- main table: one row per (backbone, size, variant, setting, K), mean ± std over seeds
    groups = {}
    for t in tests:
        groups.setdefault((t["backbone"], t["size"], t["variant"], t["setting"], t["N"], t["K"]), []).append(t)
    order = {"unet": 0, "dit": 1, "S": 0, "Sv2": 0.5, "M": 1, "L": 2, "base": 0, "condnoise": 1, "numerical": 0, "real": 1, "finetune": 2}
    keys = sorted(groups, key=lambda k: (order[k[2]], order[k[0]], order[k[1]], order[k[3]], k[5]))
    csv_rows, md_rows = [], []
    for k in keys:
        bb, size, var, setting, N, K = k
        g = groups[k]
        assert len({t["version"] for t in g}) == 1
        urs = ur_by.get((bb, size, var), []) if setting == "finetune" else []
        row = {"method": "FM (rectified flow)", "backbone": {"unet": "U-Net", "dit": "DiT"}[bb], "size": size,
               "setting": setting, "target": "full frames", "sim_noise_in_target": sim_noise_in_target(var, setting),
               "N": N, "K": K, "n_seeds": len(g), "seeds": ",".join(str(t["seed"]) for t in sorted(g, key=lambda t: t["seed"]))}
        for key, lab in METRICS:
            xs = [t["m"][key] for t in g]
            row[lab] = fmt(xs)
            row[lab + "_mean"] = statistics.mean(xs)
            row[lab + "_std"] = statistics.stdev(xs) if len(xs) > 1 else ""
        row["update_ratio"] = "; ".join(urs) if urs else ("–" if setting != "finetune" else "")
        row["dataset_version"] = g[0]["version"]
        csv_rows.append(row)
        md_rows.append([row["backbone"], size, setting, row["sim_noise_in_target"], N, K, len(g)] +
                       [row[lab] for _, lab in METRICS] + [row["update_ratio"]])

    # ---- references: Phase 4 U-Net (same official protocol, deterministic) and persistence
    refs = []
    p = json.load(open(f"{V201}/reeval/persistence.json"))
    refs.append(["persistence", "–", "–", "–", "–", "–", "–"] + [f"{p['metrics'][k]:.4g}" for k, _ in METRICS] + ["–"])
    ph4_ur = {}
    for r in csv.DictReader(open(f"{V201}/phase4_update_ratio.csv")):
        if r["model"] == "unet":
            ph4_ur.setdefault("unet", []).append(r["update_ratio"])
    for setting in ["numerical", "real", "finetune"]:
        g = [json.load(open(f)) for f in sorted(glob.glob(f"{V201}/phase4_test/unet_{setting}_s*.json"))]
        refs.append(["U-Net baseline (official, det.)", "–", setting, "yes" if setting != "real" else "n/a (no sim data)",
                     "–", "–", len(g)] + [fmt([d["metrics"][k] for d in g]) for k, _ in METRICS] +
                    ["; ".join(ph4_ur["unet"]) if setting == "finetune" else "–"])
    header = ["backbone", "size", "setting", "sim noise in target", "N", "K", "seeds"] + [lab for _, lab in METRICS] + ["Update Ratio"]
    md = "# Step 6.4 FM results (cylinder, real test, N_ar=1, dataset 2.0.1)\n\n"
    md += "All FM rows: rectified flow (spec 5.0), target = full T_out frames, EMA weights, best.pt chosen by K=1 N=10 val RMSE.\n"
    md += "Metrics on u, v (official `eval_metrics`, 2 channels, 4827 samples, metric_batch_size = all). mean ± std over seeds (sample std).\n\n"
    bad = [k[:2] in INVALID for k in keys]
    for c, b in zip(csv_rows, bad):
        c["valid"] = not b
    write_csv(f"{a.out}/phase6_main.csv", csv_rows)
    md += "## FM (base)\n\n" + md_table(header, [r for r, c, b in zip(md_rows, csv_rows, bad)
                                                   if "cond only" not in c["sim_noise_in_target"] and not b])
    md += ("\n## INVALID: U-Net-S (base_ch 40, D-028) -- kept for the record only, not a result\n\n" +
           md_table(header, [r for r, b in zip(md_rows, bad) if b]))
    md += "\n## Ablation 6.3-4: sim noise on the condition only (M tier, seed 0)\n\n" + md_table(
        header, [r for r, c in zip(md_rows, csv_rows) if "cond only" in c["sim_noise_in_target"]])
    md += "\n## References (same protocol)\n\n" + md_table(header, refs)

    # ---- K decomposition: MSE_K = A + V/K  (RMSE is sqrt of the global mean squared error here)
    dec = []
    by_run = {}
    for t in tests:
        by_run.setdefault(t["run"], {})[t["K"]] = t
    for run, d in sorted(by_run.items()):
        if 1 not in d or 5 not in d:
            continue
        m1, m5 = d[1]["m"]["rmse"] ** 2, d[5]["m"]["rmse"] ** 2
        V = (m1 - m5) / 0.8
        A = m1 - V
        dec.append({"run": os.path.basename(run), "rmse_K1": f"{d[1]['m']['rmse']:.6f}", "rmse_K5": f"{d[5]['m']['rmse']:.6f}",
                    "ratio_K1_over_K5": f"{d[1]['m']['rmse'] / d[5]['m']['rmse']:.4f}",
                    "rmse_Kinf_est": f"{math.sqrt(max(A, 0)):.6f}", "sample_spread_rms": f"{math.sqrt(max(V, 0)):.6f}",
                    "V_over_A": f"{V / A:.3f}" if A > 0 else "", "ratio_K1_over_Kinf": f"{math.sqrt(m1 / A):.4f}" if A > 0 else ""})
    write_csv(f"{a.out}/phase6_K_decomposition.csv", dec)
    md += ("\n## K decomposition\n\nMSE_K = A + V/K solved from K=1, K=5. A = error of the K→∞ sample mean, V = model sample "
           "variance. A perfect conditional model gives V = A (= tr Cov(y|c)), RMSE_K1/RMSE_K5 = √(2/1.2) = 1.291, "
           "RMSE_K1/RMSE_∞ = √2 = 1.414.\n\n" + md_table(list(dec[0]), [list(r.values()) for r in dec]))

    # ---- N sweep (N=10, 50 from the ablation dir; N=20 from the main test), K=1
    ns = load_tests(f"{FM}/phase6_ablation/N_sweep/*.json") + [t for t in tests if t["K"] == 1]
    nsw = {}
    for t in ns:
        nsw.setdefault(os.path.basename(t["run"]), {})[t["N"]] = t["m"]
    nrows = []
    for run, d in sorted(nsw.items()):
        if len(d) < 3:
            continue
        r = {"run": run}
        for N in sorted(d):
            for k, lab in [("rmse", "RMSE"), ("rel_l2_error", "RelL2"), ("high_f_error", "high"), ("ke_error", "KE")]:
                r[f"{lab}_N{N}"] = f"{d[N][k]:.5g}"
        nrows.append(r)
    write_csv(f"{a.out}/phase6_N_sweep.csv", nrows)
    md += "\n## Ablation 6.3-1: Euler steps N (K=1)\n\n" + md_table(list(nrows[0]), [list(r.values()) for r in nrows])

    write_csv(f"{a.out}/phase6_update_ratio.csv", ur)
    md += ("\n## Update Ratio (D-026 convention; FM val = 536-sample subset, N=10, K=1, every 2000 steps)\n\n" +
           md_table(["backbone", "size", "variant", "seed", "RMSE0 (real best val)", "N2", "real best = last step",
                     "N1", "UR", "finetune best val", "finetune steps"],
                    [[r["backbone"], r["size"], r["variant"], r["seed"], r["rmse0_val"], r["N2"], r["real_best_is_last"],
                      r["N1"], r["update_ratio"], r["finetune_best_val"], r["finetune_last_step"]] for r in ur]))
    open(f"{a.out}/phase6_summary.md", "w").write(md)
    plot_curves(runs, f"{a.out}/phase6_val_curves.png")
    print(md)


def plot_curves(runs, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    col = {"real": "#2a78d6", "finetune": "#eb6834"}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, bb in zip(axes, ["unet", "dit"]):
        for setting in ["real", "finetune"]:
            for seed in range(3):
                run = runs.get((bb, "M", "base", setting, seed))
                if not run:
                    continue
                c = val_curve(run)
                ax.plot([s for s, _ in c], [v for _, v in c], color=col[setting], lw=2 if seed == 0 else 1,
                        alpha=1 if seed == 0 else 0.55, label=f"{setting} (seeds 0–2)" if seed == 0 else None)
        ax.set_yscale("log")
        ax.set_title(f"{'U-Net' if bb == 'unet' else 'DiT'}-M: val RMSE (536-sample subset, N=10, K=1)", fontsize=10, loc="left")
        ax.set_xlabel("iteration of that run (finetune starts from the numerical EMA weights)", fontsize=8)
        ax.grid(True, which="both", color="#e5e5e0", lw=0.6)
        for s in ["top", "right"]:
            ax.spines[s].set_visible(False)
        ax.legend(frameon=False, fontsize=8)
    axes[0].set_ylabel("val RMSE (log)")
    fig.tight_layout()
    fig.savefig(path, dpi=130)


if __name__ == "__main__":
    main()
