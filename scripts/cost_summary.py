"""B (todo_1012): cost table (B1 as-run + B2 params/FLOPs + B3 controlled) and the RMSE-vs-NFE Pareto plot (B4).

  python scripts/cost_summary.py table    -> $R/cost/cost_table.md
  python scripts/cost_summary.py pareto   -> $R/cost/pareto.{png,csv}
"""
import csv, glob, json, os, sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

R = "/work/b314513067/pi-lfm/results"
OUT = f"{R}/cost"
BENCH = f"{R}/fm/b3_bench/bench.json"
B4 = f"{R}/fm/b4_pareto"
# U-Net baseline real test RMSE (Phase 4, 3 seeds) -- docs/REPORT_phase0-4 Section 5.2
UNET_BASELINE_RMSE = (0.01209, 0.00004)
COLORS = {"FM U-Net-M": "#2a78d6", "FM DiT-M": "#eb6834", "FM U-Net-L": "#1baf7a", "FM DiT-M x-pred p4 QK (C2)": "#4a3aa7"}
FAMILIES = list(COLORS)
FAMILY_OF = {"unet_M": "FM U-Net-M", "dit_M": "FM DiT-M", "unet_L": "FM U-Net-L", "xp4_qk": "FM DiT-M x-pred p4 QK (C2)"}
INK, MUTED = "#0b0b0b", "#8a8984"

ROWS = [  # label, bench key, as-run rows (method, setting) from asrun.csv
    ("U-Net baseline (official)", "official_unet", ("baseline unet", "real")),
    ("Transolver baseline (official)", "official_transolver", ("baseline transolver", "real")),
    ("DeepONet baseline (official)", "official_deeponet", ("baseline deeponet", "finetune")),
    ("FM U-Net-Sv2", "fm_unet_Sv2", ("FM unet-Sv2", "real")),
    ("FM U-Net-M", "fm_unet_M", ("FM unet-M", "real")),
    ("FM U-Net-L", "fm_unet_L", ("FM unet-L", "real")),
    ("FM DiT-S", "fm_dit_S", ("FM dit-S", "real")),
    ("FM DiT-M", "fm_dit_M", ("FM dit-M", "real")),
    ("FM DiT-L", "fm_dit_L", ("FM dit-L", "real")),
    ("FM DiT-M + QK-norm (C2)", "fm_dit_M_qk", None),
    ("FM DiT-M x-pred p4 + QK-norm (C2)", "fm_dit_M_xp4_qk", None),
]


def asrun():
    rows = list(csv.DictReader(open(f"{OUT}/asrun.csv")))
    out = {}
    for r in rows:
        out.setdefault((r["method"], r["setting"]), []).append(r)
    return out


def cmd_table():
    b = json.load(open(BENCH))["models"]
    ar = asrun()
    f = open(f"{OUT}/cost_table.md", "w")
    f.write(f"GPU: {json.load(open(BENCH))['gpu']} (one job, one GPU). FLOPs = one forward, batch 1, 64x128 (FM: one NFE).\n\n")
    f.write("| model | params | GFLOPs / fwd | as-run: prec, batch, s/iter, samples/s | as-run GPU-h (budget / to selected ckpt) | as-run samples (budget) | "
            "controlled train (bf16): batch, ms/step, samples/s, peak GB | infer fp32 b1 ms / fwd | infer fp32 b64 samples/s / fwd | infer bf16 b1 ms | infer bf16 b64 samples/s |\n")
    f.write("|" + "---|" * 11 + "\n")
    for label, key, ark in ROWS:
        m = b.get(key)
        if m is None:
            continue
        tc = m.get("train_controlled", {})
        tcs = (f"{tc['batch']}, {1000 * tc['median_s']:.0f}, {tc['samples_per_s']:.0f}, {tc['peak_mem_gb']:.0f}"
               if "median_s" in tc else "OOM / error")
        if key.startswith("official"):
            ta = m.get("train_as_run", {})
        if ark and ark in ar:
            rs = ar[ark]
            g = lambda k: np.mean([float(r[k]) for r in rs])
            ars = f"{rs[0]['precision']}, {rs[0]['batch']}, {g('s_per_iter'):.3f}, {g('samples_per_s'):.0f}"
            arh = f"{g('gpu_h_budget'):.2f} / {g('gpu_h_to_sel'):.2f} (n={len(rs)})"
            arn = f"{g('samples_budget') / 1e6:.2f}M"
        else:
            ars = arh = arn = "–"
        inf = lambda k: m.get(k, {})
        f.write(f"| {label} | {m['params'] / 1e6:.1f}M | {m['flops_fwd_b1'] / 1e9:.0f} | {ars} | {arh} | {arn} | {tcs} | "
                f"{1000 * inf('infer_fp32_b1').get('median_s', float('nan')):.1f} | {inf('infer_fp32_b64').get('samples_per_s', float('nan')):.0f} | "
                f"{1000 * inf('infer_bf16_b1').get('median_s', float('nan')):.1f} | {inf('infer_bf16_b64').get('samples_per_s', float('nan')):.0f} |\n")
    if "official_unet" in b and "train_as_run" in b["official_unet"]:
        t = b["official_unet"]["train_as_run"]
        f.write(f"\nOfficial U-Net, synthetic tensors, fp32, batch 12: {t['median_s']:.3f} s/step "
                f"(as-run with the official dataloader: see the as-run column).\n")
    f.close()
    print(open(f"{OUT}/cost_table.md").read())


def cmd_pareto():
    b = json.load(open(BENCH))["models"]
    lat = {"FM U-Net-M": b["fm_unet_M"]["infer_fp32_b64"]["median_s"] / 64,
           "FM DiT-M": b["fm_dit_M"]["infer_fp32_b64"]["median_s"] / 64,
           "FM U-Net-L": b["fm_unet_L"]["infer_fp32_b64"]["median_s"] / 64,
           "FM DiT-M x-pred p4 QK (C2)": b["fm_dit_M_xp4_qk"]["infer_fp32_b64"]["median_s"] / 64}
    base_t = b["official_unet"]["infer_fp32_b64"]["median_s"] / 64 if "median_s" in b["official_unet"].get("infer_fp32_b64", {}) else None
    rows = []
    for f in glob.glob(f"{B4}/*.json"):
        j = json.load(open(f))
        run = os.path.basename(j["run"].rstrip("/"))
        fam = FAMILY_OF["_".join(run.split("_")[:2])]
        nfe = j.get("nfe", j["N"] * j["K"])
        rows.append({"family": fam, "run": run, "solver": j.get("solver", "euler"), "steps": j["N"], "nfe": nfe,
                     "rmse": j["metrics"]["rmse"], "sec_per_sample_gpu": nfe * lat[fam]})
    with open(f"{OUT}/pareto.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, list(rows[0])); w.writeheader(); w.writerows(sorted(rows, key=lambda r: (r["family"], r["solver"], r["nfe"], r["run"])))
    fig, ax = plt.subplots(1, 2, figsize=(13, 6.4), sharey=True)
    for fam in FAMILIES:
        # plain Heun (last correction at t = 1, biased) is kept in the CSV only; the plot uses Heun with an Euler last step
        for solver, ls, mk in (("euler", "-", "o"), ("heun_e", "--", "s")):
            rs = [r for r in rows if r["family"] == fam and r["solver"] == solver]
            if not rs:
                continue
            nfes = sorted({r["nfe"] for r in rs})
            mean = [np.mean([r["rmse"] for r in rs if r["nfe"] == n]) for n in nfes]
            sd = [np.std([r["rmse"] for r in rs if r["nfe"] == n], ddof=1) if sum(r["nfe"] == n for r in rs) > 1 else 0 for n in nfes]
            for a, x in ((ax[0], nfes), (ax[1], [n * lat[fam] * 1000 for n in nfes])):
                a.errorbar(x, mean, yerr=sd, color=COLORS[fam], ls=ls, marker=mk, ms=6, lw=2, capsize=2,
                           label=f"{fam}, {'Heun (Euler last step)' if solver == 'heun_e' else 'Euler'}" + (f" (n={len({r['run'] for r in rs})} seeds)"))
    for a in ax:
        a.axhline(UNET_BASELINE_RMSE[0], color=INK, lw=1, ls=":")
        a.set_xscale("log"); a.grid(alpha=0.25, lw=0.5)
        for sp in ("top", "right"):
            a.spines[sp].set_visible(False)
    if base_t:
        ax[1].plot([base_t * 1000], [UNET_BASELINE_RMSE[0]], marker="*", ms=14, color=INK, ls="none", label="U-Net baseline (1 fwd)")
    ax[0].plot([1], [UNET_BASELINE_RMSE[0]], marker="*", ms=14, color=INK, ls="none", label="U-Net baseline (1 fwd)")
    ax[0].set_xlabel("NFE (velocity evaluations per sample; Heun = 2 per step, Euler last step)")
    ax[1].set_xlabel("GPU time per sample (ms; fp32, batch 64, H200, from B3)")
    ax[0].set_ylabel("real test RMSE (K = 1)")
    h, l = ax[1].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, fontsize=9, frameon=False)
    fig.suptitle("RMSE vs inference cost, 2.0.1 real test (dotted: U-Net baseline)")
    fig.tight_layout(rect=(0, 0.14, 1, 1)); fig.savefig(f"{OUT}/pareto.png", dpi=130)
    for fam in FAMILIES:
        for solver in ("euler", "heun", "heun_e"):
            rs = [r for r in rows if r["family"] == fam and r["solver"] == solver]
            print(fam, solver, [(n, round(np.mean([r["rmse"] for r in rs if r["nfe"] == n]), 5)) for n in sorted({r["nfe"] for r in rs})])


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    {"table": cmd_table, "pareto": cmd_pareto}[sys.argv[1]]()
