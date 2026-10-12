"""C (todo_1012): DiT divergence diagnostics -- figures and tables from the run logs and the diagnostic JSONs.

  python scripts/c_stability.py align    C0.1/C0.2/C0.7: every DiT/U-Net FM run, divergence step, aligned curves
  python scripts/c_stability.py diag     C0.3-C0.6 table from $R/fm/c0_diag/*.json
  python scripts/c_stability.py c1       C1 resumes: attention max logit / grad norm / val per variant
  python scripts/c_stability.py c2       C2 screening: val curves, divergence per config and seed
Outputs in $R/fm/c_stability/.
"""
import glob, json, os, sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

R = "/work/b314513067/pi-lfm/results/fm"
P = f"{R}/phase6"
OUT = f"{R}/c_stability"
SEED = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#4a3aa7"]
INK, MUTED = "#0b0b0b", "#8a8984"
DIVERGE_GN = 10.0  # grad norm above this, staying elevated = divergence onset (healthy runs stay < 2 after warmup)


def load(d):
    L = [json.loads(l) for l in open(f"{d}/log.jsonl")]
    tr = [r for r in L if "loss" in r]
    va = [r for r in L if "val" in r]
    at = [r for r in L if "attn" in r]
    return tr, va, at


def arr(rs, key):
    return np.array([r[key] for r in rs], dtype=float)


def onset(tr, after=2000):
    s, g = arr(tr, "step"), arr(tr, "grad_norm")
    gmax = arr(tr, "grad_norm_max") if "grad_norm_max" in tr[0] else g
    # sustained: the next 10 logged grad norms have median > 1 (a lone spike that recovers is a precursor, not the onset)
    idx = [i for i in np.where((s > after) & (gmax > DIVERGE_GN))[0] if np.median(g[i + 1:i + 11]) > 1]
    return int(s[idx[0]]) if idx else None


def collapsed(tr, va):
    """Diverged and did not come back: the last 2000 logged steps have median loss > 5x the run's best 1000-step median."""
    s, l = arr(tr, "step"), arr(tr, "loss")
    med = np.array([np.median(l[max(0, i - 20):i + 1]) for i in range(len(l))])
    return bool(np.median(l[s > s[-1] - 2000]) > 5 * med[s > 3000].min()) if (s > 3000).any() else False


def style(a):
    a.grid(alpha=0.25, lw=0.5)
    for sp in ("top", "right"):
        a.spines[sp].set_visible(False)


def smooth(x, w=20):
    if len(x) < w:
        return x
    return np.convolve(x, np.ones(w) / w, mode="same")


def cmd_align():
    os.makedirs(OUT, exist_ok=True)
    rows = []
    for d in sorted(glob.glob(f"{P}/*")):
        name = os.path.basename(d)
        if name.startswith("unet_S_") or not os.path.exists(f"{d}/log.jsonl") or os.path.getsize(f"{d}/log.jsonl") == 0:
            continue
        tr, va, _ = load(d)
        if not va:
            continue
        vs, vr = arr(va, "step"), np.array([r["val"]["rmse"] for r in va])
        on = onset(tr)
        s, l, g = arr(tr, "step"), arr(tr, "loss"), arr(tr, "grad_norm")
        pre = (s > 3000) & (s < (on or s[-1]) - 1000)
        rows.append({"run": name, "diverged_at": on, "collapsed": collapsed(tr, va), "best_step": int(vs[vr.argmin()]),
                     "best_val": float(vr.min()), "final_val": float(vr[-1]),
                     "gn_median_pre": float(np.median(g[pre])) if pre.any() else None,
                     "frac_gn_gt1_pre": float(np.mean(g[pre] > 1)) if pre.any() else None,
                     "lr_at_onset": float(arr(tr, "lr")[s == on][0]) if on else None})
    json.dump(rows, open(f"{OUT}/divergence_table.json", "w"), indent=1)
    with open(f"{OUT}/divergence_table.md", "w") as f:
        f.write("| run | diverged at (first logged grad norm > 10 after 2k, sustained) | stays collapsed | lr at onset | best step | best val | final val | grad norm median before onset | logged steps with grad norm > 1 before onset |\n|---|---|---|---|---|---|---|---|---|\n")
        for r in rows:
            f.write(f"| {r['run']} | {r['diverged_at'] or '–'} | {'yes' if r['collapsed'] else 'no'} | "
                    f"{('%.2e' % r['lr_at_onset']) if r['lr_at_onset'] else '–'} | {r['best_step']} | {r['best_val']:.5f} | "
                    f"{r['final_val']:.5f} | " + (f"{r['gn_median_pre']:.3f} | {100 * r['frac_gn_gt1_pre']:.1f}% |\n" if r["gn_median_pre"] is not None else "– | – |\n"))
    print(open(f"{OUT}/divergence_table.md").read())

    # aligned figure: every diverged run, x = step - onset
    div = [r for r in rows if r["diverged_at"]]
    fig, ax = plt.subplots(4, 1, figsize=(10, 11), sharex=True)
    for i, r in enumerate(div):
        tr, va, _ = load(f"{P}/{r['run']}")
        s = arr(tr, "step") - r["diverged_at"]
        c = SEED[i % len(SEED)]
        ax[0].plot(s, smooth(arr(tr, "loss"), 5), color=c, lw=1.5, label=r["run"])
        ax[1].plot(s, arr(tr, "grad_norm"), color=c, lw=1)
        ax[2].plot(s, arr(tr, "lr"), color=c, lw=1.5)
        ax[3].plot(arr(va, "step") - r["diverged_at"], [v["val"]["rmse"] for v in va], color=c, lw=1.5, marker="o", ms=3)
    for a, yl, lg in zip(ax, ["train loss (250-step mean)", "grad norm (pre-clip, logged step)", "lr", "val RMSE (subset, N=10)"],
                         [True, True, False, True]):
        style(a); a.set_ylabel(yl); a.axvline(0, color=MUTED, lw=1, ls="--")
        if lg:
            a.set_yscale("log")
    ax[0].legend(fontsize=8, frameon=False, ncol=2)
    ax[3].set_xlabel("steps relative to divergence onset (first logged grad norm > 10 that stays elevated)")
    ax[3].set_xlim(-12000, 8000)
    fig.suptitle("DiT runs that diverged, aligned at the divergence onset")
    fig.tight_layout(); fig.savefig(f"{OUT}/aligned_divergence.png", dpi=130)


def cmd_diag():
    rows = []
    for f in sorted(glob.glob(f"{R}/c0_diag/*.json")):
        j = json.load(open(f))
        for w, r in j["weights"].items():
            a = r.get("attn", {}).get("0.5")
            rows.append({"name": os.path.basename(f)[:-5], "step": j["step"], "weights": w,
                         "val_E10": r["val_euler10"], "val_E50": r["val_euler50"], "val_H10": r["val_heun10"],
                         "tbin_val": r["tbin_val"], "tbin_train": r["tbin_train"],
                         "max_logit": max(a["max_logit"]) if a else None,
                         "argmax_layer": int(np.argmax(a["max_logit"])) if a else None,
                         "min_entropy": min(a["entropy"]) if a else None, "log_n": a["log_n_tokens"] if a else None,
                         "qk_wnorm_max": max(max(r["wnorm"]["q_wnorm"]), max(r["wnorm"]["k_wnorm"])) if "wnorm" in r else None,
                         "ada_scale_max": max(a["ada_scale"]) if a else None})
    json.dump(rows, open(f"{OUT}/c0_diag_table.json", "w"), indent=1)
    tb = ["0.02", "0.3", "0.7", "0.98"]
    with open(f"{OUT}/c0_diag_table.md", "w") as f:
        f.write("| checkpoint | step | weights | val E10 | val E50 | val Heun10 | " + " | ".join(f"FM loss val/train t={t}" for t in tb)
                + " | max attn logit (layer) @t=0.5 | min layer entropy (log N) | max q/k W norm | max adaLN scale |\n")
        f.write("|" + "---|" * (6 + len(tb) + 4) + "\n")
        for r in rows:
            f.write(f"| {r['name']} | {r['step']} | {r['weights']} | {r['val_E10']:.4f} | {r['val_E50']:.4f} | {r['val_H10']:.4f} | "
                    + " | ".join(f"{r['tbin_val'][t]:.3g} / {r['tbin_train'][t]:.3g}" for t in tb) + " | "
                    + (f"{r['max_logit']:.0f} ({r['argmax_layer']}) | {r['min_entropy']:.2f} ({r['log_n']:.2f}) | {r['qk_wnorm_max']:.1f} | {r['ada_scale_max']:.2f} |"
                       if r["max_logit"] is not None else "– | – | – | – |") + "\n")
    print(open(f"{OUT}/c0_diag_table.md").read())


def attn_series(at, key="max_logit"):
    s = arr(at, "step")
    m = np.array([r["attn"][key] for r in at])  # (n, layers)
    return s, m


def cmd_c1():
    runs = sorted(glob.glob(f"{R}/c1_resume/*"))
    fig, ax = plt.subplots(4, len(runs), figsize=(3.2 * len(runs), 11), sharex="col", sharey="row")
    rows = []
    for j, d in enumerate(runs):
        name = os.path.basename(d)
        tr, va, at = load(d)
        s = arr(tr, "step")
        on = onset(tr, after=0)
        rows.append({"run": name, "start": int(s[0] - 50), "diverged_at": on, "end": int(s[-1]),
                     "val": [(int(r["step"]), r["val"]["rmse"]) for r in va],
                     "max_logit_first": max(at[0]["attn"]["max_logit"]) if at else None,
                     "max_logit_last_pre": max(at[np.searchsorted(arr(at, "step"), on) - 1]["attn"]["max_logit"]) if (at and on) else None})
        ax[0, j].plot(s, arr(tr, "grad_norm_max"), color=INK, lw=1)
        ax[0, j].set_yscale("log"); ax[0, j].set_title(name, fontsize=9)
        if at:
            st, m = attn_series(at)
            for L in range(m.shape[1]):
                ax[1, j].plot(st, m[:, L], lw=1.2 if L == 0 else 0.7, color=SEED[L % len(SEED)] if L < 6 else MUTED,
                              label=f"block {L}")
            st, e = attn_series(at, "entropy")
            for L in range(e.shape[1]):
                ax[2, j].plot(st, e[:, L], lw=1.2 if L == 0 else 0.7, color=SEED[L % len(SEED)] if L < 6 else MUTED)
            ax[1, j].set_yscale("log")
        ax[3, j].plot(arr(va, "step"), [r["val"]["rmse"] for r in va], color=INK, marker="o", ms=3)
        ax[3, j].set_yscale("log")
        for i in range(4):
            style(ax[i, j])
            if on:
                ax[i, j].axvline(on, color=MUTED, ls="--", lw=1)
        ax[3, j].set_xlabel("step")
    ax[0, 0].set_ylabel("max grad norm per 50 steps")
    ax[1, 0].set_ylabel("attention max logit (per block)")
    ax[2, 0].set_ylabel("attention entropy (nats)")
    ax[3, 0].set_ylabel("val RMSE (subset, N=10)")
    ax[1, 0].legend(fontsize=7, frameon=False)
    fig.suptitle("C1: DiT-M real resumed from best.pt (same / beta2 0.95 / lr x0.5); dashed = divergence onset")
    fig.tight_layout(); fig.savefig(f"{OUT}/c1_resume.png", dpi=120)
    json.dump(rows, open(f"{OUT}/c1_table.json", "w"), indent=1)
    for r in rows:
        print(r["run"], "start", r["start"], "diverged_at", r["diverged_at"], "logit first/last-pre",
              r["max_logit_first"], r["max_logit_last_pre"], "val", [f"{s}:{v:.4f}" for s, v in r["val"]])


C2_CFG = [("base", "current (v-pred, patch 2)"), ("qk", "+ QK-norm"), ("qk_b95", "+ QK-norm + beta2 0.95"),
          ("xp4_qk", "x-pred + patch 4 + QK-norm")]


def c2_runs(cfg):
    if cfg == "base":  # the phase-6 runs (seeds 0-2) plus the new seeds 3-5
        return [f"{P}/dit_M_real_s0b", f"{P}/dit_M_real_s1", f"{P}/dit_M_real_s2"] + sorted(glob.glob(f"{R}/c2/base_s*"))
    return sorted(glob.glob(f"{R}/c2/{cfg}_s*"))


def cmd_c2():
    # success reference: mean over seeds 0-2 of the current setting's best (pre-divergence) val RMSE
    pre_best = float(np.mean([min(r["val"]["rmse"] for r in load(d)[1]) for d in c2_runs("base")[:3]]))
    fig, ax = plt.subplots(2, 4, figsize=(18, 7.5), sharex=True)
    rows = []
    for j, (cfg, label) in enumerate(C2_CFG):
        for i, d in enumerate(c2_runs(cfg)):
            tr, va, at = load(d)
            if not va:
                continue
            vs, vr = arr(va, "step"), np.array([r["val"]["rmse"] for r in va])
            on = onset(tr)
            done = os.path.exists(f"{d}/done.json") or int(arr(tr, "step")[-1]) >= 50000
            rows.append({"config": cfg, "run": os.path.basename(d), "done": done, "last_step": int(arr(tr, "step")[-1]),
                         "diverged_at": on, "collapsed": collapsed(tr, va), "best_step": int(vs[vr.argmin()]),
                         "best_val": float(vr.min()), "final_val": float(vr[-1]),
                         "max_logit_end": max(at[-1]["attn"]["max_logit"]) if at else None})
            c = SEED[i % len(SEED)]
            ax[0, j].plot(vs, vr, color=c, lw=1.5, marker="o", ms=2.5, label=os.path.basename(d))
            if at:
                st, m = attn_series(at)
                ax[1, j].plot(st, m.max(1), color=c, lw=1.2)
        ax[0, j].axhline(pre_best, color=MUTED, ls="--", lw=1)
        ax[0, j].set_title(label, fontsize=10)
        ax[0, j].set_yscale("log"); ax[0, j].set_ylim(0.012, 0.3)
        ax[1, j].set_yscale("log")
        ax[0, j].legend(fontsize=7, frameon=False)
        ax[1, j].set_xlabel("step")
        style(ax[0, j]); style(ax[1, j])
    ax[0, 0].set_ylabel("val RMSE (subset, N=10); dashed = mean best pre-divergence val, current setting s0-2")
    ax[1, 0].set_ylabel("max attention logit over blocks\n(logged every 250 steps; new runs only)")
    fig.suptitle("C2 screening: DiT-M, real, 50k steps")
    fig.tight_layout(); fig.savefig(f"{OUT}/c2_screen.png", dpi=120)
    json.dump({"pre_best_current": pre_best, "rows": rows}, open(f"{OUT}/c2_table.json", "w"), indent=1)
    with open(f"{OUT}/c2_table.md", "w") as f:
        f.write(f"Reference: mean over seeds 0-2 of the current setting's best (pre-divergence) val = {pre_best:.5f}\n\n")
        f.write("| config | run | finished | last step | diverged at | stays collapsed | best step | best val | final val | max attn logit (last log) |\n|---|---|---|---|---|---|---|---|---|---|\n")
        for r in rows:
            f.write(f"| {r['config']} | {r['run']} | {'yes' if r['done'] else 'no'} | {r['last_step']} | {r['diverged_at'] or '–'} | "
                    f"{'yes' if r['collapsed'] else 'no'} | {r['best_step']} | {r['best_val']:.5f} | {r['final_val']:.5f} | "
                    f"{('%.0f' % r['max_logit_end']) if r['max_logit_end'] else '–'} |\n")
    print(open(f"{OUT}/c2_table.md").read())


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    {"align": cmd_align, "diag": cmd_diag, "c1": cmd_c1, "c2": cmd_c2}[sys.argv[1]]()
