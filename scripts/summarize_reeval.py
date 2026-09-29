"""Step 3.5: old-vs-new table of every re-evaluated checkpoint.

  python summarize_reeval.py --old DIR_v2.0.0 --new DIR_v2.0.1 --out-prefix PREFIX
"""
import argparse, glob, json, os

import pandas as pd

METRICS = [("rmse", "RMSE"), ("mae", "MAE"), ("rel_l2_error", "RelL2"), ("r2", "R2"), ("f_error", "fRMSE"),
           ("low_f_error", "fRMSE_low"), ("mid_f_error", "fRMSE_mid"), ("high_f_error", "fRMSE_high"),
           ("freq_error", "FE"), ("ke_error", "KE")]


def md_table(df):
    lines = ["| " + " | ".join(map(str, df.columns)) + " |", "|" + "---|" * len(df.columns)]
    lines += ["| " + " | ".join("" if v is None else str(v) for v in r) + " |" for r in df.itertuples(index=False)]
    return "\n".join(lines) + "\n"


def load(d):
    out = {}
    for p in glob.glob(os.path.join(d, "*.json")):
        r = json.load(open(p))
        out[r["job"]["name"]] = r
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--old", required=True)
    ap.add_argument("--new", required=True)
    ap.add_argument("--out-prefix", required=True)
    a = ap.parse_args()
    O, N = load(a.old), load(a.new)
    rows = []
    for name in sorted(set(O) | set(N), key=lambda n: (n != "persistence", n)):
        j = (N.get(name) or O.get(name))["job"]
        row = {"model": j.get("model"), "setting": j.get("setting"), "source": j.get("source"), "name": name,
               "ckpt": j.get("ckpt")}
        for k, lab in METRICS:
            o = O[name]["metrics"][k] if name in O else None
            n = N[name]["metrics"][k] if name in N else None
            row[f"{lab}_old"], row[f"{lab}_new"] = o, n
            row[f"{lab}_chg%"] = 100 * (n - o) / abs(o) if (o is not None and n is not None and o != 0) else None
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(a.out_prefix + ".csv", index=False)
    show = ["model", "setting", "source"] + [f"{lab}_{s}" for _, lab in METRICS for s in ("old", "new", "chg%")]
    fmt = df[show].copy()
    for c in fmt.columns[3:]:
        fmt[c] = fmt[c].map(lambda v: "" if v is None or pd.isna(v) else (f"{v:+.1f}" if c.endswith("chg%") else f"{v:.5g}"))
    with open(a.out_prefix + ".md", "w") as f:
        f.write(md_table(fmt))
    print(fmt[["model", "setting", "source", "RMSE_old", "RMSE_new", "RMSE_chg%", "RelL2_old", "RelL2_new", "RelL2_chg%",
               "R2_old", "R2_new"]].to_string(index=False))


if __name__ == "__main__":
    main()
