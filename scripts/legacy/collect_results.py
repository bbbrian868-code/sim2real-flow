#!/usr/bin/env python
"""Parse eval logs and print Table 1 / Table 2 next to the paper's cylinder values."""
import glob, os, re

import sys
PREFIX = sys.argv[1] if len(sys.argv)>1 else "/work/b314513067/pi-lfm/runs/slurm/ev-*.err"
# paper cylinder values: (rmse, relL2, fRMSE, mae, r2, ke, fe)
PAPER = {
 ("unet","sim"):      (0.0758,0.2165,0.0122,0.03591,0.62955,0.00040,195.31436),
 ("unet","real"):     (0.0700,0.0701,0.0103,0.01266,0.68363,0.00027,141.57477),
 ("unet","ft"):       (0.0632,0.0728,0.0097,0.01213,0.74257,0.00027,128.03439),
 ("deeponet","sim"):  (0.0863,0.3592,0.0132,0.04357,0.51881,0.00050,176.41255),
 ("deeponet","real"): (0.0713,0.1528,0.0108,0.02025,0.67225,0.00040,150.94789),
 ("deeponet","ft"):   (0.0661,0.1503,0.0107,0.01912,0.71811,0.00049,130.10420),
 ("transolver","sim"):(0.1121,0.3224,0.0174,0.04706,0.32880,0.00050,504.92477),
 ("transolver","real"):(0.1093,0.1887,0.0160,0.02861,0.36250,0.00047,236.56488),
 ("transolver","ft"): (0.0965,0.1723,0.0144,0.02556,0.50232,0.00045,187.44112),
}
COND = {"sim":"Simulated","real":"Real-world","ft":"Finetuning"}
PAT = re.compile(
  r"rmse: ([\d.]+), mae: ([\d.]+), rel l2 error: ([\d.]+), r2: ([-\d.]+), "
  r"ke error: ([\d.]+), f error: ([\d.]+).*?freq error: ([\d.]+)", re.S)

rows = {}
for f in sorted(glob.glob(PREFIX)):
    b = os.path.basename(f)
    m = re.match(r"[a-z]+-([a-z]+)-([a-z]+)-\d+\.err", b)
    if not m: continue
    txt = open(f, errors="replace").read()
    i = txt.find("Test results")
    if i < 0: continue
    g = PAT.search(txt[i:i+1200])
    if not g: continue
    rmse, mae, rl2, r2, ke, fe_, freq = (float(x) for x in g.groups())
    rows[(m.group(1), m.group(2))] = (rmse, rl2, fe_, mae, r2, ke, freq)

def emit(title, idxs, names):
    print(f"\n## {title}")
    hdr = "| Model | Condition | " + " | ".join(
        f"{n} ours | {n} paper | Δ%" for n in names) + " |"
    print(hdr); print("|" + "---|"*(2+3*len(names)))
    for mdl in ("unet","deeponet","transolver"):
        for c in ("sim","real","ft"):
            if (mdl,c) not in rows: continue
            o, p = rows[(mdl,c)], PAPER[(mdl,c)]
            cells = []
            for i in idxs:
                d = (o[i]-p[i])/p[i]*100 if p[i] else 0
                fmt = "{:.5f}" if abs(p[i]) < 0.01 or i in (4,) else "{:.4f}"
                if i == 6: fmt = "{:.2f}"
                cells += [fmt.format(o[i]), fmt.format(p[i]), f"{d:+.1f}"]
            print(f"| {mdl} | {COND[c]} | " + " | ".join(cells) + " |")

emit("Table 1 — RMSE / Rel L2 / fRMSE", (0,1,2), ("RMSE","RelL2","fRMSE"))
emit("Table 2 — MAE / R2 / KE / FE",    (3,4,5,6), ("MAE","R2","KE","FE"))
print(f"\n({len(rows)}/9 runs evaluated)")
