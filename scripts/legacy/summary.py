#!/usr/bin/env python
"""Consolidated cylinder results: paper vs official-checkpoint vs retrained."""
import glob, os, re, json

RUNS = "/work/b314513067/pi-lfm/runs"
# paper: rmse, relL2, fRMSE, mae, r2, ke, fe
PAPER = {
 ("unet","sim"):(0.0758,0.2165,0.0122,0.03591,0.62955,0.00040,195.31436),
 ("unet","real"):(0.0700,0.0701,0.0103,0.01266,0.68363,0.00027,141.57477),
 ("unet","ft"):(0.0632,0.0728,0.0097,0.01213,0.74257,0.00027,128.03439),
 ("deeponet","sim"):(0.0863,0.3592,0.0132,0.04357,0.51881,0.00050,176.41255),
 ("deeponet","real"):(0.0713,0.1528,0.0108,0.02025,0.67225,0.00040,150.94789),
 ("deeponet","ft"):(0.0661,0.1503,0.0107,0.01912,0.71811,0.00049,130.10420),
 ("transolver","sim"):(0.1121,0.3224,0.0174,0.04706,0.32880,0.00050,504.92477),
 ("transolver","real"):(0.1093,0.1887,0.0160,0.02861,0.36250,0.00047,236.56488),
 ("transolver","ft"):(0.0965,0.1723,0.0144,0.02556,0.50232,0.00045,187.44112),
}
PAT = re.compile(r"rmse: ([\d.]+), mae: ([\d.]+), rel l2 error: ([\d.]+), r2: ([-\d.]+), "
                 r"ke error: ([\d.]+), f error: ([\d.]+).*?freq error: ([\d.]+)", re.S)

def load(prefix):
    out = {}
    for f in glob.glob(f"{RUNS}/slurm/{prefix}-*.err"):
        m = re.match(rf"{prefix}-([a-z]+)-([a-z]+)-\d+\.err", os.path.basename(f))
        if not m: continue
        t = open(f, errors="replace").read()
        i = t.find("Test results")
        if i < 0: continue
        g = PAT.search(t[i:i+1200])
        if not g: continue
        rmse, mae, rl2, r2, ke, fe_, freq = (float(x) for x in g.groups())
        out[(m.group(1), m.group(2))] = (rmse, rl2, fe_, mae, r2, ke, freq)
    return out

OFF, RE = load("off"), load("ev")
MODELS = ["unet","deeponet","transolver"]
MODE = [("sim","Simulated Training"),("real","Real-world Training"),("ft","Real-world Finetuning")]
NAME = {"unet":"U-Net","deeponet":"DeepONet","transolver":"Transolver"}
IDX  = {"RMSE":0,"RelL2":1,"fRMSE":2,"MAE":3,"R2":4,"KE":5,"FE":6}

def f(v, k):
    return f"{v:.2f}" if k=="FE" else (f"{v:.5f}" if k in("KE","R2") else f"{v:.4f}")

def block(title, keys, src, label):
    print(f"\n### {title} — {label}\n")
    print("| Model | Training mode | " + " | ".join(f"{k} | paper | Δ%" for k in keys) + " |")
    print("|" + "---|"*(2+3*len(keys)))
    for m in MODELS:
        for c, cn in MODE:
            if (m,c) not in src: continue
            o, p = src[(m,c)], PAPER[(m,c)]
            cells=[]
            for k in keys:
                i=IDX[k]; d=(o[i]-p[i])/p[i]*100
                cells += [f(o[i],k), f(p[i],k), f"{d:+.1f}"]
            print(f"| {NAME[m]} | {cn} | " + " | ".join(cells) + " |")

for src,label in ((OFF,"official checkpoints"),(RE,"retrained here")):
    block("Table 1", ["RMSE","RelL2","fRMSE"], src, label)
    block("Table 2", ["MAE","R2","KE","FE"], src, label)

# cross-mode progression
print("\n\n### Training-mode progression (retrained)\n")
print("| Model | Metric | Simulated | Real-world | Finetuning | Sim→FT |")
print("|---|---|---|---|---|---|")
for m in MODELS:
    for k in ("RMSE","RelL2","R2"):
        i=IDX[k]
        v=[RE.get((m,c),[None]*7)[i] for c,_ in MODE]
        if any(x is None for x in v): continue
        gain = (v[2]-v[0])/v[0]*100
        arrow = "better" if (k=="R2" and gain>0) or (k!="R2" and gain<0) else "worse"
        print(f"| {NAME[m]} | {k} | {f(v[0],k)} | {f(v[1],k)} | {f(v[2],k)} | {gain:+.1f}% ({arrow}) |")

# best-val iteration table
print("\n\n### Best validation iteration (retrained, selected on val_rmse)\n")
print("| Model | Simulated | Real-world | Finetuning | budget |")
print("|---|---|---|---|---|")
import torch
EXP={"unet":"unet_cylinder","deeponet":"deeponet_cylinder","transolver":"transolver_cylinder"}
for m in MODELS:
    row=[]
    for tdt,ft in (("numerical","False"),("real","False"),("real","True")):
        g=sorted(glob.glob(f"{RUNS}/{m}/{EXP[m]}_{tdt}_{ft}/*/model_*.pth"))
        if not g: row.append("-"); continue
        try:
            bi=torch.load(g[-1],map_location="cpu",weights_only=False).get("best_iteration")
            row.append(str(bi))
        except Exception: row.append("?")
    budget = 10000 if m=="unet" else 5000
    print(f"| {NAME[m]} | {row[0]} | {row[1]} | {row[2]} | {budget} |")
