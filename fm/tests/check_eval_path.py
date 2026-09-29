"""Step 5.3(d): evaluation-path consistency (GPU job, full real test set).

A fake velocity field v(y, t, cond) = (p(cond) - y) / (1 - t), where p(cond) is the persistence prediction in
folded, target-normalized space, makes the Euler sampler land on p at t = 1 (up to float rounding). Running it
through FMPredictor + the official-mirror test loop must reproduce the persistence metrics of Step 3.2.

  python fm/tests/check_eval_path.py --data-root ROOT --out OUT.json [--ref STEP3.2_JSON]
"""
import argparse, json, os, sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from pilfm import meta
from pilfm.official_eval import Env, run_test, persistence_predictor
from fm.wrapper import FMPredictor, fold, unfold

T, C = 20, 3


class PersistenceField(torch.nn.Module):
    def __init__(self, norm):
        super().__init__()
        self.pers = persistence_predictor(norm, T_out=T)

    def forward(self, y, t, cond):
        p = fold(self.pers(unfold(cond, T, C)))
        return (p - y) / (1 - t).view(-1, 1, 1, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ref", default=None, help="Step 3.2 persistence result json on the same data root")
    args = ap.parse_args()
    device = "cuda:0"
    env = Env(args.data_root)
    norm = env.normalizer(device)
    direct = run_test(env, persistence_predictor(norm), 12, device)
    for n_steps in (1, 20):
        via = run_test(env, FMPredictor(PersistenceField(norm), T, C, n_steps=n_steps, n_samples=1), 12, device)
        rel = {k: abs(via[k] - direct[k]) / max(abs(direct[k]), 1e-30) for k in direct if isinstance(direct[k], float)}
        print(f"N={n_steps}: max rel diff over metrics = {max(rel.values()):.3e}", {k: f"{v:.1e}" for k, v in rel.items()})
        direct[f"via_fm_N{n_steps}"] = via
        direct[f"rel_diff_N{n_steps}"] = rel
    if args.ref:
        ref = json.load(open(args.ref))["metrics"]
        direct["rel_diff_vs_step3.2"] = {k: abs(direct[k] - ref[k]) / max(abs(ref[k]), 1e-30)
                                        for k in ref if isinstance(ref[k], float)}
        print("direct persistence vs Step 3.2 file:", max(direct["rel_diff_vs_step3.2"].values()))
    json.dump({"metrics": direct, "meta": meta.collect(data_root=args.data_root)}, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
