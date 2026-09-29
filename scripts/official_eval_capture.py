"""Run the unmodified official realpdebench.eval and capture eval_metrics() at full precision.

The official script logs metrics with 5 decimals. We wrap realpdebench.utils.metrics.eval_metrics at
runtime (no source change) before eval.py imports it, then run eval.py as __main__.

  python official_eval_capture.py --out OUT.json -- --config CFG --train_data_type numerical \
      --checkpoint_path CKPT --use_hf_dataset
"""
import json, os, runpy, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pilfm import meta
from pilfm.official_eval import METRIC_NAMES
import realpdebench.utils.metrics as M

argv = sys.argv[1:]
out = argv[argv.index("--out") + 1]
official_args = argv[argv.index("--") + 1:]
captured = []
_orig = M.eval_metrics


def wrapped(*a, **k):
    r = _orig(*a, **k)
    captured.append({n: float(v) for n, v in zip(METRIC_NAMES, r)})
    return r


M.eval_metrics = wrapped
sys.argv = ["realpdebench.eval", *official_args]
runpy.run_module("realpdebench.eval", run_name="__main__")

cfg = official_args[official_args.index("--config") + 1]
ckpt = official_args[official_args.index("--checkpoint_path") + 1]
import yaml
text = open(cfg).read()
root = yaml.safe_load(text)["dataset_root"]
os.makedirs(os.path.dirname(out), exist_ok=True)
json.dump({"metrics": captured[-1], "official_args": official_args,
           "meta": meta.collect(data_root=root, config=text, seed=yaml.safe_load(text)["seed"], checkpoint=ckpt)},
          open(out, "w"), indent=1)
print("captured", captured[-1])
