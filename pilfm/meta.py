"""Provenance metadata attached to every result file."""
import hashlib, json, os, socket, subprocess, datetime

OFFICIAL_REPO = "/work/b314513067/RealPDEBench"
OUR_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _git(repo, *args):
    try:
        return subprocess.check_output(["git", "-C", repo, *args], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


def repo_state(repo):
    diff = _git(repo, "diff", "HEAD")
    return {
        "commit": _git(repo, "rev-parse", "HEAD"),
        "dirty": bool(diff),
        "diff_sha256": hashlib.sha256(diff.encode()).hexdigest() if diff else None,
    }


def dataset_version(data_root):
    p = os.path.join(data_root, "version.json")
    if not os.path.exists(p):
        return None
    return json.load(open(p)).get("data_version")


def collect(data_root=None, config=None, seed=None, checkpoint=None, **extra):
    m = {
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        "host": socket.gethostname(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "official_repo": repo_state(OFFICIAL_REPO),
        "our_repo": repo_state(OUR_REPO),
        "data_root": data_root,
        "dataset_version": dataset_version(data_root) if data_root else None,
        "seed": seed,
        "checkpoint": checkpoint,
    }
    if config is not None:
        text = config if isinstance(config, str) else json.dumps(config, sort_keys=True, default=str)
        m["config"] = config
        m["config_sha256"] = hashlib.sha256(text.encode()).hexdigest()
    m.update(extra)
    return m
