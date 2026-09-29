"""Step 1.2: build NEW_DATA_ROOT for cylinder at a pinned HF revision.

Files whose content is byte-identical to OLD_DATA_ROOT become symlinks to the old file;
everything else is downloaded. Identity is checked against the remote hash:
LFS files by sha256, small (non-LFS) files by git blob sha1.

  python fetch_v201.py --trees remote_trees.json --old OLD --new NEW [--dry-run]
"""
import argparse, hashlib, json, os, sys

REPO = "AI4Science-WestlakeU/RealPDEBench"


def sha256_file(p, bs=1 << 24):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while chunk := f.read(bs):
            h.update(chunk)
    return h.hexdigest()


def git_blob_sha1(p):
    data = open(p, "rb").read()
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def matches(local, meta):
    if not os.path.isfile(local) or os.path.getsize(local) != meta["size"]:
        return False
    if meta["sha256"]:
        return sha256_file(local) == meta["sha256"]
    return git_blob_sha1(local) == meta["blob_id"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trees", required=True)
    ap.add_argument("--old", required=True)
    ap.add_argument("--new", required=True)
    ap.add_argument("--old-sha-list", help="sha256sum output for OLD (paths relative to OLD)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    trees = json.load(open(args.trees))
    rev, files = trees["new"]["revision"], trees["new"]["files"]
    wanted = {p: m for p, m in files.items()
              if p.startswith("cylinder/") or p in ("version.json", "README.md")}

    old_sha = {}
    if args.old_sha_list:
        for line in open(args.old_sha_list):
            h, rel = line.rstrip("\n").split("  ", 1)
            old_sha[rel] = h

    plan_link, plan_dl = [], []
    for p, m in sorted(wanted.items()):
        old = os.path.join(args.old, p)
        if m["sha256"] and p in old_sha:
            same = os.path.isfile(old) and os.path.getsize(old) == m["size"] and old_sha[p] == m["sha256"]
        else:
            same = matches(old, m)
        (plan_link if same else plan_dl).append(p)

    total = sum(wanted[p]["size"] for p in plan_dl)
    print(f"revision {rev}")
    print(f"symlink {len(plan_link)} files ({sum(wanted[p]['size'] for p in plan_link)/2**30:.1f} GiB)")
    print(f"download {len(plan_dl)} files ({total/2**20:.1f} MiB):")
    for p in plan_dl:
        print("   ", p, wanted[p]["size"])
    if args.dry_run:
        return

    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    from huggingface_hub import hf_hub_download

    for p in plan_link:
        dst = os.path.join(args.new, p)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        if os.path.lexists(dst):
            sys.exit(f"refusing to overwrite existing {dst}")
        os.symlink(os.path.join(args.old, p), dst)
    for p in plan_dl:
        dst = os.path.join(args.new, p)
        if os.path.lexists(dst):
            sys.exit(f"refusing to overwrite existing {dst}")
        hf_hub_download(REPO, p, repo_type="dataset", revision=rev, local_dir=args.new)

    # verify every file in NEW against the remote tree
    bad = [p for p, m in wanted.items() if not matches(os.path.join(args.new, p), m)]
    print("verify:", "OK" if not bad else f"MISMATCH {bad}")
    manifest = {"revision": rev, "symlinked": plan_link, "downloaded": plan_dl, "verify_failed": bad}
    json.dump(manifest, open(os.path.join(args.new, "fetch_manifest.json"), "w"), indent=1)
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
