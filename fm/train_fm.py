"""Train a flow-matching model on RealPDEBench cylinder (plan B: own loop, official data/normalizer/metrics).

  torchrun --nproc_per_node=G fm/train_fm.py --config fm/configs/cylinder/unet_M.yaml \
      --train-data-type numerical --out RUN_DIR [--seed 0] [--iters N] [--init-from CKPT --lr-scale 0.1]
"""
import argparse, copy, json, math, os, random, sys, time

import numpy as np
import torch
import torch.distributed as dist
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pilfm import meta
from pilfm.fast_dataset import FastCylinderHFDataset
from pilfm.official_eval import Env, run_val
from fm.build import build_model, n_params
from fm.paths import fm_loss, euler_sample
from fm.wrapper import FMPredictor, fold

T_IN, T_OUT, C = 20, 20, 3
H, W = 64, 128


def ema_decay(step):
    return min(0.9999, (1 + step) / (10 + step))


@torch.no_grad()
def ema_update(ema, model, decay):
    for pe, pm in zip(ema.parameters(), model.parameters()):
        pe.mul_(decay).add_(pm.detach(), alpha=1 - decay)


def lr_at(step, base, warmup, total):
    if step < warmup:
        return base * (step + 1) / warmup
    return 0.5 * base * (1 + math.cos(math.pi * min(1.0, (step - warmup) / max(1, total - warmup))))


def infinite(loader, sampler):
    epoch = 0
    while True:
        if sampler is not None:
            sampler.set_epoch(epoch)
        for batch in loader:
            yield batch
        epoch += 1


def val_subset_batches(env, stride, batch_size):
    idx = list(range(0, len(env.val), stride))
    sub = torch.utils.data.Subset(env.val, idx)
    loader = torch.utils.data.DataLoader(sub, batch_size=batch_size, shuffle=False, num_workers=env.num_workers)
    return [(x.clone(), y.clone()) for x, y in loader], idx


def save(path, **state):
    tmp = path + ".tmp"
    torch.save(state, tmp)
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--train-data-type", required=True, choices=["numerical", "real"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--iters", type=int, default=None, help="override train.iters")
    ap.add_argument("--val-every", type=int, default=None)
    ap.add_argument("--init-from", default=None, help="finetune: initialize model and EMA from this checkpoint's EMA")
    ap.add_argument("--lr-scale", type=float, default=1.0)
    ap.add_argument("--fixed-batch", action="store_true", help="overfit test: reuse the first batch forever")
    ap.add_argument("--no-val", action="store_true")
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--ddp-check", action="store_true",
                    help="every rank loads the same global batch and noise and trains on its slice (loss == single-GPU)")
    args = ap.parse_args()

    ddp = "RANK" in os.environ
    if ddp:
        dist.init_process_group("nccl")
        rank, world = dist.get_rank(), dist.get_world_size()
        local = int(os.environ["LOCAL_RANK"])
    else:
        rank, world, local = 0, 1, 0
    device = torch.device(f"cuda:{local}")
    torch.cuda.set_device(device)

    cfg_text = open(args.config).read()
    cfg = yaml.safe_load(cfg_text)
    tc, dc, vc = cfg["train"], cfg["data"], cfg["val"]
    iters = args.iters or tc["iters"]
    val_every = args.val_every or vc["every"]
    gbs = tc["batch_size"]
    assert gbs % world == 0
    bs = gbs // world

    random.seed(args.seed + rank); np.random.seed(args.seed + rank); torch.manual_seed(args.seed + rank)

    # data: official dataset semantics (fast subclass), official normalizer
    ds = FastCylinderHFDataset(dataset_name="cylinder", dataset_root=dc["root"], mode="train",
                               dataset_type=args.train_data_type, mask_prob=dc["mask_prob"],
                               noise_scale=dc["noise_scale"])
    use_sampler = ddp and not args.ddp_check
    sampler = torch.utils.data.distributed.DistributedSampler(ds, world, rank, shuffle=True, seed=args.seed) if use_sampler else None
    g = torch.Generator(); g.manual_seed(args.seed)
    loader = torch.utils.data.DataLoader(ds, batch_size=gbs if args.ddp_check else bs, shuffle=sampler is None,
                                         sampler=sampler, generator=g,
                                         num_workers=dc["num_workers"], pin_memory=True, drop_last=True,
                                         persistent_workers=True)
    env = Env(dc["root"], num_workers=dc["num_workers"], need_test=False, need_val=(rank == 0 and not args.no_val))
    norm = env.normalizer(device)

    model = build_model(cfg["model"], T_OUT * C + T_IN * C, T_OUT * C, H, W).to(device)
    if args.init_from:
        src = torch.load(args.init_from, map_location="cpu", weights_only=False)
        model.load_state_dict(src["ema"])
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    net = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local]) if ddp else model
    base_lr = tc["lr"] * args.lr_scale
    opt = torch.optim.AdamW(model.parameters(), lr=base_lr, betas=tuple(tc.get("betas", (0.9, 0.999))),
                            weight_decay=tc["weight_decay"])

    os.makedirs(args.out, exist_ok=True)
    info = meta.collect(data_root=dc["root"], config=cfg_text, seed=args.seed, checkpoint=args.init_from,
                        train_data_type=args.train_data_type, world_size=world, per_rank_batch=bs, iters=iters,
                        lr=base_lr, n_params=n_params(model), init_from=args.init_from)
    if rank == 0:
        json.dump(info, open(os.path.join(args.out, "run_meta.json"), "w"), indent=1)
        log = open(os.path.join(args.out, "log.jsonl"), "a")
        vb = None
        if not args.no_val:
            vb, vidx = val_subset_batches(env, vc["subset_stride"], vc.get("batch_size", 64))
            print(f"val subset: {len(vidx)} samples (stride {vc['subset_stride']})", flush=True)

    it = infinite(loader, sampler)
    fixed = None
    best = float("inf")
    t_last, n_last = time.time(), 0
    torch.cuda.reset_peak_memory_stats(device)
    for step in range(iters):
        x, y = (fixed if fixed is not None else next(it))
        if args.fixed_batch and fixed is None:
            fixed = (x, y)
        x, y = norm.preprocess(x, y)
        cond, y1 = fold(x), fold(y)
        y0 = t = None
        if args.ddp_check:
            ng = torch.Generator(device=device).manual_seed(args.seed * 100003 + step)
            y0 = torch.randn(y1.shape, device=device, generator=ng)
            t = torch.rand(y1.shape[0], device=device, generator=ng)
            sl = slice(rank * bs, (rank + 1) * bs)
            cond, y1, y0, t = cond[sl], y1[sl], y0[sl], t[sl]
        for gr in opt.param_groups:
            gr["lr"] = lr_at(step, base_lr, tc["warmup"], iters)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = fm_loss(net, y1, cond, y0, t)  # loss itself is computed in fp32 inside fm_loss
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), tc["clip_grad"])
        opt.step()
        ema_update(ema, model, ema_decay(step))
        n_last += gbs
        if ddp and (step + 1) % args.log_every == 0:
            loss = loss.detach().clone(); dist.all_reduce(loss); loss /= world

        if rank == 0 and (step + 1) % args.log_every == 0:
            dt = time.time() - t_last
            rec = {"step": step + 1, "loss": loss.item(), "grad_norm": float(gnorm), "lr": opt.param_groups[0]["lr"],
                   "samples_per_s": n_last / dt, "it_per_s": n_last / gbs / dt,
                   "peak_mem_gb": torch.cuda.max_memory_allocated(device) / 2**30}
            log.write(json.dumps(rec) + "\n"); log.flush()
            print(json.dumps(rec), flush=True)
            t_last, n_last = time.time(), 0

        last = step + 1 == iters
        if (step + 1) % val_every == 0 or last:
            if rank == 0:
                state = dict(model=model.state_dict(), ema=ema.state_dict(), opt=opt.state_dict(), step=step + 1,
                             config=cfg, meta=info, best_val_rmse=best)
                if not args.no_val:
                    pred = FMPredictor(ema, T_OUT, C, n_steps=vc["n_steps"], n_samples=1, seed=vc["noise_seed"])
                    m = run_val(env, pred, vc.get("batch_size", 64), device, metrics_device="cpu", batches=vb)
                    rec = {"step": step + 1, "val": m}
                    log.write(json.dumps(rec) + "\n"); log.flush()
                    print(f"step {step+1} val_rmse={m['rmse']:.6f}", flush=True)
                    if m["rmse"] < best:
                        best = m["rmse"]
                        state["best_val_rmse"] = best
                        save(os.path.join(args.out, "best.pt"), **state)
                save(os.path.join(args.out, "last.pt"), **state)
            if ddp:
                dist.barrier()
    if args.fixed_batch and rank == 0:
        # Step 5.4: N=20 Euler samples on the memorized batch, raw and EMA weights, rel L2 in normalized space
        x, y = norm.preprocess(*fixed)
        cond, y1 = fold(x), fold(y)
        res = {}
        for tag, net_ in (("raw", model), ("ema", ema)):
            net_.eval()
            gen = torch.Generator(device=device).manual_seed(0)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                ys = euler_sample(net_, cond, y1.shape, 20, generator=gen).float()
            rel = ((ys - y1).flatten(1).norm(dim=1) / y1.flatten(1).norm(dim=1))
            res[tag] = {"rel_l2_mean": rel.mean().item(), "rel_l2_max": rel.max().item()}
        json.dump(res, open(os.path.join(args.out, "overfit_eval.json"), "w"), indent=1)
        print("overfit eval:", res, flush=True)
    if ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
