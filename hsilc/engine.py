"""Crop sampling, training on sparse labels, tiled inference and accuracy metrics."""
from __future__ import annotations

import copy
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from .model import build_model, count_parameters

DEFAULT_CFG = dict(model="fcn", width=64, crop=96, batch=4, steps=1000, lr=1e-3, weight_decay=1e-4, ema=0.99,
                   val_every=50, seed=1)


class Crops(Dataset):
    """Random training crops. Each crop contains a training pixel of a uniformly drawn class (class-balanced sampling);
    pixels outside the training set get label -1 and are ignored by the loss. Augmentation: 90-degree rotations
    (square crops), flips, and a random gain and offset of the standardised spectra."""

    def __init__(self, x, train, n, crop, seed):
        self.x, self.y = x, train.astype(np.int64) - 1
        self.coords = [np.argwhere(train == c) for c in np.unique(train[train > 0])]
        self.n, self.crop, self.seed = n, crop, seed

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        rng = random.Random(self.seed * 1_000_003 + i)
        pts = self.coords[rng.randrange(len(self.coords))]; cy, cx = pts[rng.randrange(len(pts))]
        H, W = self.y.shape; ph, pw = min(self.crop, H), min(self.crop, W)
        top = min(max(int(cy) - rng.randrange(ph), 0), H - ph); left = min(max(int(cx) - rng.randrange(pw), 0), W - pw)
        x, y = self.x[top:top + ph, left:left + pw], self.y[top:top + ph, left:left + pw]
        if ph == pw:
            r = rng.randrange(4); x, y = np.rot90(x, r), np.rot90(y, r)
        if rng.random() < 0.5: x, y = x[:, ::-1], y[:, ::-1]
        if rng.random() < 0.5: x, y = x[::-1], y[::-1]
        x = x * rng.uniform(0.9, 1.1) + rng.uniform(-0.1, 0.1)
        return torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1))).float(), torch.from_numpy(np.ascontiguousarray(y))


# ---------------------------------------------------------------- inference
def _ramp(n, m):
    i = np.arange(n)
    return np.clip(np.minimum(i + 1, n - i) / max(m, 1), 1e-3, 1.0)


@torch.inference_mode()
def predict(model, x, device, tta=True, tile=512, overlap=64):
    """Class probabilities (C, H, W) for a normalised (H, W, B) cube; with tta the four flip views are averaged.
    Scenes larger than `tile` are processed in overlapping tiles blended with linear ramps."""
    H, W = x.shape[:2]
    t = torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1))).float()[None]

    def run(xt):
        xt = xt.to(device); outs = []
        for dims in ([(), (-1,), (-2,), (-2, -1)] if tta else [()]):
            p = torch.softmax(model(torch.flip(xt, dims) if dims else xt).float(), 1)
            outs.append(torch.flip(p, dims) if dims else p)
        return torch.stack(outs).mean(0)[0].cpu()

    if H <= tile and W <= tile:
        return run(t).numpy()
    starts = lambda n: sorted({*range(0, max(n - tile, 0) + 1, tile - overlap), max(n - tile, 0)})
    acc, wsum = None, torch.zeros(H, W)
    for top in starts(H):
        for left in starts(W):
            p = run(t[..., top:top + tile, left:left + tile]); h, w = p.shape[-2:]
            wt = torch.from_numpy(np.outer(_ramp(h, overlap), _ramp(w, overlap))).float()
            if acc is None:
                acc = torch.zeros(p.shape[0], H, W)
            acc[:, top:top + h, left:left + w] += p * wt; wsum[top:top + h, left:left + w] += wt
    return (acc / wsum).numpy()


# ---------------------------------------------------------------- metrics
def confusion(pred, labels, num_classes):
    """Confusion matrix (rows: reference, columns: prediction) over pixels with labels > 0; pred holds 0-based indices."""
    m = labels > 0
    return np.bincount((labels[m] - 1) * num_classes + pred[m], minlength=num_classes ** 2).reshape(num_classes, num_classes)


def scores(cm):
    """Overall accuracy, average (per-class) accuracy, Cohen's kappa and per-class accuracy; classes absent from the
    reference are left out of AA."""
    n = cm.sum(); oa = np.trace(cm) / n; support = cm.sum(1)
    per_class = np.where(support > 0, np.diag(cm) / np.maximum(support, 1), np.nan)
    pe = float((cm.sum(0) * support).sum()) / n ** 2
    return dict(oa=float(oa), aa=float(np.nanmean(per_class)), kappa=float((oa - pe) / (1 - pe)),
                per_class=[None if np.isnan(a) else float(a) for a in per_class], n_pixels=int(n),
                n_classes=int((support > 0).sum()))


def evaluate(prob_or_pred, labels, num_classes):
    pred = prob_or_pred.argmax(0) if prob_or_pred.ndim == 3 else prob_or_pred
    cm = confusion(pred, labels, num_classes)
    return dict(scores(cm), confusion=cm.tolist())


# ---------------------------------------------------------------- training
def train(cfg, scene, train_lab, val_lab, out_dir, device, log=print):
    """Train one model; the EMA weights with the best validation overall accuracy (ties: lowest validation cross-entropy)
    are saved with the preprocessing."""
    cfg = dict(DEFAULT_CFG, **cfg); out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(cfg["seed"]); np.random.seed(cfg["seed"]); random.seed(cfg["seed"])
    model = build_model(cfg["model"], scene.x.shape[-1], scene.num_classes, cfg["width"]).to(device)
    ema = copy.deepcopy(model).eval()
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cfg["lr"], total_steps=cfg["steps"], pct_start=0.1,
                                              div_factor=10, final_div_factor=100)
    dl = DataLoader(Crops(scene.x, train_lab, cfg["steps"] * cfg["batch"], cfg["crop"], cfg["seed"]),
                    batch_size=cfg["batch"], drop_last=True)
    hist = {"loss": [], "parameters": count_parameters(model), "config": cfg, "validation": []}
    best = {"oa": -1.0, "nll": np.inf}; t0 = time.time()
    for step, (x, y) in enumerate(dl, 1):
        model.train()
        loss = F.cross_entropy(model(x.to(device)), y.to(device), ignore_index=-1)
        opt.zero_grad(set_to_none=True); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 3.0)
        opt.step(); sch.step()
        with torch.no_grad():                                   # EMA of weights and BN statistics
            dec = min(cfg["ema"], (1 + step) / (10 + step))
            for e, p in zip(ema.parameters(), model.parameters()): e.lerp_(p, 1 - dec)
            for e, p in zip(ema.buffers(), model.buffers()):
                if e.dtype.is_floating_point: e.lerp_(p, 1 - dec)
                else: e.copy_(p)
        hist["loss"].append(loss.item())
        if step % cfg["val_every"] == 0 or step == cfg["steps"]:
            prob = predict(ema, scene.x, device, tta=False)[:, val_lab > 0]; yv = val_lab[val_lab > 0] - 1
            oa = float((prob.argmax(0) == yv).mean())
            nll = float(-np.log(np.maximum(prob[yv, np.arange(len(yv))], 1e-8)).mean())
            hist["validation"].append({"step": step, "oa": oa, "nll": nll})
            if step % (4 * cfg["val_every"]) == 0 or step == cfg["steps"]:
                log(json.dumps({"step": step, "of": cfg["steps"], "loss": round(float(np.mean(hist["loss"][-cfg["val_every"]:])), 4),
                                "val_oa": round(oa, 4), "val_nll": round(nll, 4), "minutes": round((time.time() - t0) / 60, 1)}))
            if (oa, -nll) > (best["oa"], -best["nll"]):           # ties in OA (small validation sets) -> lower NLL
                best = {"oa": oa, "nll": nll, "step": step, "state": copy.deepcopy(ema.state_dict())}
    ema.load_state_dict(best.pop("state")); hist["selected"] = best; hist["minutes"] = (time.time() - t0) / 60
    log(f"selected step {best['step']} (validation OA {100 * best['oa']:.2f}), {hist['minutes']:.1f} min")
    torch.save({"model": ema.state_dict(), "cfg": cfg, "classes": scene.classes, "bands": scene.bands,
                "mean": scene.mean, "std": scene.std, "response": scene.response, "scene": scene.name,
                "meta": {k: v for k, v in scene.meta.items() if k != "profile"}}, out_dir / "model.pt")
    (out_dir / "history.json").write_text(json.dumps(hist))
    return ema
