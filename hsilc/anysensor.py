"""AnySensor: one spectral-spatial network for any multispectral or hyperspectral sensor.

  wavelength-aware encoder   a hypernetwork maps each band's centre wavelength and FWHM (Fourier features of the
                             centre, log FWHM) to the weights of a dynamic 1x1 convolution; band contributions are
                             averaged, so any number of bands gives an embedding of fixed width
  sensor-adaptive norm       the embedding is normalised with the statistics of the current input (batch norm without
                             running statistics), i.e. per sensor during training and per scene and sensor at test time
  backbone                   the spectral-spatial FCN of model.py on top of the embedding
  random sensor simulation   every training batch is converted, through Gaussian spectral responses, into the bands of a
                             randomly drawn virtual sensor (3-16 bands in one contiguous spectral window, FWHM 10-180 nm)
                             or, with probability p_full, kept at full spectral resolution

Simulated bands are standardised with scene statistics computed in closed form from the per-band statistics and the
band correlation matrix of the hyperspectral cube, so every virtual sensor is preprocessed exactly like a real one.
"""
from __future__ import annotations

import copy
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from .data import gaussian_srf, sensor_bands
from .engine import Crops, predict
from .model import SpectralSpatialFCN, count_parameters

TEST_SENSORS = ("full", "s2", "landsat8", "rgbn", "rgb")
DEFAULT_CFG = dict(model="anysensor", width=64, crop=96, batch=4, steps=2000, lr=1e-3, weight_decay=1e-4, ema=0.99,
                   val_every=100, seed=1, rss=True, fwhm=True, p_full=0.2, kmin=3, kmax=16, wmin=10.0, wmax=180.0,
                   val_sensors=3)


class WavelengthEncoder(nn.Module):
    def __init__(self, width=64, freqs=16, hidden=128, use_fwhm=True):
        super().__init__()
        self.register_buffer("freq", torch.logspace(math.log10(0.5), math.log10(64.0), freqs))   # cycles per um
        self.mlp = nn.Sequential(nn.Linear(2 * freqs + 2, hidden), nn.GELU(), nn.Linear(hidden, hidden), nn.GELU(),
                                 nn.Linear(hidden, width))
        self.norm = nn.BatchNorm2d(width, track_running_stats=False)
        self.use_fwhm = use_fwhm

    def weights(self, bands):
        """(K, 2) centre and FWHM in nm -> (K, width) dynamic 1x1 convolution weights."""
        ang = 2 * math.pi * (bands[:, :1] / 1000.0) * self.freq
        w = torch.log(bands[:, 1:2] / 10.0) * float(self.use_fwhm)
        return self.mlp(torch.cat([ang.sin(), ang.cos(), w, w * w], 1))

    def forward(self, x, bands):
        e = torch.einsum("nkhw,kd->ndhw", x, self.weights(bands)) / x.shape[1]
        return F.relu(self.norm(e))


class AnySensorFCN(nn.Module):
    def __init__(self, num_classes, width=64, use_fwhm=True):
        super().__init__()
        self.encoder = WavelengthEncoder(width, use_fwhm=use_fwhm)
        self.fcn = SpectralSpatialFCN(width, num_classes, width)
        self.bands = None

    def set_sensor(self, bands):
        """Band descriptors (K, 2): centre and FWHM in nm of the bands fed to the next forward passes."""
        self.bands = torch.as_tensor(np.asarray(bands), dtype=torch.float32)
        return self

    def forward(self, x):
        return self.fcn(self.encoder(x, self.bands.to(x.device)))


class Simulator:
    """Turns the standardised hyperspectral cube of a scene into the standardised bands of any sensor."""

    def __init__(self, scene):
        self.wl = np.asarray(scene.meta["wavelengths"], np.float64)
        self.std = scene.std.astype(np.float64)
        z = scene.x.reshape(-1, scene.x.shape[-1]).astype(np.float64)
        self.corr = z.T @ z / len(z)                                    # band correlation of the standardised cube
        self.native = sensor_bands(self.wl, "full")[2]

    def matrix(self, response):
        """(K, B) matrix that maps standardised hyperspectral bands to standardised simulated bands."""
        m = response.astype(np.float64) * self.std[None]
        sd = np.sqrt(np.maximum(np.einsum("kb,bc,kc->k", m, self.corr, m), 1e-12))
        return (m / sd[:, None]).astype(np.float32)

    def sensor(self, name):
        """(matrix or None for full resolution, band descriptors) of a named sensor."""
        response, _, desc = sensor_bands(self.wl, name)
        return (None if response is None else self.matrix(response)), desc

    def random(self, rng, kmin=3, kmax=16, wmin=10.0, wmax=180.0):
        """A random virtual sensor. Like the common sensor families (imaging spectrometers and VNIR-SWIR satellites,
        VNIR satellites and aerial cameras, RGB cameras), its bands lie in one contiguous spectral window: the whole
        range, the visible and near-infrared part (up to 1000 nm), the visible part (up to 720 nm) or a random window at
        least 250 nm wide, with equal probability. Band centres are measured wavelengths +-5 nm, so never inside
        removed absorption bands."""
        lo, hi = self.wl.min(), self.wl.max(); u = rng.random()
        if u < 1 / 4:
            a, b = lo, hi
        elif u < 2 / 4:
            a, b = lo, min(hi, 1000.0)
        elif u < 3 / 4:
            a, b = lo, min(hi, 720.0)
        else:
            width = rng.uniform(min(250.0, hi - lo), hi - lo); a = rng.uniform(lo, hi - width); b = a + width
        pool = self.wl[(self.wl >= a) & (self.wl <= b)]
        k = int(rng.integers(kmin, kmax + 1))
        c = np.sort(np.clip(rng.choice(pool, k) + rng.uniform(-5, 5, k), lo, hi))
        w = np.exp(rng.uniform(math.log(wmin), math.log(wmax), k))
        return self.matrix(gaussian_srf(self.wl, c, w)), np.stack([c, w], 1).astype(np.float32)

    def cube(self, x, matrix):
        """Standardised (H, W, K) cube of a sensor from the standardised hyperspectral cube x (H, W, B)."""
        return x if matrix is None else (x @ matrix.T).astype(np.float32)


def predict_sensor(model, scene, sim, name, device, tta=True):
    """Class probabilities (C, H, W) of the scene as seen by the named sensor."""
    matrix, desc = sim.sensor(name)
    model.set_sensor(desc)
    return predict(model, sim.cube(scene.x, matrix), device, tta=tta)


def train_anysensor(cfg, scene, train_lab, val_lab, out_dir, device, log=print):
    """Train AnySensor with random sensor simulation. Checkpoints are selected on the validation pixels averaged over the
    full-resolution cube and `val_sensors` fixed random sensors (never the named test sensors)."""
    cfg = dict(DEFAULT_CFG, **cfg); out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(cfg["seed"]); np.random.seed(cfg["seed"]); random.seed(cfg["seed"])
    sim = Simulator(scene); rng = np.random.default_rng(cfg["seed"])
    model = AnySensorFCN(scene.num_classes, cfg["width"], cfg["fwhm"]).to(device)
    ema = copy.deepcopy(model).eval()
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cfg["lr"], total_steps=cfg["steps"], pct_start=0.1,
                                              div_factor=10, final_div_factor=100)
    dl = DataLoader(Crops(scene.x, train_lab, cfg["steps"] * cfg["batch"], cfg["crop"], cfg["seed"]),
                    batch_size=cfg["batch"], drop_last=True)
    vrng = np.random.default_rng(12345)                                 # the same validation sensors for every seed
    val = [(None, sim.native)] + [sim.random(vrng, cfg["kmin"], cfg["kmax"], cfg["wmin"], cfg["wmax"])
                                  for _ in range(cfg["val_sensors"] if cfg["rss"] else 0)]
    val = [(sim.cube(scene.x, m), d) for m, d in val]
    yv = val_lab[val_lab > 0] - 1
    hist = {"loss": [], "parameters": count_parameters(model), "config": cfg, "validation": []}
    best = {"oa": -1.0, "nll": np.inf}; t0 = time.time()
    for step, (x, y) in enumerate(dl, 1):
        if cfg["rss"] and rng.random() >= cfg["p_full"]:
            m, desc = sim.random(rng, cfg["kmin"], cfg["kmax"], cfg["wmin"], cfg["wmax"])
            x = torch.einsum("nbhw,kb->nkhw", x.to(device), torch.from_numpy(m).to(device))
        else:
            desc = sim.native; x = x.to(device)
        model.train(); model.set_sensor(desc)
        loss = F.cross_entropy(model(x), y.to(device), ignore_index=-1)
        opt.zero_grad(set_to_none=True); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 3.0)
        opt.step(); sch.step()
        with torch.no_grad():
            dec = min(cfg["ema"], (1 + step) / (10 + step))
            for e, p in zip(ema.parameters(), model.parameters()): e.lerp_(p, 1 - dec)
            for e, p in zip(ema.buffers(), model.buffers()):
                if e.dtype.is_floating_point: e.lerp_(p, 1 - dec)
                else: e.copy_(p)
        hist["loss"].append(loss.item())
        if step % cfg["val_every"] == 0 or step == cfg["steps"]:
            oas, nlls = [], []
            for cube, desc in val:
                prob = predict(ema.set_sensor(desc), cube, device, tta=False)[:, val_lab > 0]
                oas.append(float((prob.argmax(0) == yv).mean()))
                nlls.append(float(-np.log(np.maximum(prob[yv, np.arange(len(yv))], 1e-8)).mean()))
            oa, nll = float(np.mean(oas)), float(np.mean(nlls))
            hist["validation"].append({"step": step, "oa": oa, "nll": nll, "oa_per_sensor": oas})
            if step % (4 * cfg["val_every"]) == 0 or step == cfg["steps"]:
                log(json.dumps({"step": step, "of": cfg["steps"], "loss": round(float(np.mean(hist["loss"][-cfg["val_every"]:])), 4),
                                "val_oa": round(oa, 4), "val_nll": round(nll, 4), "minutes": round((time.time() - t0) / 60, 1)}))
            if (oa, -nll) > (best["oa"], -best["nll"]):
                best = {"oa": oa, "nll": nll, "step": step, "state": copy.deepcopy(ema.state_dict())}
    ema.load_state_dict(best.pop("state")); hist["selected"] = best; hist["minutes"] = (time.time() - t0) / 60
    log(f"selected step {best['step']} (validation OA {100 * best['oa']:.2f} averaged over {len(val)} sensors), "
        f"{hist['minutes']:.1f} min")
    torch.save({"model": ema.state_dict(), "cfg": cfg, "classes": scene.classes, "scene": scene.name,
                "wavelengths": sim.wl, "mean": scene.mean, "std": scene.std, "corr": sim.corr.astype(np.float32)},
               out_dir / "model.pt")
    (out_dir / "history.json").write_text(json.dumps(hist))
    return ema


def load_anysensor(path, device="cpu"):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    m = AnySensorFCN(len(ck["classes"]), ck["cfg"]["width"], ck["cfg"]["fwhm"])
    m.load_state_dict(ck["model"])
    return m.to(device).eval(), ck
