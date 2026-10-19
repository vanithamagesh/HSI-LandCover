"""Scenes, spectral simulation, normalisation and label splits.

A scene is an (H, W, B) cube with an (H, W) label map in which 0 marks unlabelled pixels and 1..C the classes.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy import io as sio


def _keep(n_total, removed):
    """Linearly spaced nominal wavelengths of an AVIRIS cube after the listed (1-based) bands were removed."""
    wl = np.linspace(400.0, 2500.0, n_total)
    return np.delete(wl, np.asarray(removed) - 1)


def _r(a, b):
    return list(range(a, b + 1))


# Benchmark scenes (GIC, University of the Basque Country). Wavelengths are nominal and linearly spaced:
# the band centres are not stored in the .mat files, so the spectral simulation below is approximate.
SCENES = {
    "IP": dict(
        name="Indian Pines", sensor="AVIRIS", gsd=20.0,
        image="Indian_pines_corrected.mat", labels="Indian_pines_gt.mat", shape=(145, 145, 200), labelled=10249,
        wavelengths=_keep(220, _r(104, 108) + _r(150, 163) + [220]),
        classes=["Alfalfa", "Corn-notill", "Corn-mintill", "Corn", "Grass-pasture", "Grass-trees", "Grass-pasture-mowed",
                 "Hay-windrowed", "Oats", "Soybean-notill", "Soybean-mintill", "Soybean-clean", "Wheat", "Woods",
                 "Buildings-Grass-Trees-Drives", "Stone-Steel-Towers"]),
    "PU": dict(
        name="Pavia University", sensor="ROSIS", gsd=1.3,
        image="PaviaU.mat", labels="PaviaU_gt.mat", shape=(610, 340, 103), labelled=42776,
        wavelengths=np.linspace(430.0, 860.0, 103),
        classes=["Asphalt", "Meadows", "Gravel", "Trees", "Painted metal sheets", "Bare soil", "Bitumen",
                 "Self-blocking bricks", "Shadows"]),
    "SA": dict(
        name="Salinas", sensor="AVIRIS", gsd=3.7,
        image="Salinas_corrected.mat", labels="Salinas_gt.mat", shape=(512, 217, 204), labelled=54129,
        wavelengths=_keep(224, _r(108, 112) + _r(154, 167) + [224]),
        classes=["Brocoli green weeds 1", "Brocoli green weeds 2", "Fallow", "Fallow rough plow", "Fallow smooth",
                 "Stubble", "Celery", "Grapes untrained", "Soil vineyard develop", "Corn senesced green weeds",
                 "Lettuce romaine 4wk", "Lettuce romaine 5wk", "Lettuce romaine 6wk", "Lettuce romaine 7wk",
                 "Vineyard untrained", "Vineyard vertical trellis"]),
}

# Sentinel-2A MSI bands at 10 m and 20 m: (centre, FWHM) in nm. The 60 m atmospheric bands (B1, B9, B10) are omitted.
S2_BANDS = {"B2": (492, 66), "B3": (560, 36), "B4": (665, 31), "B5": (704, 15), "B6": (740, 15), "B7": (783, 20),
            "B8": (833, 106), "B8A": (865, 21), "B11": (1614, 91), "B12": (2202, 175)}
RGB_BANDS = ("B4", "B3", "B2")
# Landsat-8 OLI reflective bands from the nominal band edges (USGS): (centre, width) in nm. Pan (B8) and cirrus (B9) omitted.
L8_BANDS = {"B1": (440, 20), "B2": (480, 60), "B3": (560, 60), "B4": (655, 30), "B5": (865, 30), "B6": (1610, 80),
            "B7": (2200, 180)}
SENSORS = {"s2": S2_BANDS, "landsat8": L8_BANDS, "rgbn": {b: S2_BANDS[b] for b in ("B2", "B3", "B4", "B8")},
           "rgb": {b: S2_BANDS[b] for b in RGB_BANDS}}


def gaussian_srf(wavelengths, centres, fwhms):
    """Gaussian spectral responses (one row per band, each summing to 1) sampled at the given wavelengths."""
    wl = np.asarray(wavelengths, np.float64)[None]
    c = np.asarray(centres, np.float64)[:, None]; s = np.asarray(fwhms, np.float64)[:, None] / 2.3548
    w = np.exp(-0.5 * ((wl - c) / s) ** 2)
    if (w.sum(1) < 1e-6).any():
        raise ValueError("a band lies outside the sensor range")
    return (w / w.sum(1, keepdims=True)).astype(np.float32)


def sensor_bands(wavelengths, mode):
    """Response matrix (n_out, n_bands) or None, band names and an (n_out, 2) array of centre and FWHM in nm.

    'full' keeps every band (FWHM = median band spacing); 's2', 'landsat8', 'rgbn' and 'rgb' simulate the bands of
    SENSORS[mode] whose centre lies inside the sensor range."""
    wl = np.asarray(wavelengths, dtype=np.float64)
    if mode == "full":
        fwhm = float(np.median(np.diff(wl))) if len(wl) > 1 else 10.0
        return None, [f"{w:.0f} nm" for w in wl], np.stack([wl, np.full_like(wl, fwhm)], 1).astype(np.float32)
    if mode not in SENSORS:
        raise ValueError(mode)
    bands = {b: v for b, v in SENSORS[mode].items() if wl.min() <= v[0] <= wl.max()}
    desc = np.array(list(bands.values()), np.float32)
    return gaussian_srf(wl, desc[:, 0], desc[:, 1]), list(bands), desc


def spectral_response(wavelengths, mode):
    """Gaussian spectral response matrix (n_out, n_bands) and output band names (None for 'full')."""
    response, names, _ = sensor_bands(wavelengths, mode)
    return response, names


def read_array(path, key=None):
    """(H, W[, B]) array from .mat (first non-private variable unless key is given), .npy or GeoTIFF.
    GeoTIFFs need rasterio or tifffile; with rasterio the georeferencing profile is returned as well."""
    path = Path(path); suffix = path.suffix.lower()
    if suffix == ".mat":
        m = sio.loadmat(path)
        k = key or next(k for k in m if not k.startswith("__"))
        return np.asarray(m[k]), None
    if suffix == ".npy":
        return np.load(path, allow_pickle=False), None
    if suffix in (".tif", ".tiff"):
        try:
            import rasterio
            with rasterio.open(path) as src:
                a = src.read(); profile = src.profile
            return (a[0] if a.shape[0] == 1 else a.transpose(1, 2, 0)), profile
        except ImportError:
            import tifffile
            a = tifffile.imread(path)
            if a.ndim == 3 and a.shape[0] < min(a.shape[1:]):       # bands first
                a = a.transpose(1, 2, 0)
            return a, None
    raise ValueError(f"unsupported file type: {path}")


@dataclass
class Scene:
    """Normalised cube x (H, W, B, float32), labels (H, W, int64, 0 = unlabelled) and the preprocessing used."""
    name: str
    x: np.ndarray
    labels: np.ndarray
    classes: list
    bands: list
    mean: np.ndarray
    std: np.ndarray
    response: np.ndarray | None = None
    meta: dict = field(default_factory=dict)

    @property
    def num_classes(self):
        return len(self.classes)


def preprocess(cube, response=None, mean=None, std=None):
    """Spectral resampling (optional) followed by per-band standardisation.
    Statistics are computed over all pixels of the cube unless given (e.g. from a training scene)."""
    x = cube.astype(np.float32)
    if response is not None:
        x = x @ response.T
    if mean is None:
        flat = x.reshape(-1, x.shape[-1]); mean = flat.mean(0); std = flat.std(0)
    return ((x - mean) / np.maximum(std, 1e-6)).astype(np.float32), mean.astype(np.float32), std.astype(np.float32)


def load_scene(root, key, spectral="full"):
    """Benchmark scene IP, PU or SA from the folder filled by scripts/get_data.sh."""
    s = SCENES[key]; root = Path(root)
    cube, _ = read_array(root / s["image"]); labels, _ = read_array(root / s["labels"])
    assert cube.shape == s["shape"] and int((labels > 0).sum()) == s["labelled"], \
        f"{key}: unexpected content {cube.shape}, {int((labels > 0).sum())} labelled pixels"
    response, bands = spectral_response(s["wavelengths"], spectral)
    x, mean, std = preprocess(cube, response)
    return Scene(key, x, labels.astype(np.int64), list(s["classes"]), bands, mean, std, response,
                 dict(spectral=spectral, wavelengths=s["wavelengths"].tolist()))


def load_custom(image, labels, classes=None, wavelengths=None, spectral="full", key=None):
    """Any multi- or hyperspectral raster with a label raster (0 = unlabelled, 1..C = classes).
    Spectral simulation ('s2', 'rgb') needs the band centre wavelengths in nm."""
    cube, profile = read_array(image, key); lab, _ = read_array(labels)
    if cube.ndim == 2:
        cube = cube[..., None]
    assert cube.shape[:2] == lab.shape, f"image {cube.shape} and labels {lab.shape} differ in size"
    if spectral != "full" and wavelengths is None:
        raise ValueError("spectral simulation needs --wavelengths")
    response, bands = spectral_response(wavelengths if wavelengths is not None else np.arange(cube.shape[-1]), spectral)
    if wavelengths is None:
        bands = [f"band {i + 1}" for i in range(cube.shape[-1])]
    x, mean, std = preprocess(cube, response)
    n = int(lab.max())
    classes = list(classes) if classes else [f"class {i}" for i in range(1, n + 1)]
    assert len(classes) >= n, f"{len(classes)} class names for {n} classes"
    meta = dict(spectral=spectral, profile=profile)
    if wavelengths is not None:
        meta["wavelengths"] = [float(w) for w in np.asarray(wavelengths).ravel()]
    return Scene(Path(image).stem, x, lab.astype(np.int64), classes, bands, mean, std, response, meta)


# ---------------------------------------------------------------- label splits
def _sample(labels, pool, per_class, val_frac, rng):
    """Draw min(per_class, half of the pool) labelled pixels per class from pool; val_frac of them for validation."""
    train = np.zeros_like(labels); val = np.zeros_like(labels)
    for c in range(1, int(labels.max()) + 1):
        idx = np.flatnonzero((labels.ravel() == c) & pool.ravel())
        if len(idx) == 0:
            continue
        n = min(per_class, max(len(idx) // 2, 1)); n_val = int(round(n * val_frac)) if n > 1 else 0
        pick = rng.permutation(idx)[:n]
        val.flat[pick[:n_val]] = c; train.flat[pick[n_val:]] = c
    return train, val


def split_random(labels, per_class=50, val_frac=0.2, seed=0):
    """Stratified random split: per_class labelled pixels per class (at most half of a class), of which val_frac are
    held out for validation. All remaining labelled pixels form the test set. Returns (train, val, test) label maps."""
    rng = np.random.default_rng(seed)
    train, val = _sample(labels, labels > 0, per_class, val_frac, rng)
    test = np.where((labels > 0) & (train == 0) & (val == 0), labels, 0)
    return train, val, test


def split_blocks(labels, per_class=50, val_frac=0.2, seed=0, block=None, train_frac=0.5, tries=1000):
    """Spatially disjoint split. The scene is cut into square blocks (default side min(H, W) / 8) that are assigned
    to the training side with probability train_frac. Training and validation pixels are drawn from training blocks
    only and every labelled pixel in the other blocks is a test pixel; labelled pixels of training blocks that are not
    drawn are not used. Assignments are redrawn until every class has at least two pixels on the training side."""
    H, W = labels.shape; block = block or max(8, int(round(min(H, W) / 8)))
    rng = np.random.default_rng(seed); n_cls = int(labels.max())
    gy, gx = -(-H // block), -(-W // block)
    for _ in range(tries):
        side = rng.random((gy, gx)) < train_frac
        pool = side.repeat(block, 0).repeat(block, 1)[:H, :W]
        counts = np.bincount(labels[pool & (labels > 0)], minlength=n_cls + 1)[1:]
        if counts.min() >= 2 and (labels[~pool] > 0).any():
            break
    else:
        raise RuntimeError("no block assignment puts every class on the training side; use a smaller block")
    train, val = _sample(labels, pool, per_class, val_frac, rng)
    test = np.where(~pool, labels, 0)
    return train, val, test


def make_split(labels, kind, per_class, val_frac, seed, block=None):
    if kind == "random":
        return split_random(labels, per_class, val_frac, seed)
    if kind == "block":
        return split_blocks(labels, per_class, val_frac, seed, block)
    raise ValueError(kind)
