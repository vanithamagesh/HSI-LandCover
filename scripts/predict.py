"""Classify a cube with one or more trained models (probabilities are averaged) and write the class map.

    python scripts/predict.py --weights runs/IP_full_random_fcn/seed*/model.pt --image data/Indian_pines_corrected.mat --out maps/IP

The cube is resampled and standardised exactly as during training, with the statistics of the training scene;
--scene-stats recomputes the statistics on the new cube (useful when its radiometry differs). Writes <out>.npy
(uint8, 1..C), <out>.png (colour map with legend) and, for a GeoTIFF input with rasterio installed, a georeferenced
<out>.tif.

AnySensor checkpoints (--model anysensor) work with any band layout: give the bands of the new image with --sensor
(full, s2, landsat8, rgbn, rgb; resolved against the training wavelengths) or --bands (a text file with one line per band:
centre and FWHM in nm). The image is standardised with its own statistics.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import hsilc as H  # noqa: E402
from hsilc import anysensor as A  # noqa: E402
from hsilc.viz import save_class_map  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", type=Path, nargs="+", required=True)
    ap.add_argument("--image", type=Path, required=True)
    ap.add_argument("--key", help="variable name inside a .mat file")
    ap.add_argument("--out", type=Path, required=True, help="output path without extension")
    ap.add_argument("--scene-stats", action="store_true")
    ap.add_argument("--no-tta", action="store_true")
    ap.add_argument("--sensor", help="AnySensor: named band layout of the image")
    ap.add_argument("--bands", type=Path, help="AnySensor: text file with centre and FWHM (nm) of each image band")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    anysensor = torch.load(a.weights[0], map_location="cpu", weights_only=False)["cfg"]["model"] == "anysensor"
    if anysensor:
        return predict_anysensor(a)
    models = [H.load_checkpoint(w, a.device) for w in a.weights]
    ck = models[0][1]
    assert all(c["classes"] == ck["classes"] and c["bands"] == ck["bands"] for _, c in models), \
        "all checkpoints must share classes and bands"
    cube, profile = H.read_array(a.image, a.key)
    cube = cube[..., None] if cube.ndim == 2 else cube
    n_in = cube.shape[-1] if ck["response"] is None else ck["response"].shape[1]
    assert cube.shape[-1] == n_in, f"cube has {cube.shape[-1]} bands, the models expect {n_in}"
    x, _, _ = H.preprocess(cube, ck["response"], *((None, None) if a.scene_stats else (ck["mean"], ck["std"])))
    prob = np.mean([H.predict(m, x, a.device, tta=not a.no_tta) for m, _ in models], 0)
    write(a, prob, ck["classes"], profile)


def predict_anysensor(a):
    models = [A.load_anysensor(w, a.device) for w in a.weights]
    ck = models[0][1]
    if a.bands:
        desc = np.loadtxt(a.bands, ndmin=2).astype(np.float32)
    elif a.sensor:
        desc = H.sensor_bands(ck["wavelengths"], a.sensor)[2]
    else:
        raise SystemExit("give --sensor or --bands for an AnySensor model")
    cube, profile = H.read_array(a.image, a.key)
    cube = cube[..., None] if cube.ndim == 2 else cube
    assert cube.shape[-1] == len(desc), f"cube has {cube.shape[-1]} bands, the band description has {len(desc)}"
    x = H.preprocess(cube)[0]
    prob = np.mean([H.predict(m.set_sensor(desc), x, a.device, tta=not a.no_tta) for m, _ in models], 0)
    write(a, prob, ck["classes"], profile)


def write(a, prob, classes, profile):
    labels = (prob.argmax(0) + 1).astype(np.uint8)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    np.save(a.out.with_suffix(".npy"), labels)
    save_class_map(a.out.with_suffix(".png"), labels, classes)
    if profile is not None:
        import rasterio
        p = dict(profile, count=1, dtype="uint8", nodata=0)
        with rasterio.open(a.out.with_suffix(".tif"), "w", **p) as dst:
            dst.write(labels[None])
    counts = np.bincount(labels.ravel(), minlength=len(classes) + 1)[1:]
    for name, n in zip(classes, counts):
        print(f"{name:32s} {n:9d} px  {100 * n / labels.size:6.2f} %")


if __name__ == "__main__":
    main()
