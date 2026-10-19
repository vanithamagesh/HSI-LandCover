"""Train and test on one scene for several seeds; an SVM baseline is fitted on the same splits.

    python scripts/train_eval.py --data data --scene IP --spectral full --split random --out runs
    python scripts/train_eval.py --image my_cube.tif --labels my_labels.tif --out runs     # your own raster

Seed s draws its own label split (split seed s) and initialises the network with seed s. Per class, --per-class
labelled pixels are drawn (at most half of a class); --val-frac of them are held out for checkpoint selection and
SVM hyper-parameters. All remaining labelled pixels (random split) or all labelled pixels of the test blocks
(block split) are used once, for the final test. Finished seeds are skipped on restart.

--model anysensor trains one sensor-agnostic network (hsilc/anysensor.py) on the full-resolution cube and tests it on
the hyperspectral bands and on simulated Sentinel-2, Landsat-8, RGB+NIR and RGB bands; --no-rss and --no-fwhm are its
ablations (no random sensor simulation; band centres only).
"""
import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import hsilc as H  # noqa: E402
from hsilc import anysensor as A  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("data"), help="folder with the benchmark .mat files")
    ap.add_argument("--scene", choices=sorted(H.SCENES), help="benchmark scene")
    ap.add_argument("--image", type=Path, help="own cube (.tif, .npy or .mat), used instead of --scene")
    ap.add_argument("--labels", type=Path, help="own label raster (0 = unlabelled, 1..C = classes)")
    ap.add_argument("--classes", nargs="+", help="class names of the own label raster")
    ap.add_argument("--wavelengths", type=Path, help="text file with the band centres in nm (own cube)")
    ap.add_argument("--spectral", choices=("full",) + tuple(H.SENSORS), default="full")
    ap.add_argument("--split", choices=("random", "block"), default="random")
    ap.add_argument("--block", type=int, help="block side in pixels for --split block (default min(H, W) / 8)")
    ap.add_argument("--per-class", type=int, default=50)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--model", choices=sorted(H.model.MODELS) + ["anysensor"], default="fcn")
    ap.add_argument("--no-rss", action="store_true", help="anysensor ablation: train on the full-resolution cube only")
    ap.add_argument("--no-fwhm", action="store_true", help="anysensor ablation: condition on band centres only")
    ap.add_argument("--width", type=int, default=64)
    ap.add_argument("--crop", type=int, default=96)
    ap.add_argument("--steps", type=int, help="training steps (default 1000; 2000 for anysensor)")
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--no-svm", action="store_true")
    ap.add_argument("--out", type=Path, default=Path("runs"))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    anysensor = a.model == "anysensor"
    if anysensor and a.spectral != "full":
        ap.error("anysensor is trained on the full-resolution cube (--spectral full)")
    a.steps = a.steps or (2000 if anysensor else 1000); a.no_svm = a.no_svm or anysensor
    tag = a.model + ("-norss" if a.no_rss else "") + ("-nofwhm" if a.no_fwhm else "")
    if a.scene:
        scene = H.load_scene(a.data, a.scene, a.spectral)
    elif a.image and a.labels:
        wl = np.loadtxt(a.wavelengths) if a.wavelengths else None
        scene = H.load_custom(a.image, a.labels, a.classes, wl, a.spectral)
    else:
        ap.error("give --scene or --image and --labels")
    name = f"{scene.name}_{a.spectral}_{a.split}_{tag}" + (f"_n{a.per_class}" if a.per_class != 50 else "")
    out = a.out / name; out.mkdir(parents=True, exist_ok=True)
    log = lambda m: (print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True), open(out / "log.txt", "a").write(m + "\n"))
    log(f"{scene.name}: {scene.x.shape[0]}x{scene.x.shape[1]} px, {len(scene.bands)} bands ({a.spectral}), "
        f"{scene.num_classes} classes, {a.split} split, model {tag}, {a.per_class} labelled pixels per class")
    results = {}
    for s in a.seeds:
        d = out / f"seed{s}"; done = d / "metrics.json"
        if done.exists():
            results[f"seed{s}"] = json.loads(done.read_text()); continue
        tr, va, te = H.make_split(scene.labels, a.split, a.per_class, a.val_frac, s, a.block)
        d.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(d / "split.npz", train=tr.astype(np.uint8), val=va.astype(np.uint8), test=te.astype(np.uint8))
        log(f"seed {s}: {(tr > 0).sum()} training, {(va > 0).sum()} validation, {(te > 0).sum()} test pixels")
        cfg = dict(model=a.model, width=a.width, crop=a.crop, steps=a.steps, seed=s)
        if anysensor:
            model = A.train_anysensor(dict(cfg, rss=not a.no_rss, fwhm=not a.no_fwhm), scene, tr, va, d, a.device, log)
            sim = A.Simulator(scene); sensors = {}
            for sensor in A.TEST_SENSORS:
                t0 = time.time(); p = A.predict_sensor(model, scene, sim, sensor, a.device)
                sensors[sensor] = dict(H.evaluate(p, te, scene.num_classes), bands=int(len(model.bands)),
                                       predict_seconds=time.time() - t0)
                np.save(d / f"pred_{sensor}.npy", (p.argmax(0) + 1).astype(np.uint8))
                log(f"TEST seed {s} {tag} on {sensor} ({len(model.bands)} bands): OA {100 * sensors[sensor]['oa']:.2f} "
                    f"AA {100 * sensors[sensor]['aa']:.2f} kappa {100 * sensors[sensor]['kappa']:.2f}")
            prob, t_pred = None, sensors["full"]["predict_seconds"]
        else:
            model = H.train(cfg, scene, tr, va, d, a.device, log)
            t0 = time.time(); prob = H.predict(model, scene.x, a.device); t_pred = time.time() - t0
        hist = json.loads((d / "history.json").read_text())
        net = sensors["full"] if anysensor else H.evaluate(prob, te, scene.num_classes)
        r = {"net": dict(net, parameters=hist["parameters"], train_minutes=hist["minutes"], predict_seconds=t_pred,
                         selected_step=hist["selected"]["step"]),
             "train_pixels": int((tr > 0).sum()), "val_pixels": int((va > 0).sum()), "test_pixels": int((te > 0).sum())}
        if anysensor:
            r["sensors"] = sensors
        else:
            np.save(d / "pred_net.npy", (prob.argmax(0) + 1).astype(np.uint8))
        log(f"TEST seed {s} {tag}: OA {100 * r['net']['oa']:.2f} AA {100 * r['net']['aa']:.2f} "
            f"kappa {100 * r['net']['kappa']:.2f}")
        if not a.no_svm:
            t0 = time.time(); pred, sel = H.svm(scene.x, tr, va)
            r["svm"] = dict(H.evaluate(pred, te, scene.num_classes), seconds=time.time() - t0, **sel)
            np.save(d / "pred_svm.npy", (pred + 1).astype(np.uint8))
            log(f"TEST seed {s} svm: OA {100 * r['svm']['oa']:.2f} AA {100 * r['svm']['aa']:.2f} "
                f"kappa {100 * r['svm']['kappa']:.2f}")
        done.write_text(json.dumps(r)); results[f"seed{s}"] = r
    summary = {"scene": scene.name, "spectral": a.spectral, "bands": len(scene.bands), "band_names": scene.bands,
               "split": a.split, "model": tag, "labelled_per_class": a.per_class, "seeds": a.seeds,
               "classes": scene.classes}

    def aggregate(rs, label):
        agg = {k: {"mean": float(np.mean([r[k] for r in rs])), "sd": float(np.std([r[k] for r in rs], ddof=1))
                   if len(rs) > 1 else 0.0} for k in ("oa", "aa", "kappa")}
        pc = np.array([[np.nan if v is None else v for v in r["per_class"]] for r in rs])
        agg["per_class"] = [None if np.isnan(c).all() else float(np.nanmean(c)) for c in pc.T]
        log(f"{label} mean over seeds: " + " ".join(f"{k} {100 * v['mean']:.2f} +- {100 * v['sd']:.2f}"
                                                    for k, v in agg.items() if k != "per_class"))
        return agg
    for m in ("net", "svm"):
        rs = [results[f"seed{s}"][m] for s in a.seeds if m in results[f"seed{s}"]]
        if rs:
            summary[m] = aggregate(rs, m)
    if anysensor:
        summary["sensors"] = {}
        for sensor in A.TEST_SENSORS:
            rs = [results[f"seed{s}"]["sensors"][sensor] for s in a.seeds]
            summary["sensors"][sensor] = dict(aggregate(rs, f"{tag} on {sensor}"), bands=rs[0]["bands"])
    summary["runs"] = results
    (out / "results.json").write_text(json.dumps(summary, indent=1))
    with open(out / "per_class.csv", "w", newline="") as h:
        w = csv.writer(h); w.writerow(["class"] + [m for m in ("net", "svm") if m in summary])
        for i, c in enumerate(scene.classes):
            w.writerow([c] + [summary[m]["per_class"][i] for m in ("net", "svm") if m in summary])


if __name__ == "__main__":
    main()
