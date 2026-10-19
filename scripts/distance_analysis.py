"""Distance-resolved evaluation of spectral resolution, from the runs written by train_eval.py (no retraining).

For every test pixel the Euclidean distance to the nearest training pixel is computed from the saved split, and
accuracy is reported per distance bin, per sensor and per classifier. Confidence intervals come from a cluster
bootstrap over (seed, spatial block) units, paired across sensors because all sensors share the same split.

    python scripts/distance_analysis.py --data data --runs runs --out paper

Writes paper/figures/dist_*.pdf|png, paper/tables/dist_*.tex and paper/tables/distance_analysis.json.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import hsilc as H  # noqa: E402
from hsilc import viz  # noqa: E402
import paper_figures as PF  # noqa: E402  (shared style, colours and helpers)

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

SCENES = ["IP", "PU", "SA"]
SENSORS = ["full", "s2", "landsat8", "rgbn", "rgb"]
SPLITS = ["random", "block"]
LABEL = {"full": "HSI", "s2": "S2", "landsat8": "L8", "rgbn": "RGBN", "rgb": "RGB"}
# reference categorical palette, slots 1-5 (validated: adjacent CVD and normal-vision floors pass; light slots
# 3-5 sit below 3:1 contrast, so every line is direct-labelled and carries its own marker)
COLOR = {"full": "#2a78d6", "rgb": "#eb6834", "s2": "#1baf7a", "rgbn": "#eda100", "landsat8": "#e87ba4"}
MARKER = {"full": "o", "s2": "s", "landsat8": "D", "rgbn": "^", "rgb": "v"}
INK, INK2, MUTED, GRID = PF.INK, PF.INK2, PF.MUTED, PF.GRID
EDGES = [1, 3, 6, 11, 21, 41]                 # rounded distance bins: 1-2, 3-5, 6-10, 11-20, 21-40, >40 px
BINS = ["1–2", "3–5", "6–10", "11–20", "21–40", ">40"]
NB = len(BINS)
MIN_PX = 300                                   # a bin is reported when it holds at least this many test pixels
NBOOT = 2000
GAPS = {"hsi_rgb": ("full", "rgb"), "nir": ("rgbn", "rgb"), "hsi_s2": ("full", "s2")}
PU_MEADOWS, PU_SOIL = 2, 6


# ---------------------------------------------------------------- data
def block_side(shape):
    return max(8, int(round(min(shape) / 8)))


def fields(labels):
    """Connected regions (8-connectivity) of each class: a proxy for individual fields and objects."""
    comp = np.zeros(labels.shape, np.int64); n = 0
    for c in range(1, labels.max() + 1):
        lab, k = ndimage.label(labels == c, structure=np.ones((3, 3)))
        comp[lab > 0] = lab[lab > 0] + n; n += k
    return comp


def load_split(d):
    z = np.load(d / "split.npz")
    return z["train"] > 0, z["val"] > 0, z["test"] > 0


class Scene:
    def __init__(self, sc, data):
        s = H.SCENES[sc]
        self.key, self.gsd = sc, s["gsd"]
        self.labels = H.read_array(Path(data) / s["labels"])[0].astype(np.int64)
        self.comp = fields(self.labels)
        self.blk = block_side(self.labels.shape)
        h, w = self.labels.shape
        rr, cc = np.mgrid[:h, :w]
        self.cell = (rr // self.blk) * (-(-w // self.blk)) + cc // self.blk
        self.ncell = int(self.cell.max()) + 1


def distance_bins(train):
    dist = ndimage.distance_transform_edt(~train)
    return dist, np.digitize(np.rint(dist), EDGES) - 1


def near_far(dist):
    """0: within 5 px of a training pixel, 1: 6-10 px, 2: more than 10 px (rounded distances)."""
    r = np.rint(dist)
    return np.where(r <= 5, 0, np.where(r <= 10, 1, 2)).astype(np.int64)


def seed_summary(correct, test, nf):
    out = {}
    for (model, sp), c in correct.items():
        f = test & (nf == 2); n = test & (nf == 0)
        out.setdefault(model, {})[sp] = {"oa": float(c[test].mean()), "near": float(c[n].mean()),
                                         "far": float(c[f].mean()) if f.sum() >= MIN_PX else float("nan")}
    return out


# ---------------------------------------------------------------- accumulation
class Acc:
    """Correct / total counts per (seed x spatial block) cluster and distance bin (or any other grouping)."""
    def __init__(self, ngroups):
        self.n = []; self.c = {}; self.logd = np.zeros(ngroups); self.ng = ngroups

    def add(self, cluster, group, mask, correct_by_key, logd=None):
        k = cluster[mask]; g = group[mask]; ncl = int(cluster.max()) + 1
        tot = np.zeros((ncl, self.ng)); np.add.at(tot, (k, g), 1)
        keep = tot.sum(1) > 0
        self.n.append(tot[keep])
        for key, corr in correct_by_key.items():
            cc = np.zeros((ncl, self.ng)); np.add.at(cc, (k, g), corr[mask].astype(float))
            self.c.setdefault(key, []).append(cc[keep])
        if logd is not None:
            np.add.at(self.logd, g, logd[mask])

    def done(self):
        self.N = np.concatenate(self.n); self.C = {k: np.concatenate(v) for k, v in self.c.items()}
        self.tot = self.N.sum(0); self.meanlogd = self.logd / np.maximum(self.tot, 1)
        return self

    def acc(self, key):
        return self.C[key].sum(0) / np.maximum(self.tot, 1)

    def oa(self, key):
        return self.C[key].sum() / self.N.sum()


def bootstrap(acc, keys, rng, nboot=NBOOT):
    """Bootstrap distributions of per-group accuracy, resampling clusters with replacement (paired over keys)."""
    K = acc.N.shape[0]
    W = rng.multinomial(K, np.full(K, 1 / K), size=nboot).astype(float)
    n = W @ acc.N
    out = {k: (W @ acc.C[k]) / np.maximum(n, 1) for k in keys}
    out["_oa"] = {k: (W @ acc.C[k].sum(1)) / (W @ acc.N.sum(1)) for k in keys}
    return out


def ci(x):
    return [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))]


def slope(gap, x, w):
    """Weighted least-squares slope of gap on log2 distance: points gained per doubling of distance."""
    w = w / w.sum(); xm = (w * x).sum()
    return ((w * (x - xm)) * gap).sum() / ((w * (x - xm) ** 2).sum())


# ---------------------------------------------------------------- analysis
def analyse(runs, data, rng):
    R = {"bins": BINS, "edges_px": EDGES, "min_px": MIN_PX, "nboot": NBOOT, "scenes": {}}
    cache = {}
    for sc in SCENES:
        S = Scene(sc, data); cache[sc] = S
        out = R["scenes"][sc] = {"gsd_m": S.gsd, "block_px": S.blk, "splits": {}}
        for split in SPLITS:
            dist_acc = Acc(NB); seen_acc = Acc(2); nf_acc = Acc(3); val_d, test_d, fieldseen = [], [], []
            cls_rows = []; per_seed = []
            for seed in (1, 2, 3):
                base = Path(runs) / f"{sc}_full_{split}_fcn" / f"seed{seed}"
                train, val, test = load_split(base)
                for sp in SENSORS:     # every sensor must share the split, otherwise the comparison is not paired
                    t2, v2, te2 = load_split(Path(runs) / f"{sc}_{sp}_{split}_fcn" / f"seed{seed}")
                    assert (t2 == train).all() and (te2 == test).all(), (sc, sp, split, seed)
                dist, b = distance_bins(train)
                cluster = (seed - 1) * S.ncell + S.cell
                seen_ids = np.unique(S.comp[train]); seen = np.isin(S.comp, seen_ids).astype(np.int64)
                correct = {}
                for sp in SENSORS:
                    d = Path(runs) / f"{sc}_{sp}_{split}_fcn" / f"seed{seed}"
                    for model, f in (("fcn", "pred_net.npy"), ("svm", "pred_svm.npy")):
                        p = np.load(d / f).astype(np.int64)
                        correct[(model, sp)] = p == S.labels
                        if sc == "PU" and model == "fcn":
                            cls_rows.append((sp, seed, p, b, test))
                dist_acc.add(cluster, b, test, correct, np.log2(np.maximum(dist, 1)))
                seen_acc.add(cluster, seen, test, correct)
                nf = near_far(dist); nf_acc.add(cluster, nf, test, correct)
                per_seed.append(seed_summary(correct, test, nf))
                val_d.append(dist[val]); test_d.append(dist[test]); fieldseen.append(seen[test])
            dist_acc.done(); seen_acc.done(); nf_acc.done()
            keys = list(dist_acc.C)
            bd = bootstrap(dist_acc, keys, rng); bs = bootstrap(seen_acc, keys, rng); bn = bootstrap(nf_acc, keys, rng)
            valid = dist_acc.tot >= MIN_PX
            td = np.concatenate(test_d); vd = np.concatenate(val_d)
            o = out["splits"][split] = {
                "test_px": int(td.size), "bin_px": dist_acc.tot.astype(int).tolist(), "valid": valid.tolist(),
                "bin_mean_log2d": dist_acc.meanlogd.tolist(),
                "test_dist": {"median": float(np.median(td)), "p90": float(np.percentile(td, 90)),
                              "le5": float((td <= 5).mean()), "le10": float((td <= 10).mean()),
                              "median_m": float(np.median(td) * S.gsd)},
                "val_dist": {"median": float(np.median(vd)), "le5": float((vd <= 5).mean()),
                             "le10": float((vd <= 10).mean())},
                "seen_frac": float(np.concatenate(fieldseen).mean()),
                "nearfar_px": nf_acc.tot.astype(int).tolist(), "per_seed": per_seed,
                "acc": {}, "oa": {}, "gap": {}, "seen": {}}
            for (model, sp) in keys:
                o["acc"].setdefault(model, {})[sp] = {"mean": dist_acc.acc((model, sp)).tolist(),
                                                      "ci": [ci(bd[(model, sp)][:, k]) for k in range(NB)]}
                o["oa"].setdefault(model, {})[sp] = {"mean": float(dist_acc.oa((model, sp))),
                                                     "ci": ci(bd["_oa"][(model, sp)])}
                o["seen"].setdefault(model, {})[sp] = {
                    "unseen": float(seen_acc.acc((model, sp))[0]), "seen": float(seen_acc.acc((model, sp))[1]),
                    "ci_unseen": ci(bs[(model, sp)][:, 0]), "ci_seen": ci(bs[(model, sp)][:, 1])}
            x = dist_acc.meanlogd
            for model in ("fcn", "svm"):
                for g, (a, b_) in GAPS.items():
                    gap = dist_acc.acc((model, a)) - dist_acc.acc((model, b_))
                    gb = bd[(model, a)] - bd[(model, b_)]
                    entry = {"mean": gap.tolist(), "ci": [ci(gb[:, k]) for k in range(NB)],
                             "overall": float(dist_acc.oa((model, a)) - dist_acc.oa((model, b_))),
                             "overall_ci": ci(bd["_oa"][(model, a)] - bd["_oa"][(model, b_)])}
                    if valid.sum() >= 3:
                        w = dist_acc.tot[valid]
                        entry["slope"] = float(slope(gap[valid], x[valid], w))
                        sl = np.array([slope(gb[i, valid], x[valid], w) for i in range(NBOOT)])
                        entry["slope_ci"] = ci(sl); entry["slope_p_le0"] = float((sl <= 0).mean())
                    gn = bn[(model, a)] - bn[(model, b_)]
                    near = nf_acc.acc((model, a))[0] - nf_acc.acc((model, b_))[0]
                    far = nf_acc.acc((model, a))[2] - nf_acc.acc((model, b_))[2]
                    entry["nearfar"] = {"near": float(near), "far": float(far), "ci_near": ci(gn[:, 0]), "ci_far": ci(gn[:, 2]),
                                        "diff": float(far - near), "ci_diff": ci(gn[:, 2] - gn[:, 0]),
                                        "far_valid": bool(nf_acc.tot[2] >= MIN_PX),
                                        "seeds": [[p[model][a]["near"] - p[model][b_]["near"],
                                                   p[model][a]["far"] - p[model][b_]["far"]] for p in per_seed]}
                    sg = bs[(model, a)] - bs[(model, b_)]
                    entry["seen"] = {"unseen": float(seen_acc.acc((model, a))[0] - seen_acc.acc((model, b_))[0]),
                                     "seen": float(seen_acc.acc((model, a))[1] - seen_acc.acc((model, b_))[1]),
                                     "ci_unseen": ci(sg[:, 0]), "ci_seen": ci(sg[:, 1])}
                    o["gap"].setdefault(model, {})[g] = entry
            if sc == "PU":
                o["meadows"] = meadows_by_distance(cls_rows, S.labels)
        reweight(out)
    R["sign_test"] = sign_test(R)
    return R, cache


def sign_test(R):
    """Seed-level comparisons of the FCN's HSI-RGB gain far from (>10 px) and near (<=5 px) the training pixels."""
    from scipy.stats import binomtest
    res = {}
    for model in ("fcn", "svm"):
        for g in ("hsi_rgb", "nir"):
            rows = []
            for sc in SCENES:
                for split in SPLITS:
                    for k, (near, far) in enumerate(R["scenes"][sc]["splits"][split]["gap"][model][g]["nearfar"]["seeds"]):
                        if np.isfinite(far):
                            rows.append((sc, split, k + 1, near, far))
            wins = sum(f > n for *_, n, f in rows)
            res[f"{model}_{g}"] = {"far_greater": wins, "n": len(rows),
                                   "p_one_sided": float(binomtest(wins, len(rows), 0.5, alternative="greater").pvalue),
                                   "rows": rows}
    return res


def meadows_by_distance(rows, labels):
    """Pavia University: Meadows accuracy and the share of Meadows test pixels labelled Bare soil, per distance bin."""
    res = {}
    for sp in SENSORS:
        hit = np.zeros(NB); soil = np.zeros(NB); tot = np.zeros(NB)
        for s, seed, p, b, test in rows:
            if s != sp:
                continue
            m = test & (labels == PU_MEADOWS)
            np.add.at(tot, b[m], 1); np.add.at(hit, b[m], (p[m] == PU_MEADOWS)); np.add.at(soil, b[m], (p[m] == PU_SOIL))
        res[sp] = {"acc": (hit / np.maximum(tot, 1)).tolist(), "to_soil": (soil / np.maximum(tot, 1)).tolist(),
                   "px": tot.astype(int).tolist()}
    return res


def robustness(runs, cache):
    """Sign test of the far-versus-near HSI-RGB gain under other distance definitions and thresholds."""
    from scipy.stats import binomtest
    variants = {"train, near<=5, far>10": ("train", 5, 10), "train, near<=3, far>15": ("train", 3, 15),
                "train, near<=5, far>20": ("train", 5, 20), "same class, near<=5, far>10": ("same", 5, 10),
                "train+validation, near<=5, far>10": ("trainval", 5, 10)}
    res = {name: {"fcn": [], "svm": []} for name in variants}
    for sc in SCENES:
        S = cache[sc]; lab = S.labels
        for split in SPLITS:
            for seed in (1, 2, 3):
                train, val, test = load_split(Path(runs) / f"{sc}_full_{split}_fcn" / f"seed{seed}")
                dist = {"train": ndimage.distance_transform_edt(~train),
                        "trainval": ndimage.distance_transform_edt(~(train | val))}
                same = np.full(lab.shape, np.inf)
                for c in range(1, lab.max() + 1):
                    m = lab == c
                    if (train & m).any():
                        same[m] = ndimage.distance_transform_edt(~(train & m))[m]
                dist["same"] = same
                ok = {(m, sp): np.load(Path(runs) / f"{sc}_{sp}_{split}_fcn" / f"seed{seed}" / f).astype(np.int64) == lab
                      for sp in ("full", "rgb") for m, f in (("fcn", "pred_net.npy"), ("svm", "pred_svm.npy"))}
                for name, (kind, lo, hi) in variants.items():
                    r = np.rint(dist[kind]); near, far = test & (r <= lo), test & (r > hi)
                    if far.sum() < MIN_PX or near.sum() < MIN_PX:
                        continue
                    for m in ("fcn", "svm"):
                        g = lambda mask: ok[(m, "full")][mask].mean() - ok[(m, "rgb")][mask].mean()
                        res[name][m].append([sc, split, seed, float(g(near)), float(g(far))])
    out = {}
    for name, d in res.items():
        out[name] = {}
        for m, rows in d.items():
            wins = sum(f > n for *_, n, f in rows)
            out[name][m] = {"far_greater": wins, "n": len(rows),
                            "p_one_sided": float(binomtest(wins, len(rows), 0.5, alternative="greater").pvalue) if rows else None}
        print(f"robustness {name}: FCN {out[name]['fcn']['far_greater']}/{out[name]['fcn']['n']} "
              f"(p={out[name]['fcn']['p_one_sided']:.4f}), SVM {out[name]['svm']['far_greater']}/{out[name]['svm']['n']} "
              f"(p={out[name]['svm']['p_one_sided']:.3f})")
    return out


def reweight(out):
    """Block-split accuracy per distance bin, re-weighted to the random split's distance distribution.
    If distance explains the protocol effect, this reproduces the random-split numbers."""
    rnd, blk = out["splits"]["random"], out["splits"]["block"]
    p = np.array(rnd["bin_px"], float); p /= p.sum()
    nb = np.array(blk["bin_px"], float)
    support = (nb >= 100) | (p < 0.005)
    rw = {"support": bool(support.all()), "fcn": {}, "svm": {}}
    for model in ("fcn", "svm"):
        for sp in SENSORS:
            a = np.array(blk["acc"][model][sp]["mean"])
            rw[model][sp] = {"block": blk["oa"][model][sp]["mean"], "random": rnd["oa"][model][sp]["mean"],
                             "block_at_random_distances": float((p * a).sum())}
        for g, (a, b) in GAPS.items():
            obs_r = rnd["gap"][model][g]["overall"]; obs_b = blk["gap"][model][g]["overall"]
            pred = rw[model][a]["block_at_random_distances"] - rw[model][b]["block_at_random_distances"]
            rw[model][g] = {"random": obs_r, "block": obs_b, "block_at_random_distances": pred,
                            "explained": (obs_b - pred) / (obs_b - obs_r) if abs(obs_b - obs_r) > 1e-9 else None}
    out["reweight"] = rw


def budget(runs, cache):
    """Random split, all bands: accuracy per distance bin for 10, 25, 50 and 100 labelled pixels per class."""
    res = {}
    for sc in SCENES:
        S = cache[sc]; res[sc] = {}
        for n in (10, 25, 50, 100):
            name = f"{sc}_full_random_fcn" + ("" if n == 50 else f"_n{n}")
            acc = Acc(NB); dd = []
            for seed in (1, 2, 3):
                d = Path(runs) / name / f"seed{seed}"
                train, val, test = load_split(d); dist, b = distance_bins(train)
                correct = {m: np.load(d / f).astype(np.int64) == S.labels for m, f in (("fcn", "pred_net.npy"), ("svm", "pred_svm.npy"))}
                acc.add((seed - 1) * S.ncell + S.cell, b, test, correct); dd.append(dist[test])
            acc.done(); dd = np.concatenate(dd)
            res[sc][n] = {"bin_px": acc.tot.astype(int).tolist(), "median_dist": float(np.median(dd)),
                          "fcn": acc.acc("fcn").tolist(), "svm": acc.acc("svm").tolist(),
                          "oa_fcn": float(acc.oa("fcn")), "oa_svm": float(acc.oa("svm"))}
        for lo, hi in ((10, 100), (10, 50), (50, 100)):
            res[sc][f"decomp_{lo}_{hi}"] = {m: decompose(res[sc][lo], res[sc][hi], m) for m in ("fcn", "svm")}
    return res


def decompose(a, b, model):
    """Split the OA change between two label budgets into a part due to the shift of the distance distribution
    (at the accuracy per bin of the smaller budget) and a part due to better accuracy at a given distance."""
    pa = np.array(a["bin_px"], float); pa /= pa.sum(); pb = np.array(b["bin_px"], float); pb /= pb.sum()
    A, Bv = np.array(a[model]), np.array(b[model])
    shift = float(((pb - pa) * A).sum()); better = float((pb * (Bv - A)).sum())
    return {"total": float(b[f"oa_{model}"] - a[f"oa_{model}"]), "distance": shift, "per_distance": better}


def separability(data, rng, per_class=1500):
    """Jeffries-Matusita distance between class pairs for each simulated sensor (Gaussian classes, Ledoit-Wolf
    covariances, standardised bands). Independent of any classifier."""
    from sklearn.covariance import LedoitWolf
    res = {}
    for sc in SCENES:
        res[sc] = {}
        for sp in SENSORS:
            scene = H.load_scene(data, sc, spectral=sp); X = scene.x.reshape(-1, scene.x.shape[-1]).astype(np.float64)
            y = scene.labels.ravel(); stats = {}
            for c in range(1, y.max() + 1):
                idx = np.flatnonzero(y == c)
                idx = rng.choice(idx, min(per_class, idx.size), replace=False)
                lw = LedoitWolf().fit(X[idx]); stats[c] = (X[idx].mean(0), lw.covariance_)
            jm = {}
            for i in stats:
                for j in stats:
                    if j <= i:
                        continue
                    (m1, s1), (m2, s2) = stats[i], stats[j]; s = (s1 + s2) / 2; dm = m1 - m2
                    bh = dm @ np.linalg.solve(s, dm) / 8 + 0.5 * (np.linalg.slogdet(s)[1]
                                                                   - 0.5 * (np.linalg.slogdet(s1)[1] + np.linalg.slogdet(s2)[1]))
                    jm[f"{i}-{j}"] = float(2 * (1 - np.exp(-bh)))
            v = np.array(list(jm.values()))
            res[sc][sp] = {"mean": float(v.mean()), "min": float(v.min()), "below_1.9": float((v < 1.9).mean()),
                           "pairs": jm}
            print(f"JM {sc} {sp}: mean {v.mean():.3f}, min {v.min():.3f}, pairs < 1.9: {(v < 1.9).sum()}/{v.size}")
    return res


# ---------------------------------------------------------------- figures
def style(ax, grid="y"):
    PF.style(ax, grid)


def distaxis(ax, ticks=(1, 2, 5, 10, 20, 50)):
    """Logarithmic distance axis with plain tick labels."""
    from matplotlib.ticker import FixedLocator, NullLocator, FixedFormatter
    ax.set_xscale("log"); lo, hi = ax.get_xlim(); t = [v for v in ticks if lo <= v <= hi * 1.001]
    ax.xaxis.set_major_locator(FixedLocator(t)); ax.xaxis.set_major_formatter(FixedFormatter([str(v) for v in t]))
    ax.xaxis.set_minor_locator(NullLocator())


def example_seed(R):
    """The Pavia University block-split seed whose FCN HSI-RGB gap is closest to the three-seed mean."""
    ps = R["scenes"]["PU"]["splits"]["block"]["per_seed"]
    gaps = np.array([p["fcn"]["full"]["oa"] - p["fcn"]["rgb"]["oa"] for p in ps])
    return int(np.argmin(np.abs(gaps - gaps.mean()))) + 1


def fig_concept(out, data, runs, cache, seed):
    """Pavia University: training pixels and the distance of each test pixel to the nearest one."""
    S = cache["PU"]; img = PF.composite("PU", data)
    fig, axes = plt.subplots(1, 3, figsize=(PF.COL2, 3.4), gridspec_kw=dict(width_ratios=[1, 1, 1]))
    cmap = LinearSegmentedColormap.from_list("d", ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
    axes[0].imshow(img, interpolation="nearest"); axes[0].set_title("(a) False-colour composite\n ")
    for ax, split, title in ((axes[1], "random", "(b) Random split"), (axes[2], "block", "(c) Block split")):
        train, val, test = load_split(Path(runs) / f"PU_full_{split}_fcn" / f"seed{seed}")
        dist = ndimage.distance_transform_edt(~train)
        rgba = np.ones(S.labels.shape + (3,)) * 0.94
        ax.imshow(rgba, interpolation="nearest")
        im = ax.imshow(np.ma.masked_where(~test, np.minimum(dist, 60)), cmap=cmap, vmin=0, vmax=60, interpolation="nearest")
        ty, tx = np.nonzero(train); ax.scatter(tx, ty, s=1.6, c=INK, lw=0)
        if split == "block":
            h, w = S.labels.shape
            for k in range(S.blk, w, S.blk):
                ax.axvline(k - 0.5, color="white", lw=0.5)
            for k in range(S.blk, h, S.blk):
                ax.axhline(k - 0.5, color="white", lw=0.5)
        med = np.median(dist[test])
        ax.set_title(f"{title}\nmedian distance {med:.0f} px ({med * S.gsd:.0f} m)")
    for ax in axes:
        ax.set_axis_off()
    cb = fig.colorbar(im, ax=axes[1:], shrink=0.7, pad=0.02, extend="max")
    cb.set_label("Distance of test pixel to nearest training pixel (px)", color=INK2); cb.outline.set_visible(False)
    fig.legend(handles=[Line2D([], [], marker="o", ls="", color=INK, ms=2.5, label="training pixel")],
               loc="lower center", ncol=1, bbox_to_anchor=(0.62, -0.01), labelcolor=INK2)
    out.save(fig, "dist_concept")


def fig_cdf(out, runs, cache):
    fig, axes = plt.subplots(1, 3, figsize=(PF.COL2, 2.2), sharey=True)
    for ax, sc in zip(axes, SCENES):
        for split, ls in (("random", (0, (4, 2))), ("block", "-")):
            td, vd = [], []
            for seed in (1, 2, 3):
                train, val, test = load_split(Path(runs) / f"{sc}_full_{split}_fcn" / f"seed{seed}")
                dist = ndimage.distance_transform_edt(~train); td.append(dist[test]); vd.append(dist[val])
            for d, col, lab in ((np.concatenate(td), INK, "test"), (np.concatenate(vd), MUTED, "validation")):
                x = np.sort(d); y = np.arange(1, x.size + 1) / x.size
                ax.plot(x, y, color=col, ls=ls, lw=1.4 if lab == "test" else 1.0, label=f"{lab}, {split}")
        ax.set_xscale("log"); ax.set_xlim(1, 200); distaxis(ax, (1, 2, 5, 10, 20, 50, 100, 200)); ax.set_title(H.SCENES[sc]["name"])
        ax.axvline(10, color=GRID, lw=0.8, zorder=0)
        ax.set_xlabel("Distance to nearest training pixel (px)"); style(ax, "both")
    axes[0].set_ylabel("Cumulative share of pixels")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.12), labelcolor=INK2)
    fig.tight_layout(); out.save(fig, "dist_cdf")


def fig_accuracy(out, R):
    fig, axes = plt.subplots(2, 3, figsize=(PF.COL2, 4.3), sharex=False)
    for j, sc in enumerate(SCENES):
        for i, split in enumerate(SPLITS):
            ax = axes[i, j]; o = R["scenes"][sc]["splits"][split]; v = np.array(o["valid"])
            x = 2 ** np.array(o["bin_mean_log2d"])[v]
            ends = []
            for sp in SENSORS:
                y = 100 * np.array(o["acc"]["fcn"][sp]["mean"])[v]
                ax.plot(x, y, color=COLOR[sp], marker=MARKER[sp], ms=3.2, lw=1.4, mec="white", mew=0.4)
                ends.append((y[-1], sp, x[-1]))
            for y, sp, xe in label_positions(ends, ax):
                ax.annotate(LABEL[sp], (xe, y), xytext=(4, 0), textcoords="offset points", va="center", fontsize=6,
                            color=INK2)
            ax.set_xscale("log"); style(ax, "both")
            ax.set_title(f"{H.SCENES[sc]['name']}, {split} split")
            if i == 1:
                ax.set_xlabel("Distance to nearest training pixel (px)")
            if j == 0:
                ax.set_ylabel("FCN overall accuracy (%)")
            ax.set_xlim(right=ax.get_xlim()[1] * 1.6); distaxis(ax)
    fig.tight_layout(); out.save(fig, "dist_accuracy")


def label_positions(ends, ax, min_gap=None):
    """Spread direct labels vertically so they do not overlap (in data units)."""
    ends = sorted(ends)
    lo, hi = ax.get_ylim() if min_gap is None else (0, 0)
    gap = min_gap or max(1.0, (max(e[0] for e in ends) - min(e[0] for e in ends) + 8) / 9)
    ys = [e[0] for e in ends]
    for k in range(1, len(ys)):
        ys[k] = max(ys[k], ys[k - 1] + gap)
    return [(y, e[1], e[2]) for y, e in zip(ys, ends)]


def fig_gain(out, R):
    fig, axes = plt.subplots(2, 3, figsize=(PF.COL2, 4.3))
    rows = (("hsi_rgb", "HSI minus RGB (points)"), ("nir", "RGBN minus RGB (points)"))
    series = (("fcn", "block", INK, "-", "o", "FCN, block"), ("fcn", "random", INK, (0, (4, 2)), "o", "FCN, random"),
              ("svm", "block", MUTED, "-", "s", "SVM, block"), ("svm", "random", MUTED, (0, (4, 2)), "s", "SVM, random"))
    for i, (g, ylab) in enumerate(rows):
        for j, sc in enumerate(SCENES):
            ax = axes[i, j]
            for model, split, col, ls, mk, lab in series:
                o = R["scenes"][sc]["splits"][split]; v = np.array(o["valid"])
                x = 2 ** np.array(o["bin_mean_log2d"])[v]
                e = o["gap"][model][g]; y = 100 * np.array(e["mean"])[v]; c = 100 * np.array(e["ci"])[v]
                if model == "fcn":
                    ax.fill_between(x, c[:, 0], c[:, 1], color=col, alpha=0.10 if split == "block" else 0.06, lw=0)
                ax.plot(x, y, color=col, ls=ls, marker=mk, ms=3 if model == "fcn" else 2.6, lw=1.4 if model == "fcn" else 1.0,
                        mfc=col if split == "block" else "white", mec=col, label=lab)
            ax.axhline(0, color=PF.AXIS, lw=0.8); ax.set_xscale("log"); style(ax, "both"); distaxis(ax)
            if i == 0:
                ax.set_title(H.SCENES[sc]["name"])
            if i == 1:
                ax.set_xlabel("Distance to nearest training pixel (px)")
            if j == 0:
                ax.set_ylabel(ylab)
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.04), labelcolor=INK2)
    fig.tight_layout(rect=(0, 0.03, 1, 1)); out.save(fig, "dist_gain")


def fig_seen(out, R):
    """Dumbbell: HSI minus RGB gap of the FCN on test pixels whose field holds a training pixel versus not."""
    fig, axes = plt.subplots(1, 2, figsize=(PF.COL2, 1.7), sharey=True)
    rows = [(sc, split) for sc in SCENES for split in SPLITS if 1 - R["scenes"][sc]["splits"][split]["seen_frac"] >= 0.02]
    for ax, model in zip(axes, ("fcn", "svm")):
        for k, (sc, split) in enumerate(rows):
            o = R["scenes"][sc]["splits"][split]; s = o["gap"][model]["hsi_rgb"]["seen"]
            y = len(rows) - 1 - k
            ax.plot([100 * s["seen"], 100 * s["unseen"]], [y, y], color=PF.AXIS, lw=1.2, zorder=1)
            ax.errorbar(100 * s["seen"], y, xerr=[[100 * (s["seen"] - s["ci_seen"][0])], [100 * (s["ci_seen"][1] - s["seen"])]],
                        fmt="o", mfc="white", mec=INK, ecolor=MUTED, ms=4, lw=0.8, zorder=2)
            ax.errorbar(100 * s["unseen"], y, xerr=[[100 * (s["unseen"] - s["ci_unseen"][0])], [100 * (s["ci_unseen"][1] - s["unseen"])]],
                        fmt="o", color=INK, ecolor=MUTED, ms=4, lw=0.8, zorder=2)
            frac = 100 * o["seen_frac"]
            if model == "fcn":
                ax.annotate(f"{frac:.0f}% seen", (1.0, y), xycoords=("axes fraction", "data"), xytext=(4, 0),
                            textcoords="offset points", va="center", fontsize=6, color=MUTED)
        ax.axvline(0, color=PF.AXIS, lw=0.8); style(ax, "x")
        ax.set_yticks(range(len(rows)), [f"{H.SCENES[sc]['name']}, {split}" for sc, split in rows][::-1])
        ax.set_xlabel("HSI minus RGB overall accuracy (points)"); ax.set_title(model.upper())
    handles = [Line2D([], [], marker="o", ls="", mfc="white", mec=INK, label="test pixels in a field with training pixels"),
               Line2D([], [], marker="o", ls="", color=INK, label="test pixels in a field without training pixels")]
    fig.legend(handles=handles, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.14), labelcolor=INK2)
    fig.tight_layout(); out.save(fig, "dist_seen")


def fig_maps(out, data, runs, cache, seed):
    """Pavia University, block split: where the spectrum decides."""
    S = cache["PU"]; d = Path(runs); seed = f"seed{seed}"
    train, val, test = load_split(d / "PU_full_block_fcn" / seed)
    dist = ndimage.distance_transform_edt(~train)
    ph = np.load(d / "PU_full_block_fcn" / seed / "pred_net.npy").astype(np.int64)
    pr = np.load(d / "PU_rgb_block_fcn" / seed / "pred_net.npy").astype(np.int64)
    okh, okr = (ph == S.labels) & test, (pr == S.labels) & test
    bg = np.full(S.labels.shape + (3,), 0.95)
    fig, axes = plt.subplots(1, 5, figsize=(PF.COL2, 2.9))
    axes[0].imshow(PF.composite("PU", data), interpolation="nearest"); axes[0].set_title("(a) False colour")
    axes[1].imshow(viz.colorize(S.labels, 9), interpolation="nearest"); axes[1].set_title("(b) Reference")
    for ax, ok, title in ((axes[2], okh, "(c) HSI errors"), (axes[3], okr, "(d) RGB errors")):
        im = bg.copy(); im[test] = (0.72, 0.72, 0.70); im[test & ~ok] = (0.82, 0.18, 0.18)
        ax.imshow(im, interpolation="nearest"); ty, tx = np.nonzero(train); ax.scatter(tx, ty, s=0.8, c=INK, lw=0)
        ax.set_title(f"{title}: OA {100 * ok.sum() / test.sum():.1f}%")
    im = bg.copy(); im[test] = (0.85, 0.85, 0.83)
    im[okh & ~okr] = (0.16, 0.47, 0.84); im[okr & ~okh] = (0.92, 0.41, 0.20)
    axes[4].imshow(im, interpolation="nearest"); axes[4].set_title("(e) Disagreements")
    for ax in axes:
        ax.set_axis_off()
    handles = [Patch(color=(0.72, 0.72, 0.70), label="correct"), Patch(color=(0.82, 0.18, 0.18), label="wrong"),
               Line2D([], [], marker="o", ls="", color=INK, ms=2.5, label="training pixel"),
               Patch(color=(0.16, 0.47, 0.84), label="only HSI correct"), Patch(color=(0.92, 0.41, 0.20), label="only RGB correct")]
    fig.legend(handles=handles, loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.02), labelcolor=INK2)
    fig.tight_layout(rect=(0, 0.06, 1, 1)); out.save(fig, "dist_maps")
    near, far = test & (dist <= 10), test & (dist > 20)
    return {"only_hsi": int((okh & ~okr).sum()), "only_rgb": int((okr & ~okh).sum()),
            "only_hsi_far_share": float(((okh & ~okr) & far).sum() / max((okh & ~okr).sum(), 1)),
            "test_far_share": float(far.sum() / test.sum()), "oa_hsi": float(okh.sum() / test.sum()),
            "oa_rgb": float(okr.sum() / test.sum()),
            "near": {"hsi": float(okh[near].mean()), "rgb": float(okr[near].mean())},
            "far": {"hsi": float(okh[far].mean()), "rgb": float(okr[far].mean())}}


def fig_mechanism(out, data, R):
    cube, labels = PF.scene_data("PU", data); wl = H.SCENES["PU"]["wavelengths"]
    fig, axes = plt.subplots(1, 3, figsize=(PF.COL2, 2.1))
    ax = axes[0]
    stats = {}
    for c, col, name in ((PU_MEADOWS, "#008300", "Meadows"), (PU_SOIL, "#8a6d3b", "Bare soil")):
        X = cube[labels == c]; m, s = X.mean(0), X.std(0); stats[c] = (m, s, len(X))
        ax.plot(wl, m, color=col, lw=1.4, label=name); ax.fill_between(wl, m - s, m + s, color=col, alpha=0.15, lw=0)
    shade(ax)
    ax.set_xlabel("Wavelength (nm, nominal)"); ax.set_ylabel("Pixel value (DN)"); ax.set_title("(a) Class spectra, mean ± SD")
    ax.legend(loc="upper left", labelcolor=INK2); style(ax)
    ax = axes[1]
    (m1, s1, n1), (m2, s2, n2) = stats[PU_MEADOWS], stats[PU_SOIL]
    dval = (m1 - m2) / np.sqrt(((n1 - 1) * s1 ** 2 + (n2 - 1) * s2 ** 2) / (n1 + n2 - 2))
    ax.plot(wl, dval, color=INK, lw=1.4); ax.axhline(0, color=PF.AXIS, lw=0.8); shade(ax)
    ax.set_xlabel("Wavelength (nm, nominal)"); ax.set_ylabel("Cohen's d (Meadows minus Bare soil)")
    ax.set_title("(b) Separation per band"); style(ax)
    ax = axes[2]; o = R["scenes"]["PU"]["splits"]["block"]; v = np.array(o["valid"])
    x = 2 ** np.array(o["bin_mean_log2d"])[v]; ends = []
    for sp in SENSORS:
        y = 100 * np.array(o["meadows"][sp]["to_soil"])[v]
        ax.plot(x, y, color=COLOR[sp], marker=MARKER[sp], ms=3, lw=1.4, mec="white", mew=0.4); ends.append((y[-1], sp, x[-1]))
    for y, sp, xe in label_positions(ends, ax):
        ax.annotate(LABEL[sp], (xe, y), xytext=(4, 0), textcoords="offset points", va="center", fontsize=6, color=INK2)
    ax.set_xscale("log"); ax.set_xlim(right=ax.get_xlim()[1] * 1.6); style(ax, "both"); distaxis(ax)
    ax.set_xlabel("Distance to nearest training pixel (px)"); ax.set_ylabel("Meadows labelled Bare soil (%)")
    ax.set_title("(c) FCN, block split")
    fig.tight_layout(); out.save(fig, "dist_mechanism")
    k_nir = int(np.argmin(np.abs(wl - 833))); k_red = int(np.argmin(np.abs(wl - 665))); k_grn = int(np.argmin(np.abs(wl - 560)))
    return {"cohen_d_833nm": float(dval[k_nir]), "cohen_d_665nm": float(dval[k_red]), "cohen_d_560nm": float(dval[k_grn]),
            "cohen_d_max": float(dval.max()), "wl_at_max": float(wl[int(np.argmax(dval))])}


def shade(ax):
    for name in ("B2", "B3", "B4", "B8"):
        c, f = H.S2_BANDS[name]
        ax.axvspan(c - f / 2, c + f / 2, color="#d9d8d3" if name != "B8" else "#c7dcf5", alpha=0.5, lw=0, zorder=0)
        ax.annotate(name, (c, 1.0), xycoords=("data", "axes fraction"), ha="center", va="bottom", fontsize=5.5, color=MUTED)


def fig_budget(out, B):
    ramp = {10: "#86b6ef", 25: "#3987e5", 50: "#1c5cab", 100: "#0d366b"}
    fig, axes = plt.subplots(1, 3, figsize=(PF.COL2, 2.1), sharey=False)
    for ax, sc in zip(axes, SCENES):
        for n in (10, 25, 50, 100):
            o = B[sc][n]; v = np.array(o["bin_px"]) >= MIN_PX
            x = np.array([1.5, 4, 8, 15, 30, 60])[v]
            ax.plot(x, 100 * np.array(o["fcn"])[v], color=ramp[n], marker="o", ms=3, lw=1.4,
                    label=f"{n} px/class (median {o['median_dist']:.0f} px)")
        ax.set_xscale("log"); style(ax, "both"); distaxis(ax); ax.set_title(H.SCENES[sc]["name"])
        ax.set_xlabel("Distance to nearest training pixel (px)")
        ax.legend(loc="lower left", labelcolor=INK2, fontsize=5.5, handlelength=1.2)
    axes[0].set_ylabel("FCN overall accuracy (%)")
    fig.tight_layout(); out.save(fig, "dist_budget")


# ---------------------------------------------------------------- tables
def f1(x):
    return f"{100 * x:.1f}"


def first(sc, show):
    return r"\multirow{2}{*}{" + H.SCENES[sc]["name"] + "}" if show else ""


def gci(v, c):
    return rf"{100 * v:+.1f} ({100 * c[0]:+.1f}, {100 * c[1]:+.1f})"


def tables(out, R, B, JM):
    T = out.tab
    # distance distributions
    rows = []
    for sc in SCENES:
        for split in SPLITS:
            o = R["scenes"][sc]["splits"][split]; t, v = o["test_dist"], o["val_dist"]
            rows.append(rf"{first(sc, split == 'random')} & {split} & "
                        rf"{t['median']:.1f} & {t['median_m']:.0f} & {100 * t['le5']:.0f} & {100 * t['le10']:.0f} & "
                        rf"{100 * o['seen_frac']:.0f} & {v['median']:.1f} \\")
        rows.append(r"\midrule")
    rows = rows[:-1]
    tab(T / "dist_distribution.tex", "dist_distribution",
        "Distance of test and validation pixels to the nearest training pixel (three seeds pooled)",
        "llrrrrrr", [r" & & \multicolumn{5}{c}{Test pixels} & Validation \\", r"\cmidrule(lr){3-7}",
                     r"Scene & Split & Median (px) & Median (m) & $\le$5 px (\%) & $\le$10 px (\%) & Seen field (\%) & Median (px) \\"],
        rows, wide=True)
    # near / far
    rows = []
    for sc in SCENES:
        for split in SPLITS:
            o = R["scenes"][sc]["splits"][split]; cells = []
            for model in ("fcn", "svm"):
                e = o["gap"][model]["hsi_rgb"]["nearfar"]
                if e["far_valid"]:
                    wins = sum(f > n for n, f in e["seeds"] if np.isfinite(f))
                    cells += [gci(e["near"], e["ci_near"]), gci(e["far"], e["ci_far"]), gci(e["diff"], e["ci_diff"]), f"{wins}/3"]
                else:
                    cells += [gci(e["near"], e["ci_near"]), "--", "--", "--"]
            rows.append(rf"{first(sc, split == 'random')} & {split} & " + " & ".join(cells) + r" \\")
        rows.append(r"\midrule")
    rows = rows[:-1]
    tab(T / "dist_nearfar.tex", "dist_nearfar",
        "HSI minus RGB overall accuracy (points) for test pixels within 5 px of a training pixel (near) and more than "
        "10 px away (far), with 95\\% cluster-bootstrap intervals; seeds: number of seeds with a larger gain far than near",
        "llrrrrrrrr", [r" & & \multicolumn{4}{c}{FCN} & \multicolumn{4}{c}{SVM} \\",
                       r"\cmidrule(lr){3-6}\cmidrule(lr){7-10}",
                       r"Scene & Split & Near & Far & Far$-$near & Seeds & Near & Far & Far$-$near & Seeds \\"], rows, wide=True,
        size="scriptsize", sep=2)
    # seen / unseen
    rows = []
    for sc in SCENES:
        for split in SPLITS:
            o = R["scenes"][sc]["splits"][split]
            s = o["seen"]["fcn"]; g = o["gap"]["fcn"]["hsi_rgb"]["seen"]
            if 1 - o["seen_frac"] < 0.02:
                rows.append(rf"{first(sc, split == 'random')} & {split} & {100 * o['seen_frac']:.1f} & {f1(s['full']['seen'])} & "
                            rf"{f1(s['rgb']['seen'])} & {100 * g['seen']:+.1f} & \multicolumn{{3}}{{c}}{{too few pixels}} \\")
                continue
            rows.append(rf"{first(sc, split == 'random')} & {split} & {100 * o['seen_frac']:.1f} & {f1(s['full']['seen'])} & {f1(s['rgb']['seen'])} & "
                        rf"{100 * g['seen']:+.1f} & {f1(s['full']['unseen'])} & {f1(s['rgb']['unseen'])} & {100 * g['unseen']:+.1f} \\")
        rows.append(r"\midrule")
    rows = rows[:-1]
    tab(T / "dist_seen.tex", "dist_seen",
        "FCN overall accuracy (\\%) on test pixels inside a field that holds a training pixel (seen) and inside other fields (unseen)",
        "llrrrrrrr", [r" & & & \multicolumn{3}{c}{Seen fields} & \multicolumn{3}{c}{Unseen fields} \\",
                      r"\cmidrule(lr){4-6}\cmidrule(lr){7-9}",
                      r"Scene & Split & Share (\%) & HSI & RGB & Gap & HSI & RGB & Gap \\"], rows, wide=True)
    # reweighting
    rows = []
    for sc in SCENES:
        rw = R["scenes"][sc]["reweight"]
        for g, name in (("hsi_rgb", "HSI$-$RGB"), ("nir", "RGBN$-$RGB")):
            e = rw["fcn"][g]
            ex = "--" if e["explained"] is None else f"{100 * e['explained']:.0f}"
            rows.append(rf"{first(sc, g == 'hsi_rgb')} & {name} & {100 * e['random']:+.1f} & {100 * e['block']:+.1f} & "
                        rf"{100 * e['block_at_random_distances']:+.1f} & {ex} \\")
        rows.append(r"\midrule")
    rows = rows[:-1]
    tab(T / "dist_reweight.tex", "dist_reweight",
        "Spectral gain of the FCN (points) under each split, the block-split gain re-weighted to the distance "
        "distribution of the random split, and the share of the protocol difference explained by distance",
        "llrrrr", [r"Scene & Gain & Random & Block & Re-weighted & Expl. (\%) \\"], rows, size="scriptsize", sep=2)
    # separability
    rows = []
    for sc in SCENES:
        cells = [f"{JM[sc][sp]['mean']:.2f} / {100 * JM[sc][sp]['below_1.9']:.0f}" for sp in SENSORS]
        rows.append(rf"{H.SCENES[sc]['name']} & " + " & ".join(cells) + r" \\")
    tab(T / "dist_jm.tex", "dist_jm",
        "Spectral separability without a classifier: mean Jeffries--Matusita distance over all class pairs / share of "
        "pairs below 1.9 (\\%)", "lrrrrr", [r"Scene & HSI & S2 & L8 & RGBN & RGB \\"], rows)


def tab(path, label, caption, align, head, rows, wide=False, size="footnotesize", sep=3):
    env = "table*" if wide else "table"
    lines = [rf"\begin{{{env}}}[!t]", r"\centering", rf"\caption{{{caption}}}", rf"\label{{tab:{label}}}", f"\\{size}",
             rf"\setlength{{\tabcolsep}}{{{sep}pt}}", rf"\begin{{tabular}}{{{align}}}", r"\toprule", *head, r"\midrule", *rows,
             r"\bottomrule", r"\end{tabular}", rf"\end{{{env}}}"]
    path.write_text("\n".join(lines) + "\n"); print(f"table   {label}")


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data"); ap.add_argument("--runs", default="runs"); ap.add_argument("--out", default="paper")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed); out = PF.Out(a.out)
    R, cache = analyse(a.runs, a.data, rng)
    R["budget"] = budget(a.runs, cache)
    R["robustness"] = robustness(a.runs, cache)
    R["separability"] = separability(a.data, rng)
    R["example_seed"] = seed = example_seed(R)
    fig_concept(out, a.data, a.runs, cache, seed); fig_cdf(out, a.runs, cache); fig_accuracy(out, R); fig_gain(out, R)
    fig_seen(out, R); R["maps_PU"] = fig_maps(out, a.data, a.runs, cache, seed); R["mechanism_PU"] = fig_mechanism(out, a.data, R)
    fig_budget(out, R["budget"]); tables(out, R, R["budget"], R["separability"])
    for sc in SCENES:   # pair names are long; keep only the summaries in the JSON
        for sp in SENSORS:
            pairs = R["separability"][sc][sp].pop("pairs")
            if sc == "PU":
                R["separability"][sc][sp]["meadows_soil"] = pairs[f"{min(PU_MEADOWS, PU_SOIL)}-{max(PU_MEADOWS, PU_SOIL)}"]
    (out.tab / "distance_analysis.json").write_text(json.dumps(R, indent=1))
    print("wrote", out.tab / "distance_analysis.json")


if __name__ == "__main__":
    main()
