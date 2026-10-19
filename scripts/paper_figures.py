"""Every figure and table for the paper, from the runs written by train_eval.py.

    python scripts/paper_figures.py --data data --runs runs --out paper

Writes paper/figures/<name>.pdf (vector) and <name>.png (300 dpi), paper/tables/<name>.tex (LaTeX, booktabs) and
paper/tables/results.md (all tables in Markdown). Figures and tables whose runs are missing are skipped with a note,
so the script can be run at any time. Figures that need no runs (architecture, data, spectra, split protocol) are
always drawn.

Colours follow the method across all figures: SVM blue, MLP aqua, FCN orange.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import hsilc as H  # noqa: E402
from hsilc import viz  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.patches import FancyBboxPatch, Patch  # noqa: E402

INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df", "#b9b8b3"
METHODS = {"svm": ("SVM", viz.HUES[0]), "mlp": ("MLP", viz.HUES[2]), "fcn": ("FCN", viz.HUES[1]),
           "anysensor": ("AnySensor", viz.HUES[6]), "anysensor-norss": ("AnySensor w/o RSS", viz.HUES[6]),
           "anysensor-nofwhm": ("AnySensor, centres only", viz.HUES[6])}
SPECTRAL = {"full": "Hyperspectral", "s2": "Sentinel-2-like", "landsat8": "Landsat-8-like", "rgbn": "RGB+NIR",
            "rgb": "RGB"}
SHORT = {"full": "HSI", "s2": "S2", "landsat8": "L8", "rgbn": "RGBN", "rgb": "RGB"}
SCENES = ["IP", "PU", "SA"]
COL1, COL2 = 3.5, 7.16                                  # single- and double-column widths (in)
SEQ = LinearSegmentedColormap.from_list("seq", ["#ffffff", "#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#104281", "#0d366b"])
CONFUSED = {"IP": (2, 11), "PU": (3, 8), "SA": (8, 15)}  # spectrally similar class pairs highlighted in the spectra
PAIR_COLORS = (INK, viz.HUES[7])
SPLIT_COLORS = {"train": INK, "val": viz.HUES[4], "test": "#a8a7a2", "unused": "#e1e0db"}

plt.rcParams.update({
    "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "legend.fontsize": 7, "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.titlecolor": INK, "legend.frameon": False, "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.dpi": 300,
    "figure.facecolor": "white", "savefig.facecolor": "white"})


# ---------------------------------------------------------------- helpers
class Out:
    def __init__(self, root):
        self.fig = Path(root) / "figures"; self.tab = Path(root) / "tables"
        self.fig.mkdir(parents=True, exist_ok=True); self.tab.mkdir(parents=True, exist_ok=True); self.md = []

    def save(self, fig, name):
        fig.savefig(self.fig / f"{name}.pdf", bbox_inches="tight"); fig.savefig(self.fig / f"{name}.png", bbox_inches="tight")
        plt.close(fig); print(f"figure  {name}")

    def table(self, name, caption, header, rows, align=None):
        align = align or "l" * len(header)
        esc = lambda t: str(t).replace("&", r"\&").replace("%", r"\%").replace("_", r"\_").replace("±", r"$\pm$") \
            .replace("κ", r"$\kappa$").replace("Δ", r"$\Delta$").replace("✓", r"\checkmark").replace("–", "--") \
            .replace("≈", r"$\approx$").replace("×", r"$\times$")
        cell = lambda c: rf"\textbf{{{esc(str(c)[4:])}}}" if str(c).startswith("**: ") else esc(c)
        tex = [r"\begin{table}[t]", r"\centering", rf"\caption{{{esc(caption)}}}", rf"\label{{tab:{name}}}",
               r"\small", rf"\begin{{tabular}}{{{align}}}", r"\toprule", " & ".join(esc(h) for h in header) + r" \\",
               r"\midrule"]
        for r in rows:
            if r == "---":
                tex.append(r"\midrule"); continue
            tex.append(" & ".join(cell(c) for c in r) + r" \\")
        tex += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
        (self.tab / f"{name}.tex").write_text("\n".join(tex) + "\n")
        md = [f"### {caption}", "", "| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
        md += ["| " + " | ".join((f"**{c[4:]}**" if str(c).startswith("**: ") else str(c)) for c in r) + " |"
               for r in rows if r != "---"]
        self.md.append("\n".join(md)); print(f"table   {name}")


def skip(what, why):
    print(f"skipped {what}: {why}")


def style(ax, grid="y"):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    if grid:
        ax.grid(axis=grid, color=GRID, lw=0.6, zorder=0); ax.set_axisbelow(True)
    ax.tick_params(length=2, width=0.6)


def load_runs(runs):
    res = {}
    for p in sorted(Path(runs).glob("*/results.json")):
        r = json.loads(p.read_text()); r["dir"] = p.parent
        res[(r["scene"], r["spectral"], r["split"], r["model"], r.get("labelled_per_class", 50))] = r
    return res


def method_result(res, sc, method, sp="full", split="random", n=50):
    """(summary dict with oa/aa/kappa mean+sd and per_class, run dict) of one method, or (None, None)."""
    if method == "svm":
        for model in ("fcn", "mlp"):
            r = res.get((sc, sp, split, model, n))
            if r and "svm" in r:
                return r["svm"], r
        return None, None
    r = res.get((sc, sp, split, method, n))
    return (r["net"], r) if r and "net" in r else (None, None)


def seed_values(run, key, k):
    return [v[key][k] for v in run["runs"].values() if key in v]


def pm(s, k, digits=2):
    return f"{100 * s[k]['mean']:.{digits}f} ± {100 * s[k]['sd']:.{digits}f}"


_cubes = {}


def scene_data(sc, data):
    if sc not in _cubes:
        s = H.SCENES[sc]
        cube = H.read_array(Path(data) / s["image"])[0].astype(np.float32)
        labels = H.read_array(Path(data) / s["labels"])[0].astype(np.int64)
        _cubes[sc] = (cube, labels)
    return _cubes[sc]


def composite(sc, data):
    """False-colour infrared composite (bands nearest 860, 650, 550 nm), 2-98 % stretch."""
    cube, _ = scene_data(sc, data); wl = H.SCENES[sc]["wavelengths"]
    idx = [int(np.argmin(np.abs(wl - w))) for w in (860, 650, 550)]
    img = cube[..., idx]
    lo, hi = np.percentile(img.reshape(-1, 3), [2, 98], axis=0)
    return np.clip((img - lo) / (hi - lo + 1e-6), 0, 1)


# ---------------------------------------------------------------- figures without runs
def fig_architecture(out):
    fig, ax = plt.subplots(figsize=(COL2, 2.9)); ax.set_xlim(-0.1, 14.3); ax.set_ylim(-0.2, 5.4); ax.set_axis_off()

    def box(x, y, w, h, title, sub, fill="#f0efec", edge=AXIS):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12", fc=fill, ec=edge, lw=0.8))
        ax.text(x + w / 2, y + h * 0.74, title, ha="center", va="center", fontsize=6.5, color=INK, weight="bold")
        ax.text(x + w / 2, y + h * 0.36, sub, ha="center", va="center", fontsize=5.3, color=INK2, linespacing=1.25)

    def arrow(x0, y0, x1, y1):
        ax.annotate("", (x1, y1), (x0, y0), arrowprops=dict(arrowstyle="-|>", color=INK2, lw=0.8, shrinkA=0, shrinkB=0,
                                                            mutation_scale=7))
    enc_fill, dec_fill = "#e4effc", "#fde7dd"
    w, h, top, bot = 2.2, 1.35, 3.5, 0.4
    box(0.0, top, 1.5, h, "Input", "H × W × B\nbands")
    box(1.8, top, 1.9, h, "Spectral stem", "2 × (1×1 conv,\nBN, ReLU), 64 ch")
    xs = [4.0 + i * 2.55 for i in range(4)]
    enc = [("Stage 1 (1/1)", "residual block\n64 ch"), ("Stage 2 (1/2)", "stride-2 conv,\nresidual, 96 ch"),
           ("Stage 3 (1/4)", "stride-2 conv,\nresidual, 128 ch"), ("Stage 4 (1/8)", "stride-2 conv, residual,\n128 ch + global context")]
    for x, (t, s) in zip(xs, enc):
        box(x, top, w, h, t, s, enc_fill)
    arrow(1.5, top + h / 2, 1.8, top + h / 2); arrow(3.7, top + h / 2, xs[0], top + h / 2)
    for i in range(3):
        arrow(xs[i] + w, top + h / 2, xs[i + 1], top + h / 2)
    dec = ["Fuse (1/1)", "Fuse (1/2)", "Fuse (1/4)", "Lateral (1/8)"]
    for i, x in enumerate(xs):
        sub = "1×1 conv,\n64 ch" if i == 3 else "upsample + 1×1 lateral,\nadd, 3×3 conv, 64 ch"
        box(x, bot, w, h, dec[i], sub, dec_fill)
        arrow(x + w / 2, top, x + w / 2, bot + h)
    for i in (3, 2, 1):
        arrow(xs[i], bot + h / 2, xs[i - 1] + w, bot + h / 2)
    box(1.8, bot, 1.9, h, "Classifier", "dropout 0.1,\n1×1 conv → C")
    box(0.0, bot, 1.5, h, "Class map", "H × W")
    arrow(xs[0], bot + h / 2, 3.7, bot + h / 2); arrow(1.8, bot + h / 2, 1.5, bot + h / 2)
    ax.text(xs[0], top + h + 0.2, "Encoder", fontsize=7, color=INK2)
    ax.text(xs[0], bot - 0.4, "Top-down decoder", fontsize=7, color=INK2)
    out.save(fig, "fig_architecture")


def fig_data(out, data):
    for sc in SCENES:
        s = H.SCENES[sc]; _, labels = scene_data(sc, data); n = len(s["classes"])
        Hh, W = labels.shape; pw = 2.3; ph = pw * Hh / W
        fig, axes = plt.subplots(1, 2, figsize=(2 * pw + 0.3, ph + 0.4))
        axes[0].imshow(composite(sc, data), interpolation="nearest")
        axes[0].set_title("(a) False-colour composite (≈860/650/550 nm)")
        axes[1].imshow(viz.colorize(labels, n), interpolation="nearest")
        axes[1].set_title(f"(b) Reference labels ({int((labels > 0).sum()):,} px)")
        for ax in axes:
            ax.set_axis_off()
        counts = np.bincount(labels.ravel(), minlength=n + 1)[1:]
        cols = viz.class_colors(n) / 255.0
        handles = [Patch(fc=c, ec="none", label=f"{i + 1} {name} ({k:,})") for i, (c, name, k) in
                   enumerate(zip(cols, s["classes"], counts))]
        fig.legend(handles=handles, loc="center left", bbox_to_anchor=(1.0, 0.5), labelcolor=INK2, fontsize=6.5,
                   handlelength=1.0, handleheight=0.9)
        fig.suptitle(f"{s['name']} ({s['sensor']}, {Hh} × {W} px, {s['shape'][2]} bands, {s['gsd']} m)", fontsize=8,
                     color=INK, y=1.0)
        fig.tight_layout(); out.save(fig, f"fig_data_{sc}")


def fig_spectra(out, data):
    fig, axes = plt.subplots(1, 3, figsize=(COL2, 2.4))
    for ax, sc in zip(axes, SCENES):
        s = H.SCENES[sc]; cube, labels = scene_data(sc, data); wl = s["wavelengths"]
        gap = np.diff(wl) > 3 * np.median(np.diff(wl))
        x = np.insert(wl, np.flatnonzero(gap) + 1, np.nan)
        lo, hi = wl.min(), wl.max(); last = -np.inf
        for b, (c, f) in H.S2_BANDS.items():
            if lo <= c <= hi:
                ax.axvspan(c - f / 2, c + f / 2, color="#eeede9", lw=0, zorder=0)
                if c - last > 0.05 * (hi - lo):          # label only bands that do not crowd the previous label
                    ax.text(c, 1.02, b, transform=ax.get_xaxis_transform(), ha="center", va="bottom", fontsize=5,
                            color=MUTED)
                    last = c
        means = np.stack([cube[labels == c].mean(0) for c in range(1, len(s["classes"]) + 1)]) / 1000.0
        for c, m in enumerate(means, 1):
            if c not in CONFUSED[sc]:
                ax.plot(x, np.insert(m, np.flatnonzero(gap) + 1, np.nan), color="#c9c8c2", lw=0.6, zorder=1)
        for c, col in zip(CONFUSED[sc], PAIR_COLORS):
            ax.plot(x, np.insert(means[c - 1], np.flatnonzero(gap) + 1, np.nan), color=col, lw=1.4, zorder=2,
                    label=f"{c} {s['classes'][c - 1]}")
        ax.set_title(s["name"], pad=12); ax.set_xlabel("Wavelength (nm, nominal)"); style(ax, None)
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.3), fontsize=6, labelcolor=INK2, handlelength=1.2)
    axes[0].set_ylabel("Mean value (×10³)")
    fig.tight_layout(); out.save(fig, "fig_spectra")


def fig_split_protocol(out, data, sc="IP", seed=1):
    _, labels = scene_data(sc, data); Hh, W = labels.shape
    fig, axes = plt.subplots(1, 2, figsize=(COL2 * 0.75, 2.9))
    block = max(8, int(round(min(Hh, W) / 8)))
    for ax, kind in zip(axes, ("random", "block")):
        tr, va, te = H.make_split(labels, kind, 50, 0.2, seed)
        img = np.empty((Hh, W, 3)); img[:] = matplotlib.colors.to_rgb(viz.BACKGROUND)
        img[(labels > 0)] = matplotlib.colors.to_rgb(SPLIT_COLORS["unused"])
        img[te > 0] = matplotlib.colors.to_rgb(SPLIT_COLORS["test"])
        ax.imshow(img, interpolation="nearest")
        for lab, key in ((tr, "train"), (va, "val")):
            yy, xx = np.nonzero(lab)
            ax.scatter(xx, yy, s=1.6, c=SPLIT_COLORS[key], lw=0, zorder=3)
        if kind == "block":
            for k in range(0, max(Hh, W) + 1, block):
                ax.axhline(k - 0.5, color="white", lw=0.5); ax.axvline(k - 0.5, color="white", lw=0.5)
            ax.set_xlim(-0.5, W - 0.5); ax.set_ylim(Hh - 0.5, -0.5)
        ax.set_title(f"({'a' if kind == 'random' else 'b'}) {kind} split: {(tr > 0).sum()} train, {(va > 0).sum()} val, "
                     f"{(te > 0).sum():,} test", fontsize=7)
        ax.set_axis_off()
    handles = [Patch(color=SPLIT_COLORS[k], label=l) for k, l in (("train", "training"), ("val", "validation"),
                                                                  ("test", "test"), ("unused", "not used"))]
    fig.legend(handles=handles, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.02), labelcolor=INK2)
    fig.tight_layout(rect=(0, 0.06, 1, 1)); out.save(fig, f"fig_split_protocol_{sc}")


# ---------------------------------------------------------------- figures from runs
def bars(ax, groups, series, ylabel=True, labels=False):
    """groups: list of x labels; series: list of (name, colour, means, sds, hatch)."""
    n = len(series); bw = 0.8 / n; xs = np.arange(len(groups))
    for j, (name, col, means, sds, hatch) in enumerate(series):
        pos = xs - 0.4 + bw * (j + 0.5)
        m = np.array([np.nan if v is None else v for v in means], float)
        ax.bar(pos, m, bw * 0.92, color=col if not hatch else "white", edgecolor=col, hatch=hatch, lw=0.8 if hatch else 0,
               label=name, zorder=2)
        ax.errorbar(pos, m, [0 if v is None else v for v in sds], fmt="none", ecolor=INK2, elinewidth=0.7, capsize=1.5,
                    zorder=3)
        if labels:
            for p, v in zip(pos, m):
                if not np.isnan(v):
                    ax.text(p, v + 1.2, f"{v:.1f}", ha="center", va="bottom", fontsize=5, color=INK2, rotation=90)
    ax.set_xticks(xs, groups); ax.set_ylim(0, 112); ax.set_yticks(range(0, 101, 20)); style(ax)
    if ylabel:
        ax.set_ylabel("Overall accuracy (%)")


def fig_spectral_accuracy(out, res):
    scenes = [sc for sc in SCENES if method_result(res, sc, "fcn")[0]]
    if not scenes or not any(method_result(res, sc, "fcn", sp)[0] for sc in scenes for sp in ("s2", "rgb")):
        return skip("fig_spectral_accuracy", "needs --spectral s2/rgb runs")
    fig, axes = plt.subplots(1, len(scenes), figsize=(COL2, 2.5), sharey=True, squeeze=False)
    for i, (ax, sc) in enumerate(zip(axes[0], scenes)):
        sps = [sp for sp in SPECTRAL if method_result(res, sc, "fcn", sp)[0]]
        groups = [f"{SHORT[sp]}\n({method_result(res, sc, 'fcn', sp)[1]['bands']})" for sp in sps]
        series = []
        for m in ("svm", "fcn"):
            st = [method_result(res, sc, m, sp)[0] for sp in sps]
            series.append((METHODS[m][0], METHODS[m][1], [100 * s["oa"]["mean"] if s else None for s in st],
                           [100 * s["oa"]["sd"] if s else None for s in st], None))
        bars(ax, groups, series, ylabel=i == 0); ax.set_title(H.SCENES[sc]["name"])
    handles, names = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, names, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.02), labelcolor=INK2)
    fig.tight_layout(rect=(0, 0.07, 1, 1)); out.save(fig, "fig_spectral_accuracy")


def fig_random_vs_block(out, res):
    scenes = [sc for sc in SCENES if method_result(res, sc, "fcn", "full", "block")[0]]
    if not scenes:
        return skip("fig_random_vs_block", "needs --split block runs")
    fig, axes = plt.subplots(1, len(scenes), figsize=(COL2, 2.6), sharey=True, squeeze=False)
    for i, (ax, sc) in enumerate(zip(axes[0], scenes)):
        sps = [sp for sp in SPECTRAL if method_result(res, sc, "fcn", sp, "block")[0] and method_result(res, sc, "fcn", sp)[0]]
        series = []
        for m in ("svm", "fcn"):
            for split, hatch in (("random", None), ("block", "//////")):
                st = [method_result(res, sc, m, sp, split)[0] for sp in sps]
                series.append((f"{METHODS[m][0]}, {split}", METHODS[m][1], [100 * s["oa"]["mean"] if s else None for s in st],
                               [100 * s["oa"]["sd"] if s else None for s in st], hatch))
        bars(ax, [SHORT[sp] for sp in sps], series, ylabel=i == 0); ax.set_title(H.SCENES[sc]["name"])
    handles, names = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, names, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.03), labelcolor=INK2)
    fig.tight_layout(rect=(0, 0.07, 1, 1)); out.save(fig, "fig_random_vs_block")


def fig_maps(out, res, data, seed=1):
    for sc in SCENES:
        fcn, run = method_result(res, sc, "fcn")
        if not fcn:
            skip(f"fig_maps_{sc}", "needs the full/random FCN run"); continue
        d = Path(run["dir"]) / f"seed{seed}"; labels = scene_data(sc, data)[1]; classes = H.SCENES[sc]["classes"]
        test = np.load(d / "split.npz")["test"].astype(np.int64); n = len(classes)
        metrics = json.loads((d / "metrics.json").read_text())
        preds = [("SVM", np.load(d / "pred_svm.npy").astype(np.int64), metrics["svm"]["oa"])]
        _, mrun = method_result(res, sc, "mlp")
        if mrun and (Path(mrun["dir"]) / f"seed{seed}" / "pred_net.npy").exists():
            dm = Path(mrun["dir"]) / f"seed{seed}"
            mt = json.loads((dm / "metrics.json").read_text())
            preds.append(("MLP", np.load(dm / "pred_net.npy").astype(np.int64), mt["net"]["oa"]))
        preds.append(("FCN", np.load(d / "pred_net.npy").astype(np.int64), metrics["net"]["oa"]))
        Hh, W = labels.shape; cols = 1 + len(preds); pw = (COL2 - 1.7) / cols; ph = pw * Hh / W
        fig, axes = plt.subplots(2, cols, figsize=(COL2, 2 * ph + 0.6))
        axes[0, 0].imshow(viz.colorize(labels, n), interpolation="nearest"); axes[0, 0].set_title("Reference")
        axes[1, 0].imshow(composite(sc, data), interpolation="nearest"); axes[1, 0].set_title("False colour")
        for j, (name, p, oa) in enumerate(preds, 1):
            axes[0, j].imshow(viz.colorize(p, n), interpolation="nearest"); axes[0, j].set_title(f"{name} map")
            axes[1, j].imshow(viz.error_image(p, test), interpolation="nearest")
            axes[1, j].set_title(f"{name} errors (OA {100 * oa:.1f} %)", fontsize=6.5)
        for ax in axes.ravel():
            ax.set_axis_off()
        handles = viz.legend_handles(classes) + [Patch(color=viz.CORRECT, label="correct (test)"),
                                                 Patch(color=viz.WRONG, label="misclassified (test)")]
        fig.legend(handles=handles, loc="center left", bbox_to_anchor=(1.0, 0.5), labelcolor=INK2, fontsize=6,
                   handlelength=1.0, handleheight=0.9)
        fig.tight_layout(); out.save(fig, f"fig_maps_{sc}")


def fig_confusion(out, res):
    scenes = [sc for sc in SCENES if method_result(res, sc, "fcn")[0]]
    if not scenes:
        return skip("fig_confusion", "needs full/random FCN runs")
    fig, axes = plt.subplots(2, len(scenes), figsize=(COL2, 5.3), squeeze=False)
    for j, sc in enumerate(scenes):
        for i, m in enumerate(("svm", "fcn")):
            st, run = method_result(res, sc, m); key = "svm" if m == "svm" else "net"
            cm = np.sum([np.array(v[key]["confusion"]) for v in run["runs"].values()], 0).astype(float)
            cm = 100 * cm / np.maximum(cm.sum(1, keepdims=True), 1); n = len(cm); ax = axes[i, j]
            im = ax.imshow(cm, cmap=SEQ, vmin=0, vmax=100, interpolation="nearest")
            ticks = np.arange(n); ax.set_xticks(ticks, ticks + 1, fontsize=5); ax.set_yticks(ticks, ticks + 1, fontsize=5)
            ax.tick_params(length=0)
            for s in ax.spines.values():
                s.set_visible(False)
            for a in range(n):
                for b in range(n):
                    if cm[a, b] >= 5 and a != b:
                        ax.text(b, a, f"{cm[a, b]:.0f}", ha="center", va="center", fontsize=4.2,
                                color="white" if cm[a, b] > 55 else INK)
            ax.set_title(f"{H.SCENES[sc]['name']}\n{METHODS[m][0]}, OA {pm(st, 'oa', 1)} %", fontsize=6.5)
            if j == 0:
                ax.set_ylabel("Reference class")
            if i == 1:
                ax.set_xlabel("Predicted class")
    cb = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.6, pad=0.02)
    cb.set_label("Share of reference pixels (%)", color=INK2); cb.outline.set_visible(False); cb.ax.tick_params(labelsize=6)
    fig.text(0.01, 0.005, "Rows normalised; summed over seeds; off-diagonal cells ≥ 5 % are labelled.", fontsize=6, color=MUTED)
    out.save(fig, "fig_confusion")


def fig_learning_curve(out, res):
    ns = sorted({k[4] for k in res if k[1] == "full" and k[2] == "random" and k[3] == "fcn"})
    if len(ns) < 2:
        return skip("fig_learning_curve", "needs runs with several --per-class values")
    fig, axes = plt.subplots(1, 3, figsize=(COL2, 2.3), sharey=True)
    for i, (ax, sc) in enumerate(zip(axes, SCENES)):
        for m in ("svm", "fcn"):
            pts = [(n, method_result(res, sc, m, n=n)[0]) for n in ns]
            pts = [(n, s) for n, s in pts if s]
            if not pts:
                continue
            x = [n for n, _ in pts]; y = [100 * s["oa"]["mean"] for _, s in pts]; e = [100 * s["oa"]["sd"] for _, s in pts]
            ax.errorbar(x, y, e, color=METHODS[m][1], lw=1.5, marker="o", ms=3.5, capsize=1.5, elinewidth=0.7,
                        label=METHODS[m][0])
        ax.set_xscale("log"); ax.set_xticks(ns, [str(n) for n in ns]); ax.minorticks_off()
        ax.set_title(H.SCENES[sc]["name"]); ax.set_xlabel("Labelled pixels per class"); style(ax, "both")
        if i == 0:
            ax.set_ylabel("Overall accuracy (%)")
    axes[0].legend(loc="best", labelcolor=INK2)
    fig.tight_layout(); out.save(fig, "fig_learning_curve")


def fig_training_curves(out, res):
    scenes = [sc for sc in SCENES if method_result(res, sc, "fcn")[0]]
    if not scenes:
        return skip("fig_training_curves", "needs full/random FCN runs")
    fig, axes = plt.subplots(2, len(scenes), figsize=(COL2, 3.6), sharex="col", squeeze=False)
    col = METHODS["fcn"][1]
    for j, sc in enumerate(scenes):
        run = method_result(res, sc, "fcn")[1]
        for d in sorted(Path(run["dir"]).glob("seed*/history.json")):
            h = json.loads(d.read_text()); loss = np.array(h["loss"]); k = min(25, len(loss))
            sm = np.convolve(loss, np.ones(k) / k, mode="valid")
            axes[0, j].plot(np.arange(k, len(loss) + 1), sm, color=col, lw=0.9, alpha=0.85)
            v = h["validation"]; axes[1, j].plot([e["step"] for e in v], [100 * e["oa"] for e in v], color=col, lw=0.9,
                                                 alpha=0.85)
            sel = h["selected"]; axes[1, j].plot(sel["step"], 100 * sel["oa"], "o", ms=3.5, mfc="white", mec=INK, mew=0.8)
        axes[0, j].set_yscale("log"); axes[0, j].set_title(H.SCENES[sc]["name"])
        axes[1, j].set_xlabel("Training step")
        for i in range(2):
            style(axes[i, j], "both")
    axes[0, 0].set_ylabel("Training loss (CE)"); axes[1, 0].set_ylabel("Validation OA (%)")
    axes[1, -1].plot([], [], "o", ms=3.5, mfc="white", mec=INK, mew=0.8, label="selected checkpoint")
    axes[1, -1].plot([], [], color=col, lw=0.9, label="one line per seed"); axes[1, -1].legend(loc="lower right", labelcolor=INK2)
    fig.tight_layout(); out.save(fig, "fig_training_curves")


def fig_per_class(out, res):
    for sc in SCENES:
        sts = [(m, method_result(res, sc, m)[0]) for m in ("svm", "mlp", "fcn", "anysensor")]
        sts = [(m, s) for m, s in sts if s]
        if not any(m == "fcn" for m, _ in sts):
            skip(f"fig_per_class_{sc}", "needs the full/random FCN run"); continue
        classes = H.SCENES[sc]["classes"]; n = len(classes); ys = np.arange(n); bh = 0.8 / len(sts)
        fig, ax = plt.subplots(figsize=(COL1 + 0.6, 0.22 * n * len(sts) / 2 + 0.9))
        for j, (m, s) in enumerate(sts):
            v = [np.nan if a is None else 100 * a for a in s["per_class"]]
            ax.barh(ys - 0.4 + bh * (j + 0.5), v, bh * 0.9, color=METHODS[m][1], label=METHODS[m][0], zorder=2)
        ax.set_yticks(ys, [f"{i + 1} {c}" for i, c in enumerate(classes)], fontsize=6); ax.invert_yaxis()
        ax.set_xlim(0, 100); ax.set_xlabel("Per-class accuracy (%)"); style(ax, "x")
        ax.set_title(f"{H.SCENES[sc]['name']} (mean of seeds, random split)")
        ax.legend(loc="lower left", bbox_to_anchor=(0, 1.04), ncol=len(sts), labelcolor=INK2)
        fig.tight_layout(); out.save(fig, f"fig_per_class_{sc}")


# ---------------------------------------------------------------- AnySensor
def anysensor_result(res, sc, sensor, split="random", tag="anysensor"):
    r = res.get((sc, "full", split, tag, 50))
    return r["sensors"][sensor] if r and "sensors" in r else None


def fig_anysensor_method(out):
    fig, ax = plt.subplots(figsize=(COL2, 2.7)); ax.set_xlim(-0.1, 15.1); ax.set_ylim(-0.1, 5.2); ax.set_axis_off()

    def box(x, y, w, h, title, sub, fill="#f0efec"):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12", fc=fill, ec=AXIS, lw=0.8))
        ax.text(x + w / 2, y + h * 0.72, title, ha="center", va="center", fontsize=6.3, color=INK, weight="bold")
        ax.text(x + w / 2, y + h * 0.33, sub, ha="center", va="center", fontsize=5.2, color=INK2, linespacing=1.25)

    def arrow(x0, y0, x1, y1):
        ax.annotate("", (x1, y1), (x0, y0), arrowprops=dict(arrowstyle="-|>", color=INK2, lw=0.8, mutation_scale=7))
    new, h = "#ece9fb", 1.3
    box(0.0, 3.5, 2.5, h, "Hyperspectral crop", "B bands,\nstandardised")
    box(2.9, 3.5, 3.5, h, "Random sensor simulation", "3-16 Gaussian bands in one window,\nFWHM 10-180 nm", new)
    box(6.8, 3.5, 2.6, h, "Simulated bands", "x_1 ... x_K, standardised\nin closed form")
    box(2.9, 0.6, 3.5, h, "Band descriptors", "centre λ_k and FWHM w_k\nof each band", new)
    box(6.8, 0.6, 2.6, h, "Hypernetwork", "Fourier(λ_k), log w_k\n→ MLP → weights a_k", new)
    box(9.8, 2.05, 2.6, h, "Dynamic 1×1 conv", "e = (1/K) Σ_k x_k a_k,\nsensor-adaptive BN", new)
    box(12.8, 2.05, 2.2, h, "FCN backbone", "spectral-spatial\nencoder-decoder → map")
    arrow(2.5, 4.15, 2.9, 4.15); arrow(6.4, 4.15, 6.8, 4.15); arrow(4.65, 3.5, 4.65, 1.9); arrow(6.4, 1.25, 6.8, 1.25)
    arrow(9.4, 4.15, 10.4, 3.35); arrow(9.4, 1.25, 10.4, 2.05); arrow(12.4, 2.7, 12.8, 2.7)
    ax.text(0.0, 1.9, "training: a new random\nsensor for every batch\n\ntest: hyperspectral,\nSentinel-2, Landsat-8,\n"
            "RGB+NIR, RGB\n(never seen in training)", fontsize=5.6, color=INK2, va="center")
    out.save(fig, "fig_anysensor_method")


def fig_sensors(out, sc="IP"):
    """Random virtual sensors as drawn during training and the five test sensors, over the wavelengths of a scene."""
    wl = H.SCENES[sc]["wavelengths"]; x = np.linspace(380, 2520, 2000)
    fake = H.Scene(sc, np.zeros((2, 2, len(wl)), np.float32), np.zeros((2, 2), np.int64), ["a"], [], np.zeros(len(wl)),
                   np.ones(len(wl)), meta={"wavelengths": list(wl)})
    from hsilc.anysensor import Simulator
    sim = Simulator(fake); rng = np.random.default_rng(3)
    rows = [(f"random sensor {i + 1}", sim.random(rng)[1], METHODS["anysensor"][1]) for i in range(4)]
    rows += [(SPECTRAL[s], H.sensor_bands(wl, s)[2], METHODS["fcn"][1]) for s in ("s2", "landsat8", "rgbn", "rgb")]
    fig, axes = plt.subplots(len(rows), 1, figsize=(COL2, 0.42 * len(rows) + 0.5), sharex=True)
    gaps = np.flatnonzero(np.diff(wl) > 3 * np.median(np.diff(wl)))
    for ax, (name, desc, col) in zip(axes, rows):
        for c, f in desc:
            ax.fill_between(x, np.exp(-0.5 * ((x - c) / (f / 2.3548)) ** 2), color=col, alpha=0.35, lw=0)
            ax.plot(x, np.exp(-0.5 * ((x - c) / (f / 2.3548)) ** 2), color=col, lw=0.7)
        for g in gaps:
            ax.axvspan(wl[g], wl[g + 1], color="#e4e3df", lw=0, zorder=0)
        ax.set_ylim(0, 1.15); ax.set_yticks([]); style(ax, None); ax.spines["left"].set_visible(False)
        ax.text(-0.01, 0.5, f"{name} ({len(desc)})", transform=ax.transAxes, ha="right", va="center", fontsize=6.5,
                color=INK2)
    axes[-1].set_xlim(380, 2520); axes[-1].set_xlabel("Wavelength (nm)")
    axes[0].set_title(f"Training: random virtual sensors (violet). Test: real sensor layouts (orange). Grey: removed "
                      f"water-absorption bands ({H.SCENES[sc]['name']}).", fontsize=6.5, color=INK2, loc="left")
    fig.tight_layout(h_pad=0.2); out.save(fig, "fig_sensors")


def fig_anysensor(out, res):
    splits = [sp for sp in ("random", "block") if any(anysensor_result(res, sc, "full", sp) for sc in SCENES)]
    if not splits:
        return skip("fig_anysensor", "needs --model anysensor runs")
    fig, axes = plt.subplots(len(splits), 3, figsize=(COL2, 2.5 * len(splits) + 0.3), sharey=True, squeeze=False)
    for i, split in enumerate(splits):
        for j, sc in enumerate(SCENES):
            ax = axes[i, j]; sens = [s for s in SPECTRAL if anysensor_result(res, sc, s, split)]
            if not sens:
                ax.set_axis_off(); continue
            series = []
            fc = [method_result(res, sc, "fcn", s, split)[0] for s in sens]
            series.append(("FCN, one model per sensor", METHODS["fcn"][1], [100 * f["oa"]["mean"] if f else None for f in fc],
                           [100 * f["oa"]["sd"] if f else None for f in fc], None))
            for tag, hatch in (("anysensor", None), ("anysensor-norss", "//////"), ("anysensor-nofwhm", "....")):
                st = [anysensor_result(res, sc, s, split, tag) for s in sens]
                if any(st):
                    series.append((METHODS[tag][0] + (" (one model)" if tag == "anysensor" else ""), METHODS[tag][1],
                                   [100 * v["oa"]["mean"] if v else None for v in st],
                                   [100 * v["oa"]["sd"] if v else None for v in st], hatch))
            bars(ax, [SHORT[s] for s in sens], series, ylabel=j == 0)
            ax.set_title(f"{H.SCENES[sc]['name']}, {split} split")
    handles, names = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, names, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.02), labelcolor=INK2)
    fig.tight_layout(rect=(0, 0.06, 1, 1)); out.save(fig, "fig_anysensor")


def anysensor_tables(out, res):
    for split in ("random", "block"):
        rows = []
        for sc in SCENES:
            for s in SPECTRAL:
                a = anysensor_result(res, sc, s, split)
                if not a:
                    continue
                f = method_result(res, sc, "fcn", s, split)[0]; sv = method_result(res, sc, "svm", s, split)[0]
                ab = [anysensor_result(res, sc, s, split, t) for t in ("anysensor-norss", "anysensor-nofwhm")]
                rows.append([H.SCENES[sc]["name"], SPECTRAL[s], a["bands"], pm(sv, "oa") if sv else "–",
                             pm(f, "oa") if f else "–", pm(a, "oa"),
                             f"{100 * (a['oa']['mean'] - f['oa']['mean']):+.2f}" if f else "–"]
                            + [pm(v, "oa") if v else "–" for v in ab])
            rows.append("---")
        if any(r != "---" for r in rows):
            out.table(f"anysensor_{split}", f"One AnySensor model versus one model per sensor ({split} split, overall "
                      "accuracy %); Δ: AnySensor minus per-sensor FCN; ablations without random sensor simulation (RSS) "
                      "and without FWHM conditioning",
                      ["Scene", "Sensor", "Bands", "SVM", "FCN per sensor", "AnySensor", "Δ", "w/o RSS", "centres only"],
                      rows[:-1], "llrllllll")
        else:
            skip(f"table anysensor_{split}", "needs --model anysensor runs")


# ---------------------------------------------------------------- tables
def tables(out, res, data):
    rows = []
    for sc in SCENES:
        s = H.SCENES[sc]; labels = scene_data(sc, data)[1]
        tr, va, te = H.split_random(labels, 50, 0.2, 1); tb, vb, eb = H.split_blocks(labels, 50, 0.2, 1)
        rows.append([s["name"], s["sensor"], f"{s['shape'][0]} × {s['shape'][1]}", s["shape"][2], s["gsd"],
                     len(s["classes"]), f"{s['labelled']:,}", f"{(tr > 0).sum()} / {(va > 0).sum()} / {(te > 0).sum():,}",
                     f"{(tb > 0).sum()} / {(vb > 0).sum()} / {(eb > 0).sum():,}"])
    out.table("datasets", "Scenes and label splits (seed 1): training / validation / test pixels",
              ["Scene", "Sensor", "Size (px)", "Bands", "GSD (m)", "Classes", "Labelled px", "Random split", "Block split"],
              rows, "llrrrrrrr")

    rows = []
    for b, (c, f) in H.S2_BANDS.items():
        rows.append([b, c, f] + ["✓" if H.SCENES[sc]["wavelengths"].min() <= c <= H.SCENES[sc]["wavelengths"].max()
                                 else "–" for sc in SCENES] + ["✓" if b in H.RGB_BANDS else "–"])
    out.table("simulated_bands", "Simulated Sentinel-2A bands (Gaussian response, centre and FWHM in nm)",
              ["Band", "Centre", "FWHM", "IP", "PU", "SA", "RGB"], rows, "lrrcccc")

    rows = []
    for sc in SCENES:
        sts = [(m, *method_result(res, sc, m)) for m in ("svm", "mlp", "fcn", "anysensor")]
        sts = [x for x in sts if x[1]]
        if not sts:
            continue
        best = max(s["oa"]["mean"] for _, s, _ in sts)
        for m, s, run in sts:
            key = "svm" if m == "svm" else "net"
            if m == "svm":
                extra = ["–", f"{np.mean(seed_values(run, key, 'seconds')):.1f} s" if seed_values(run, key, 'seconds') else "–", "–"]
            else:
                p = seed_values(run, key, "parameters"); t = seed_values(run, key, "train_minutes")
                q = seed_values(run, key, "predict_seconds")
                extra = [f"{p[0] / 1e6:.2f} M" if p else "–", f"{np.mean(t):.1f} min" if t else "–",
                         f"{np.mean(q):.2f} s" if q else "–"]
            oa = pm(s, "oa")
            rows.append([H.SCENES[sc]["name"], METHODS[m][0], ("**: " + oa) if s["oa"]["mean"] == best else oa,
                         pm(s, "aa"), pm(s, "kappa")] + extra)
        rows.append("---")
    if rows:
        out.table("main", "Accuracy with all hyperspectral bands, random split, 50 labelled pixels per class "
                          "(mean ± SD over three seeds, %); fit time for SVM, training and inference time for networks",
                  ["Scene", "Method", "OA", "AA", "κ", "Params", "Train", "Inference"], rows[:-1], "llllllrr")
    else:
        skip("table main", "needs full/random runs")

    rows = []
    for sc in SCENES:
        ref = method_result(res, sc, "fcn")[0]
        for sp in SPECTRAL:
            s, run = method_result(res, sc, "fcn", sp)
            if not s:
                continue
            sv = method_result(res, sc, "svm", sp)[0]
            delta = f"{100 * (s['oa']['mean'] - ref['oa']['mean']):+.2f}" if ref and sp != "full" else "–"
            rows.append([H.SCENES[sc]["name"], SPECTRAL[sp], run["bands"], pm(sv, "oa") if sv else "–", pm(s, "oa"),
                         pm(s, "aa"), pm(s, "kappa"), delta])
        rows.append("---")
    if any(r != "---" and r[1] != SPECTRAL["full"] for r in rows):
        out.table("spectral", "Effect of spectral resolution (random split, %); ΔOA: FCN relative to all bands",
                  ["Scene", "Input", "Bands", "SVM OA", "FCN OA", "FCN AA", "FCN κ", "ΔOA"], rows[:-1], "llrlllll")
    else:
        skip("table spectral", "needs --spectral s2/rgb runs")

    rows = []
    for sc in SCENES:
        for sp in SPECTRAL:
            r, b = method_result(res, sc, "fcn", sp)[0], method_result(res, sc, "fcn", sp, "block")
            if not (r and b[0]):
                continue
            rs, bs = method_result(res, sc, "svm", sp)[0], method_result(res, sc, "svm", sp, "block")[0]
            ncls = sorted({v["net"]["n_classes"] for v in b[1]["runs"].values()})
            rows.append([H.SCENES[sc]["name"], SPECTRAL[sp], pm(rs, "oa"), pm(bs, "oa"), pm(r, "oa"), pm(b[0], "oa"),
                         f"{100 * (b[0]['oa']['mean'] - r['oa']['mean']):+.2f}", "/".join(map(str, ncls))])
    if rows:
        out.table("block", "Random versus spatially disjoint (block) split: overall accuracy (%); "
                           "ΔOA: FCN block minus random; classes present in the block test set per seed",
                  ["Scene", "Input", "SVM random", "SVM block", "FCN random", "FCN block", "ΔOA", "Test classes"], rows,
                  "llllllrl")
    else:
        skip("table block", "needs --split block runs")

    for sc in SCENES:
        sts = [(m, method_result(res, sc, m)[0]) for m in ("svm", "mlp", "fcn", "anysensor")]
        sts = [(m, s) for m, s in sts if s]
        if not sts:
            continue
        labels = scene_data(sc, data)[1]; counts = np.bincount(labels.ravel())[1:]; rows = []
        for i, c in enumerate(H.SCENES[sc]["classes"]):
            vals = [s["per_class"][i] for _, s in sts]
            best = max(v for v in vals if v is not None) if any(v is not None for v in vals) else None
            rows.append([i + 1, c, f"{counts[i]:,}"] + [("–" if v is None else (("**: " if v == best else "") + f"{100 * v:.2f}"))
                                                       for v in vals])
        rows.append("---")
        rows.append(["", "OA", ""] + [pm(s, "oa") for _, s in sts]); rows.append(["", "AA", ""] + [pm(s, "aa") for _, s in sts])
        rows.append(["", "κ", ""] + [pm(s, "kappa") for _, s in sts])
        out.table(f"per_class_{sc}", f"Per-class accuracy on {H.SCENES[sc]['name']} (random split, all bands, %)",
                  ["#", "Class", "Pixels"] + [METHODS[m][0] for m, _ in sts], rows, "rlr" + "r" * len(sts))

    ns = sorted({k[4] for k in res if k[1] == "full" and k[2] == "random" and k[3] == "fcn"})
    if len(ns) > 1:
        rows = []
        for sc in SCENES:
            for n in ns:
                sv, fc = method_result(res, sc, "svm", n=n)[0], method_result(res, sc, "fcn", n=n)[0]
                if fc:
                    rows.append([H.SCENES[sc]["name"], n, pm(sv, "oa") if sv else "–", pm(fc, "oa"), pm(fc, "aa"), pm(fc, "kappa")])
            rows.append("---")
        out.table("learning_curve", "Accuracy versus labelled pixels per class (random split, all bands, %)",
                  ["Scene", "Pixels/class", "SVM OA", "FCN OA", "FCN AA", "FCN κ"], rows[:-1], "lrllll")
    else:
        skip("table learning_curve", "needs runs with several --per-class values")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("data"))
    ap.add_argument("--runs", type=Path, default=Path("runs"))
    ap.add_argument("--out", type=Path, default=Path("paper"))
    a = ap.parse_args(); out = Out(a.out); res = load_runs(a.runs)
    print(f"{len(res)} finished configurations in {a.runs}")
    fig_architecture(out); fig_data(out, a.data); fig_spectra(out, a.data); fig_split_protocol(out, a.data)
    fig_spectral_accuracy(out, res); fig_random_vs_block(out, res); fig_maps(out, res, a.data); fig_confusion(out, res)
    fig_learning_curve(out, res); fig_training_curves(out, res); fig_per_class(out, res)
    fig_anysensor_method(out); fig_sensors(out); fig_anysensor(out, res)
    tables(out, res, a.data); anysensor_tables(out, res)
    (out.tab / "results.md").write_text("# Results\n\nMean ± SD over seeds 1-3 (each seed draws its own label split).\n\n"
                                         + "\n\n".join(out.md) + "\n")
    print("tables  results.md")


if __name__ == "__main__":
    main()
