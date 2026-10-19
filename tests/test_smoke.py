"""Fast checks on a synthetic scene: splits, spectral simulation, metrics, a short training run and prediction."""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import hsilc as H  # noqa: E402


def synthetic(h=40, w=48, bands=30, classes=3, seed=0):
    rng = np.random.default_rng(seed)
    labels = np.zeros((h, w), np.int64)
    labels[2:18, 2:20] = 1; labels[22:38, 4:24] = 2; labels[5:35, 30:46] = 3
    means = rng.normal(size=(classes + 1, bands)) * 2
    cube = means[labels] + rng.normal(scale=0.5, size=(h, w, bands))
    return cube.astype(np.float32), labels


def test_random_split_is_disjoint_and_complete():
    _, labels = synthetic()
    tr, va, te = H.split_random(labels, per_class=10, val_frac=0.2, seed=1)
    assert not ((tr > 0) & (va > 0)).any() and not ((tr > 0) & (te > 0)).any() and not ((va > 0) & (te > 0)).any()
    assert ((tr > 0) | (va > 0) | (te > 0)).sum() == (labels > 0).sum()
    for c in (1, 2, 3):
        assert (tr == c).sum() == 8 and (va == c).sum() == 2


def test_block_split_keeps_test_blocks_apart():
    _, labels = synthetic()
    tr, va, te = H.split_blocks(labels, per_class=10, val_frac=0.2, seed=3, block=8)
    used = (tr > 0) | (va > 0)
    blocks = lambda m: {(y // 8, x // 8) for y, x in np.argwhere(m)}
    assert not blocks(used) & blocks(te > 0)
    assert set(np.unique(tr[tr > 0])) == {1, 2, 3}


def test_spectral_response_rows_sum_to_one():
    wl = H.SCENES["IP"]["wavelengths"]
    r, names = H.spectral_response(wl, "s2")
    assert names[0] == "B2" and "B12" in names and np.allclose(r.sum(1), 1)
    r, names = H.spectral_response(H.SCENES["PU"]["wavelengths"], "s2")
    assert "B11" not in names and len(names) == 7
    assert H.spectral_response(wl, "rgb")[1] == ["B4", "B3", "B2"]


def test_scores_match_hand_computation():
    cm = np.array([[8, 2], [1, 9]])
    s = H.scores(cm)
    assert s["oa"] == pytest.approx(0.85) and s["aa"] == pytest.approx(0.85)
    pe = (9 * 10 + 11 * 10) / 400
    assert s["kappa"] == pytest.approx((0.85 - pe) / (1 - pe))


def test_tiled_prediction_matches_whole_scene():
    torch.manual_seed(0)
    m = H.build_model("fcn", 30, 3, width=16).eval()
    x = np.random.default_rng(0).normal(size=(40, 48, 30)).astype(np.float32)
    whole = H.predict(m, x, "cpu", tta=False)
    tiled = H.predict(m, x, "cpu", tta=False, tile=32, overlap=16)
    assert whole.shape == tiled.shape == (3, 40, 48)
    assert np.allclose(tiled.sum(0), 1, atol=1e-5)
    assert (whole.argmax(0) == tiled.argmax(0)).mean() > 0.8


def test_short_training_learns_separable_classes(tmp_path):
    cube, labels = synthetic()
    x, mean, std = H.preprocess(cube)
    scene = H.Scene("toy", x, labels, ["a", "b", "c"], [f"b{i}" for i in range(30)], mean, std)
    tr, va, te = H.split_random(labels, per_class=10, val_frac=0.2, seed=0)
    model = H.train(dict(width=16, crop=32, steps=60, val_every=20), scene, tr, va, tmp_path, "cpu", log=lambda m: None)
    r = H.evaluate(H.predict(model, x, "cpu"), te, 3)
    assert r["oa"] > 0.9
    m2, ck = H.load_checkpoint(tmp_path / "model.pt")
    assert ck["classes"] == ["a", "b", "c"]
    assert np.allclose(H.predict(m2, x, "cpu"), H.predict(model, x, "cpu"), atol=1e-5)
