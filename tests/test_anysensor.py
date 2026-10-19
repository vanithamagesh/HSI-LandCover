"""AnySensor: closed-form sensor simulation, variable band counts and a short training run on a synthetic scene."""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import hsilc as H  # noqa: E402
from hsilc import anysensor as A  # noqa: E402
from test_smoke import synthetic  # noqa: E402

WL = np.linspace(400, 1000, 30)


def scene():
    cube, labels = synthetic()
    cube = cube + 5.0                                            # positive, sensor-like values
    x, mean, std = H.preprocess(cube)
    return cube, H.Scene("toy", x, labels, ["a", "b", "c"], [f"{w:.0f}" for w in WL], mean, std,
                         meta={"wavelengths": WL.tolist()})


def test_simulated_sensor_equals_preprocessing_of_that_sensor():
    cube, sc = scene(); sim = A.Simulator(sc)
    for name in ("s2", "landsat8", "rgbn", "rgb"):
        matrix, desc = sim.sensor(name)
        response, _, desc2 = H.sensor_bands(WL, name)
        ref = H.preprocess(cube, response)[0]
        assert np.allclose(sim.cube(sc.x, matrix), ref, atol=1e-3)
        assert np.array_equal(desc, desc2)
    assert sim.cube(sc.x, sim.sensor("full")[0]) is sc.x


def test_one_model_accepts_any_number_of_bands():
    m = A.AnySensorFCN(3, width=16).eval()
    for k in (3, 7, 30):
        desc = np.stack([np.linspace(450, 950, k), np.full(k, 40.0)], 1)
        out = m.set_sensor(desc)(torch.randn(1, k, 40, 48))
        assert out.shape == (1, 3, 40, 48)


def test_random_sensors_stay_in_range():
    _, sc = scene(); sim = A.Simulator(sc); rng = np.random.default_rng(0)
    for _ in range(20):
        matrix, desc = sim.random(rng)
        assert 3 <= len(desc) <= 16 and matrix.shape == (len(desc), len(WL))
        assert desc[:, 0].min() >= WL.min() and desc[:, 0].max() <= WL.max()
        assert 10 <= desc[:, 1].min() and desc[:, 1].max() <= 180


def test_short_training_works_for_unseen_sensors(tmp_path):
    _, sc = scene()
    tr, va, te = H.split_random(sc.labels, per_class=10, val_frac=0.2, seed=0)
    model = A.train_anysensor(dict(width=16, crop=32, steps=120, val_every=40), sc, tr, va, tmp_path, "cpu",
                              log=lambda m: None)
    sim = A.Simulator(sc)
    for name in ("full", "s2", "rgb"):
        assert H.evaluate(A.predict_sensor(model, sc, sim, name, "cpu"), te, 3)["oa"] > 0.8, name
    m2, ck = A.load_anysensor(tmp_path / "model.pt")
    assert np.allclose(A.predict_sensor(m2, sc, sim, "rgb", "cpu"), A.predict_sensor(model, sc, sim, "rgb", "cpu"), atol=1e-5)
