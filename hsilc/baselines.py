"""Classical spectral baseline: RBF support vector machine on the standardised spectra of single pixels."""
from __future__ import annotations

import itertools

import numpy as np
from sklearn.svm import SVC

C_GRID = (1, 10, 100, 1000, 10000)
GAMMA_GRID = (0.1, 0.3, 1, 3, 10)          # multiples of 1 / bands


def svm(x, train_lab, val_lab, chunk=50_000):
    """C and gamma are selected by validation OA; the selected model is fitted on the training pixels only.
    Returns (0-based class index map (H, W), selection dict)."""
    H, W, B = x.shape; X = x.reshape(-1, B)
    tr, va = train_lab.ravel() > 0, val_lab.ravel() > 0
    ytr, yva = train_lab.ravel()[tr] - 1, val_lab.ravel()[va] - 1
    best = None
    for c, g in itertools.product(C_GRID, GAMMA_GRID):
        clf = SVC(C=c, gamma=g / B).fit(X[tr], ytr)
        oa = float((clf.predict(X[va]) == yva).mean())
        if best is None or oa > best[0]:
            best = (oa, c, g, clf)
    oa, c, g, clf = best
    pred = np.concatenate([clf.predict(X[i:i + chunk]) for i in range(0, len(X), chunk)])
    return pred.reshape(H, W).astype(np.int64), {"val_oa": oa, "C": c, "gamma": g / B}
