"""Modal tests: calibration against the published measured frequencies."""

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sutong_fe import params as P
from sutong_fe.geometry import build_geometry
from sutong_fe.model import FEModel
from sutong_fe.modal import modal_analysis, pairing_report, mae, \
    eigen_residual_check


def _scales():
    """Use the stored calibration result when available."""
    path = os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "results", "calibration_result.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


_S = _scales()
_GEO = build_geometry(_S)
MODEL = FEModel(_GEO)
MODEL.shape_find()
MODES = modal_analysis(MODEL, n_modes=36)
ROWS = pairing_report(MODES)


def test_all_measured_modes_paired():
    assert all(r["f_fe"] is not None for r in ROWS)


def test_calibration_accuracy():
    assert mae(ROWS) < 5.0
    assert max(abs(r["err_pct"]) for r in ROWS) < 5.0


def test_mode_symmetry_matches_published_labels():
    for r in ROWS:
        want = "sym" if r["label"].startswith("VS") else "antisym"
        assert r["symmetry"] == want


def test_floating_mode():
    floats = [m for m in MODES if m["kind"] == "float"]
    assert floats, "longitudinal floating mode not found"
    ref = P.REFERENCE_3D["floating"]
    assert abs(floats[0]["f"] - ref) / ref < 0.25


def test_no_spurious_negative_modes():
    assert all(m["lam"] > 0 for m in MODES)


def test_eigen_residuals():
    assert eigen_residual_check(MODEL, MODES) < 1e-6


def test_mode_orthogonality():
    M = MODEL.M
    worst = 0.0
    for i, mi in enumerate(MODES):
        for mj in MODES[i + 1:]:
            pi = np.zeros(MODEL.n_dof); pi[MODEL.free] = mi["phi_free"]
            pj = np.zeros(MODEL.n_dof); pj[MODEL.free] = mj["phi_free"]
            worst = max(worst, abs(float(pi @ (M @ pj))))
    assert worst < 1e-8


def test_rebuild_determinism():
    geo2 = build_geometry(_S)
    m2 = FEModel(geo2)
    m2.shape_find()
    modes2 = modal_analysis(m2, n_modes=36)
    assert np.allclose(MODEL.tension, m2.tension, rtol=0, atol=1e-6)
    for a, b in zip(MODES[:20], modes2[:20]):
        assert abs(a["f"] - b["f"]) < 1e-10
