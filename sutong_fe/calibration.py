"""
Bounded model updating of the 2-D Sutong model against the measured
vertical-bending frequencies.

Five physical scale factors are updated within documented bounds
(deck mass, ballast magnitude, cable steel area, pylon bending inertia,
girder bending inertia).  The objective is the relative frequency error of
the seven measured vertical modes; every objective evaluation re-runs the
full physics pipeline - geometry, assembly, dead-load shape finding with
Ernst cable modulus, and the linearised eigen-problem - so the updated
model stays statically and dynamically consistent by construction.
"""

import numpy as np
from scipy.optimize import least_squares

from . import params as P
from .geometry import build_geometry
from .model import FEModel
from .modal import modal_analysis, pairing_report


# ---------------------------------------------------------------------------
def evaluate(scales, ernst_iters=4):
    """Full pipeline for one parameter set; returns (model, modes)."""
    geo = build_geometry(scales)
    model = FEModel(geo)
    model.shape_find(iters=ernst_iters)
    modes = modal_analysis(model, n_modes=36)
    return model, modes


def paired_frequencies(modes):
    """FE frequencies paired to the measured targets by shape (symmetry and
    main-span half-wave count); missing pairs return None."""
    rows = pairing_report(modes)
    return [r["f_fe"] for r in rows]


def calibrate(verbose=True, ernst_iters=4):
    """Bounded least-squares update of the five scale factors."""
    f_meas = np.array([f for (_, f, _) in P.MEASURED_VERTICAL_MODES])
    names = P.CALIBRATION["names"]
    x0 = np.array(P.CALIBRATION["x0"], dtype=float)
    lo = np.array(P.CALIBRATION["lower"], dtype=float)
    hi = np.array(P.CALIBRATION["upper"], dtype=float)

    evals = [0]

    def residual(x):
        evals[0] += 1
        scales = dict(zip(names, x))
        try:
            _, modes = evaluate(scales, ernst_iters=ernst_iters)
            f_fe = paired_frequencies(modes)
        except Exception as exc:                     # keep the optimiser alive
            if verbose:
                print(f"  evaluation failed ({exc}); penalty residual")
            return np.full(len(f_meas), 10.0)
        r = np.zeros(len(f_meas))
        bad = 0
        for i, f in enumerate(f_fe):
            if f is None:
                r[i] = 10.0
                bad += 1
            else:
                r[i] = (f - f_meas[i]) / f_meas[i]
        if verbose:
            print(f"  eval {evals[0]:3d}: scales "
                  f"{dict((n, round(v, 3)) for n, v in zip(names, x))} "
                  f"-> max|err| {100 * np.max(np.abs(r)):.2f} %"
                  + (f"  [{bad} unpaired]" if bad else ""))
        return r

    # three starts guard against the weak local minima created by the
    # side-span / global mode interleaving of the 2-D model
    starts = [x0,
              x0 + np.array([0.05, 0.15, 0.20, 0.20]),
              x0 + np.array([-0.05, 0.05, 0.05, 0.10])]
    best = None
    for x_start in starts:
        r = least_squares(residual, x_start, bounds=(lo, hi),
                          diff_step=0.03, xtol=1e-4, ftol=1e-4, gtol=1e-4)
        if best is None or r.cost < best.cost:
            best = r
    return dict(zip(names, best.x)), best
