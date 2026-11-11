"""Static (dead-load) tests of the shape-found bridge."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sutong_fe import params as P
from sutong_fe.geometry import build_geometry
from sutong_fe.model import FEModel

_GEO = build_geometry()
MODEL = FEModel(_GEO)
MODEL.shape_find()


def test_global_equilibrium():
    W = MODEL.total_weight()
    rs = MODEL.reaction_summary()
    assert abs(rs["pylons"] + rs["piers"] - W) / W < 1e-9
    rx = float(np.sum(MODEL.reactions[0::3]))
    assert abs(rx) < 1e-3 * W


def test_prestress_produces_tension():
    assert MODEL.tension.min() > 0.0
    sr = MODEL.stress_ratios()
    assert 0.10 <= sr.min() and sr.max() <= 0.45
    # no stay exceeds the documented maximum stay force (+ allowance);
    # element tension is merged over the two cable planes
    cap = P.SHAPE_FINDING["force_cap_margin"] * \
        P.SHAPE_FINDING["force_cap_plane"]
    assert MODEL.tension.max() <= 2.0 * cap + 1e4      # N


def test_deck_is_shape_found():
    xs_d, w_d = MODEL.deck_deflection_profile()
    main = (xs_d > P.PYLON_X[0]) & (xs_d < P.PYLON_X[1])
    assert np.max(np.abs(w_d[main])) < 0.20           # m, main span
    assert np.max(np.abs(w_d[~main])) < 0.40          # m, side spans (camber)
    assert MODEL.pylon_top_drift() < 0.05              # m


def test_force_flow():
    # the girder works as a compression chord: max |N| at the pylons,
    # zero at mid-span and at the abutments
    deck_N = np.array([MODEL.beam_axial[be.eid] for be in _GEO.beams
                       if be.group == "deck"])
    xs = np.array([0.5 * (_GEO.nodes[be.n1].x + _GEO.nodes[be.n2].x)
                   for be in _GEO.beams if be.group == "deck"])
    assert deck_N.min() < -150e6                       # MN-scale compression
    near_mid = np.abs(xs - 1044.0) < 8.0
    near_end = (xs < 12.0) | (xs > 2076.0)
    assert np.abs(deck_N[near_mid]).max() < 15e6   # paired anchors
    assert np.abs(deck_N[near_end]).max() < 5e6


def test_pylon_compression():
    pyl_N = [MODEL.beam_axial[be.eid] for be in _GEO.beams
             if be.group == "pylon"]
    # base compression ~= deck carried by cables + pylon self weight
    assert -1300e6 < min(pyl_N) < -600e6


def test_live_load_serviceability():
    ll = MODEL.live_load_check()
    assert ll["stress_ratios"].max() <= 0.45
    assert ll["min_tension"] > 0
    assert ll["max_deflection"] < 1.2


def test_bearing_reactions():
    """Auxiliary and transition piers carry a meaningful share of the deck."""
    rs = MODEL.reaction_summary()
    assert 100e6 < rs["piers"] < 600e6
    assert 1500e6 < rs["pylons"] < 2100e6
