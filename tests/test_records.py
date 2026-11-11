"""Simulation tests: sensor layout, ambient excitation and record generation.

A reduced modal basis keeps the suite fast; the full-fidelity record checks
(number of scenarios, independence statistics) live in the stage-2 driver,
which runs the complete pipeline and its 19 verification checks.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sutong_fe import params as P
from sutong_fe.geometry import build_geometry
from sutong_fe.model import FEModel
from sutong_fe.modal import modal_analysis
from sutong_fe.sensors import build_channels
from sutong_fe.excitation import WindField, TrafficStream, AmbientExcitation
from sutong_fe.simulate import simulate_record, add_sensor_noise

# one calibrated-geometry model, shared by the module (built once)
GEO = build_geometry()
MODEL = FEModel(GEO)
MODEL.shape_find()
MODES = [m for m in modal_analysis(MODEL, n_modes=40)
         if m["f"] <= P.EXCITATION["f_modes_max"]]


# ---------------------------------------------------------------------------
def test_channel_composition():
    channels, R = build_channels(MODEL)
    assert len(channels) == P.SENSORS["n_channels"] == 64
    kinds = [c["kind"] for c in channels]
    assert kinds.count("deck-vertical") == 43
    assert kinds.count("deck-longitudinal") == 5
    assert kinds.count("pylon") == 8
    assert kinds.count("cable-axial") == 8
    assert len({c["tag"] for c in channels}) == 64
    assert len({c["ch"] for c in channels}) == 64
    assert R.shape == (64, MODEL.n_free)


def test_channel_geometry_consistency():
    channels, R = build_channels(MODEL)
    dm = MODEL.dof_map
    for c in channels:
        node = GEO.nodes[c["node"]]
        assert abs(node.x - c["x"]) < 1e-9
        assert abs(node.y - c["y"]) < 1e-9
        if c["kind"] == "deck-vertical":
            # exactly one unit entry, on the node's vertical DOF
            row = R.getrow(c["ch"])
            assert row.nnz == 1 and row.indices[0] == dm[3 * c["node"] + 1]
        elif c["kind"] == "deck-longitudinal":
            row = R.getrow(c["ch"])
            assert row.nnz == 1 and row.indices[0] == dm[3 * c["node"]]
        elif c["kind"] == "pylon":
            row = R.getrow(c["ch"])
            assert row.nnz == 1
    # the 43 deck-vertical channels are symmetric about mid-span
    xs = sorted(c["x"] for c in channels if c["kind"] == "deck-vertical")
    assert xs[0] == 36.0 and xs[-1] == 2052.0
    assert all(abs((xa + xb) / 2.0 - 1044.0) < 1e-9
               for xa, xb in zip(xs, xs[::-1]))


def test_cable_axial_projection():
    channels, R = build_channels(MODEL)
    for c in channels:
        if c["kind"] != "cable-axial":
            continue
        row = R.getrow(c["ch"])
        # projection of the deck-anchor motion on the cable axis
        assert row.nnz == 2
        dof_u = MODEL.dof_map[3 * c["node"]]
        dof_w = MODEL.dof_map[3 * c["node"] + 1]
        w = dict(zip(row.indices, row.data))
        # two stays can share the anchor node (mid-span); the weights must
        # match the axis of one of them exactly
        cands = [k for k in GEO.cables if k.n1 == c["node"]]
        assert any(abs(w[dof_u] - np.cos(k.theta)) < 1e-9 and
                   abs(w[dof_w] - np.sin(k.theta)) < 1e-9
                   for k in cands)


# ---------------------------------------------------------------------------
def test_wind_field_statistics():
    rng = np.random.default_rng(11)
    wf = WindField(rng, 600.0, 200.0)
    # a single realisation fluctuates around the target variance with the
    # long correlation time of the low-frequency band, so the spatial
    # average is compared with a 10 % tolerance
    for comp, I in ((wf.u, P.EXCITATION["I_u"]),
                     (wf.w, P.EXCITATION["I_w"])):
        target = I * P.EXCITATION["wind_U"]
        sd = float(np.mean(np.std(comp, axis=1)))
        assert abs(sd - target) / target < 0.10, (sd, target)
        # band limit: essentially no energy beyond wind_f_max
        n = comp.shape[1]
        spec = np.abs(np.fft.rfft(comp[0])) ** 2
        f = np.fft.rfftfreq(n, 1.0 / 200.0)
        beyond = float(np.sum(spec[f > P.EXCITATION["wind_f_max"] * 1.2]))
        assert beyond / np.sum(spec) < 1e-4


def test_wind_draws_are_seed_dependent():
    a = WindField(np.random.default_rng(5), 120.0, 200.0)
    b = WindField(np.random.default_rng(5), 120.0, 200.0)
    c = WindField(np.random.default_rng(6), 120.0, 200.0)
    assert np.array_equal(a.u, b.u) and np.array_equal(a.w, b.w)
    assert not np.array_equal(a.u, c.u)


def test_traffic_stream():
    a = TrafficStream(np.random.default_rng(7), 600.0)
    assert len(a.axles) > 10
    # entries start inside the pre-history window (vehicles are already on
    # the bridge when a record begins) and stay inside the time horizon
    v_min = P.EXCITATION["car_speed"][0] - 3.0 * P.EXCITATION["car_speed"][1]
    pre = P.TOTAL_LENGTH / max(v_min, 5.0) + 20.0
    assert all(-pre - 1.0 <= ax["t_entry"] <= 600.0 for ax in a.axles)
    assert all(12.0 <= ax["v"] for ax in a.axles)
    assert any(ax["force"] > 2.0e5 for ax in a.axles)   # trucks present
    b = TrafficStream(np.random.default_rng(8), 600.0)
    assert [ax["t_entry"] for ax in a.axles] != [ax["t_entry"]
                                                 for ax in b.axles]


def test_forces_shape_and_finiteness():
    rng = np.random.default_rng(9)
    exc = AmbientExcitation(MODEL, rng, 130.0, 200.0)
    t = 60.0 + np.arange(2000) / 200.0
    F = exc.forces(t)
    assert F.shape == (exc.n_rows(), len(t))
    assert np.isfinite(F).all()
    # traffic is downward: the deck-vertical force block has negative parts
    nd = len(exc.deck_w_rows)
    assert F[:nd].min() < 0.0


# ---------------------------------------------------------------------------
def test_record_determinism_and_shape():
    rng_a = np.random.default_rng((2024, 30, 1))
    _, acc_a = simulate_record(MODEL, MODES, rng_a, T_burn=30.0,
                                fs_burn=50.0, T_rec=6.0, fs_rec=200.0)
    rng_b = np.random.default_rng((2024, 30, 1))
    _, acc_b = simulate_record(MODEL, MODES, rng_b, T_burn=30.0,
                               fs_burn=50.0, T_rec=6.0, fs_rec=200.0)
    assert acc_a.shape == (64, 1200)
    assert np.array_equal(acc_a, acc_b)          # bitwise deterministic
    assert np.isfinite(acc_a).all()
    # independent seeds give independent records
    rng_c = np.random.default_rng((2024, 30, 2))
    _, acc_c = simulate_record(MODEL, MODES, rng_c, T_burn=30.0,
                               fs_burn=50.0, T_rec=6.0, fs_rec=200.0)
    assert not np.array_equal(acc_a, acc_c)
    # accelerations of an ambient record stay in a physical band
    rms = float(np.sqrt(np.mean(acc_a ** 2)))
    assert 1.0e-4 < rms < 5.0e-2


def test_sensor_noise():
    acc = np.zeros((64, 1000))
    rng = np.random.default_rng(3)
    noisy = add_sensor_noise(acc, rng)
    assert abs(float(np.std(noisy)) - P.EXCITATION["noise_std"]) / \
        P.EXCITATION["noise_std"] < 0.05
