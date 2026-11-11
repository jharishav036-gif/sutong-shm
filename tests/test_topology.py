"""Topology tests: geometry, cable layout and the non-crossing property.

These tests are deliberately independent of the solver: they verify the
model *topology* (the cable-stayed bridge layout itself) from published
dimensions only.
"""

import itertools
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sutong_fe import params as P
from sutong_fe.geometry import build_geometry
from sutong_fe.elements import segment_intersection

GEO = build_geometry()


def test_span_layout():
    assert sum(P.SPAN_LAYOUT) == 2088.0
    assert P.PYLON_X == [500.0, 1588.0]
    assert P.PIERS[0]["x"] == 0.0 and P.PIERS[-1]["x"] == 2088.0


def test_cable_count_and_levels():
    assert len(GEO.cables) == 136
    for p in (0, 1):
        for side in ("river", "shore"):
            n = sum(1 for c in GEO.cables if c.pylon == p and c.side == side)
            assert n == 34
    levels = {c.level for c in GEO.cables}
    assert levels == set(range(1, 35))


def test_anchor_spacings():
    for p in (0, 1):
        riv = sorted(abs(c.x_deck - P.PYLON_X[p]) for c in GEO.cables
                     if c.pylon == p and c.side == "river")
        assert np.allclose(np.diff(riv), 16.0)
        assert np.isclose(riv[-1], 544.0)            # mid-span anchor
        sho = sorted(abs(c.x_deck - P.PYLON_X[p]) for c in GEO.cables
                     if c.pylon == p and c.side == "shore")
        gaps = np.diff(sho)
        assert np.all(np.isclose(gaps, 16.0) | np.isclose(gaps, 12.0))
        assert np.isclose(sho[-1], 480.0)            # farthest shore anchor


def test_anchorage_zone():
    lo = min(c.y_anchor for c in GEO.cables)
    hi = max(c.y_anchor for c in GEO.cables)
    assert lo > P.PYLON["junction_y"]                # above the leg junction
    assert hi < 300.4                                # below the pylon top
    dy = (hi - lo) / 33.0
    assert 2.0 <= dy <= 3.0                          # documented 2.25-4.05 m


def test_monotone_anchor_heights():
    """Anchorage height increases with deck distance: the structural reason
    the fans cannot cross."""
    for p in (0, 1):
        for side in ("river", "shore"):
            cs = sorted((c for c in GEO.cables
                         if c.pylon == p and c.side == side),
                        key=lambda c: abs(c.x_deck - P.PYLON_X[p]))
            hs = [c.y_anchor for c in cs]
            assert all(b > a for a, b in zip(hs, hs[1:]))


def test_no_cable_crossings():
    segs = GEO.cable_segments()
    for (c1, a1, b1), (c2, a2, b2) in itertools.combinations(segs, 2):
        assert not segment_intersection(a1, b1, a2, b2), \
            f"cables {c1.eid} and {c2.eid} cross"


def test_cable_lengths_vs_as_built():
    Ls = sorted(c.length for c in GEO.cables)
    assert abs(Ls[0] - P.CABLES["doc_shortest_m"]) / \
        P.CABLES["doc_shortest_m"] < 0.02
    assert abs(Ls[-1] - P.CABLES["doc_longest_m"]) / \
        P.CABLES["doc_longest_m"] < 0.04


def test_cable_spec_ladder():
    wires = sorted({c.n_wires for c in GEO.cables})
    assert wires == [139, 151, 187, 199, 223, 241, 283, 313]
    # specification grows with anchor level (force demand)
    for p in (0, 1):
        for side in ("river", "shore"):
            ws = [c.n_wires for c in sorted(
                (c for c in GEO.cables if c.pylon == p and c.side == side),
                key=lambda c: c.level)]
            assert all(b >= a for a, b in zip(ws, ws[1:]))


def test_mirror_symmetry():
    mirror = {round(2088.0 - c.x_deck, 6): c for c in GEO.cables}
    for c in GEO.cables:
        m = mirror[round(c.x_deck, 6)]
        assert m.y_anchor == c.y_anchor
        assert m.side == c.side
    steel = sum(c.area * c.length for c in GEO.cables) * 7850.0
    assert 5.5e6 < steel < 7.0e6                      # kg, documented supply
