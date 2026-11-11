"""Scenario tests: the 37-case catalogue and the exactness of damage application.

The tests verify that the documented damage families hit exactly the elements
they describe - and nothing else - so the scenario descriptions that ship
with the dataset and the implemented damage are the same thing.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sutong_fe import params as P
from sutong_fe.geometry import build_geometry
from sutong_fe.scenarios import (SCENARIO_LIST, N_SCENARIOS, FAMILIES,
                                 apply_damage, severity_pct)

GEO = build_geometry()


def test_catalogue_composition():
    assert N_SCENARIOS == 37
    assert len(FAMILIES) == 12
    healthy = [s for s in SCENARIO_LIST if s["family"] is None]
    assert len(healthy) == 1 and healthy[0]["sid"] == 0
    codes = [s["code"] for s in SCENARIO_LIST]
    assert len(set(codes)) == 37
    for fam in FAMILIES:
        rows = [s for s in SCENARIO_LIST if s["family"] == fam]
        assert [r["severity_label"] for r in rows] == \
            ["light", "moderate", "severe"]
    # scenario ids are contiguous and ordered family by family
    assert [s["sid"] for s in SCENARIO_LIST] == list(range(37))


def test_severity_percentages():
    for fam in FAMILIES:
        light, moderate, severe = [
            severity_pct(fam, k) for k in (1, 2, 3)]
        if fam in ("F11", "F12"):
            assert (light, moderate, severe) == (3, 10, 25)
        else:
            assert (light, moderate, severe) == (5, 15, 35)


def test_stay_families_hit_the_documented_cables():
    n_lev = P.CABLES["n_levels"]

    def ids(family, **kw):
        sc = next(s for s in SCENARIO_LIST
                  if s["family"] == family and s["severity_label"] == "light")
        mods, _ = apply_damage(build_geometry(), sc)
        return [eid for (kind, eid, _, _) in mods if kind == "cable"]

    # F1: the single longest river stay of the low pylon (mid-span anchor)
    f1 = ids("F1")
    assert len(f1) == 1
    assert GEO.cables[f1[0]].pylon == 0
    assert GEO.cables[f1[0]].side == "river"
    assert GEO.cables[f1[0]].level == n_lev
    # F2: both level-34 river stays, one from each pylon, mid-span anchor
    f2 = ids("F2")
    assert len(f2) == 2
    assert {GEO.cables[i].pylon for i in f2} == {0, 1}
    assert all(GEO.cables[i].level == n_lev for i in f2)
    assert GEO.cables[f2[0]].n1 == GEO.cables[f2[1]].n1   # shared anchor node
    # F3: the eight outermost levels of each of the four fans
    f3 = ids("F3")
    assert len(f3) == 32
    assert all(GEO.cables[i].level > n_lev - 8 for i in f3)
    assert {GEO.cables[i].pylon for i in f3} == {0, 1}
    assert {GEO.cables[i].side for i in f3} == {"river", "shore"}


def test_girder_families_hit_the_documented_bands():
    def beam_ids(family):
        sc = next(s for s in SCENARIO_LIST
                  if s["family"] == family and s["severity_label"] == "light")
        mods, _ = apply_damage(build_geometry(), sc)
        return [eid for (kind, eid, _, _) in mods if kind == "beam"]

    def mid_x(eid):
        be = GEO.beams[eid]
        return 0.5 * (GEO.nodes[be.n1].x + GEO.nodes[be.n2].x)

    f4 = beam_ids("F4")
    assert all(abs(mid_x(e) - 1044.0) <= 50.0 + 1e-9 for e in f4)
    f5 = beam_ids("F5")
    qp = P.SCENARIOS["quarter_point_x"]
    assert all(abs(mid_x(e) - qp) <= 50.0 + 1e-9 for e in f5)
    f6 = beam_ids("F6")
    dp = P.SCENARIOS["deck_band_pylon"]
    assert all(min(abs(mid_x(e) - xp) for xp in P.PYLON_X) <= dp + 1e-9
               for e in f6)
    # every girder family damages deck elements, and only deck elements
    for fam in ("F4", "F5", "F6"):
        assert beam_ids(fam)
        assert all(GEO.beams[e].group == "deck" for e in beam_ids(fam))


def test_pylon_and_bearing_families():
    def mods_of(family):
        sc = next(s for s in SCENARIO_LIST
                  if s["family"] == family and s["severity_label"] == "severe")
        return apply_damage(build_geometry(), sc)[0]

    # F7: base segment of the low pylon only
    f7 = [eid for (k, eid, _, _) in mods_of("F7")]
    assert f7 and all(GEO.beams[e].group == "pylon" for e in f7)
    assert all(GEO.nodes[GEO.beams[e].n1].x == P.PYLON_X[0] for e in f7)
    for e in f7:
        ym = 0.5 * (GEO.nodes[GEO.beams[e].n1].y +
                    GEO.nodes[GEO.beams[e].n2].y)
        assert ym <= P.SCENARIOS["pylon_base_to_y"]
    # F8: both transition bearings; F9: the four auxiliary bearings
    f8 = [eid for (k, eid, _, _) in mods_of("F8")]
    f9 = [eid for (k, eid, _, _) in mods_of("F9")]
    assert len(f8) == 2 and len(f9) == 4
    assert all(GEO.beams[e].group == "bearing" for e in f8 + f9)
    # F10: upper columns of both pylons above the documented elevation
    f10 = [eid for (k, eid, _, _) in mods_of("F10")]
    assert f10 and all(GEO.beams[e].group == "pylon" for e in f10)
    assert {GEO.nodes[GEO.beams[e].n1].x for e in f10} == set(P.PYLON_X)
    for e in f10:
        ym = 0.5 * (GEO.nodes[GEO.beams[e].n1].y +
                    GEO.nodes[GEO.beams[e].n2].y)
        assert ym > P.SCENARIOS["pylon_top_from_y"]
    # F11 / F12 combine the documented elements of their parents
    f11 = mods_of("F11")
    cables = [eid for (k, eid, _, _) in f11 if k == "cable"]
    beams = [eid for (k, eid, _, _) in f11 if k == "beam"]
    assert len(cables) == 1 and beams
    f12 = mods_of("F12")
    pylon_ids = [eid for (k, eid, _, _) in f12 if k == "beam"]
    bearing_ids = [eid for (k, eid, _, _) in f12 if k == "bearing"]
    assert pylon_ids and len(bearing_ids) == 2


def test_damage_modifies_only_the_listed_properties():
    pristine = build_geometry()
    for sc in SCENARIO_LIST:
        if sc["family"] is None:
            continue
        geo = build_geometry()
        mods, _ = apply_damage(geo, sc)
        listed = {eid for (_, eid, _, _) in mods}
        for i, (c, cb) in enumerate(zip(geo.cables, pristine.cables)):
            if i in listed:
                continue
            assert c.area == cb.area and c.rho_line == cb.rho_line
        for i, (be, bb) in enumerate(zip(geo.beams, pristine.beams)):
            if i in listed:
                continue
            assert be.I == bb.I and be.A == bb.A
        # the applied factor is exactly 1 - severity / 100
        d = 1.0 - sc["severity_pct"] / 100.0
        for (kind, eid, prop, factor) in mods:
            assert abs(factor - d) < 1e-15
            if kind == "cable":
                assert abs(geo.cables[eid].area - pristine.cables[eid].area * d) \
                    < 1e-6
