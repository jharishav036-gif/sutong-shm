"""
Damage scenario definitions for the simulation stage.

The study uses 37 scenarios: one healthy state and twelve damage families
(F1..F12) at three severity levels each (light / moderate / severe), which
gives 1 + 12 x 3 = 37.

Every family is defined by the exact elements and properties it degrades, on
the model as it exists in the vertical plane:

* F1  single stay degradation        - the longest river stay of the low pylon
  (level 34, the stay anchored at mid-span);
* F2  symmetric stay pair            - the two level-34 river stays that share
  the mid-span anchor node;
* F3  fan-end degradation            - the eight outermost (highest-level)
  stays of each of the four fans, 32 stays in total (about 25 % of every
  fan);
* F4/F5/F6 girder bending loss       - deck elements in a 100 m band at
  mid-span, at the low-side quarter point, and in 60 m bands at the two
  pylon axes;
* F7  pylon base loss                - base segment of the low pylon;
* F8  deck-end bearing loss          - the two transition-pier bearings;
* F9  auxiliary bearing loss         - the four auxiliary-pier bearings;
* F10 pylon top loss                 - both upper columns above the anchor
  zone;
* F11/F12 combined                   - F1 + F4 and F7 + F8 at reduced levels.

The out-of-plane restraint states of the original family list cannot exist
in a vertical-plane model; the bearing and pylon-top families above degrade
the in-plane restraint and stiffness that those substructures actually
provide here.  The degradation is a property loss (strand area loss in a
stay, section loss / cracking in concrete and steel, bearing stiffness
loss): stiffness and capacity drop together, which is the physical meaning
of the property-loss percentages.

The application function returns the exact list of modifications it made,
so the driver can verify that the implemented damage and the documented
description are the same thing.
"""

import numpy as np

from . import params as P


# ---------------------------------------------------------------------------
# family catalogue
# ---------------------------------------------------------------------------
FAMILIES = {
    "F1": "single stay degradation (longest river stay, low pylon)",
    "F2": "symmetric stay pair degradation (both mid-span river stays)",
    "F3": "fan-end degradation (8 outermost stays per fan, 32 stays)",
    "F4": "girder bending loss at mid-span (+/- 50 m)",
    "F5": "girder bending loss at the low-side quarter point (+/- 50 m)",
    "F6": "girder bending loss at both pylon anchors (+/- 30 m)",
    "F7": "low pylon base bending loss (base segment)",
    "F8": "deck-end bearing stiffness loss (both transition piers)",
    "F9": "auxiliary bearing stiffness loss (4 interior bearings)",
    "F10": "pylon upper-column bending loss (both pylons)",
    "F11": "combined stay + girder loss at mid-span (F1 + F4)",
    "F12": "combined pylon base + deck-end bearing loss (F7 + F8)",
}

# Four severity classes of the learning task: healthy = 0, light = 1,
# moderate = 2, severe = 3 (the operational classes of the study).
SEVERITIES = {"light": 1, "moderate": 2, "severe": 3}


def severity_pct(family, severity_class):
    """Property-loss percentage of a family at a severity class (1..3)."""
    key = "combined" if family in ("F11", "F12") else "standard"
    return P.SCENARIOS[key][severity_class - 1]


def build_scenario_list():
    """The 37 scenarios in a fixed order: healthy first, then F1..F12 each at
    light, moderate, severe level."""
    out = [dict(sid=0, code="H", family=None, family_name="healthy state",
                severity_class=0, severity_pct=0, severity_label="healthy")]
    sid = 1
    for fam in sorted(FAMILIES, key=lambda f: int(f[1:])):
        for label in ("light", "moderate", "severe"):
            out.append(dict(
                sid=sid, code=f"{fam}_{label}", family=fam,
                family_name=FAMILIES[fam], severity_class=SEVERITIES[label],
                severity_pct=severity_pct(fam, SEVERITIES[label]),
                severity_label=label))
            sid += 1
    return out


SCENARIO_LIST = build_scenario_list()
N_SCENARIOS = len(SCENARIO_LIST)


# ---------------------------------------------------------------------------
# damage application
# ---------------------------------------------------------------------------
def _deck_region_ids(geo, x_lo, x_hi):
    """Deck element ids whose midpoints lie inside [x_lo, x_hi]."""
    out = []
    for be in geo.beams:
        if be.group != "deck":
            continue
        xm = 0.5 * (geo.nodes[be.n1].x + geo.nodes[be.n2].x)
        if x_lo - 1e-9 <= xm <= x_hi + 1e-9:
            out.append(be.eid)
    return out


def _pylon_region_ids(geo, pylon=None, y_lo=None, y_hi=None):
    """Pylon element ids of one or both pylons within an elevation band."""
    out = []
    for be in geo.beams:
        if be.group != "pylon":
            continue
        if pylon is not None and geo.nodes[be.n1].x != P.PYLON_X[pylon]:
            continue
        ym = 0.5 * (geo.nodes[be.n1].y + geo.nodes[be.n2].y)
        if y_lo is not None and ym <= y_lo - 1e-9:
            continue
        if y_hi is not None and ym >= y_hi + 1e-9:
            continue
        out.append(be.eid)
    return out


def _bearing_ids(geo, roles):
    out = []
    for be in geo.beams:
        if be.group == "bearing":
            pr = next(p for p in P.PIERS if p["label"] == be.label.split("-")[-1])
            if pr["role"] in roles:
                out.append(be.eid)
    return out


def _stay_ids(geo, pylon=None, side=None, levels=None):
    out = []
    for c in geo.cables:
        if pylon is not None and c.pylon != pylon:
            continue
        if side is not None and c.side != side:
            continue
        if levels is not None and c.level not in levels:
            continue
        out.append(c.eid)
    return out


def _family_targets(geo, family):
    """Elements touched by a family, as a dict of property -> element ids.

    Returns (cable_ids, beam_ids, bearing_ids, description).
    """
    n_lev = P.CABLES["n_levels"]
    outer = set(range(n_lev - P.SCENARIOS["fan_loss_levels"] + 1, n_lev + 1))
    xm, dq = P.SCENARIOS["deck_band_main"], P.SCENARIOS["deck_band_quarter"]
    dp, qp = P.SCENARIOS["deck_band_pylon"], P.SCENARIOS["quarter_point_x"]

    if family == "F1":
        ids = _stay_ids(geo, pylon=0, side="river", levels={n_lev})
        return ids, [], [], "stay L34 river, low pylon (mid-span anchor)"
    if family == "F2":
        ids = _stay_ids(geo, side="river", levels={n_lev})
        return ids, [], [], "both level-34 river stays (shared mid-span anchor)"
    if family == "F3":
        ids = _stay_ids(geo, levels=outer)
        lo_lev = n_lev - P.SCENARIOS["fan_loss_levels"] + 1
        desc = (f"{len(ids)} outermost stays (levels "
                f"{lo_lev}..{n_lev})")
        return ids, [], [], desc
    if family == "F4":
        ids = _deck_region_ids(geo, 1044.0 - xm, 1044.0 + xm)
        return [], ids, [], f"girder elements at mid-span +/- {xm:.0f} m"
    if family == "F5":
        ids = _deck_region_ids(geo, qp - dq, qp + dq)
        return [], ids, [], f"girder elements at x = {qp:.0f} +/- {dq:.0f} m"
    if family == "F6":
        ids = _deck_region_ids(geo, P.PYLON_X[0] - dp, P.PYLON_X[0] + dp) + \
              _deck_region_ids(geo, P.PYLON_X[1] - dp, P.PYLON_X[1] + dp)
        return [], ids, [], f"girder elements at both pylon axes +/- {dp:.0f} m"
    if family == "F7":
        ids = _pylon_region_ids(geo, pylon=0, y_hi=P.SCENARIOS["pylon_base_to_y"])
        return [], ids, [], "low pylon base segment (0-12 m)"
    if family == "F8":
        ids = _bearing_ids(geo, {"transition"})
        return [], [], ids, "both transition-pier bearings"
    if family == "F9":
        ids = _bearing_ids(geo, {"auxiliary"})
        return [], [], ids, "four auxiliary-pier bearings"
    if family == "F10":
        ids = _pylon_region_ids(geo, y_lo=P.SCENARIOS["pylon_top_from_y"])
        desc = (f"both upper columns above y = "
                f"{P.SCENARIOS['pylon_top_from_y']:.1f} m")
        return [], ids, [], desc
    if family == "F11":
        cids = _stay_ids(geo, pylon=0, side="river", levels={n_lev})
        bids = _deck_region_ids(geo, 1044.0 - xm, 1044.0 + xm)
        return cids, bids, [], "stay L34 river + girder at mid-span"
    if family == "F12":
        bids = _pylon_region_ids(geo, pylon=0,
                                 y_hi=P.SCENARIOS["pylon_base_to_y"])
        gids = _bearing_ids(geo, {"transition"})
        return [], bids, gids, "low pylon base + both deck-end bearings"
    if family is None:
        return [], [], [], "no damage (healthy)"
    raise ValueError(f"unknown family {family}")


def apply_damage(geo, scenario):
    """Degrade the geometry of one scenario in place.

    Returns the modification record: a list of
    ``(kind, eid, property, factor)`` tuples, kind being 'cable' or 'beam'.
    Cable degradation scales the merged steel area (stiffness, capacity and
    self-weight together, the model of strand loss); girder / pylon
    degradation scales the bending inertia; bearing degradation scales the
    axial stiffness of the link.
    """
    fam = scenario["family"]
    d = 1.0 - scenario["severity_pct"] / 100.0
    cable_ids, beam_ids, bearing_ids, desc = _family_targets(geo, fam)

    mods = []
    for eid in cable_ids:
        c = geo.cables[eid]
        c.area *= d
        c.rho_line *= d
        mods.append(("cable", eid, "area", d))
    for eid in beam_ids:
        geo.beams[eid].I *= d
        mods.append(("beam", eid, "I", d))
    for eid in bearing_ids:
        geo.beams[eid].A *= d
        mods.append(("bearing", eid, "A", d))
    return mods, desc


def damage_summary(scenario):
    """Human-readable one-line description of a scenario's damage."""
    if scenario["family"] is None:
        return "healthy structure, as-built dead-load state"
    return (f"{scenario['family']} {scenario['severity_label']} "
            f"({scenario['severity_pct']} % property loss)")
