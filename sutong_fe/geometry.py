"""
Geometry and connectivity generation for the Sutong Bridge 2-D model.

The module builds the complete node / element / cable layout from the
published bridge arrangement.  The stay-cable connectivity is generated
programmatically from the fan geometry - anchorage heights on the pylon are
strictly monotone in the deck-anchor distance - which makes crossing cables
geometrically impossible; the automated topology checks in ``tests`` verify
this property rather than assume it.
"""

from dataclasses import dataclass, field

import numpy as np

from . import params as P


@dataclass
class Node:
    nid: int
    x: float
    y: float
    group: str          # 'deck' | 'pylon' | 'pier'
    label: str = ""


@dataclass
class BeamElement:
    eid: int
    n1: int
    n2: int
    E: float
    A: float
    I: float
    rho_line: float     # kg/m, mass per unit length
    group: str          # 'deck' | 'pylon' | 'pier' | 'bearing'
    label: str = ""


@dataclass
class CableElement:
    eid: int
    n1: int             # deck anchor node
    n2: int             # pylon anchor node
    pylon: int          # 0 or 1
    side: str           # 'river' (towards mid-span) | 'shore'
    level: int          # 1 (nearest pylon) .. 34 (farthest)
    x_deck: float
    y_anchor: float
    length: float       # chord length (m)
    theta: float        # angle from deck node to pylon node (rad)
    n_wires: int
    area: float         # m^2, merged over both cable planes
    rho_line: float     # kg/m, merged over both planes


@dataclass
class Geometry:
    nodes: list = field(default_factory=list)
    beams: list = field(default_factory=list)
    cables: list = field(default_factory=list)
    deck_node_x: dict = field(default_factory=dict)     # x -> nid
    pylon_anchor_nodes: dict = field(default_factory=dict)  # (pylon, level) -> nid
    pier_nodes: dict = field(default_factory=dict)      # x -> (base, top)
    info: dict = field(default_factory=dict)

    # ------------------------------------------------------------------
    def node_by_x(self, x):
        return self.nodes[self.deck_node_x[x]]

    # ------------------------------------------------------------------
    def cable_segments(self):
        """Deck and pylon anchor points of every cable, for plotting and
        for the intersection tests."""
        segs = []
        for c in self.cables:
            p_deck = np.array([c.x_deck, P.DECK_Y])
            p_tower = np.array([P.PYLON_X[c.pylon], c.y_anchor])
            segs.append((c, p_deck, p_tower))
        return segs


# ---------------------------------------------------------------------------
def _pylon_ys():
    """Node elevations along one pylon: segment boundaries, all cable-anchor
    levels and an ~12 m infill in the lower and middle segments."""
    ys = {0.0, P.PYLON["junction_y"], 300.4}
    n_lev = P.CABLES["n_levels"]
    y0, y1 = P.CABLES["anchor_y0"], P.CABLES["anchor_y1"]
    dy = (y1 - y0) / (n_lev - 1)
    levels = [y0 + dy * k for k in range(n_lev)]
    ys.update(levels)
    y = 0.0
    while y < P.PYLON["junction_y"] - 1e-9:
        y += 12.0
        ys.add(min(y, P.PYLON["junction_y"]))
    return sorted(ys), levels


def _pylon_props(y):
    """Area, in-plane bending inertia and mass per metre at elevation y
    (per-leg values doubled below the leg junction)."""
    for (ya, yb, A0, A1, I0, I1, nlegs) in P.PYLON["segments"]:
        if ya - 1e-9 <= y <= yb + 1e-9:
            t = 0.0 if yb == ya else (y - ya) / (yb - ya)
            A = A0 + (A1 - A0) * t
            I = I0 + (I1 - I0) * t
            return nlegs * A, nlegs * I, nlegs * A * P.PYLON["rho"]
    raise ValueError(f"elevation {y} outside the pylon")


def _wires_for_level(level):
    for rng, wires in P.CABLES["level_wires"]:
        if level in rng:
            return wires
    raise ValueError(f"no PES7 specification mapped for level {level}")


def _deck_line_mass(x, ballast_scale=1.0):
    """Line mass of the girder at position x (kg/m).  Standard segments in
    the main span and the 300 m side spans; thickened tail segments over the
    auxiliary spans, which also carry the design ballast (the documented
    counterweight placement, taken down by the pier bearings)."""
    tail = (x < P.GIRDER["tail_length"] - 1e-9) or \
           (x > P.TOTAL_LENGTH - P.GIRDER["tail_length"] + 1e-9)
    if tail:
        return P.GIRDER["m_tail"] + ballast_scale * \
            P.GIRDER["ballast_total"] / (2.0 * P.GIRDER["tail_length"])
    return P.GIRDER["m_main"]


def build_geometry(scales=None):
    """Assemble the full 2-D model geometry.

    ``scales`` is a dict of calibration factors (default all 1.0):
    deck_mass, ballast, cable_area, pylon_I, deck_I.
    """
    s = dict(deck_mass=1.0, ballast=1.0, cable_area=1.0, pylon_I=1.0,
             deck_I=1.0)
    if scales:
        s.update(scales)

    geo = Geometry()
    nodes = geo.nodes

    def add_node(x, y, group, label=""):
        n = Node(len(nodes), x, y, group, label)
        nodes.append(n)
        return n.nid

    # --- deck ------------------------------------------------------------
    nx = int(round(P.TOTAL_LENGTH / P.DECK_MESH))
    for i in range(nx + 1):
        x = i * P.DECK_MESH
        nid = add_node(x, P.DECK_Y, "deck", f"D{i:03d}")
        geo.deck_node_x[round(x, 6)] = nid

    tail_region = lambda x: (x < P.GIRDER["tail_length"]) or \
        (x > P.TOTAL_LENGTH - P.GIRDER["tail_length"])
    for i in range(nx):
        xa = i * P.DECK_MESH
        xb = (i + 1) * P.DECK_MESH
        xm = 0.5 * (xa + xb)
        A = P.GIRDER["A_side"] if tail_region(xm) else P.GIRDER["A_main"]
        I = P.GIRDER["I_side"] if tail_region(xm) else P.GIRDER["I_main"]
        m = _deck_line_mass(xm, s["ballast"])
        geo.beams.append(BeamElement(
            len(geo.beams), geo.deck_node_x[round(xa, 6)],
            geo.deck_node_x[round(xb, 6)],
            P.GIRDER["E"], A, I * s["deck_I"], s["deck_mass"] * m,
            "deck", f"deck-{i:03d}"))

    # --- pylons ----------------------------------------------------------
    pylon_ys, anchor_levels = _pylon_ys()
    for p, xp in enumerate(P.PYLON_X):
        prev = add_node(xp, pylon_ys[0], "pylon", f"P{p}-base")
        level_map = {round(y, 6): k + 1 for k, y in enumerate(anchor_levels)}
        for y in pylon_ys[1:]:
            nid = add_node(xp, y, "pylon", f"P{p}-y{y:.1f}")
            # element properties are filled in the loop below (section averages)
            geo.beams.append(BeamElement(
                len(geo.beams), prev, nid, P.PYLON["E"], 0.0, 0.0, 0.0,
                "pylon", f"pylon{p}"))
            lev = level_map.get(round(y, 6))
            if lev is not None:
                geo.pylon_anchor_nodes[(p, lev)] = nid
            prev = nid

    # fix pylon element properties (average over each element)
    for el in geo.beams:
        if el.group != "pylon":
            continue
        ya = nodes[el.n1].y
        yb = nodes[el.n2].y
        ym = 0.5 * (ya + yb)
        A, I, mline = _pylon_props(ym)
        el.A = A
        el.I = I * s["pylon_I"]
        el.rho_line = mline

    # --- piers and bearings ----------------------------------------------
    for pr in P.PIERS:
        aux = pr["role"] == "auxiliary"
        A = P.PIER_PROPS["A_aux"] if aux else P.PIER_PROPS["A_tra"]
        I = P.PIER_PROPS["I_aux"] if aux else P.PIER_PROPS["I_tra"]
        base = add_node(pr["x"], 0.0, "pier", f"{pr['label']}-base")
        top = add_node(pr["x"], P.PIER_TOP_Y, "pier", f"{pr['label']}-top")
        geo.beams.append(BeamElement(
            len(geo.beams), base, top, P.PIER_PROPS["E"], A, I,
            A * P.PIER_PROPS["rho"], "pier", pr["label"]))
        geo.pier_nodes[round(pr["x"], 6)] = (base, top)
        # vertical-only bearing link to the deck node at the same x
        geo.beams.append(BeamElement(
            len(geo.beams), top, geo.deck_node_x[round(pr["x"], 6)],
            P.PIER_PROPS["E"], P.BEARING["EA"] / P.PIER_PROPS["E"],
            0.0, 0.0, "bearing", f"brg-{pr['label']}"))

    # --- stay cables ------------------------------------------------------
    n_lev = P.CABLES["n_levels"]
    y0, y1 = P.CABLES["anchor_y0"], P.CABLES["anchor_y1"]
    dy = (y1 - y0) / (n_lev - 1)
    for p, xp in enumerate(P.PYLON_X):
        direction = 1.0 if p == 0 else -1.0       # towards mid-span
        for lev in range(1, n_lev + 1):
            y_a = y0 + dy * (lev - 1)
            tower_nid = geo.pylon_anchor_nodes[(p, lev)]
            # deck anchor offset from the pylon: 16 m at every level on the
            # river (main-span) side; on the shore side 16 m up to level 18
            # and 12 m through the auxiliary spans thereafter
            dx_river = 16.0 * lev
            dx_shore = 16.0 * lev if lev <= 18 else 288.0 + 12.0 * (lev - 18)
            for side, sgn, dx in (("river", direction, dx_river),
                                  ("shore", -direction, dx_shore)):
                x_d = round(xp + sgn * dx, 6)
                deck_nid = geo.deck_node_x[x_d]
                theta = float(np.arctan2(y_a - P.DECK_Y, xp - x_d))
                L = float(np.hypot(xp - x_d, y_a - P.DECK_Y))
                wires = _wires_for_level(lev)
                area = 2.0 * wires * P.CABLES["wire_area"] * s["cable_area"]
                geo.cables.append(CableElement(
                    len(geo.cables), deck_nid, tower_nid, p, side, lev,
                    x_d, y_a, L, theta, wires, area,
                    area * P.CABLES["rho"]))

    geo.info = dict(
        n_nodes=len(nodes),
        n_beams=len(geo.beams),
        n_cables=len(geo.cables),
        anchor_dy=dy,
        anchor_levels=anchor_levels,
    )
    return geo
