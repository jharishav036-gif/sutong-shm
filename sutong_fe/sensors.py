"""
Sensor layout of the monitoring-system subset used by the simulation stage.

64 channels in the vertical plane of the model:

* 43 deck-vertical accelerometers, symmetric about mid-span at 48 m spacing
  (36 m .. 2052 m, mid-span included, expansion joints excluded);
* 5 deck-longitudinal accelerometers at the two expansion joints, the two
  side-span midpoints and mid-span - the channels that see the
  longitudinally floating motion of the girder;
* 8 pylon bidirectional accelerometers at both pylons, top and lowest
  anchor level, horizontal and vertical;
* 8 stay-cable axial channels: the deck-anchor acceleration projected on the
  axis of the eight fan-end stays (levels 34 and 33 of each pylon and side),
  the channels that track the tension rate of the critical stays.

The module returns the channel table together with the measurement operator
(a 64 x n_free sparse matrix of free-DOF rows), which is applied to the
modal accelerations to produce the recorded signals.
"""

import numpy as np
import scipy.sparse as sp

from . import params as P


def build_channels(model):
    """Channel table and measurement operator of the 64-channel subset.

    Returns ``(channels, R)`` where ``channels`` is a list of dicts (ch, tag,
    kind, node, x, y, direction) and ``R`` is a csr matrix with one row per
    channel mapping free-DOF accelerations to the channel signal.
    """
    geo = model.geo
    dm = model.dof_map
    channels = []
    rows, cols, vals = [], [], []

    def add(tag, kind, nid, x, y, direction, dof_terms):
        ch = len(channels)
        channels.append(dict(ch=ch, tag=tag, kind=kind, node=nid, x=x, y=y,
                              direction=direction))
        for dof, w in dof_terms:
            fd = dm[dof]
            if fd >= 0:
                rows.append(ch); cols.append(fd); vals.append(w)

    def node(x):
        return geo.deck_node_x[round(x, 6)]

    # --- 43 deck-vertical ----------------------------------------------
    step = P.SENSORS["deck_vertical_step"]
    xs = sorted({round(1044.0 + s * step * j, 6)
                 for s in (1.0, -1.0) for j in range(0, 22)})
    xs = [x for x in xs if 0.0 <= x <= P.TOTAL_LENGTH]
    for k, x in enumerate(xs):
        nid = node(x)
        add(f"D{k + 1:02d}", "deck-vertical", nid, x, P.DECK_Y,
            "vertical", [(3 * nid + 1, 1.0)])

    # --- 5 deck-longitudinal --------------------------------------------
    for k, x in enumerate(P.SENSORS["deck_longitudinal_x"]):
        nid = node(x)
        add(f"A{k + 1:02d}", "deck-longitudinal", nid, x, P.DECK_Y,
            "longitudinal", [(3 * nid, 1.0)])

    # --- 8 pylon bidirectional ------------------------------------------
    for p, xp in enumerate(P.PYLON_X):
        for y in P.SENSORS["pylon_heights"]:
            nid = _pylon_node(geo, p, y)
            for dof_off, name in ((0, "horizontal"), (1, "vertical")):
                add(f"P{p + 1}{name[0].upper()}{int(y)}", "pylon", nid,
                    xp, y, name, [(3 * nid + dof_off, 1.0)])

    # --- 8 stay-cable axial ----------------------------------------------
    cable_map = {}
    for c in geo.cables:
        cable_map[(c.pylon, c.side, c.level)] = c
    k = 0
    for p in (0, 1):
        for side in ("river", "shore"):
            for lev in P.SENSORS["cable_channels_levels"]:
                c = cable_map[(p, side, lev)]
                nid = c.n1
                cth = c.theta
                add(f"C{k + 1:02d}", "cable-axial", nid, c.x_deck, P.DECK_Y,
                    f"axis of {side} stay L{lev} (pylon {p + 1})",
                    [(3 * nid, float(np.cos(cth))),
                     (3 * nid + 1, float(np.sin(cth)))])
                k += 1

    R = sp.coo_matrix((vals, (rows, cols)),
                      shape=(len(channels), model.n_free)).tocsr()
    assert len(channels) == P.SENSORS["n_channels"]
    return channels, R


def _pylon_node(geo, pylon, y):
    """The pylon node at elevation y (nodes exist at every anchor level and
    at the top; a small tolerance handles rounding)."""
    xp = P.PYLON_X[pylon]
    best, bd = None, None
    for n in geo.nodes:
        if n.group == "pylon" and n.x == xp:
            dd = abs(n.y - y)
            if bd is None or dd < bd:
                best, bd = n, dd
    if bd > 1e-6:
        raise ValueError(f"no pylon node within {bd:.3g} m of y = {y}")
    return best.nid
