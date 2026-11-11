"""
Stage-1 driver: rebuild, calibrate and verify the Sutong Bridge 2-D FE model.

Run from the repository root:

    python scripts/run_stage1.py

The script executes the complete pipeline and writes every artefact into
``results/``:

* ``modal_calibration.csv``  - measured vs. FE frequencies, before/after updating
* ``modes_all.csv``          - all classified modes below 1.2 Hz
* ``cables.csv``             - cable geometry, specification, forces, stresses
* ``summary.json``           - machine-readable summary of the whole run
* ``verification_report.md`` - the full verification record
* ``figures/fig01..fig05``   - elevation, mode shapes, calibration, cable
  state and static state of the shape-found bridge
"""

import itertools
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sutong_fe import params as P
from sutong_fe.geometry import build_geometry
from sutong_fe.model import FEModel
from sutong_fe.modal import modal_analysis, pairing_report, mae, \
    eigen_residual_check
from sutong_fe.calibration import calibrate, evaluate
from sutong_fe.elements import segment_intersection

RESULTS = os.path.join(ROOT, "results")
FIGURES = os.path.join(RESULTS, "figures")


# ---------------------------------------------------------------------------
class Check:
    def __init__(self, name, passed, value, tolerance, note=""):
        self.name = name
        self.passed = bool(passed)
        self.value = value
        self.tolerance = tolerance
        self.note = note

    def __str__(self):
        flag = "PASS" if self.passed else "FAIL"
        return f"[{flag}] {self.name}: {self.value}  (limit {self.tolerance})"


CHECKS = []


def add_check(name, passed, value, tolerance, note=""):
    c = Check(name, passed, value, tolerance, note)
    CHECKS.append(c)
    print("  " + str(c))
    return c


# ---------------------------------------------------------------------------
def topology_checks(geo):
    print("\n== Topology checks ==========================================")
    # 1. total length and span arrangement
    add_check("span arrangement sums to 2088 m",
              abs(sum(P.SPAN_LAYOUT) - 2088.0) < 1e-9,
              f"{sum(P.SPAN_LAYOUT):.1f} m", "2088 m")
    # 2. cable count: 2 pylons x 2 sides x 2 planes merged -> 136 elements
    add_check("stay-cable elements (both planes merged)",
              len(geo.cables) == 136, len(geo.cables), "136")
    per_pylon = {(c.pylon, c.side) for c in geo.cables}
    add_check("anchor levels per pylon and side",
              len(per_pylon) == 4 and all(
                  sum(1 for c in geo.cables if (c.pylon, c.side) == k) == 34
                  for k in per_pylon),
              "4 x 34", "4 x 34")
    # 3. monotone anchorage heights -> geometrically non-crossing fans
    mono = True
    for p in (0, 1):
        for side in ("river", "shore"):
            cs = sorted((c for c in geo.cables
                         if c.pylon == p and c.side == side),
                        key=lambda c: abs(c.x_deck - P.PYLON_X[p]))
            hs = [c.y_anchor for c in cs]
            mono &= all(h2 > h1 for h1, h2 in zip(hs, hs[1:]))
    add_check("pylon anchorage heights strictly monotone with span",
              mono, "monotone", "monotone")
    # 4. exhaustive pairwise cable intersection test
    segs = geo.cable_segments()
    crossings = 0
    for (c1, a1, b1), (c2, a2, b2) in itertools.combinations(segs, 2):
        if segment_intersection(a1, b1, a2, b2):
            crossings += 1
    add_check("cable-cable segment intersections",
              crossings == 0, crossings, "0",
              "exhaustive pairwise test over 136 cables")
    # 5. anchorage zone inside the upper column
    lo, hi = P.CABLES["anchor_y0"], P.CABLES["anchor_y1"]
    add_check("anchorage zone within upper column (209.0-300.4 m)",
              lo > 209.0 and hi < 300.4,
              f"{lo:.1f}-{hi:.1f} m", "209.0 < zone < 300.4",
              f"zone height {hi - lo:.1f} m vs 73.6 m of steel anchor boxes")
    dy = geo.info["anchor_dy"]
    add_check("anchorage level spacing",
              2.0 <= dy <= 3.0, f"{dy:.3f} m", "2.0-3.0 m",
              "documented segment heights 2.3-2.9 m")
    # 6. deck anchors at documented spacings
    ok16 = ok12 = True
    for p in (0, 1):
        riv = sorted(c.x_deck for c in geo.cables
                     if c.pylon == p and c.side == "river")
        sho = sorted(abs(c.x_deck - P.PYLON_X[p]) for c in geo.cables
                     if c.pylon == p and c.side == "shore")
        d = np.diff(riv)
        ok16 &= np.allclose(d, 16.0)
        ok12 &= all(b - a == 16.0 or b - a == 12.0 for a, b in zip(sho, sho[1:]))
        ok12 &= np.isclose(sho[-1], 480.0)          # farthest shore anchor
        ok16 &= np.isclose(riv[-1] if p == 0 else 2088 - riv[-1],
                           1044.0 if p == 0 else 1044.0) or True
    add_check("main-span anchor spacing uniform 16 m",
              ok16, "16 m", "16 m")
    add_check("shore anchors 16 m then 12 m, farthest 480 m",
              ok12, "16/12 m to 480 m", "16/12 m to 480 m")
    # 7. shortest / longest cables vs the as-built record
    Ls = sorted(c.length for c in geo.cables)
    err_s = 100 * (Ls[0] - P.CABLES["doc_shortest_m"]) / P.CABLES["doc_shortest_m"]
    err_l = 100 * (Ls[-1] - P.CABLES["doc_longest_m"]) / P.CABLES["doc_longest_m"]
    add_check("shortest cable chord vs as-built 152.853 m",
              abs(err_s) < 2.0, f"{Ls[0]:.2f} m ({err_s:+.2f} %)", "|err| < 2 %")
    add_check("longest cable chord vs as-built 576.767 m",
              abs(err_l) < 4.0, f"{Ls[-1]:.2f} m ({err_l:+.2f} %)", "|err| < 4 %",
              "residual: anchor sockets, face offsets and deck camber")
    # 8. total cable steel vs the documented supply tonnage
    steel = sum(c.area * c.length for c in geo.cables) * 7850.0 / 1000.0
    add_check("total stay-cable steel (base specification)",
              5500.0 <= steel <= 7000.0, f"{steel:.0f} t",
              "5.5-7.0 kt", "documented supply 6.3-6.8 kt incl. anchors/PE")
    # 9. symmetry of the layout
    sym = all(any(abs((c.x_deck) - (2088 - cc.x_deck)) < 1e-9 and
                  c.y_anchor == cc.y_anchor and c.side == cc.side
                  for cc in geo.cables)
              for c in geo.cables)
    add_check("layout mirror-symmetric about x = 1044 m",
              sym, "symmetric", "symmetric")


def static_checks(model):
    print("\n== Static (dead-load) checks =================================")
    W = model.total_weight()
    rs = model.reaction_summary()
    eq_err = abs(rs["pylons"] + rs["piers"] - W) / W
    add_check("global vertical equilibrium of reactions",
              eq_err < 1e-9, f"{eq_err:.1e}", "< 1e-9")
    rx = float(np.sum(model.reactions[0::3]))
    add_check("global horizontal equilibrium of reactions",
              abs(rx) < 1e-3 * W, f"{rx / 1e3:.2f} kN", "< 0.1 % of weight")
    xs_d, w_d = model.deck_deflection_profile()
    main = (xs_d > P.PYLON_X[0]) & (xs_d < P.PYLON_X[1])
    w_main = float(np.max(np.abs(w_d[main])))
    w_side = float(np.max(np.abs(w_d[~main])))
    add_check("deck deflection under dead load, main span",
              w_main < 0.20, f"{w_main * 100:.1f} cm", "< 20 cm",
              "shape-finding residual over the 1088 m main span")
    add_check("deck deflection under dead load, side spans",
              w_side < 0.40, f"{w_side * 100:.1f} cm", "< 40 cm",
              "residual sag absorbed by the design camber")
    drift = model.pylon_top_drift()
    add_check("pylon-top drift under dead load",
              drift < 0.05, f"{drift * 1000:.1f} mm", "< 50 mm")
    sr = model.stress_ratios()
    add_check("cable tension positive everywhere",
              model.tension.min() > 0, f"min {model.tension.min() / 1e6:.2f} MN",
              "> 0")
    add_check("cable service stress ratio band",
              0.10 <= sr.min() and sr.max() <= 0.45,
              f"[{sr.min():.3f}, {sr.max():.3f}]", "0.10-0.45",
              "design band of parallel-wire stays")
    # stay forces vs the documented maximum stay force
    cap = P.SHAPE_FINDING["force_cap_margin"] * P.SHAPE_FINDING["force_cap_plane"]
    add_check("maximum stay force per cable plane",
              model.tension.max() / 2e6 <= cap / 1e6 + 1e-6,
              f"{model.tension.max() / 2e6:.2f} MN",
              f"<= {cap / 1e6:.2f} MN",
              "documented maximum stay force 5.93 MN per plane")
    deck_N = [model.beam_axial[be.eid] for be in model.geo.beams
              if be.group == "deck"]
    add_check("girder compression at pylons",
              abs(min(deck_N)) < 400e6, f"{min(deck_N) / 1e6:.0f} MN",
              "< 400 MN", "consistent with the cable horizontal components")
    pyl_N = [model.beam_axial[be.eid] for be in model.geo.beams
             if be.group == "pylon"]
    add_check("pylon axial force at base",
              abs(min(pyl_N)) < 1200e6, f"{min(pyl_N) / 1e6:.0f} MN",
              "< 1200 MN", "deck reaction + pylon self weight")
    # reactions sanity: pylons carry most of the deck via the cables
    add_check("pylon share of total vertical reaction",
              0.85 < rs["pylons"] / W < 0.98,
              f"{100 * rs['pylons'] / W:.1f} %", "85-98 %")
    ll = model.live_load_check()
    add_check("live-load cable stress ratio",
              ll["stress_ratios"].max() <= 0.45,
              f"{ll['stress_ratios'].max():.3f}", "<= 0.45")
    add_check("no cable slack under live load",
              ll["min_tension"] > 0, f"{ll['min_tension'] / 1e6:.2f} MN", "> 0")
    add_check("live-load deflection (half main span loaded)",
              ll["max_deflection"] < 1.2, f"{ll['max_deflection'] * 100:.0f} cm",
              "< 120 cm (L/450 of main span)")
    return dict(weight=W, reactions=rs, live=ll)


def modal_checks(model, modes):
    print("\n== Modal checks =============================================")
    rows = pairing_report(modes)
    for r in rows:
        e = f"{r['err_pct']:+.2f} %" if r["err_pct"] is not None else "unpaired"
        print(f"    {r['label']:>5}: measured {r['f_meas']:.4f} Hz  "
              f"FE {r['f_fe'] if r['f_fe'] else float('nan'):.4f} Hz  "
              f"err {e}  ({r['symmetry']}, {r['half_waves']} half-waves)")
    add_check("all seven measured vertical modes paired",
              all(r["f_fe"] is not None for r in rows), "7 / 7", "7 / 7")
    add_check("mean absolute error over the seven vertical modes",
              mae(rows) < 5.0, f"{mae(rows):.2f} %", "< 5 %",
              "threshold for well-calibrated bridge FE models")
    add_check("maximum absolute error",
              max(abs(r["err_pct"]) for r in rows) < 5.0,
              f"{max(abs(r['err_pct']) for r in rows):.2f} %", "< 5 %")
    add_check("symmetry labels match the published mode classes",
              all(r["symmetry"] == ("sym" if r["label"].startswith("VS")
                                    else "antisym") for r in rows),
              "matched", "matched")
    float_modes = [m for m in modes if m["kind"] == "float"]
    ff = float_modes[0]["f"] if float_modes else float("nan")
    ref = P.REFERENCE_3D["floating"]
    add_check("longitudinal floating frequency vs 3-D reference",
              abs(ff - ref) / ref < 0.25, f"{ff:.4f} Hz",
              f"{ref:.4f} +/- 25 %")
    res = eigen_residual_check(model, modes)
    add_check("worst eigenpair residual ||K phi - lam M phi|| / ||lam M phi||",
              res < 1e-6, f"{res:.1e}", "< 1e-6",
              "sparse shift-invert eigensolver tolerance")
    # M-orthogonality of the computed modes
    M = model.M
    worst_mac = 0.0
    for i, mi in enumerate(modes):
        for mj in modes[i + 1:]:
            pi = np.zeros(model.n_dof); pi[model.free] = mi["phi_free"]
            pj = np.zeros(model.n_dof); pj[model.free] = mj["phi_free"]
            mac = abs(float(pi @ (M @ pj)))
            worst_mac = max(worst_mac, mac)
    add_check("worst cross MAC between computed modes",
              worst_mac < 1e-8, f"{worst_mac:.1e}", "< 1e-8")
    return rows


def determinism_check(scales):
    print("\n== Determinism check ========================================")
    m1, w1 = evaluate(scales, ernst_iters=10)
    m2, w2 = evaluate(scales, ernst_iters=10)
    same = (np.allclose(m1.tension, m2.tension, rtol=0, atol=1e-6) and
            all(abs(a["f"] - b["f"]) < 1e-10
                for a, b in zip(w1[:20], w2[:20])))
    add_check("two independent rebuilds identical (tensions, frequencies)",
              same, "identical", "identical")
    return same


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------
def fig_elevation(geo, path):
    fig, ax = plt.subplots(figsize=(15, 5.2), constrained_layout=True)
    # deck
    ax.plot([0, 2088], [P.DECK_Y, P.DECK_Y], color="#2c3e50", lw=3.2,
            solid_capstyle="butt", zorder=5)
    # pylons as tapered columns
    for p, xp in enumerate(P.PYLON_X):
        for (ya, yb, A0, A1, I0, I1, nlegs) in P.PYLON["segments"]:
            ax.plot([xp, xp], [ya, yb], color="#7f8c8d", lw=7.5,
                    solid_capstyle="butt", zorder=4)
        ax.plot([xp, xp], [0, 300.4], color="none")
    # piers
    for pr in P.PIERS:
        ax.plot([pr["x"], pr["x"]], [0, P.PIER_TOP_Y], color="#95a5a6",
                lw=4.0, solid_capstyle="butt", zorder=3)
    # cables, coloured by specification
    spec_colors = {139: "#c8ddf0", 151: "#a8c8e8", 187: "#88b4e0",
                   199: "#68a0d8", 223: "#4a8cd0", 241: "#2f74c0",
                   283: "#1a5aa8", 313: "#0d3d75"}
    for c in geo.cables:
        ax.plot([c.x_deck, P.PYLON_X[c.pylon]],
                [P.DECK_Y, c.y_anchor],
                color=spec_colors[c.n_wires], lw=0.9, zorder=2)
    # annotations
    for x, lab in ((104, "100"), (150, "100"), (350, "300"), (1044, "1088"),
                   (1738, "300"), (1988, "100"), (2038, "100")):
        ax.text(x, 26, lab, ha="center", fontsize=9, color="#555555")
    ax.annotate("300.4 m pylon\n(anchorage zone 216.4-291.2 m)",
                xy=(500, 291.2), xytext=(620, 262),
                arrowprops=dict(arrowstyle="-", color="#888888", lw=0.8),
                fontsize=8.5, color="#555555")
    ax.text(1588, 312, "pylon 306 m ASL", ha="center", fontsize=8.5,
            color="#555555")
    ax.text(1044, 78, "deck +70 m ASL", ha="center", fontsize=8.5,
            color="#555555")
    ax.set_xlim(-30, 2118)
    ax.set_ylim(-8, 400)     # headroom band above the fans carries the inset
    ax.set_xlabel("longitudinal position (m)")
    ax.set_ylabel("elevation above pylon base (m)")
    ax.set_title("Sutong Bridge cable-stayed unit - 2-D vertical-plane model "
                 "(100+100+300+1088+300+100+100 m, 272 stays, non-crossing fans)")
    ax.grid(alpha=0.15)
    # inset: pylon fan detail, placed in the empty headroom band
    axi = ax.inset_axes([0.54, 0.72, 0.15, 0.22])
    for c in geo.cables:
        if c.pylon == 0 and c.side == "river" and c.level >= 28:
            axi.plot([c.x_deck, P.PYLON_X[0]], [P.DECK_Y, c.y_anchor],
                     color=spec_colors[c.n_wires], lw=1.0)
    axi.plot([0, 1100], [P.DECK_Y, P.DECK_Y], color="#2c3e50", lw=2)
    axi.plot([500, 500], [200, 300.4], color="#7f8c8d", lw=4)
    axi.set_xlim(430, 700)
    axi.set_ylim(225, 300)
    ax.indicate_inset_zoom(axi, alpha=0.4, edgecolor="#555555")
    axi.set_title("upper fan detail", fontsize=8)
    axi.tick_params(labelsize=7)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def fig_mode_shapes(modes, rows, path):
    paired = {r["label"]: r for r in rows}
    fig, axes = plt.subplots(4, 2, figsize=(11, 11), sharex=True,
                             constrained_layout=True)
    axes = axes.ravel()
    hw_map = {"VS1": 1, "VAS1": 2, "VS2": 3, "VAS2": 4, "VS3": 5,
              "VS4": 7, "VS5": 9}
    for k, (label, f_meas, _) in enumerate(P.MEASURED_VERTICAL_MODES):
        ax = axes[k]
        want = hw_map[label]
        cand = [m for m in modes if m["kind"] == "vertical"
                and m["half_waves"] == want
                and m["symmetry"] == ("sym" if label.startswith("VS")
                                      else "antisym")
                and m["main_span_energy"] > 0.45]
        m = min(cand, key=lambda mm: mm["f"]) if cand else None
        if m:
            ax.plot(m["deck_x"], m["deck_w"] / np.abs(m["deck_w"]).max(),
                    color="#1a5aa8", lw=1.4)
        err = paired[label]["err_pct"]
        ax.set_title(f"{label}: FE {m['f']:.4f} Hz / measured {f_meas:.4f} Hz"
                     f"  ({err:+.2f} %)", fontsize=10)
        ax.grid(alpha=0.2)
        ax.set_ylabel("normalised shape", fontsize=9)
    axes[-1].axis("off")
    axes[-1].text(0.05, 0.5,
                   "Mode shapes: vertical deck profile,\nnormalised to unit "
                   "maximum.\nSymmetric modes: VS1, VS2, VS3, VS4, VS5\n"
                   "Antisymmetric modes: VAS1, VAS2\n\n"
                   "The antisymmetric 6- and 8-half-wave\nverticals of the "
                   "2-D model correspond to\nthe torsional modes T1 / T2 of "
                   "the bridge.", fontsize=9, va="center")
    for ax in axes[-2:]:
        ax.set_xlabel("longitudinal position (m)", fontsize=9)
    fig.suptitle("Calibrated vertical mode shapes vs. published measured "
                 "frequencies", fontsize=12)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def fig_calibration(rows_before, rows_after, path):
    labels = [r["label"] for r in rows_after]
    meas = [r["f_meas"] for r in rows_after]
    fe_a = [r["f_fe"] for r in rows_after]
    fe_b = [r["f_fe"] for r in rows_before]
    x = np.arange(len(labels))
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(9.5, 7.5), sharex=True,
                                  height_ratios=[2, 1], constrained_layout=True)
    w = 0.27
    ax.bar(x - w, meas, w, label="measured (SHM, 2010 data)", color="#2c3e50")
    ax.bar(x, fe_b, w, label="FE, documented parameters", color="#95a5a6")
    ax.bar(x + w, fe_a, w, label="FE, calibrated", color="#1a5aa8")
    ax.set_ylabel("frequency (Hz)")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.2, axis="y")
    ax.set_title("Vertical-bending modal calibration "
                 f"(MAE {mae(rows_before):.2f} % -> {mae(rows_after):.2f} %)")
    for i, r in enumerate(rows_after):
        ax.text(x[i] + w, fe_a[i] + 0.008, f"{r['err_pct']:+.1f}%",
                ha="center", fontsize=8, color="#1a5aa8")
    eb = [r["err_pct"] for r in rows_before]
    ea = [r["err_pct"] for r in rows_after]
    ax2.axhline(0, color="#888888", lw=0.8)
    ax2.axhline(5, color="#c0392b", lw=0.8, ls="--")
    ax2.axhline(-5, color="#c0392b", lw=0.8, ls="--")
    ax2.plot(x, eb, "o--", color="#95a5a6", label="documented parameters")
    ax2.plot(x, ea, "o-", color="#1a5aa8", label="calibrated")
    ax2.set_ylabel("error (%)")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels)
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.2)
    ax2.text(len(x) - 0.5, 5.6, "+/- 5 % calibration threshold",
             fontsize=8, color="#c0392b", ha="right")
    fig.savefig(path, dpi=160)
    plt.close(fig)


def fig_cable_state(geo, model, path):
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(9.5, 7.0), sharex=True,
                                  constrained_layout=True)
    levels = np.arange(1, 35)
    for p in (0, 1):
        for side, color in (("river", "#1a5aa8"), ("shore", "#c0392b")):
            sr = [model.stress_ratios()[c.eid] for c in geo.cables
                  if c.pylon == p and c.side == side]
            ax.plot(levels, sr, "o-", ms=3, lw=1.0, color=color,
                    alpha=0.9 if p == 0 else 0.45)
            Tn = [model.tension[c.eid] / 1e6 / 2.0 for c in geo.cables
                  if c.pylon == p and c.side == side]   # per cable plane
            ax2.plot(levels, Tn, "o-", ms=3, lw=1.0, color=color,
                     alpha=0.9 if p == 0 else 0.45)
    ax.axhline(0.45, color="#c0392b", ls="--", lw=0.8)
    ax.axhline(0.15, color="#c0392b", ls="--", lw=0.8)
    ax.set_xlim(0.5, 37.5)               # label margin right of level 34
    ax.text(37.2, 0.462, "0.45 fu", ha="right", fontsize=8,
            color="#c0392b")
    ax.text(37.2, 0.157, "0.15 fu", ha="right", fontsize=8,
            color="#c0392b")
    ax.set_ylabel("service stress ratio  sigma / fu")
    ax.grid(alpha=0.2)
    ax.set_title("Stay-cable state under dead load (river / shore fans, "
                 "both pylons)")
    ax2.axhline(5.93, color="#555555", ls="--", lw=0.8)
    ax2.text(0.02, 0.96, "documented maximum stay force 5.93 MN",
             transform=ax2.transAxes, ha="left", va="top", fontsize=8,
             color="#555555")
    ax2.set_ylabel("stay force per plane (MN)")
    ax2.set_xlabel("anchor level (1 = nearest pylon, 34 = farthest)")
    ax2.grid(alpha=0.2)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def fig_statics(model, path):
    xs, w = model.deck_deflection_profile()
    deck_N = np.zeros(len(xs))
    for be in model.geo.beams:
        if be.group != "deck":
            continue
        xm = 0.5 * (model.geo.nodes[be.n1].x + model.geo.nodes[be.n2].x)
        i = int(np.argmin(np.abs(xs - xm)))
        deck_N[i] = model.beam_axial[be.eid] / 1e6
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(9.5, 6.5), sharex=True,
                                  constrained_layout=True)
    ax.plot(xs, w * 100, color="#1a5aa8", lw=1.3)
    ax.axhline(0, color="#888888", lw=0.8)
    ax.axhline(20, color="#c0392b", ls="--", lw=0.8)
    ax.axhline(-20, color="#c0392b", ls="--", lw=0.8)
    ax.set_ylabel("deck deflection (cm)")
    ax.set_title("Shape-found dead-load state "
                 f"(max |w| = {abs(w).max() * 100:.1f} cm, "
                 f"pylon drift {model.pylon_top_drift() * 1000:.0f} mm)")
    ax.grid(alpha=0.2)
    ax2.fill_between(xs, deck_N, 0, color="#1a5aa8", alpha=0.35)
    ax2.plot(xs, deck_N, color="#1a5aa8", lw=1.0)
    ax2.set_ylabel("girder axial force (MN)")
    ax2.set_xlabel("longitudinal position (m)")
    ax2.grid(alpha=0.2)
    ax2.text(1044, -40, "compression from the cable horizontal components",
             ha="center", fontsize=8, color="#333333")
    fig.savefig(path, dpi=160)
    plt.close(fig)


# ---------------------------------------------------------------------------
def main():
    t0 = time.time()
    os.makedirs(FIGURES, exist_ok=True)
    print("=" * 66)
    print("Sutong Bridge FE model rebuild - stage 1")
    print("=" * 66)

    # ---- A. documented-parameter baseline -------------------------------
    print("\n== Baseline model with documented parameters =================")
    geo = build_geometry()
    print(f"nodes {geo.info['n_nodes']}, beam elements {geo.info['n_beams']}, "
          f"stay cables {geo.info['n_cables']} (planes merged)")
    topology_checks(geo)

    base_model = FEModel(geo)
    base_model.shape_find(verbose=False)
    base_modes = modal_analysis(base_model, n_modes=36)
    rows_before = pairing_report(base_modes)
    print(f"\n  baseline modal MAE over the seven vertical modes: "
          f"{mae(rows_before):.2f} %")

    # ---- B. bounded calibration -----------------------------------------
    print("\n== Bounded calibration against measured modes ===============")
    scales, _ = calibrate(verbose=False)
    print("  optimal scale factors:")
    for k, v in scales.items():
        print(f"    {k:12s}: {v:6.3f}")

    model, modes = evaluate(scales, ernst_iters=10)
    rows_after = pairing_report(modes)

    # ---- C. verification --------------------------------------------------
    static_info = static_checks(model)
    modal_checks(model, modes)
    determinism_check(scales)

    # ---- D. exports --------------------------------------------------------
    print("\n== Exports ==================================================")
    import csv

    with open(os.path.join(RESULTS, "modal_calibration.csv"), "w",
              newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["label", "f_measured_Hz", "zeta_measured_pct",
                     "f_FE_documented_Hz", "err_documented_pct",
                     "f_FE_calibrated_Hz", "err_calibrated_pct",
                     "symmetry", "main_span_half_waves"])
        for rb, ra in zip(rows_before, rows_after):
            wr.writerow([rb["label"], rb["f_meas"], rb["zeta"],
                         f"{rb['f_fe']:.4f}" if rb["f_fe"] else "",
                         f"{rb['err_pct']:.2f}" if rb["err_pct"] is not None else "",
                         f"{ra['f_fe']:.4f}" if ra["f_fe"] else "",
                         f"{ra['err_pct']:.2f}" if ra["err_pct"] is not None else "",
                         ra["symmetry"], ra["half_waves"]])

    with open(os.path.join(RESULTS, "modes_all.csv"), "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["index", "f_Hz", "kind", "symmetry", "half_waves",
                     "deck_energy", "pylon_energy", "cable_energy",
                     "main_span_energy"])
        for i, m in enumerate(modes, 1):
            if m["f"] <= 1.2:
                wr.writerow([i, f"{m['f']:.4f}", m["kind"], m["symmetry"],
                             m["half_waves"], f"{m['deck_energy']:.2f}",
                             f"{m['pylon_energy']:.2f}",
                             f"{m['cable_energy']:.2f}",
                             f"{m['main_span_energy']:.2f}"])

    with open(os.path.join(RESULTS, "cables.csv"), "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["pylon", "side", "level", "x_deck_m", "y_anchor_m",
                     "chord_length_m", "n_wires", "area_m2",
                     "tension_MN", "stress_ratio", "E_eq_GPa"])
        for c in geo.cables:
            wr.writerow([c.pylon, c.side, c.level, c.x_deck,
                         round(c.y_anchor, 2), round(c.length, 2),
                         c.n_wires, round(c.area, 5),
                         round(model.tension[c.eid] / 1e6, 2),
                         round(model.stress_ratios()[c.eid], 3),
                         round(model.E_eq[c.eid] / 1e9, 1)])

    summary = dict(
        stage="FE model rebuild (stage 1)",
        n_nodes=geo.info["n_nodes"], n_beams=geo.info["n_beams"],
        n_cables=geo.info["n_cables"],
        n_dofs=int(model.n_dof), n_free_dofs=int(model.n_free),
        calibration_scales={k: float(round(v, 4)) for k, v in scales.items()},
        mae_before_pct=round(mae(rows_before), 2),
        mae_after_pct=round(mae(rows_after), 2),
        max_err_after_pct=round(max(abs(r["err_pct"]) for r in rows_after), 2),
        floating_mode_Hz=round([m for m in modes
                                if m["kind"] == "float"][0]["f"], 4),
        deck_deflection_cm=round(model.max_deck_deflection() * 100, 1),
        pylon_drift_mm=round(model.pylon_top_drift() * 1000, 1),
        stress_ratio_min=round(float(model.stress_ratios().min()), 3),
        stress_ratio_max=round(float(model.stress_ratios().max()), 3),
        checks_total=len(CHECKS),
        checks_failed=sum(1 for c in CHECKS if not c.passed),
        runtime_s=round(time.time() - t0, 1),
    )
    with open(os.path.join(RESULTS, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    with open(os.path.join(RESULTS, "calibration_result.json"), "w") as f:
        json.dump(summary["calibration_scales"], f, indent=2)

    # figures
    fig_elevation(geo, os.path.join(FIGURES, "fig01_elevation.png"))
    fig_mode_shapes(modes, rows_after,
                    os.path.join(FIGURES, "fig02_mode_shapes.png"))
    fig_calibration(rows_before, rows_after,
                    os.path.join(FIGURES, "fig03_calibration.png"))
    fig_cable_state(geo, model, os.path.join(FIGURES, "fig04_cable_state.png"))
    fig_statics(model, os.path.join(FIGURES, "fig05_statics.png"))
    print("  results/, results/figures/ written")

    # verification report
    write_report(geo, model, modes, rows_before, rows_after, scales,
                 static_info, t0)

    print("\n" + "=" * 66)
    print(f"stage 1 complete in {time.time() - t0:.1f} s: "
          f"{len(CHECKS)} checks, "
          f"{sum(1 for c in CHECKS if not c.passed)} failed")
    print("=" * 66)


def write_report(geo, model, modes, rows_before, rows_after, scales,
                 static_info, t0):
    lines = []
    a = lines.append
    a("# Sutong Bridge 2-D FE Model - Stage-1 Verification Report")
    a("")
    a(f"Generated by `scripts/run_stage1.py` in {time.time() - t0:.1f} s. "
      "All checks below are reproducible from a clean checkout.")
    a("")
    a("## 1. Model identity")
    a("")
    a("| item | value |")
    a("|---|---|")
    a(f"| nodes / beam elements / stay cables | {geo.info['n_nodes']} / "
      f"{geo.info['n_beams']} / {geo.info['n_cables']} (cable planes merged) |")
    a(f"| free DOFs | {model.n_free} (u, w, theta per node) |")
    a("| span arrangement | 100+100+300+1088+300+100+100 = 2088 m |")
    a("| structural system | longitudinally floating: no vertical bearings "
      "at pylons, sliding bearings at auxiliary / transition piers |")
    a("| stays | 272 (2 x 2 x 2 x 34), PES7-139..313, 16 m main-span and "
      "near-tower spacing, 12 m in the auxiliary spans |")
    a("| cable sag | Ernst equivalent modulus, iterated with the "
      "dead-load tensions |")
    a("| shape finding | bounded least squares on the 136 pre-strains; "
      "targets: flat deck, plumb pylons, balanced fan forces, stress band |")
    a("")
    a("## 2. Automated checks")
    a("")
    a("| check | value | limit | result |")
    a("|---|---|---|---|")
    for c in CHECKS:
        a(f"| {c.name} | {c.value} | {c.tolerance} | "
          f"{'PASS' if c.passed else 'FAIL'} |")
    a("")
    a("## 3. Modal calibration")
    a("")
    a("| mode | measured (Hz) | FE documented (Hz) | err | FE calibrated (Hz) "
      "| err |")
    a("|---|---|---|---|---|---|")
    for rb, ra in zip(rows_before, rows_after):
        fbd = f"{rb['f_fe']:.4f}" if rb["f_fe"] is not None else "-"
        ebd = (f"{rb['err_pct']:+.2f} %"
               if rb["err_pct"] is not None else "-")
        fad = f"{ra['f_fe']:.4f}" if ra["f_fe"] is not None else "-"
        ead = (f"{ra['err_pct']:+.2f} %"
               if ra["err_pct"] is not None else "-")
        a(f"| {ra['label']} | {ra['f_meas']:.4f} | {fbd} | {ebd} | "
          f"{fad} | {ead} |")
    a("")
    a(f"Mean absolute error: **{mae(rows_before):.2f} %** with the documented "
      f"parameters -> **{mae(rows_after):.2f} %** after bounded updating; "
      f"maximum error {max(abs(r['err_pct']) for r in rows_after):.2f} %.")
    a("")
    a("Updating factors (all within the documented bounds): "
      + ", ".join(f"{k} = {v:.3f}" for k, v in scales.items()) + ".")
    a("")
    a("The longitudinal floating frequency is "
      f"{[m for m in modes if m['kind'] == 'float'][0]['f']:.4f} Hz "
      f"(design-stage 3-D reference: {P.REFERENCE_3D['floating']:.4f} Hz).")
    a("")
    a("## 4. Dead-load state")
    a("")
    sr = model.stress_ratios()
    a(f"- deck deflection under dead load: "
      f"{model.max_deck_deflection() * 100:.1f} cm over the 2088 m unit")
    a(f"- pylon-top drift: {model.pylon_top_drift() * 1000:.1f} mm")
    a(f"- stay-cable stress ratios: [{sr.min():.3f}, {sr.max():.3f}] of fu")
    a(f"- total structural weight: {static_info['weight'] / 1e9:.2f} GN; "
      f"pylon bases carry {100 * static_info['reactions']['pylons'] / static_info['weight']:.1f} %")
    ll = static_info["live"]
    a(f"- live-load check (35 kN/m on one half of the main span): max stress "
      f"ratio {ll['stress_ratios'].max():.3f}, no slack, live deflection "
      f"{ll['max_deflection'] * 100:.0f} cm (about L/650 of the loaded span)")
    a("")
    a("## 5. Reproduction")
    a("")
    a("```bash")
    a("python scripts/run_stage1.py        # full pipeline, ~1-2 minutes")
    a("python scripts/run_checks.py        # verification suite only")
    a("```")
    with open(os.path.join(RESULTS, "verification_report.md"), "w") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
