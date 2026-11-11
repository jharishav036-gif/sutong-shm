"""
Modal analysis of the 2-D Sutong model and automatic mode classification.

The generalised eigen-problem  K phi = omega^2 M phi  is solved on the
free DOFs with a sparse shift-invert solver.  Every mode is then classified
from its mass-energy distribution:

* ``vertical``   - girder vertical bending (the calibration targets);
* ``float``      - longitudinal floating motion of the deck;
* ``pylon``      - pylon-dominated modes;
* ``deck_long``  - deck longitudinal / axial modes.

Vertical modes are further labelled symmetric or antisymmetric about the
bridge centreline, which lets them be paired one-to-one with the published
measured modes (VS1, VAS1, VS2, ...).
"""

import numpy as np
import scipy.sparse.linalg as spla

from . import params as P


# ---------------------------------------------------------------------------
def modal_analysis(model, n_modes=36, sigma=0.05):
    """Solve the eigen-problem and classify the modes.

    Returns a list of dicts sorted by frequency with fields:
    f, kind, symmetry, deck_energy, pylon_energy, cable_energy,
    half_waves, deck_w (vertical profile over the deck nodes), deck_x.
    """
    K, M = model.assemble_modal()
    Kff = K[model.free][:, model.free].tocsc()
    Mff = M[model.free][:, model.free].tocsc()
    vals, vecs = spla.eigsh(Kff, k=n_modes, M=Mff, sigma=sigma,
                            which="LM", tol=1e-10)

    order = np.argsort(vals)
    deck_w_dofs = np.array([3 * n + 1 for n in model.deck_nids])
    deck_u_dofs = np.array([3 * n for n in model.deck_nids])

    # tributary deck mass per node (for main-span energy share)
    deck_mass = np.zeros(len(model.deck_nids))
    nid_to_idx = {n: k for k, n in enumerate(model.deck_nids)}
    for be in model.geo.beams:
        if be.group != "deck":
            continue
        n1, n2 = model.geo.nodes[be.n1], model.geo.nodes[be.n2]
        L = float(np.hypot(n2.x - n1.x, n2.y - n1.y))
        deck_mass[nid_to_idx[be.n1]] += be.rho_line * L / 2.0
        deck_mass[nid_to_idx[be.n2]] += be.rho_line * L / 2.0
    main_span = (model.deck_xs > 544.0) & (model.deck_xs < 1544.0)

    # group membership of every dof (for mass-energy bookkeeping)
    group = np.zeros(model.n_dof, dtype=int)     # 0 other, 1 deck, 2 pylon, 3 cable
    for n in model.geo.nodes:
        g = {"deck": 1, "pylon": 2, "pier": 0}[n.group]
        group[3 * n.nid:3 * n.nid + 3] = g
    for i in range(model.nc):
        group[model.c_dof[i]] = 3
    group_free = group[model.free]

    modes = []
    for idx in order:
        lam = vals[idx]
        if lam <= 0:
            continue
        v = vecs[:, idx]
        v = v / np.sqrt(v @ (Mff @ v))          # M-orthonormal
        Mv = Mff @ v
        tot = float(v @ Mv)
        e = {}
        for g, name in ((1, "deck"), (2, "pylon"), (3, "cable")):
            e[name] = float(np.sum(Mv[group_free == g] * v[group_free == g]) / tot)

        phi = np.zeros(model.n_dof)
        phi[model.free] = v
        w = phi[deck_w_dofs]
        u_d = phi[deck_u_dofs]
        e_w = float(np.sum(Mv[model.dof_map[deck_w_dofs]] ** 2))
        e_u = float(np.sum(Mv[model.dof_map[deck_u_dofs]] ** 2))

        if e["deck"] > 0.45 and e_w > 2.0 * e_u:
            kind = "vertical"
        elif e["pylon"] > 0.5:
            kind = "pylon"
        elif e["deck"] > 0.45 and e_u >= e_w:
            kind = "float"
        else:
            kind = "deck_long"

        # symmetry about the bridge centreline (deck vertical profile)
        xs = model.deck_xs
        w_m = w - w.mean()
        sym = float(np.dot(w_m, w_m[::-1]) /
                    (np.linalg.norm(w_m) * np.linalg.norm(w_m[::-1]) + 1e-30))
        symmetry = "sym" if sym > 0.6 else ("antisym" if sym < -0.6 else "mixed")

        # half-wave count over the main span
        main = (xs > 516.0) & (xs < 1572.0)
        wm = w[main]
        sgn = np.sign(wm[np.abs(wm) > 0.05 * np.max(np.abs(wm))])
        half_waves = int(np.sum(sgn[1:] != sgn[:-1])) + 1 if len(sgn) else 0

        # share of the deck vertical modal energy inside the main span
        ms_energy = float(np.sum(deck_mass[main_span] * w[main_span] ** 2) /
                          (np.sum(deck_mass * w ** 2) + 1e-30))

        modes.append(dict(
            f=float(np.sqrt(lam) / (2 * np.pi)),
            kind=kind,
            symmetry=symmetry,
            sym_index=sym,
            deck_energy=e["deck"],
            pylon_energy=e["pylon"],
            cable_energy=e["cable"],
            half_waves=half_waves,
            main_span_energy=ms_energy,
            deck_w=w,
            deck_x=xs,
            phi_free=v,
            lam=float(lam),
        ))
    modes.sort(key=lambda m: m["f"])
    return modes


def vertical_modes(modes, n=None):
    out = [m for m in modes if m["kind"] == "vertical"]
    return out[:n] if n else out


def pairing_report(modes):
    """Pair classified vertical modes with the measured targets.

    The published measured modes are the global vertical modes of the main
    span, alternating symmetric / antisymmetric and identified from the
    main-span sensor array.  The pairing therefore takes, per symmetry
    class, the main-span-dominant vertical modes (modal energy share above
    0.6) in frequency order: symmetric #1..#5 -> VS1..VS5, antisymmetric
    #1..#2 -> VAS1, VAS2.  Additional antisymmetric and overtone verticals
    of the 2-D model either couple into the torsional modes T1 / T2 of the
    real bridge or are side-span dominated, and stay outside the pairing.
    """
    vm = [m for m in modes if m["kind"] == "vertical"
          and m["main_span_energy"] > 0.6]
    sym_modes = [m for m in vm if m["symmetry"] == "sym"]
    antisym_modes = [m for m in vm if m["symmetry"] == "antisym"]
    rows = []
    counters = {"VS": 0, "VAS": 0}
    for label, f_meas, zeta in P.MEASURED_VERTICAL_MODES:
        cls = "VS" if label.startswith("VS") else "VAS"
        pool = sym_modes if cls == "VS" else antisym_modes
        idx = counters[cls]
        counters[cls] += 1
        if idx < len(pool):
            m = pool[idx]
            err = 100.0 * (m["f"] - f_meas) / f_meas
            rows.append(dict(label=label, f_meas=f_meas, zeta=zeta,
                             f_fe=m["f"], err_pct=err,
                             symmetry=m["symmetry"],
                             half_waves=m["half_waves"]))
        else:
            rows.append(dict(label=label, f_meas=f_meas, zeta=zeta,
                             f_fe=None, err_pct=None, symmetry="-",
                             half_waves=None))
    return rows


def mae(rows):
    e = [abs(r["err_pct"]) for r in rows if r["err_pct"] is not None]
    return float(np.mean(e)) if e else float("nan")


def eigen_residual_check(model, modes):
    """|| (K phi - lambda M phi)_free || / || lambda M phi || for every mode.

    The residual is evaluated on the free DOFs only: at the constrained DOFs
    the same expression equals the support reactions by construction.
    """
    K = model.K_modal
    M = model.M
    worst = 0.0
    for m in modes:
        lam = m["lam"]
        phi_full = np.zeros(model.n_dof)
        phi_full[model.free] = m["phi_free"]
        r = (K @ phi_full - lam * (M @ phi_full))[model.free]
        denom = np.linalg.norm(lam * (M @ phi_full)[model.free])
        worst = max(worst, float(np.linalg.norm(r) / max(denom, 1e-30)))
    return worst
