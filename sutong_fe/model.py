"""
Sparse assembly, dead-load shape finding and serviceability checks for the
Sutong Bridge 2-D vertical-plane model.

Statics
-------
The bridge is a longitudinally floating system: the girder carries no
vertical bearings at the pylons (cables only) and rests on one-way sliding
bearings at the auxiliary and transition piers.  Dead-load equilibrium is
established by a bounded least-squares *shape finding*: the 136 cable
pre-strains are the unknowns, the residuals are the deck deflections, the
pylon-top drift, the horizontal-force balance at each pylon and a light
regularisation towards the tributary-stress state.  Cable sag is represented
by the Ernst equivalent modulus, iterated to convergence with the cable
tensions.

Modal analysis
--------------
The eigen-problem is formed on the linearised stiffness about the dead-load
state (elastic stiffness + geometric stiffness from the dead-load element
forces) with the consistent mass matrix.
"""

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.optimize import lsq_linear

from . import params as P
from . import elements as el
from .geometry import _deck_line_mass


# ---------------------------------------------------------------------------
class FEModel:
    """Assembled 2-D finite-element model of the cable-stayed unit."""

    def __init__(self, geo):
        self.geo = geo
        self.n_nodes = len(geo.nodes)
        self.n_dof = 3 * self.n_nodes

        # --- fixed DOFs: pylon and pier bases ----------------------------
        fixed_nodes = [n.nid for n in geo.nodes
                       if n.group in ("pylon", "pier") and abs(n.y) < 1e-9]
        self.fixed = sorted(d for nid in fixed_nodes
                            for d in range(3 * nid, 3 * nid + 3))
        self._fixed_pos = {d: k for k, d in enumerate(self.fixed)}
        free_mask = np.ones(self.n_dof, dtype=bool)
        free_mask[self.fixed] = False
        self.free = np.flatnonzero(free_mask)
        self.dof_map = -np.ones(self.n_dof, dtype=int)
        self.dof_map[self.free] = np.arange(len(self.free))
        self.n_free = len(self.free)

        # --- bookkeeping --------------------------------------------------
        self.deck_nids = [n.nid for n in geo.nodes if n.group == "deck"]
        self.deck_xs = np.array([geo.nodes[n].x for n in self.deck_nids])
        self.pylon_top_nids = [n.nid for n in geo.nodes
                               if n.group == "pylon" and abs(n.y - 300.4) < 1e-6]

        # cable geometry arrays
        nc = len(geo.cables)
        self.nc = nc
        self.c_dof = np.zeros((nc, 4), dtype=int)        # global dofs
        self.c_theta = np.zeros(nc)
        self.c_L = np.zeros(nc)
        self.c_A = np.zeros(nc)
        self.c_rho = np.zeros(nc)
        self.c_Lx = np.zeros(nc)
        for i, c in enumerate(geo.cables):
            self.c_dof[i] = [3 * c.n1, 3 * c.n1 + 1, 3 * c.n2, 3 * c.n2 + 1]
            self.c_theta[i] = c.theta
            self.c_L[i] = c.length
            self.c_A[i] = c.area
            self.c_rho[i] = c.rho_line
            self.c_Lx[i] = abs(P.PYLON_X[c.pylon] - c.x_deck)
        self.c_dof_free = self.dof_map[self.c_dof]        # free-reduced dofs

        # --- gravity load vector -----------------------------------------
        self.F_g = self._gravity_vector()

        # --- state (filled by shape finding) ------------------------------
        self.eps0 = np.zeros(nc)
        self.tension = np.zeros(nc)
        self.E_eq = np.full(nc, P.CABLES["E"])
        self.u = np.zeros(self.n_dof)
        self.beam_axial = {}
        self.reactions = np.zeros(len(self.fixed))

    # ------------------------------------------------------------------
    # assembly helpers
    # ------------------------------------------------------------------
    def _beam_angle(self, n1, n2):
        return float(np.arctan2(n2.y - n1.y, n2.x - n1.x))

    def _beam_dofs(self, be):
        return list(range(3 * be.n1, 3 * be.n1 + 3)) + \
               list(range(3 * be.n2, 3 * be.n2 + 3))

    def _add_block(self, rows, cols, vals, dofs, Km):
        for a in range(Km.shape[0]):
            for b in range(Km.shape[1]):
                if abs(Km[a, b]) > 0.0:
                    rows.append(dofs[a]); cols.append(dofs[b]); vals.append(Km[a, b])

    def _bearing_truss(self, be):
        """Bearing link as a two-node truss (vertical pin-pin bar): no
        rotational DOFs, vertical force transfer only.  The axial stiffness is
        taken from the element itself (E * A), so a degrading bearing can be
        represented by scaling the element's properties."""
        n1, n2 = self.geo.nodes[be.n1], self.geo.nodes[be.n2]
        L = float(np.hypot(n2.x - n1.x, n2.y - n1.y))
        theta = float(np.arctan2(n2.y - n1.y, n2.x - n1.x))
        dofs = [3 * be.n1, 3 * be.n1 + 1, 3 * be.n2, 3 * be.n2 + 1]
        K, M = el.cable_matrices(be.E * be.A, 0.0, L, theta)
        return dofs, K, M

    def _assemble_elastic(self, E_eq):
        rows, cols, vals = [], [], []
        geo = self.geo
        for be in geo.beams:
            if be.group == "bearing":
                dofs, K, _ = self._bearing_truss(be)
                self._add_block(rows, cols, vals, dofs, K)
                continue
            n1, n2 = geo.nodes[be.n1], geo.nodes[be.n2]
            L = float(np.hypot(n2.x - n1.x, n2.y - n1.y))
            alpha = self._beam_angle(n1, n2)
            K, _ = el.beam_matrices(be.E, be.A, be.I, be.rho_line, L, alpha)
            self._add_block(rows, cols, vals, self._beam_dofs(be), K)
        for i in range(self.nc):
            EA = E_eq[i] * self.c_A[i]
            K, _ = el.cable_matrices(EA, 0.0, self.c_L[i], self.c_theta[i])
            self._add_block(rows, cols, vals, list(self.c_dof[i]), K)
        return sp.coo_matrix((vals, (rows, cols)),
                             shape=(self.n_dof, self.n_dof)).tocsc()

    def _assemble_mass(self):
        rows, cols, vals = [], [], []
        geo = self.geo
        for be in geo.beams:
            if be.group == "bearing":
                continue                      # massless link
            n1, n2 = geo.nodes[be.n1], geo.nodes[be.n2]
            L = float(np.hypot(n2.x - n1.x, n2.y - n1.y))
            alpha = self._beam_angle(n1, n2)
            _, M = el.beam_matrices(be.E, be.A, be.I, be.rho_line, L, alpha)
            self._add_block(rows, cols, vals, self._beam_dofs(be), M)
        for i in range(self.nc):
            _, M = el.cable_matrices(0.0, self.c_rho[i], self.c_L[i],
                                     self.c_theta[i])
            self._add_block(rows, cols, vals, list(self.c_dof[i]), M)
        return sp.coo_matrix((vals, (rows, cols)),
                             shape=(self.n_dof, self.n_dof)).tocsc()

    def _gravity_vector(self):
        F = np.zeros(self.n_dof)
        geo = self.geo
        for be in geo.beams:
            if be.rho_line == 0.0 or be.group == "bearing":
                continue
            n1, n2 = geo.nodes[be.n1], geo.nodes[be.n2]
            L = float(np.hypot(n2.x - n1.x, n2.y - n1.y))
            alpha = self._beam_angle(n1, n2)
            F[self._beam_dofs(be)] += el.beam_gravity_load(be.rho_line, L, alpha)
        for i in range(self.nc):
            F[self.c_dof[i]] += el.cable_gravity_load(self.c_rho[i], self.c_L[i])
        return F

    def _prestress_matrix(self, E_eq):
        """Equivalent nodal forces per unit cable pre-strain (global)."""
        rows, cols, vals = [], [], []
        for i in range(self.nc):
            f = el.cable_prestress_force(E_eq[i] * self.c_A[i], 1.0,
                                         self.c_L[i], self.c_theta[i])
            for a in range(4):
                rows.append(self.c_dof[i, a]); cols.append(i); vals.append(f[a])
        return sp.coo_matrix((vals, (rows, cols)),
                             shape=(self.n_dof, self.nc)).tocsc()

    # ------------------------------------------------------------------
    # strain / force recovery
    # ------------------------------------------------------------------
    def _cable_strain_free(self, v_free):
        d = v_free[self.c_dof_free]
        c, s = np.cos(self.c_theta), np.sin(self.c_theta)
        return ((d[:, 2] - d[:, 0]) * c + (d[:, 3] - d[:, 1]) * s) / self.c_L

    def _cable_strain_full(self, u_full):
        d = u_full[self.c_dof]
        c, s = np.cos(self.c_theta), np.sin(self.c_theta)
        return ((d[:, 2] - d[:, 0]) * c + (d[:, 3] - d[:, 1]) * s) / self.c_L

    def _cable_tension(self, u_full, eps0, E_eq):
        return E_eq * self.c_A * (self._cable_strain_full(u_full) + eps0)

    def _beam_axial_forces(self, u):
        out = {}
        geo = self.geo
        for be in geo.beams:
            if be.group == "bearing":
                continue
            n1, n2 = geo.nodes[be.n1], geo.nodes[be.n2]
            L = float(np.hypot(n2.x - n1.x, n2.y - n1.y))
            alpha = self._beam_angle(n1, n2)
            c, s = np.cos(alpha), np.sin(alpha)
            du = (u[3 * be.n2] - u[3 * be.n1]) * c + \
                 (u[3 * be.n2 + 1] - u[3 * be.n1 + 1]) * s
            out[be.eid] = be.E * be.A * du / L
        return out

    # ------------------------------------------------------------------
    # shape finding
    # ------------------------------------------------------------------
    def _eps_ref(self):
        """Reference pre-strain from the tributary dead-load demand of each
        cable (merged over the two cable planes)."""
        eps_ref = np.zeros(self.nc)
        anchors = {}
        for c in self.geo.cables:
            anchors.setdefault((c.pylon, c.side), []).append(c.x_deck)
        for c in self.geo.cables:
            xs = sorted(anchors[(c.pylon, c.side)])
            i = xs.index(c.x_deck)
            last = (i == len(xs) - 1)
            # half spacing towards the neighbour, or 8 m at a fan boundary;
            # the mid-span anchor is shared with the mirror cable of the
            # opposite fan, which claims the far-side tributary.
            left = (c.x_deck - xs[i - 1]) / 2.0 if i > 0 else 8.0
            if last:
                right = 0.0 if c.side == "river" else 8.0
            else:
                right = (xs[i + 1] - c.x_deck) / 2.0
            trib = left + right
            m = _deck_line_mass(c.x_deck)
            V = m * 9.81 * trib
            T = V / max(np.sin(self.c_theta[c.eid]), 1e-3)
            eps_ref[c.eid] = T / (P.CABLES["E"] * self.c_A[c.eid])
        return eps_ref

    def _solve_prestrain(self, E_eq, eps_ref):
        """One bounded least-squares solve of the cable pre-strains at the
        equivalent modulus ``E_eq``.  Returns (eps0, u_g, S).

        Row groups (weighted):
        * deck deflection - tight in the main span, loose in the side spans
          (residual side-span sag is absorbed by the design camber);
        * pylon-top drift;
        * horizontal-force balance at each pylon;
        * force fidelity - each stay within ``force_tol`` of its tributary
          design force (this is what keeps the back-stays at the documented
          service level instead of merely inside the stress bounds);
        * regularisation toward the tributary pre-strain state.
        """
        cfg = P.SHAPE_FINDING
        csn = np.abs(np.cos(self.c_theta))
        pyl = np.array([c.pylon for c in self.geo.cables])
        side = np.array([c.side for c in self.geo.cables])
        deck_w_dofs = self.dof_map[np.array([3 * n + 1 for n in self.deck_nids])]
        top_u_dofs = self.dof_map[np.array([3 * n for n in self.pylon_top_nids])]

        # per-cable bounds: service-stress band, further capped by the
        # documented maximum stay force (per cable plane, with a small
        # modelling allowance) - this redistributes the pylon H-balance
        # over all shore stays instead of dumping it on the flattest two
        lo = cfg["stress_lo"] * P.CABLES["fu"] / E_eq
        hi = np.minimum(cfg["stress_hi"] * P.CABLES["fu"],
                        cfg["force_cap_margin"] * 2.0 *
                        cfg["force_cap_plane"]) / (E_eq * self.c_A)
        K = self._assemble_elastic(E_eq)
        lu = spla.splu(K[self.free][:, self.free].tocsc())
        u_g = lu.solve(self.F_g[self.free])
        S = lu.solve(np.asarray(self._prestress_matrix(E_eq)[self.free].todense()))

        # linear tension map  N(eps0) = N0 + Jn @ eps0
        N0 = E_eq * self.c_A * self._cable_strain_free(u_g)
        Jn = np.zeros((self.nc, self.nc))
        for j in range(self.nc):
            Jn[:, j] = E_eq * self.c_A * self._cable_strain_free(S[:, j])
        Jn[np.arange(self.nc), np.arange(self.nc)] += E_eq * self.c_A

        w_deck = np.where(
            (self.deck_xs > P.PYLON_X[0]) & (self.deck_xs < P.PYLON_X[1]),
            cfg["w_deck_main"], cfg["w_deck_side"])
        A_rows = [S[deck_w_dofs, :] * w_deck[:, None]]
        b_rows = [-u_g[deck_w_dofs] * w_deck]
        A_rows.append(cfg["w_pylon_drift"] * S[top_u_dofs, :])
        b_rows.append(-cfg["w_pylon_drift"] * u_g[top_u_dofs])
        for p in (0, 1):
            riv = (pyl == p) & (side == "river")
            sho = (pyl == p) & (side == "shore")
            H_ref = max(np.sum(np.clip(N0[riv], 0, None) * csn[riv]) +
                        np.sum(np.clip(N0[sho], 0, None) * csn[sho]), 1.0)
            const = np.sum(N0[riv] * csn[riv]) - np.sum(N0[sho] * csn[sho])
            coeff = (np.sum(Jn[riv] * csn[riv][:, None], axis=0) -
                     np.sum(Jn[sho] * csn[sho][:, None], axis=0))
            scale = cfg["w_h_balance"] * H_ref
            A_rows.append((coeff / scale)[None, :])
            b_rows.append(np.array([-const / scale]))
        # force fidelity toward the tributary design forces
        T_trib = E_eq * self.c_A * eps_ref
        tolF = cfg["force_tol"] * np.maximum(T_trib, 1.0)
        A_rows.append((Jn / tolF[:, None]) * cfg["w_force"])
        b_rows.append(((T_trib - N0) / tolF) * cfg["w_force"])
        # regularisation toward the tributary pre-strains
        A_rows.append(cfg["w_regular"] * np.eye(self.nc))
        b_rows.append(cfg["w_regular"] * eps_ref)

        A_ls = np.vstack(A_rows)
        b_ls = np.concatenate([np.atleast_1d(r) for r in b_rows])
        sol = lsq_linear(A_ls, b_ls, bounds=(lo, hi), tol=1e-12,
                         max_iter=300)
        return sol.x, u_g, S

    def shape_find(self, iters=None, verbose=False):
        """Bounded least-squares determination of the cable pre-strains.

        The bounds are expressed in force space through the current Ernst
        modulus, so a softer effective modulus automatically admits a larger
        pre-strain for the same service stress; the modulus update itself is
        damped to keep the iteration stable.  After the Ernst iteration
        converges, one more bounded solve is performed AT the converged
        modulus, so the service-stress bounds hold exactly in the delivered
        state (and the two stays sharing the mid-span anchor node are
        re-symmetrised by the regularisation).
        """
        cfg = P.SHAPE_FINDING
        iters = iters or cfg["ernst_iters"]
        eps_ref = self._eps_ref()

        E_eq = self.E_eq.copy()
        for it in range(iters):
            self.eps0, u_g, S = self._solve_prestrain(E_eq, eps_ref)
            u_full = np.zeros(self.n_dof)
            u_full[self.free] = u_g + S @ self.eps0
            self.u = u_full
            self.tension = E_eq * self.c_A * \
                (self._cable_strain_full(u_full) + self.eps0)

            E_target = np.array([
                el.ernst_modulus(P.CABLES["E"], P.CABLES["rho"], self.c_A[i],
                                 self.c_Lx[i], max(self.tension[i], 1.0))
                for i in range(self.nc)])
            if verbose:
                print(f"    shape-finding iter {it + 1}: "
                      f"max|w_deck| = {self.max_deck_deflection() * 100:.2f} cm, "
                      f"sigma/fu in [{self.stress_ratios().min():.3f}, "
                      f"{self.stress_ratios().max():.3f}]")
            if np.max(np.abs(E_target - E_eq) / P.CABLES["E"]) < 1e-4:
                E_eq = E_target
                break
            damp = cfg["ernst_damp"]
            E_eq = E_eq + damp * (E_target - E_eq)

        # ---- final bounded solve at the converged modulus ---------------
        self.eps0, _, _ = self._solve_prestrain(E_eq, eps_ref)

        # ---- final consistent state ------------------------------------
        self.E_eq = E_eq
        self.K_el = self._assemble_elastic(E_eq)
        RHS = self.F_g + self._prestress_matrix(E_eq).dot(self.eps0)
        u = np.zeros(self.n_dof)
        u[self.free] = spla.spsolve(
            self.K_el[self.free][:, self.free].tocsc(), RHS[self.free])
        self.u = u
        self.tension = self._cable_tension(u, self.eps0, E_eq)
        self.beam_axial = self._beam_axial_forces(u)
        self._compute_reactions(RHS)
        return self

    # ------------------------------------------------------------------
    # post-damage equilibrium (frozen pre-strains)
    # ------------------------------------------------------------------
    def reequilibrate(self):
        """Re-solve the dead-load equilibrium after the element properties
        have been modified in place (damage).

        The pre-strains ``self.eps0`` and the Ernst moduli ``self.E_eq`` are
        the as-built values determined by the shape finding of the pristine
        structure: a damage event changes stiffness, not the manufactured
        cable lengths, and the equivalent moduli of the stays are held at
        their as-built dead-load values - the standard modelling practice for
        cable-stayed bridges, since the modulus-tension coupling of the Ernst
        relation is a second-order effect next to the stiffness loss itself.
        (Iterating the modulus to self-consistency with the redistributed
        forces is numerically unreliable for the 544 m river stays of this
        bridge: the sag-stiffness feedback diverges for any relaxation
        factor, so the frozen-modulus solve is also the only robust choice.)

        The post-damage state is therefore ONE exact linear solve of
        gravity plus the manufactured pre-strains on the damaged structure -
        deterministic, free of iteration settings, and identical in every
        solver respect to the healthy state.  With no property modified it
        reproduces the shape-found state bit for bit.
        """
        E_eq = self.E_eq.copy()
        self.K_el = self._assemble_elastic(E_eq)
        RHS = self.F_g + self._prestress_matrix(E_eq).dot(self.eps0)
        u = np.zeros(self.n_dof)
        u[self.free] = spla.spsolve(
            self.K_el[self.free][:, self.free].tocsc(), RHS[self.free])
        self.u = u
        self.tension = self._cable_tension(u, self.eps0, E_eq)
        self.beam_axial = self._beam_axial_forces(u)
        self._compute_reactions(RHS)
        return self

    def equilibrium_residual(self):
        """Relative residual of the current dead-load state on the free
        DOFs (safety net for the scenario pipeline)."""
        RHS = self.F_g + self._prestress_matrix(self.E_eq).dot(self.eps0)
        r = (self.K_el.dot(self.u) - RHS)[self.free]
        return float(np.linalg.norm(r) /
                     max(np.linalg.norm(RHS[self.free]), 1e-30))

    # ------------------------------------------------------------------
    # state queries
    # ------------------------------------------------------------------
    def max_deck_deflection(self):
        w = self.u[[3 * n + 1 for n in self.deck_nids]]
        return float(np.max(np.abs(w)))

    def deck_deflection_profile(self):
        w = self.u[[3 * n + 1 for n in self.deck_nids]]
        return self.deck_xs, w

    def stress_ratios(self):
        return self.tension / (self.c_A * P.CABLES["fu"])

    def pylon_top_drift(self):
        return float(np.max(np.abs(self.u[[3 * n for n in self.pylon_top_nids]])))

    def _compute_reactions(self, RHS):
        R = self.K_el.dot(self.u) - RHS
        self.reactions = R[self.fixed]

    def total_weight(self):
        W = 0.0
        for be in self.geo.beams:
            if be.rho_line == 0.0:
                continue
            n1, n2 = self.geo.nodes[be.n1], self.geo.nodes[be.n2]
            L = float(np.hypot(n2.x - n1.x, n2.y - n1.y))
            W += be.rho_line * 9.81 * L
        for i in range(self.nc):
            W += self.c_rho[i] * 9.81 * self.c_L[i]
        return W

    def reaction_summary(self):
        """Vertical reactions grouped by support family (N)."""
        out = {"pylons": 0.0, "piers": 0.0}
        for n in self.geo.nodes:
            if abs(n.y) > 1e-9:
                continue
            r = self.reactions[self._fixed_pos[3 * n.nid + 1]]
            if n.group == "pylon":
                out["pylons"] += r
            elif n.group == "pier":
                out["piers"] += r
        return out

    # ------------------------------------------------------------------
    # modal analysis
    # ------------------------------------------------------------------
    def assemble_modal(self):
        rows, cols, vals = [], [], []
        geo = self.geo
        for be in geo.beams:
            if be.group == "bearing":
                continue                      # massless link, no sag stiffness
            n1, n2 = geo.nodes[be.n1], geo.nodes[be.n2]
            L = float(np.hypot(n2.x - n1.x, n2.y - n1.y))
            alpha = self._beam_angle(n1, n2)
            Kg = el.beam_geometric_stiffness(self.beam_axial.get(be.eid, 0.0),
                                             L, alpha)
            self._add_block(rows, cols, vals, self._beam_dofs(be), Kg)
        for i in range(self.nc):
            Kg = el.cable_geometric_stiffness(max(self.tension[i], 0.0),
                                              self.c_L[i], self.c_theta[i])
            self._add_block(rows, cols, vals, list(self.c_dof[i]), Kg)
        Kgeo = sp.coo_matrix((vals, (rows, cols)),
                             shape=(self.n_dof, self.n_dof)).tocsc()
        self.M = self._assemble_mass()
        self.K_modal = self.K_el + Kgeo
        return self.K_modal, self.M

    # ------------------------------------------------------------------
    # live-load serviceability check
    # ------------------------------------------------------------------
    def live_load_check(self):
        q = P.LIVE_LOAD["q"]
        F = self.F_g.copy()
        geo = self.geo
        for be in geo.beams:
            if be.group != "deck":
                continue
            n1, n2 = geo.nodes[be.n1], geo.nodes[be.n2]
            xm = 0.5 * (n1.x + n2.x)
            if P.LIVE_LOAD["x_start"] - 1e-6 <= xm <= P.LIVE_LOAD["x_end"] + 1e-6:
                L = float(np.hypot(n2.x - n1.x, n2.y - n1.y))
                F[self._beam_dofs(be)] += el.line_load_vertical(-q, L, 0.0)
        RHS = F + self._prestress_matrix(self.E_eq).dot(self.eps0)
        u = np.zeros(self.n_dof)
        u[self.free] = spla.spsolve(
            self.K_el[self.free][:, self.free].tocsc(), RHS[self.free])
        N = self._cable_tension(u, self.eps0, self.E_eq)
        w_live = u[[3 * n + 1 for n in self.deck_nids]]
        return dict(stress_ratios=N / (self.c_A * P.CABLES["fu"]),
                    min_tension=float(N.min()),
                    max_tension=float(N.max()),
                    max_deflection=float(np.max(np.abs(w_live))))
