"""
Element matrices for the 2-D vertical-plane finite-element model.

Two element families are used:

* ``beam``   - Euler-Bernoulli frame elements with 3 DOFs per node
               (u, w, theta). Used for the girder, pylons and piers.
* ``cable``  - two-node truss elements with 2 DOFs per node, representing a
               stay cable (both cable planes merged) with the Ernst
               equivalent modulus for cable sag and a tension-dependent
               geometric-stiffness term.

All matrices are returned in GLOBAL coordinates; the small rotation of each
element is handled here.
"""

import numpy as np

# DOF order everywhere: [u1, w1, th1, u2, w2, th2] for beams (global) and
# [u1, w1, u2, w2] for trusses (global).


def _rotation_2d(alpha):
    """Rotation matrix T(alpha) mapping LOCAL to GLOBAL dofs
    (u_global = T @ u_local) for a 2-D frame element."""
    c, s = np.cos(alpha), np.sin(alpha)
    T = np.zeros((6, 6))
    T[0, 0] = T[1, 1] = T[3, 3] = T[4, 4] = c
    T[0, 1] = T[3, 4] = -s
    T[1, 0] = T[4, 3] = s
    T[2, 2] = T[5, 5] = 1.0
    return T


def beam_matrices(E, A, I, rho_line, L, alpha):
    """Stiffness, consistent mass and geometric-stiffness matrices of a 2-D
    Euler-Bernoulli frame element in global coordinates.

    Parameters
    ----------
    E, A, I : Young's modulus, area, bending inertia of the section.
    rho_line : mass per unit length (kg/m) of the element.
    L : element length (m).
    alpha : angle of the element axis from the global x axis (rad).
    """
    EA, EI = E * A, E * I

    k = np.zeros((6, 6))
    # axial part (local dofs 0 and 3)
    k[0, 0] = k[3, 3] = EA / L
    k[0, 3] = k[3, 0] = -EA / L
    # bending part (local dofs 1, 2, 4, 5)
    b = EI / L ** 3
    kb = b * np.array([
        [12.0,   6 * L,  -12.0,   6 * L],
        [6 * L,  4 * L ** 2, -6 * L,  2 * L ** 2],
        [-12.0, -6 * L,   12.0,  -6 * L],
        [6 * L,  2 * L ** 2, -6 * L,  4 * L ** 2],
    ])
    for a, i in ((0, 1), (1, 2), (2, 4), (3, 5)):
        for bb, j in ((0, 1), (1, 2), (2, 4), (3, 5)):
            k[i, j] = kb[a, bb]

    m = np.zeros((6, 6))
    ma = rho_line * L / 6.0
    m[0, 0] = m[3, 3] = 2.0 * ma
    m[0, 3] = m[3, 0] = 1.0 * ma
    mb = rho_line * L / 420.0
    mbb = mb * np.array([
        [156.0,     22 * L,   54.0,    -13 * L],
        [22 * L,    4 * L ** 2, 13 * L,  -3 * L ** 2],
        [54.0,      13 * L,   156.0,   -22 * L],
        [-13 * L,  -3 * L ** 2, -22 * L,  4 * L ** 2],
    ])
    for a, i in ((0, 1), (1, 2), (2, 4), (3, 5)):
        for bb, j in ((0, 1), (1, 2), (2, 4), (3, 5)):
            m[i, j] = mbb[a, bb]

    T = _rotation_2d(alpha)
    K = T @ k @ T.T
    M = T @ m @ T.T
    return K, M


def beam_geometric_stiffness(N, L, alpha):
    """Geometric stiffness of a frame element with axial force N
    (tension positive; tension stiffens the transverse response)."""
    kg = np.zeros((6, 6))
    g = N / L
    gg = g * np.array([
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 6.0 / 5.0, L / 10.0, 0.0, -6.0 / 5.0, L / 10.0],
        [0.0, L / 10.0, 2 * L ** 2 / 15.0, 0.0, -L / 10.0, -L ** 2 / 30.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, -6.0 / 5.0, -L / 10.0, 0.0, 6.0 / 5.0, -L / 10.0],
        [0.0, L / 10.0, -L ** 2 / 30.0, 0.0, -L / 10.0, 2 * L ** 2 / 15.0],
    ])
    T = _rotation_2d(alpha)
    return T @ gg @ T.T


def beam_gravity_load(rho_line, L, alpha):
    """Consistent nodal load vector of self weight (global, N).

    The body force (0, -rho*g) is projected onto the local frame, so the
    routine is correct for horizontal, vertical and inclined elements.
    """
    gx, gy = 0.0, -rho_line * 9.81
    c, s = np.cos(alpha), np.sin(alpha)
    p = gx * c + gy * s          # local axial distributed load
    q = -gx * s + gy * c         # local transverse distributed load
    f = np.zeros(6)
    f[0] = p * L / 2.0
    f[3] = p * L / 2.0
    f[1] = q * L / 2.0
    f[2] = q * L ** 2 / 12.0
    f[4] = q * L / 2.0
    f[5] = -q * L ** 2 / 12.0
    T = _rotation_2d(alpha)
    return T @ f


def line_load_vertical(q_line, L, alpha):
    """Consistent nodal load of a vertical distributed load q_line (N/m)
    on a beam element (used for deck dead load and live load)."""
    c, s = np.cos(alpha), np.sin(alpha)
    p = q_line * s               # local axial component
    q = q_line * c               # local transverse component
    f = np.zeros(6)
    f[0] = p * L / 2.0
    f[3] = p * L / 2.0
    f[1] = q * L / 2.0
    f[2] = q * L ** 2 / 12.0
    f[4] = q * L / 2.0
    f[5] = -q * L ** 2 / 12.0
    T = _rotation_2d(alpha)
    return T @ f


def cable_matrices(EA, rho_line, L, theta):
    """Elastic stiffness and consistent mass of a stay-cable truss element.

    theta : angle of the cable axis from the horizontal (rad), pointing from
            the deck node to the pylon node.
    """
    c, s = np.cos(theta), np.sin(theta)
    d = np.array([-c, -s, c, s])
    K = (EA / L) * np.outer(d, d)

    m = np.zeros((4, 4))
    ma = rho_line * L / 6.0
    # consistent mass in local axial dofs rotated to global
    T = np.zeros((4, 4))
    T[0, 0] = T[1, 1] = T[2, 2] = T[3, 3] = c
    T[0, 1] = T[2, 3] = -s
    T[1, 0] = T[3, 2] = s
    ml = np.zeros((4, 4))
    ml[0, 0] = ml[2, 2] = 2.0 * ma
    ml[0, 2] = ml[2, 0] = 1.0 * ma
    M = T @ ml @ T.T
    return K, M


def cable_geometric_stiffness(T_force, L, theta):
    """Geometric stiffness of a taut cable (T_force > 0, tension in N).

    With n = [-sin, cos, sin, -cos] mapping the nodal displacements to the
    transverse elongation of the chord,  K_g = (T/L) * n n^T.
    """
    c, s = np.cos(theta), np.sin(theta)
    n = np.array([-s, c, s, -c])
    return (T_force / L) * np.outer(n, n)


def cable_prestress_force(EA, eps0, L, theta):
    """Equivalent nodal forces of a cable pre-strain eps0 (shortening
    positive, i.e. pre-tension).  Node 1 = deck anchor, node 2 = pylon
    anchor; e = (cos theta, sin theta) points from node 1 to node 2."""
    c, s = np.cos(theta), np.sin(theta)
    T = EA * eps0
    return T * np.array([c, s, -c, -s])


def cable_gravity_load(rho_line, L):
    """Self weight of a cable lumped as nodal forces (N, global)."""
    W = rho_line * L * 9.81
    return np.array([0.0, -W / 2.0, 0.0, -W / 2.0])


def ernst_modulus(E, rho, A, Lx, tension):
    """Ernst equivalent modulus of a sagging stay cable.

    E_eq = E / (1 + (gamma * Lx)^2 * E / (12 * sigma^3 * ... )) in stress
    form; written here with the tension N as

        E_eq = E / (1 + (w * Lx)^2 * E * A / (12 * N^3))

    with w = rho*g*A the cable weight per unit length and Lx the horizontal
    projection of the cable.  Returns E when the tension is non-positive
    (slack cables keep the bare modulus; they are flagged elsewhere).
    """
    if tension <= 0.0:
        return E
    w = rho * 9.81 * A
    ratio = (w * Lx) ** 2 * E * A / (12.0 * tension ** 3)
    return E / (1.0 + ratio)


def segment_intersection(p1, p2, p3, p4, eps=1e-9):
    """True if segments p1-p2 and p3-p4 properly cross (shared endpoints are
    excluded via eps).  Used by the topology checks."""
    d1 = p2 - p1
    d2 = p4 - p3
    denom = d1[0] * d2[1] - d1[1] * d2[0]
    if abs(denom) < eps:
        return False
    t = ((p3 - p1)[0] * d2[1] - (p3 - p1)[1] * d2[0]) / denom
    u = ((p3 - p1)[0] * d1[1] - (p3 - p1)[1] * d1[0]) / denom
    return (eps < t < 1 - eps) and (eps < u < 1 - eps)
