"""
Modal-superposition simulation of the monitoring records.

The dynamic response about the (healthy or damaged) dead-load state is
expanded on the eigenmodes of that state,

    u(x, t) = sum_i  phi_i(x) q_i(t),
    qdd_i + 2 zeta omega_i qd_i + omega_i^2 q_i = phi_i^T f(t),

with uniform modal damping ``zeta`` (2 % of critical, identical for every
mode and every scenario) and the ambient forces of
``excitation.AmbientExcitation``.  Each modal oscillator is integrated by
the exact zero-order-hold discretisation of its state-space form, so the
scheme is unconditionally stable and introduces no numerical damping.

A burn-in interval (600 s at 50 Hz) precedes the 60 s record (200 Hz): the
response of the lowest mode of the bridge, the 0.07 Hz longitudinal
floating mode, needs several minutes of wash-out before the record starts
at a stationary ambient state rather than at a decaying transient.

The recorded signals are the 64 sensor channels of ``sensors.build_channels``
(accelerations in m/s^2) plus white sensor noise.
"""

import numpy as np
import scipy.linalg as sla

from . import params as P


# ---------------------------------------------------------------------------
class ModalSimulator:
    """Exact ZOH modal integration of one scenario's modes."""

    def __init__(self, Phi_loaded, Phi_channels, omega, zeta):
        """
        Parameters
        ----------
        Phi_loaded : (n_loaded, n_modes)
            mode shapes restricted to the loaded free DOFs (force rows).
        Phi_channels : (n_channels, n_modes)
            measurement operator applied to the mode shapes.
        omega : (n_modes,) angular frequencies (rad/s).
        """
        self.PhiL = np.asarray(Phi_loaded)
        self.PhiC = np.asarray(Phi_channels)
        self.omega = np.asarray(omega)
        self.zeta = float(zeta)
        self.n_modes = len(omega)
        self._disc = {}

    # -- discretisation -----------------------------------------------------
    def _discretisation(self, dt):
        if dt not in self._disc:
            Ad = np.empty((self.n_modes, 2, 2))
            Bd = np.empty((self.n_modes, 2))
            for i in range(self.n_modes):
                w, z = self.omega[i], self.zeta
                A = np.array([[0.0, 1.0], [-w * w, -2.0 * z * w]])
                Ad[i] = sla.expm(A * dt)
                A_inv = np.linalg.inv(A)
                Bd[i] = A_inv @ (Ad[i] - np.eye(2)) @ np.array([0.0, 1.0])
            self._disc[dt] = (Ad, Bd)
        return self._disc[dt]

    # -- one integration step over a block of modal forces ------------------
    def _integrate_block(self, pblock, dt, store=None):
        """Integrate the modal states through a block of modal forces
        (n_modes x n_steps, ZOH per step).  Optionally store the modal
        accelerations of every step."""
        Ad, Bd = self._discretisation(dt)
        a00 = Ad[:, 0, 0]; a01 = Ad[:, 0, 1]
        a10 = Ad[:, 1, 0]; a11 = Ad[:, 1, 1]
        b0 = Bd[:, 0]; b1 = Bd[:, 1]
        c_damp = 2.0 * self.zeta * self.omega
        c_stiff = self.omega ** 2
        q = self.q; qd = self.qd
        if store is not None:
            acc = np.empty((self.n_modes, pblock.shape[1]))
            for k in range(pblock.shape[1]):
                pk = pblock[:, k]
                acc[:, k] = pk - c_damp * qd - c_stiff * q
                qn = a00 * q + a01 * qd + b0 * pk
                qdn = a10 * q + a11 * qd + b1 * pk
                q, qd = qn, qdn
            self.q, self.qd = q, qd
            store[:] = acc
        else:
            for k in range(pblock.shape[1]):
                pk = pblock[:, k]
                qn = a00 * q + a01 * qd + b0 * pk
                qdn = a10 * q + a11 * qd + b1 * pk
                q, qd = qn, qdn
            self.q, self.qd = q, qd

    # -- full record ---------------------------------------------------------
    def simulate(self, excitation, T_burn, fs_burn, T_rec, fs_rec,
                 chunk_s=20.0):
        """Run burn-in + record and return the channel accelerations.

        ``excitation.forces(t)`` supplies the forces at master-grid times;
        burn-in and record may use different integration rates (the times
        must lie on the master grid of the excitation).
        """
        self.q = np.zeros(self.n_modes)
        self.qd = np.zeros(self.n_modes)

        # burn-in (no storage needed, only the end state matters)
        t0 = 0.0
        dt_b = 1.0 / fs_burn
        n_step_b = int(round(T_burn * fs_burn))
        n_chunk_b = max(int(round(chunk_s * fs_burn)), 1)
        for start in range(0, n_step_b, n_chunk_b):
            n = min(n_chunk_b, n_step_b - start)
            t = (t0 + start + np.arange(n)) * dt_b
            F = excitation.forces(t)
            self._integrate_block(self.PhiL.T @ F, dt_b)

        # record
        n_step_r = int(round(T_rec * fs_rec))
        dt_r = 1.0 / fs_rec
        n_chunk_r = max(int(round(chunk_s * fs_rec)), 1)
        acc = np.empty((self.n_modes, n_step_r))
        for start in range(0, n_step_r, n_chunk_r):
            n = min(n_chunk_r, n_step_r - start)
            t = T_burn + (start + np.arange(n)) * dt_r
            F = excitation.forces(t)
            block = np.ascontiguousarray(acc[:, start:start + n])
            self._integrate_block(self.PhiL.T @ F, dt_r, store=block)
            acc[:, start:start + n] = block

        return self.PhiC @ acc


# ---------------------------------------------------------------------------
def simulate_record(model, modes, rng, T_burn=None, fs_burn=None,
                    fs_rec=None, T_rec=None, chunk_s=20.0):
    """Simulate one seeded record of a scenario.

    Returns ``(channels, acc)``: the channel table and the (64 x N) noise
    -free acceleration matrix in m/s^2.
    """
    cfg = P.EXCITATION
    T_burn = cfg["burn_in_s"] if T_burn is None else T_burn
    fs_burn = cfg["burn_in_fs"] if fs_burn is None else fs_burn
    fs_rec = cfg["fs"] if fs_rec is None else fs_rec
    T_rec = cfg["duration"] if T_rec is None else T_rec

    from .sensors import build_channels
    from .excitation import AmbientExcitation

    channels, R = build_channels(model)

    Phi = np.column_stack([m["phi_free"] for m in modes])
    omega = 2.0 * np.pi * np.array([m["f"] for m in modes])

    exc = AmbientExcitation(model, rng, T_burn + T_rec, fs_rec)
    rows = exc.loaded_rows()
    PhiL = Phi[rows, :]

    sim = ModalSimulator(PhiL, R @ Phi, omega, cfg["zeta"])
    acc = sim.simulate(exc, T_burn, fs_burn, T_rec, fs_rec,
                       chunk_s=chunk_s)
    return channels, acc


def add_sensor_noise(acc, rng, std=None):
    cfg = P.EXCITATION
    std = cfg["noise_std"] if std is None else std
    return acc + std * rng.standard_normal(acc.shape)
