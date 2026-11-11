"""
Blind operational modal identification of the simulated records.

Three independent identification engines are provided, all operating on the
raw multi-channel accelerations without any knowledge of the underlying
finite-element model:

* frequency-domain decomposition (FDD) - SVD of the cross-spectral
  matrix at every frequency line, peak-picking of the first singular
  value, mode shape = first singular vector at the peak;
* covariance-driven stochastic subspace identification (SSI-COV) -
  block-Toeplitz covariance of the reference channels, observability
  matrix from its SVD, state matrix by shifted-block least squares, poles
  from the eigenvalues, stabilisation over increasing model order;
* the least-squares complex-frequency-domain common-denominator fit
  (p-LSCF in its scalar form) - a shared denominator polynomial fitted to
  the reference-channel half-spectra, poles from its roots, stabilisation
  over the three records of a scenario.

They exist so the frequency-domain quantities used by the study (the
modal-amplitude ratio in the feature vector, and the identified-vs-FE
frequency table) come from real identification runs rather than from the
simulation kernel that generated the data.
"""

import numpy as np

from . import params as P


# ---------------------------------------------------------------------------
# shared helpers
# ---------------------------------------------------------------------------
def cross_psd_matrix(record, fs, nperseg, nfft=None):
    """One-sided cross-PSD tensor of a record: (n_ch, n_ch, bins).

    Welch with a Hann window and 50 % overlap; ``record`` is
    (n_ch, n_samples).
    """
    if nfft is None:
        nfft = nperseg
    step = nperseg // 2
    n_seg = (record.shape[1] - nperseg) // step + 1
    win = np.hanning(nperseg)
    X = np.empty((record.shape[0], n_seg, nfft // 2 + 1), dtype=complex)
    for s in range(n_seg):
        s0 = s * step
        block = record[:, s0: s0 + nperseg] * win[None, :]
        X[:, s] = np.fft.rfft(block, n=nfft, axis=1)
    Sxx = np.einsum("isk,jsk->ijk", X, np.conj(X)) / (
        n_seg * np.sum(win ** 2))
    f = np.fft.rfftfreq(nfft, 1.0 / fs)
    return f, Sxx


def _pole_from_discrete(mu, fs):
    """(f_n Hz, zeta) of a discrete state eigenvalue mu."""
    ln = np.log(mu.astype(complex))
    wn = abs(ln) * fs                    # rad/s, undamped
    if wn <= 0:
        return 0.0, 1.0
    zeta = -ln.real / abs(ln)
    return abs(wn) / (2.0 * np.pi), float(np.clip(zeta, -1.0, 1.0))


# ---------------------------------------------------------------------------
# FDD
# ---------------------------------------------------------------------------
def fdd_first_sv_peaks(records, fs, band, n_targets=12):
    """The ``n_targets`` largest peaks of the first singular value of the
    cross-spectral matrix inside ``band``, with their shapes (first
    singular vectors).

    ``records`` is (n_rec, n_ch, n_samples); every record contributes one
    Hann-windowed full-length periodogram and the cross-PSD matrices are
    averaged over the records (the standard FDD practice when several
    records of the same state exist).  With 60 s records the window width
    resolves the closest mode pair of this bridge (0.183 / 0.227 Hz).

    Returns a list of (f, s1, shape) tuples, largest first.
    """
    n = records.shape[2]
    nfft = 4 * n
    win = np.hanning(n)
    f = np.fft.rfftfreq(nfft, 1.0 / fs)
    sel = np.flatnonzero((f >= band[0]) & (f <= band[1]))
    Ssum = np.zeros((records.shape[1], records.shape[1], len(sel)),
                    dtype=complex)
    for rec in range(records.shape[0]):
        X = np.fft.rfft(records[rec] * win[None, :], n=nfft, axis=1)
        Ssum += np.einsum("ik,jk->ijk", X[:, sel], np.conj(X[:, sel]))
    Ssum /= records.shape[0] * np.sum(win ** 2)

    s1 = np.empty(len(sel))
    shapes = np.empty((len(sel), records.shape[1]), dtype=complex)
    for q in range(len(sel)):
        _, sv, vt = np.linalg.svd(Ssum[:, :, q])
        s1[q] = sv[0]
        shapes[q] = vt[0].conj()
    fl = f[sel]
    peaks = []
    for q in range(1, len(sel) - 1):
        if s1[q] > s1[q - 1] and s1[q] >= s1[q + 1]:
            d = 0.5 * (np.log(s1[q + 1]) - np.log(s1[q - 1])) / (
                np.log(s1[q - 1]) - 2 * np.log(s1[q]) + np.log(s1[q + 1])
                + 1e-30)
            d = float(np.clip(d, -1.0, 1.0))
            f_pk = fl[q] + d * (fl[1] - fl[0])
            peaks.append((float(f_pk), float(s1[q]), shapes[q]))
    peaks.sort(key=lambda t: -t[1])
    return peaks[:n_targets]


# ---------------------------------------------------------------------------
# SSI-COV
# ---------------------------------------------------------------------------
def ssi_cov(record, fs, ref_idx, i_blocks=80, orders=None, f_band=None):
    """Covariance-driven SSI on the reference channels.

    The records are decimated to an effective rate of at most 25 Hz
    (poly-phase FIR anti-alias filtering) before the covariance sequence
    is formed, so the block lags span several periods of the sub-1 Hz
    bridge modes; with 80 block rows of 8 reference channels the lags
    reach 6.4 s, more than one full period of the 0.19 Hz first vertical.

    Returns the stabilised pole list [(f, zeta), ...].  ``i_blocks`` is
    the number of block rows of the Toeplitz matrix; ``orders`` the model
    orders scanned for stabilisation.
    """
    from scipy.signal import resample_poly

    y = record[list(ref_idx)].astype(np.float64)
    q_dec = max(1, int(fs // 25))
    if q_dec > 1:
        y = resample_poly(y, 1, q_dec, axis=1)
        fs_eff = fs / q_dec
    else:
        fs_eff = fs
    r, n = y.shape
    i_max = i_blocks
    lags = np.arange(0, 2 * i_max)      # k = 0 .. 2i-1 (Toeplitz reach)
    # covariance sequence R_k (r x r), k = 0 .. 2i-2
    R = np.empty((len(lags), r, r))
    for q, k in enumerate(lags):
        a = y[:, : n - k] if k > 0 else y
        b = y[:, k:]
        R[q] = (a @ b.T) / (n - k)

    def toeplitz_blocks(i):
        # T[p, q] = R_{i + p - q}  (Peeters & De Roeck stacking)
        T = np.empty((i * r, i * r), dtype=float)
        for p in range(i):
            for q in range(i):
                k = i + p - q
                T[p * r:(p + 1) * r, q * r:(q + 1) * r] = R[k]
        return T

    if orders is None:
        orders = list(range(4, 2 * i_max - 1, 2))
    if f_band is None:
        f_band = (0.05, 1.5)

    prev_poles = None
    stable = []
    T = toeplitz_blocks(i_max)
    U, S, _ = np.linalg.svd(T, full_matrices=False)
    all_poles = []
    n_orders_used = 0
    for n_state in orders:
        n_st = min(n_state, len(S))
        O = U[:, : n_st] * np.sqrt(S[: n_st])       # (i r x n_st)
        if O.shape[0] < 2 * r:
            continue
        n_orders_used += 1
        O_low, O_up = O[:- r], O[r:]
        # O_up = A O_low  (least squares)
        A = np.linalg.pinv(O_low) @ O_up
        mus = np.linalg.eigvals(A)
        for mu in mus:
            if abs(mu.imag) <= 1e-12:
                continue
            f_n, zeta = _pole_from_discrete(mu, fs_eff)
            if f_band[0] <= f_n <= f_band[1] and 0.0 < zeta < 0.05:
                all_poles.append((f_n, zeta))

    # stabilisation: physical poles are pooled over all scanned orders and
    # clustered by frequency (gap < 0.5 %); a cluster counts as stable
    # when it collects poles from a large fraction of the scanned orders,
    # the stabilisation-diagram column of the classical implementations
    all_poles.sort()
    clusters = []
    for f_n, zeta in all_poles:
        if clusters and abs(f_n - clusters[-1][0]) / clusters[-1][0] < 0.005:
            c = clusters[-1]
            c[0] = (c[0] * c[2] + f_n) / (c[2] + 1)
            c[1] = (c[1] * c[2] + zeta) / (c[2] + 1)
            c[2] += 1
        else:
            clusters.append([f_n, zeta, 1])
    min_count = max(3, int(0.4 * n_orders_used))
    stable = [(f, z) for f, z, cnt in clusters if cnt >= min_count]
    return [(round(f, 4), round(z, 4)) for f, z in sorted(stable)]


# ---------------------------------------------------------------------------
# p-LSCF (scalar common-denominator LSCF)
# ---------------------------------------------------------------------------
def p_lscf(record, fs, band, order=28):
    """Poles of the scalar common-denominator fit on the channel
    half-spectra (the least-squares complex-frequency-domain method in
    its monic single-denominator form).

    The records are decimated to an effective rate of 5 Hz before the
    fit: the bridge modes sit far below the 100 Hz Nyquist of the raw
    records, and a polynomial basis on the unit circle is too
    ill-conditioned there to stabilise sub-0.25 Hz poles at practical
    orders.  Even at 5 Hz the method stabilises only the strongest of
    the sub-0.7 Hz modes - an honest negative capability that the
    comparison table reports as it is.

    Returns [(f, zeta), ...] inside ``band``.
    """
    from scipy.signal import resample_poly

    q_dec = max(1, int(round(fs / 5.0)))
    yd = resample_poly(record.astype(np.float64), 1, q_dec, axis=1)
    fs_d = fs / q_dec
    nperseg = yd.shape[1]
    nfft = 4 * nperseg
    f, Sxx = cross_psd_matrix(yd, fs_d, nperseg, nfft)
    sel = np.flatnonzero((f >= band[0]) & (f <= band[1]))
    fgrid = f[sel]
    Y = np.stack([Sxx[k, k, sel] for k in range(yd.shape[0])])

    # basis z^j on the unit circle at the selected lines
    z = np.exp(2j * np.pi * fgrid / fs_d)
    Phi = np.vander(z, order + 1, increasing=True)

    # eliminated-numerator LS residual, stacked over the channels
    Mlist = []
    for k in range(Y.shape[0]):
        D = Phi * Y[k][:, None]
        P = Phi @ np.linalg.pinv(Phi)
        Mlist.append(D - P @ D)
    M = np.vstack(Mlist)

    # monic constraint on the highest-power coefficient
    M0, mv = M[:, :-1], M[:, -1]
    N_mat = np.conj(M0.T) @ M0
    rhs = -np.conj(M0.T) @ mv
    a = np.concatenate([np.linalg.solve(N_mat + 1e-9 * np.eye(order), rhs),
                        [1.0]])
    roots = np.roots(a)
    poles = []
    for mu in roots:
        if abs(mu.imag) <= 1e-9:
            continue
        f_n, zeta = _pole_from_discrete(mu, fs_d)
        if band[0] <= f_n <= band[1] and 0.0 < zeta < 0.05:
            poles.append((f_n, zeta))
    poles.sort()
    return poles


def stabilise_poles(per_record_poles, tol_f=0.025, min_records=2):
    """Cross-record stabilisation of identification results.

    The pole lists of the individual records (each already stabilised
    over the internal model orders) are pooled and clustered with a
    relative frequency tolerance; a cluster is kept when it collects
    poles from at least ``min_records`` records.  The tolerance is wider
    than the intra-record stabilisation band because the pole scatter
    between 60 s ambient records of the same state is 2-4 %.
    """
    entries = []
    for i, poles in enumerate(per_record_poles):
        for f_n, zeta in poles:
            entries.append((f_n, zeta, i))
    entries.sort()
    clusters = []
    for f_n, zeta, rec in entries:
        if clusters and abs(f_n - clusters[-1][0]) / clusters[-1][0] < tol_f:
            c = clusters[-1]
            c[0] = (c[0] * c[3] + f_n) / (c[3] + 1)
            c[1] = (c[1] * c[3] + zeta) / (c[3] + 1)
            c[2].add(rec)
            c[3] += 1
        else:
            clusters.append([f_n, zeta, {rec}, 1])
    return [(round(f, 4), round(z, 4)) for f, z, recs, _ in clusters
            if len(recs) >= min_records]
