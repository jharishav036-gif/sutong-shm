"""
Damage-sensitive feature extraction from the simulated records.

Every 20 s window of every record is transformed into the 387-dimensional
feature vector of the study:

* 320 CWT energies - the continuous wavelet transform of each of the 64
  channels with the complex Morlet wavelet (w0 = 6), evaluated at 50
  logarithmically spaced scales whose pseudo-frequencies cover 0.05-10 Hz
  exactly, aggregated into the five energy bands (0.05-0.20, 0.20-0.50,
  0.50-1.00, 1.00-2.00, 2.00-10.00 Hz) - 64 x 5 = 320 numbers per window;
* 64 Hilbert envelope decays - the least-squares slope of log|z(t)| of the
  analytic signal over the window, one per channel;
* 1 SVD entropy - the Shannon entropy of the normalised squared singular
  values of the 64 x 4000 response matrix of the window;
* 1 cross-spectral phase deviation - the mean over the adjacent
  deck-vertical sensor pairs of the standard deviation of the cross-spectral
  phase over 0.1-2 Hz;
* 1 modal-amplitude ratio - the ratio of the blindly identified mode-1
  (longitudinal floating) amplitude at the deck mid-span longitudinal
  sensor to the identified mode-2 (first vertical) amplitude at the deck
  mid-span vertical sensor (frequency-domain decomposition).

Numerical notes
---------------
The CWT is evaluated by FFT convolution over the full 60 s record
extended by a symmetric reflection of at least 60 s (and at least the
wavelet half-support, whichever is larger) on each side, so the
convolution stays linear over the analysed windows; the residual edge
sensitivity of the end windows at the lowest scales is accepted and
documented rather than hidden.  Band energies are integrated
as dt * sum |W(a, t)|^2 with the L2-normalised wavelet
W(a, b) = a^-1/2 sum x(t) psi*((t - b) / a), the standard Torrance-Compo
convention; the scale-frequency mapping f = (w0 + sqrt(2 + w0^2)) / (4 pi s)
follows the practical guide the study cites.
"""

import numpy as np
import scipy.fft as sfft
from scipy.signal import hilbert

from . import params as P


# ---------------------------------------------------------------------------
# Morlet CWT machinery
# ---------------------------------------------------------------------------
def morlet_scale_frequency(s_sec, w0):
    """Pseudo-frequency (Hz) of a Morlet wavelet of scale ``s_sec`` seconds."""
    return (w0 + np.sqrt(2.0 + w0 * w0)) / (4.0 * np.pi * s_sec)


def morlet_scales(fs, w0, n_scales, f_lo, f_hi):
    """``n_scales`` log-spaced scales whose pseudo-frequencies span the band.

    Returns ``(scales_sec, freqs_hz)``; the frequencies run from ``f_hi``
    down to ``f_lo`` because the scale grows as the frequency falls.
    """
    const = (w0 + np.sqrt(2.0 + w0 * w0)) / (4.0 * np.pi)
    s_hi = const / f_lo          # largest scale  -> lowest frequency
    s_lo = const / f_hi          # smallest scale -> highest frequency
    scales = np.geomspace(s_lo, s_hi, n_scales)
    freqs = const / scales
    return scales, freqs


def _morlet_wavelet(s_samples, w0):
    """The L2-normalised Morlet daughter psi_a(t) = a^-1/2 psi(t / a),
    sampled at the record rate, centred at zero time.

    The wavelet is truncated at |t| > 3.5 a, where the envelope is below
    e^(-6.125) ~ 0.2 % of its peak (an energy truncation of ~4e-6).
    """
    a = float(s_samples)
    half = int(np.ceil(3.5 * a))
    t = np.arange(-half, half + 1)
    arg = t / a
    psi = (np.pi ** -0.25) * np.exp(1j * w0 * arg) * np.exp(-0.5 * arg ** 2)
    return psi.astype(np.complex128)


def _wavelet_frame(s_samples, w0, n_fft, pad):
    """FFT of the Morlet daughter of one scale, centred at the frame index
    ``pad`` (the time origin of the record inside the padded frame)."""
    psi = _morlet_wavelet(s_samples, w0)
    frame = np.zeros(n_fft, dtype=np.complex128)
    start = pad - (len(psi) - 1) // 2
    frame[start: start + len(psi)] = psi
    return np.fft.fft(frame, n_fft)


def record_scale_powers(x, scales, w0, pad_min=0):
    """|W(a, t)|^2 spectrograms of a 1-D signal, one row per scale.

    The signal is extended by symmetric reflection on each side before
    the FFT convolution; the reflection padding is at least ``pad_min``
    samples and at least the truncated half-support of the largest
    wavelet, whichever is larger, so the convolution stays linear over
    the whole original range.  The returned power has the length of the
    original signal (the padding is discarded).  ``scales`` are in
    samples.

    The CWT W(a, b) = sum x(t) psi*((t - b)/a) is the correlation of the
    signal with the complex wavelet: ifft(fft(x) conj(fft(psi))); with
    the wavelet centred at ``pad`` the CWT value of record sample j
    appears at output index j (verified against the direct sum).
    """
    n = len(x)
    half_max = int(np.ceil(3.5 * float(np.max(scales))))
    pad = int(max(pad_min, half_max + 2))
    n_fft = int(2 ** np.ceil(np.log2(n + 2 * pad)))
    ext = np.concatenate([x[pad:0:-1], x, x[n - 2: n - 2 - pad:-1]])
    Xc = np.fft.fft(ext, n_fft)
    PSIs = [np.conj(_wavelet_frame(s, w0, n_fft, pad)) for s in scales]

    rows = np.empty((len(scales), n))
    for k in range(len(scales)):
        W = np.fft.ifft(Xc * PSIs[k])
        rows[k] = np.abs(W[: n]) ** 2
    return rows


def cwt_scale_spectrogram(x, scales, w0, pad_min=0):
    """|W(a, t)|^2 spectrograms of a 1-D signal, one row per scale.

    See :func:`record_scale_powers` for the padding and the alignment
    convention; this wrapper is kept for single-signal use (figures).
    """
    return record_scale_powers(x, scales, w0, pad_min)


# ---------------------------------------------------------------------------
# feature families
# ---------------------------------------------------------------------------
def cwt_band_energies(record, fs, channels_kinds):
    """Band energies of one record: ``(n_windows, 64, 5)``.

    ``record`` is the (64, n_samples) acceleration matrix.  The scalogram
    is computed once per channel over the whole record and summed inside
    each window and each frequency band.
    """
    F = P.FEATURES
    w0 = F["cwt_w0"]
    scales_sec, freqs = morlet_scales(
        fs, w0, F["cwt_n_scales"], F["cwt_f_lo"], F["cwt_f_hi"])
    scales_samples = scales_sec * fs
    bands = F["bands"]

    # band index of every scale; the bands partition the scale range
    # (half-open [lo, hi), the last band closed at the 10 Hz top edge)
    inner_edges = np.array([b[1] for b in bands[:-1]])
    band_of = np.searchsorted(inner_edges, freqs, side="right")
    if freqs.min() < bands[0][0] or freqs.max() > bands[-1][1]:
        raise ValueError("scale frequency outside the band partition")

    pad = int(round(F["cwt_pad_s"] * fs))          # floor; extended by
    n_win = F["n_windows"]                          # the wavelet support
    n_win_samples = int(round(F["window_s"] * fs))
    dt = 1.0 / fs

    # wavelet spectra shared by every channel of the record.  The CWT
    # kernel runs in single precision (scipy pocketfft): the relative
    # accuracy of the band-energy sums is ~1e-4, negligible against the
    # feature standardisation of the learning stage.
    half_max = int(np.ceil(3.5 * float(np.max(scales_samples))))
    pad = int(max(pad, half_max + 2))
    n = record.shape[1]
    n_fft = int(2 ** np.ceil(np.log2(n + 2 * pad)))
    PSIs = np.conj(np.stack(
        [_wavelet_frame(s, w0, n_fft, pad)
         for s in scales_samples])).astype(np.complex64)

    win_slices = [slice(w * n_win_samples, (w + 1) * n_win_samples)
                  for w in range(n_win)]
    out = np.zeros((n_win, len(channels_kinds), len(bands)))
    for ch in range(len(channels_kinds)):
        x = np.ascontiguousarray(record[ch], dtype=np.float32)
        ext = np.concatenate([x[pad:0:-1], x, x[n - 2: n - 2 - pad:-1]])
        Xc = sfft.fft(ext, n_fft)
        # accumulate the band-window sums directly, one scale at a time
        for k in range(len(PSIs)):
            W = sfft.ifft(Xc * PSIs[k])
            pw = np.abs(W[: n]) ** 2
            j = band_of[k]
            for w, sl in enumerate(win_slices):
                out[w, ch, j] += dt * float(np.sum(pw[sl], dtype=np.float64))
    return out


def hilbert_decay(record, fs):
    """LS slope of log|z(t)| per channel per window: ``(n_windows, 64)``."""
    F = P.FEATURES
    eps = F["hilbert_eps"]
    n_win = F["n_windows"]
    n_win_samples = int(round(F["window_s"] * fs))
    t = np.arange(n_win_samples) / fs
    out = np.zeros((n_win, record.shape[0]))
    for w in range(n_win):
        seg = record[:, w * n_win_samples: (w + 1) * n_win_samples]
        for ch in range(seg.shape[0]):
            z = hilbert(seg[ch].astype(np.float64))
            env = np.abs(z) + eps
            out[w, ch] = float(np.polyfit(t, np.log(env), 1)[0])
    return out


def svd_entropy(record, fs):
    """Shannon entropy (nats) of the normalised squared singular values
    of the window response matrices: ``(n_windows,)``."""
    F = P.FEATURES
    n_win = F["n_windows"]
    n_win_samples = int(round(F["window_s"] * fs))
    out = np.zeros(n_win)
    for w in range(n_win):
        X = record[:, w * n_win_samples: (w + 1) * n_win_samples]
        s = np.linalg.svdvals(X.astype(np.float64))
        p = s ** 2
        p /= p.sum()
        out[w] = float(-np.sum(p * np.log(p)))
    return out


def phase_deviation(record, fs, pair_index):
    """Cross-spectral phase deviation of adjacent sensor pairs.

    ``pair_index`` lists the (i, j) channel pairs entering the statistic
    (the adjacent deck-vertical sensors of the layout, in x order).  For
    every pair the standard deviation of arg S_ij(f) over the 0.1-2 Hz band
    is computed with Welch cross-PSDs; the scalar is the mean over pairs.
    Returns ``(n_windows,)``.
    """
    F = P.FEATURES
    n_win = F["n_windows"]
    n_win_samples = int(round(F["window_s"] * fs))
    nperseg = F["phase_nperseg"]
    out = np.zeros(n_win)
    for w in range(n_win):
        seg = record[:, w * n_win_samples: (w + 1) * n_win_samples]
        f, X = _segment_fft(seg, fs, nperseg)
        sel = (f >= F["phase_f_lo"]) & (f <= F["phase_f_hi"])
        devs = []
        for i, j in pair_index:
            S = np.mean(X[i] * np.conj(X[j]), axis=0)
            phi = np.angle(S[sel])
            devs.append(float(np.std(phi)))
        out[w] = float(np.mean(devs))
    return out


def _segment_fft(seg, fs, nperseg):
    """Segmented FFTs of a multi-channel window (Hann window, 50 % overlap).

    Returns ``(f, X)`` with ``X`` of shape (n_channels, n_segments, bins).
    """
    step = nperseg // 2
    n_seg = (seg.shape[1] - nperseg) // step + 1
    win = np.hanning(nperseg)
    X = np.empty((seg.shape[0], n_seg, nperseg // 2 + 1), dtype=complex)
    for s in range(n_seg):
        s0 = s * step
        block = seg[:, s0: s0 + nperseg] * win[None, :]
        X[:, s] = np.fft.rfft(block, axis=1)
    f = np.fft.rfftfreq(nperseg, 1.0 / fs)
    return f, X


def _fdd_singular_peak(seg, fs, band):
    """First singular value spectrum of a window and its peak inside a
    frequency band.

    Returns ``(f_peak, s1_peak, u_peak)`` - the frequency, the first
    singular value and the first singular vector (64-channel mode shape
    estimate) at the peak of the FDD singular-value spectrum.
    """
    F = P.FEATURES
    nfft = F["fdd_nfft_mult"] * seg.shape[1]
    win = np.hanning(seg.shape[1])
    X = np.fft.rfft(seg * win[None, :], n=nfft, axis=1)   # (ch, bins)
    Sxx = np.einsum("ik,jk->ijk", X, np.conj(X))
    f = np.fft.rfftfreq(nfft, 1.0 / fs)
    sel = (f >= band[0]) & (f <= band[1])
    s1 = np.empty(sel.sum())
    us = np.empty((sel.sum(), seg.shape[0]), dtype=complex)
    for q, idx in enumerate(np.flatnonzero(sel)):
        _, sv, vt = np.linalg.svd(Sxx[:, :, idx])
        s1[q] = sv[0]
        us[q] = vt[0].conj()
    k = int(np.argmax(s1))
    return float(f[sel][k]), float(s1[k]), us[k]


def modal_amplitude_ratio(record, fs, ch_mode1, ch_mode2):
    """Ratio of identified mode-1 to mode-2 amplitude at the deck mid-span.

    Mode 1 (the longitudinally floating mode) is picked in the
    ``mode1_band`` and evaluated at the mid-span longitudinal channel;
    mode 2 (the first vertical) is picked in the ``mode2_band`` and
    evaluated at the mid-span vertical channel.  Each shape is the first
    singular vector of the window's cross-spectral matrix at the peak,
    normalised to unit Euclidean norm.  Returns ``(n_windows,)``.
    """
    F = P.FEATURES
    n_win = F["n_windows"]
    n_win_samples = int(round(F["window_s"] * fs))
    out = np.zeros(n_win)
    for w in range(n_win):
        seg = record[:, w * n_win_samples: (w + 1) * n_win_samples]
        _, _, u1 = _fdd_singular_peak(seg, fs, F["mode1_band"])
        _, _, u2 = _fdd_singular_peak(seg, fs, F["mode2_band"])
        a1 = abs(u1[ch_mode1]) / np.linalg.norm(u1)
        a2 = abs(u2[ch_mode2]) / np.linalg.norm(u2)
        out[w] = float(a1 / max(a2, 1e-12))
    return out


# ---------------------------------------------------------------------------
# window assembly
# ---------------------------------------------------------------------------
def extract_window_features(record, fs, channel_kinds, pair_index,
                            ch_mode1, ch_mode2):
    """The 387 features of every window of one record.

    Returns ``(features, layout)`` where ``features`` has shape
    ``(n_windows, 387)`` ordered as
    [cwt_bands (64 x 5) | hilbert (64) | svd | phase | modal_ratio], and
    ``layout`` maps the block names to their column slices.
    """
    cwt = cwt_band_energies(record, fs, channel_kinds)      # (W, 64, 5)
    hdt = hilbert_decay(record, fs)                          # (W, 64)
    svd_e = svd_entropy(record, fs)                          # (W,)
    pha = phase_deviation(record, fs, pair_index)            # (W,)
    mar = modal_amplitude_ratio(record, fs, ch_mode1, ch_mode2)  # (W,)

    n_win, n_ch, n_bands = cwt.shape
    feats = np.empty((n_win, n_ch * n_bands + n_ch + 3))
    feats[:, : n_ch * n_bands] = cwt.reshape(n_win, n_ch * n_bands)
    feats[:, n_ch * n_bands: n_ch * n_bands + n_ch] = hdt
    feats[:, -3] = svd_e
    feats[:, -2] = pha
    feats[:, -1] = mar
    layout = dict(
        cwt=slice(0, n_ch * n_bands),
        hilbert=slice(n_ch * n_bands, n_ch * n_bands + n_ch),
        scalars=slice(n_ch * n_bands + n_ch, n_ch * n_bands + n_ch + 3),
        n_features=feats.shape[1])
    return feats, layout


FEATURE_DIM = 64 * 5 + 64 + 3      # 387


def adjacent_vertical_pairs(channel_kinds):
    """Index pairs of consecutive deck-vertical channels in channel order.

    The deck-vertical channels are sorted by their position in the
    channel list, which follows the x order of the layout; every
    consecutive pair is one adjacent-sensor pair of the phase statistic.
    """
    idx = [k for k, kind in enumerate(channel_kinds)
           if kind == "deck-vertical"]
    return list(zip(idx[:-1], idx[1:]))
