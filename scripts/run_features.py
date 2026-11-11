"""
Stage-3 driver: damage-sensitive feature extraction and blind modal
identification of the regenerated records.

Run from the repository root:

    python scripts/run_features.py          # full run, ~5 minutes
    python scripts/run_features.py --quick  # 3 scenarios, 1 record each

Every 60 s record of every scenario (three records per scenario, 111
records in total) is split into its three 20 s windows and each window is
transformed into the 387-dimensional feature vector of the study: 320 CWT
band energies (complex Morlet, w0 = 6, 50 log-spaced scales covering
0.05-10 Hz, five energy bands per channel), 64 Hilbert log-envelope
decays, the SVD entropy, the cross-spectral phase deviation and the
modal-amplitude ratio from blind frequency-domain decomposition.  The
resulting 333 window samples carry their scenario / family / severity
labels and are what the learning stage of the study trains on.

The driver also runs the three blind identification engines (FDD,
SSI-COV, p-LSCF) on the healthy records and tabulates them against the
calibrated FE frequencies, and regenerates the data figures of the study
from the actual records: the healthy PSD panel, the CWT scalogram, the
band-energy violins by severity class, the damage inventory matrix, the
modal-frequency shifts of the 36 damaged scenarios and the representative
time-series panel.

Artefacts:

* ``data/features/features_windows.npz``  - the 333 x 387 feature matrix
  with labels and grouping keys;
* ``results/modal_id_comparison.csv``     - FE vs FDD / SSI-COV / p-LSCF;
* ``results/feature_descriptions.csv``    - the feature inventory table;
* ``results/figures/fig10..fig16``        - data figures;
* ``results/feature_report.md`` and the stage-3 block of
  ``results/summary.json``.
"""

import csv
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
from scipy.signal import welch

from sutong_fe import params as P
from sutong_fe import features as FE
from sutong_fe import modal_id as MI
from sutong_fe.scenarios import SCENARIO_LIST

RESULTS = os.path.join(ROOT, "results")
FIGURES = os.path.join(RESULTS, "figures")
RECORDS = os.path.join(ROOT, "data", "scenario_records")
FEATDIR = os.path.join(ROOT, "data", "features")

SEVERITY_NAMES = {0: "healthy", 1: "light", 2: "moderate", 3: "severe"}
CLASS_OF = {"healthy": 0, "light": 1, "moderate": 2, "severe": 3}


# ---------------------------------------------------------------------------
class Check:
    def __init__(self, name, passed, value, tolerance):
        self.name = name
        self.passed = bool(passed)
        self.value = value
        self.tolerance = tolerance


def check_header(checks, title):
    print(f"\n== {title} " + "=" * max(0, 60 - len(title)))
    for c in checks:
        mark = "PASS" if c.passed else "FAIL"
        print(f"  [{mark}] {c.name}: {c.value}  ({c.tolerance})")
    n_fail = sum(1 for c in checks if not c.passed)
    print(f"  -> {len(checks)} checks, {n_fail} failed")
    return n_fail


def read_records(path_only=True):
    """The 37 scenario npz files in fixed scenario order."""
    files = []
    for sc in SCENARIO_LIST:
        name = f"scenario_{sc['sid']:02d}_{sc['code']}.npz"
        files.append((sc, os.path.join(RECORDS, name)))
    return files


# ---------------------------------------------------------------------------
# extraction
# ---------------------------------------------------------------------------
def channel_positions():
    """(x, y) of the 64 channels from the healthy npz metadata."""
    d = np.load(os.path.join(RECORDS, "scenario_00_H.npz"))
    return (d["channel_x_m"], d["channel_y_m"],
            list(d["channel_tags"]), list(d["channel_kinds"]))


def midspan_channels(kinds, tags):
    """Channel indices of the mid-span vertical and longitudinal sensors."""
    xs, _, tags, kinds = _channel_table()
    vert = [i for i, k in enumerate(kinds) if k == "deck-vertical"]
    longi = [i for i, k in enumerate(kinds) if k == "deck-longitudinal"]
    i_vert = min(vert, key=lambda i: abs(xs[i] - 1044.0))
    i_long = min(longi, key=lambda i: abs(xs[i] - 1044.0))
    return i_vert, i_long


def _channel_table():
    d = np.load(os.path.join(RECORDS, "scenario_00_H.npz"))
    return (d["channel_x_m"], d["channel_y_m"],
            list(d["channel_tags"]), list(d["channel_kinds"]))


def _extract_one_record(task):
    """Worker: features of every window of one record of one scenario."""
    sc, path, rec = task
    d = np.load(path)
    acc = d["acc"].astype(np.float64)
    fs = float(d["fs"])
    kinds = list(d["channel_kinds"])
    xs = d["channel_x_m"]
    vert = [i for i, k in enumerate(kinds) if k == "deck-vertical"]
    longi = [i for i, k in enumerate(kinds) if k == "deck-longitudinal"]
    i_vert = min(vert, key=lambda i: abs(xs[i] - 1044.0))
    i_long = min(longi, key=lambda i: abs(xs[i] - 1044.0))
    pairs = FE.adjacent_vertical_pairs(kinds)
    feats, _ = FE.extract_window_features(
        acc[rec], fs, kinds, pairs, i_long, i_vert)
    return (sc["sid"], rec, sc, feats)


def extract_all(quick=False):
    """The full 333 x 387 feature matrix with labels."""
    xs, ys, tags, kinds = _channel_table()
    i_vert, i_long = midspan_channels(kinds, tags)

    tasks = []
    files = read_records()
    if quick:
        files = files[:3]
    for sc, path in files:
        d = np.load(path)
        n_rec = int(d["acc"].shape[0])
        if quick:
            n_rec = min(n_rec, 1)
        for rec in range(n_rec):
            tasks.append((sc, path, rec))

    t0 = time.time()
    results = []
    n_proc = 1 if quick else min(2, os.cpu_count() or 1)
    if n_proc > 1:
        import multiprocessing as mp
        with mp.Pool(n_proc) as pool:
            for out in pool.imap_unordered(_extract_one_record, tasks):
                results.append(out)
                print(f"  features: sid {out[0]} rec {out[1]} done "
                      f"({len(results)}/{len(tasks)}, "
                      f"{time.time() - t0:.0f} s elapsed)")
    else:
        for task in tasks:
            out = _extract_one_record(task)
            results.append(out)
            print(f"  features: sid {out[0]} rec {out[1]} done "
                  f"({len(results)}/{len(tasks)}, "
                  f"{time.time() - t0:.0f} s elapsed)")

    # deterministic sample order: scenario order, then record, then window
    order = {(t[0]["sid"], t[2]): k for k, t in enumerate(tasks)}
    results.sort(key=lambda r: order[(r[0], r[1])])

    rows = []
    for sid, rec, sc, feats in results:
        for w in range(feats.shape[0]):
            rows.append(dict(
                sid=sc["sid"], code=sc["code"],
                family=sc["family"] or "H",
                sev_class=sc["severity_class"],
                sev_label=sc["severity_label"],
                record=rec, window=w, feats=feats[w]))
    X = np.stack([r["feats"] for r in rows])
    return rows, X, (i_vert, i_long), xs, ys, tags, kinds


def save_features(rows, X, path):
    os.makedirs(FEATDIR, exist_ok=True)
    np.savez_compressed(
        path,
        X=X.astype(np.float64),
        sid=np.array([r["sid"] for r in rows], dtype=np.int64),
        code=np.array([r["code"] for r in rows]),
        family=np.array([r["family"] for r in rows]),
        sev_class=np.array([r["sev_class"] for r in rows], dtype=np.int64),
        sev_label=np.array([r["sev_label"] for r in rows]),
        record=np.array([r["record"] for r in rows], dtype=np.int64),
        window=np.array([r["window"] for r in rows], dtype=np.int64),
        feature_names=np.array(_feature_names()))


def _feature_names():
    names = []
    for ch in range(64):
        for b in range(5):
            names.append(f"cwt_band{b + 1}_ch{ch:02d}")
    for ch in range(64):
        names.append(f"hilbert_decay_ch{ch:02d}")
    names += ["svd_entropy", "xspectral_phase_dev", "modal_amp_ratio"]
    return names


# ---------------------------------------------------------------------------
# blind modal identification table
# ---------------------------------------------------------------------------
def modal_id_table():
    """FDD / SSI-COV / p-LSCF identification of the healthy records vs
    the calibrated FE frequencies."""
    d = np.load(os.path.join(RECORDS, "scenario_00_H.npz"))
    acc = d["acc"].astype(np.float64)
    fs = float(d["fs"])

    with open(os.path.join(RESULTS, "modal_calibration.csv")) as fh:
        fe_rows = list(csv.DictReader(fh))
    fe_freqs = [float(r["f_FE_calibrated_Hz"]) for r in fe_rows]

    # reference channels for the time-domain engines: fifteen
    # deck-vertical sensors spanning the full deck at 144 m spacing (the
    # 96 m wavelength of the highest target mode is above the Nyquist
    # wavelength of the 144 m reference grid)
    xs, _, tags, kinds = _channel_table()
    vert = [i for i, k in enumerate(kinds) if k == "deck-vertical"]
    refs = vert[::3]

    # FDD: peaks of the first singular value of the cross-PSD matrix
    # averaged over the three records of the healthy state
    peaks = MI.fdd_first_sv_peaks(acc, fs, (0.10, 0.80), n_targets=12)
    common = [round(f, 4) for f, _, _ in peaks]

    # SSI-COV: pole list per record, stabilised over the three records
    ssi_sets = []
    for rec in range(acc.shape[0]):
        poles = MI.ssi_cov(acc[rec], fs, refs)
        ssi_sets.append(poles)
    ssi_stable = MI.stabilise_poles(ssi_sets)

    # p-LSCF: scalar common denominator per record (the fit band is wider
    # than the reporting band, for a well-conditioned polynomial basis)
    plscf_sets = []
    for rec in range(acc.shape[0]):
        poles = MI.p_lscf(acc[rec], fs, (0.05, 5.0))
        plscf_sets.append(poles)
    plscf_stable = MI.stabilise_poles(plscf_sets)

    def matched(freqs, targets, tol=0.025):
        out = []
        for ft in targets:
            hit = min(freqs, key=lambda f: abs(f - ft)) if freqs else None
            out.append(round(hit, 4)
                       if hit is not None and abs(hit - ft) / ft < tol
                       else "")
        return out

    rows = [dict(method="FE (calibrated reference)",
                 freqs=[round(f, 4) for f in fe_freqs],
                 n_stable=len(fe_freqs))]
    rows.append(dict(method="FDD", freqs=matched(common, fe_freqs),
                     n_stable=len(common)))
    rows.append(dict(method="SSI-COV", freqs=matched(
        [f for f, _ in ssi_stable], fe_freqs), n_stable=len(ssi_stable)))
    rows.append(dict(method="p-LSCF", freqs=matched(
        [f for f, _ in plscf_stable], fe_freqs), n_stable=len(plscf_stable)))
    return rows, fe_freqs


def write_modal_id_csv(rows, path):
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["method", "f1_Hz", "f2_Hz", "f3_Hz", "f4_Hz",
                    "f5_Hz", "f6_Hz", "f7_Hz", "n_stable"])
        for r in rows:
            fs = [str(f) for f in r["freqs"]] + [""] * (7 - len(r["freqs"]))
            w.writerow([r["method"]] + fs[:7] + [r["n_stable"]])


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------
def fig_psd_healthy(path):
    d = np.load(os.path.join(RECORDS, "scenario_00_H.npz"))
    acc = d["acc"].astype(np.float64)
    fs = float(d["fs"])
    xs, _, tags, kinds = _channel_table()
    picks = [("mid-span deck vertical", 21),
             ("quarter-point deck vertical",
              min((i for i, k in enumerate(kinds)
                   if k == "deck-vertical"), key=lambda i: abs(xs[i] - 776)),
             ),
             ("mid-span deck longitudinal", 45),
             ("pylon-top horizontal", 48),
             ("stay axial", 56)]
    fig, axes = plt.subplots(len(picks), 1, figsize=(9.5, 1.75 * len(picks)),
                              sharex=True)
    for ax, (label, ch) in zip(axes, picks):
        f, psd = welch(np.mean(acc[:, ch, :], axis=0), fs=fs,
                       nperseg=8192)
        ax.semilogy(f, psd, lw=0.9, color="#1f77b4")
        ax.set_ylabel(f"{tags[ch]}\n({label})", fontsize=7)
        ax.set_xlim(0, 2.0)
        ax.grid(alpha=0.3)
    axes[0].set_title("Averaged PSD of the healthy records "
                     "(mean of 3 x 60 s), selected channels")
    axes[-1].set_xlabel("frequency (Hz)")
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def fig_cwt_scalogram(path):
    d = np.load(os.path.join(RECORDS, "scenario_00_H.npz"))
    x = d["acc"][0, 21].astype(np.float64)
    fs = float(d["fs"])
    F = P.FEATURES
    scales, freqs = FE.morlet_scales(
        fs, F["cwt_w0"], F["cwt_n_scales"], F["cwt_f_lo"], F["cwt_f_hi"])
    power = FE.cwt_scale_spectrogram(x, scales * fs, F["cwt_w0"],
                                     int(round(F["cwt_pad_s"] * fs)))
    t = np.arange(len(x)) / fs
    fig, ax = plt.subplots(figsize=(9.5, 5.0), constrained_layout=True)
    im = ax.pcolormesh(t, freqs, np.log10(power + 1e-30),
                       cmap="magma", shading="auto")
    for lo, hi in F["bands"]:
        ax.axhline(lo, color="w", lw=0.6, alpha=0.5)
        ax.axhline(hi, color="w", lw=0.6, alpha=0.5)
    ax.set_yscale("log")
    ax.set_ylim(F["cwt_f_lo"], F["cwt_f_hi"])
    ax.set_xlabel("time (s)")
    ax.set_ylabel("pseudo-frequency (Hz)")
    ax.set_title("Morlet CWT scalogram, mid-span deck vertical\n"
                 "healthy record 1 (50 scales, band edges in white)")
    fig.colorbar(im, ax=ax, label="log10 |W(a,t)|^2")
    fig.savefig(path, dpi=170)
    plt.close(fig)


def fig_band_violins(X, sev, path):
    classes = [0, 1, 2, 3]
    fig, axes = plt.subplots(1, 5, figsize=(13.5, 3.9), sharey=True,
                             constrained_layout=True)
    names = ["healthy", "light", "moderate", "severe"]
    for b in range(5):
        data = [np.log10(X[sev == c][:, 64 * b: 64 * (b + 1)].ravel()
                        + 1e-30) for c in classes]
        parts = axes[b].violinplot(data, positions=range(4), showmedians=True)
        for pc in parts["bodies"]:
            pc.set_alpha(0.55)
        axes[b].set_xticks(range(4))
        axes[b].set_xticklabels(names, rotation=20, fontsize=8)
        axes[b].set_title(f"band {b + 1}")
        axes[b].grid(alpha=0.3)
    axes[0].set_ylabel("log10 band energy, (m/s$^2$)$^2$")
    fig.suptitle("CWT band energies by severity class "
                 "(333 window samples, 64 channels pooled)")
    fig.savefig(path, dpi=170, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


def fig_damage_inventory(path):
    fams = sorted({sc["family"] for sc in SCENARIO_LIST if sc["family"]},
                  key=lambda f: (f[0], int(f[1:])))   # F2 before F10
    M = np.full((len(fams), 3), np.nan)
    for sc in SCENARIO_LIST:
        if sc["family"]:
            M[fams.index(sc["family"]), sc["severity_class"] - 1] = \
                sc["severity_pct"]
    fig, ax = plt.subplots(figsize=(7.8, 5.6), constrained_layout=True)
    im = ax.imshow(M, cmap="YlOrRd", aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if np.isfinite(M[i, j]):
                ax.text(j, i, f"{M[i, j]:.0f}%", ha="center", va="center",
                        fontsize=9)
    ax.set_xticks(range(3))
    ax.set_xticklabels(["light", "moderate", "severe"], fontsize=10)
    ax.set_yticks(range(len(fams)))
    ax.set_yticklabels(fams)
    ax.set_title("Damage inventory: property loss (%)\n"
                 "of the 12 families x 3 severities")
    fig.colorbar(im, ax=ax, label="property loss (%)")
    fig.savefig(path, dpi=170)
    plt.close(fig)


def fig_modal_shifts(path):
    with open(os.path.join(RESULTS, "scenario_modal_shifts.csv")) as fh:
        rows = list(csv.DictReader(fh))
    labels = ["VS1", "VAS1", "VS2", "VAS2", "VS3", "VS4", "VS5"]
    codes = []
    for r in rows:
        if r["sid"] != "0" and r["code"] not in codes:
            codes.append(r["code"])
    shift = {lab: [] for lab in labels}
    for r in rows:
        if r["sid"] == "0":
            continue
        shift[r["label"]].append(float(r["shift_pct"]))
    fig, ax = plt.subplots(figsize=(13.5, 5.0), constrained_layout=True)
    n = len(codes)
    for k, lab in enumerate(labels):
        vals = np.array(shift[lab])
        ax.plot(range(n), vals, marker="o", ms=3, lw=1.0, label=lab)
    ax.axhline(0, color="0.4", lw=0.8)
    ax.set_xticks(range(n))
    ax.set_xticklabels(codes, rotation=90, fontsize=7)
    ax.set_ylabel("frequency shift vs healthy (%)")
    ax.set_title("MAC-matched shifts of the seven measured vertical modes "
                 "(36 damaged scenarios)")
    ax.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    ax.grid(alpha=0.3)
    fig.savefig(path, dpi=170)
    plt.close(fig)


def fig_timeseries(path):
    xs, _, tags, kinds = _channel_table()
    picks = [21, 22, 20, 45, 47, 48, 52, 56]     # 8 representative channels
    scen = [("scenario_00_H.npz", "healthy"),
            ("scenario_03_F1_severe.npz", "F1 severe (stay loss)"),
            ("scenario_12_F4_severe.npz", "F4 severe (girder loss)")]
    fig, axes = plt.subplots(3, 1, figsize=(10.5, 7.2), sharex=True,
                             sharey=False)
    t = None
    for ax, (fname, label) in zip(axes, scen):
        d = np.load(os.path.join(RECORDS, fname))
        seg = d["acc"][0].astype(np.float64)[:, : 2000]     # first 10 s
        t = np.arange(seg.shape[1]) / float(d["fs"])
        offsets = np.arange(len(picks)) * 6e-3
        for k, ch in enumerate(picks):
            ax.plot(t, seg[ch] + offsets[k], lw=0.35,
                    color="0.25" if k % 2 else "0.55")
        ax.text(0.995, 0.92, label, transform=ax.transAxes, ha="right",
                fontsize=9)
        ax.set_ylabel("acc + offset (m/s$^2$)")
        ax.set_ylim(-4e-3, offsets[-1] + 4e-3)
    axes[-1].set_xlabel("time (s)")
    axes[0].set_title("Representative channels of three scenarios "
                      "(8 channels, stacked, first 10 s of record 1)")
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


# ---------------------------------------------------------------------------
def write_feature_descriptions(path):
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["feature", "dim", "unit", "channel-wise", "definition"])
        w.writerow(["cwt_band1..5", 320, "(m/s^2)^2 s", "yes (64 x 5)",
                    "CWT energy per channel in the bands "
                    "0.05-0.20 / 0.20-0.50 / 0.50-1.00 / 1.00-2.00 / "
                    "2.00-10.00 Hz (Morlet w0=6, 50 log-spaced scales)"])
        w.writerow(["hilbert_decay", 64, "log/s", "yes",
                    "LS slope of the log Hilbert envelope per window"])
        w.writerow(["svd_entropy", 1, "nats", "no",
                    "Shannon entropy of the squared singular values of "
                    "the 64 x 4000 window matrix"])
        w.writerow(["xspectral_phase_dev", 1, "rad", "no",
                    "mean over the 42 adjacent deck-vertical pairs of the "
                    "std of arg S_ij(f) over 0.1-2 Hz"])
        w.writerow(["modal_amp_ratio", 1, "-", "no",
                    "identified mode-1 (floating) amplitude at the "
                    "mid-span longitudinal sensor over the identified "
                    "mode-2 (first vertical) amplitude at the mid-span "
                    "vertical sensor (FDD shapes, unit normalised)"])
        w.writerow(["total per window", 387, "-", "-", ""])


def load_summary():
    path = os.path.join(RESULTS, "summary.json")
    if os.path.exists(path):
        with open(path) as fh:
            return json.load(fh)
    return {}


def main():
    quick = "--quick" in sys.argv
    t0 = time.time()
    os.makedirs(FEATDIR, exist_ok=True)
    os.makedirs(FIGURES, exist_ok=True)

    print("== Stage 3: feature extraction "
          "=================================")
    rows, X, (i_vert, i_long), xs, ys, tags, kinds = extract_all(quick)

    n_expected = len(rows)
    feat_path = os.path.join(FEATDIR, "features_windows.npz")
    save_features(rows, X, feat_path)
    sev = np.array([r["sev_class"] for r in rows])

    checks = []
    n_exp = (3 if quick else 37) * 3 * P.FEATURES["n_windows"]
    checks.append(Check("window samples extracted", len(rows) == n_exp,
                        f"{len(rows)}", f"{n_exp}"))
    checks.append(Check("feature dimension per window",
                        X.shape[1] == FE.FEATURE_DIM,
                        X.shape[1], f"{FE.FEATURE_DIM}"))
    finite = bool(np.all(np.isfinite(X)))
    checks.append(Check("all features finite", finite, "finite",
                       "no NaN / inf"))
    cwt = X[:, : 320]
    checks.append(Check("CWT band energies strictly positive",
                        bool(np.all(cwt > 0)),
                        f"min {cwt.min():.3e}", "> 0"))
    scales, freqs = FE.morlet_scales(
        200.0, P.FEATURES["cwt_w0"], P.FEATURES["cwt_n_scales"],
        P.FEATURES["cwt_f_lo"], P.FEATURES["cwt_f_hi"])
    lo_edges = [b[0] for b in P.FEATURES["bands"]]
    hi_edges = [b[1] for b in P.FEATURES["bands"]]
    covered = freqs.min() >= P.FEATURES["bands"][0][0] and \
        freqs.max() <= P.FEATURES["bands"][-1][1]
    checks.append(Check("scale frequencies inside the band partition",
                        covered,
                        f"{freqs.min():.4f}..{freqs.max():.2f} Hz",
                        "0.05..10 Hz"))
    svd_e = X[:, -3]
    checks.append(Check("SVD entropy bounds",
                        bool(np.all((svd_e > 0) & (svd_e <= np.log(64)))),
                        f"{svd_e.min():.3f}..{svd_e.max():.3f} nats",
                        "(0, ln 64]"))
    pha = X[:, -2]
    checks.append(Check("phase deviation range",
                        bool(np.all((pha >= 0) & (pha <= np.pi))),
                        f"{pha.min():.3f}..{pha.max():.3f} rad",
                        "[0, pi]"))
    mar = X[:, -1]
    checks.append(Check("modal-amplitude ratio varies across scenarios",
                        float(np.std(mar) / max(np.mean(mar), 1e-30)) > 1e-3,
                        f"cv {np.std(mar) / np.mean(mar):.3f}",
                        "std/mean > 0.001 (no dead feature)"))
    hdt = X[:, 320: 384]
    checks.append(Check("Hilbert slopes finite and small",
                        bool(np.all(np.abs(hdt) < 1.0)),
                        f"|slope| max {np.abs(hdt).max():.3f}",
                        "< 1 log/s"))
    frac12 = cwt.reshape(len(rows), 64, 5)[:, :, : 2].sum(axis=(1, 2)) / \
        cwt.reshape(len(rows), 64, 5).sum(axis=(1, 2))
    checks.append(Check("bands 1-2 carry the dominant CWT energy",
                        float(np.median(frac12)) > 0.30,
                        f"median {np.median(frac12) * 100:.1f} %",
                        "> 30 %"))
    # determinism of one window recomputed from the record
    d0 = np.load(os.path.join(RECORDS, "scenario_00_H.npz"))
    pairs = FE.adjacent_vertical_pairs(kinds)
    ref, _ = FE.extract_window_features(
        d0["acc"][1].astype(np.float64), float(d0["fs"]), kinds, pairs,
        i_long, i_vert)
    idx = next(k for k, r in enumerate(rows)
              if r["sid"] == 0 and r["record"] == 1 and r["window"] == 0)
    same = float(np.max(np.abs(ref[0] - X[idx])))
    checks.append(Check("deterministic recomputation of one window",
                        same == 0.0, f"max diff {same:.1e}",
                        "0 (bitwise)"))

    # --- blind modal identification table -------------------------------
    print("  modal identification (FDD / SSI-COV / p-LSCF)...")
    mi_rows, fe_freqs = modal_id_table()
    write_modal_id_csv(mi_rows, os.path.join(RESULTS,
                                             "modal_id_comparison.csv"))
    fdd_row = mi_rows[1]
    ssi_row = mi_rows[2]
    n_fdd_matched = sum(1 for f in fdd_row["freqs"] if f != "")
    n_ssi_matched = sum(1 for f in ssi_row["freqs"] if f != "")
    checks.append(Check("FDD identifies the verticals of the healthy "
                        "records",
                        n_fdd_matched >= 5,
                        f"{n_fdd_matched} of 7 within 2.5 %",
                        ">= 5"))
    checks.append(Check("SSI-COV identifies the verticals of the healthy "
                        "records",
                        n_ssi_matched >= 5,
                        f"{n_ssi_matched} of 7 within 2.5 %",
                        ">= 5"))

    # --- figures ---------------------------------------------------------
    print("  figures...")
    fig_psd_healthy(os.path.join(FIGURES, "fig10_psd_healthy.png"))
    fig_cwt_scalogram(os.path.join(FIGURES, "fig11_cwt_scalogram.png"))
    fig_band_violins(X, sev, os.path.join(FIGURES,
                                          "fig12_band_violins.png"))
    fig_damage_inventory(os.path.join(FIGURES, "fig14_damage_inventory.png"))
    fig_modal_shifts(os.path.join(FIGURES, "fig15_modal_shifts.png"))
    fig_timeseries(os.path.join(FIGURES, "fig16_timeseries.png"))
    write_feature_descriptions(os.path.join(RESULTS,
                                           "feature_descriptions.csv"))

    n_fail = check_header(checks, "Stage-3 feature verification")

    # --- report + summary ------------------------------------------------
    with open(os.path.join(RESULTS, "feature_report.md"), "w") as fh:
        fh.write("# Sutong Bridge 2-D FE Model - Stage-3 Feature Report\n\n")
        fh.write(f"Generated by ``scripts/run_features.py`` in "
                 f"{time.time() - t0:.0f} s. "
                 f"{len(rows)} window samples "
                 f"({len(rows) // P.FEATURES['n_windows']} records) x 387 "
                 "features.\n\n")
        fh.write("## 1. Feature families\n\n")
        fh.write("| family | dim | definition |\n|---|---|---|\n")
        fh.write("| CWT band energies | 320 | complex Morlet (w0 = 6), "
                  "50 log-spaced scales 0.05-10 Hz, five bands per channel "
                  "|\n")
        fh.write("| Hilbert envelope decay | 64 | LS slope of log|z(t)| per "
                  "channel |\n")
        fh.write("| SVD entropy | 1 | Shannon entropy of the squared "
                  "singular values |\n")
        fh.write("| cross-spectral phase deviation | 1 | mean over the 42 "
                  "adjacent deck-vertical pairs |\n")
        fh.write("| modal-amplitude ratio | 1 | blind FDD mode-1 / mode-2 "
                  "amplitude at mid-span |\n\n")
        fh.write("## 2. Automated checks\n\n")
        fh.write("| check | value | limit | result |\n|---|---|---|---|\n")
        for c in checks:
            mark = "PASS" if c.passed else "FAIL"
            fh.write(f"| {c.name} | {c.value} | {c.tolerance} | {mark} |\n")
        fh.write("\n## 3. Blind modal identification of the healthy "
                 "records\n\n")
        fh.write("| method | identified f1..f7 (Hz) | n stable |\n"
                 "|---|---|---|\n")
        for r in mi_rows:
            fs_ = ", ".join(str(f) for f in r["freqs"])
            fh.write(f"| {r['method']} | {fs_} | {r['n_stable']} |\n")

    summary = load_summary()
    summary["stage_3"] = dict(
        n_window_samples=int(len(rows)),
        n_records=int(len(rows) // P.FEATURES["n_windows"]),
        n_features=int(X.shape[1]),
        cwt_scales=int(P.FEATURES["cwt_n_scales"]),
        cwt_range_hz=[P.FEATURES["cwt_f_lo"], P.FEATURES["cwt_f_hi"]],
        checks_total=len(checks),
        checks_failed=n_fail,
        runtime_s=round(time.time() - t0, 1))
    with open(os.path.join(RESULTS, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)

    print(f"\nstage 3 done in {time.time() - t0:.0f} s "
          f"({n_fail} failed checks)")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
