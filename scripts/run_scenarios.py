"""
Stage-2 driver: regenerate the 37 damage scenarios of the study.

Run from the repository root:

    python scripts/run_scenarios.py          # full run, ~15-20 minutes
    python scripts/run_scenarios.py --quick  # 4 scenarios, short records

Every scenario - the healthy state included - goes through the *same*
pipeline: calibrated geometry, the as-built dead-load state from the
stage-1 shape finding, damage applied to the element properties, a
post-damage equilibrium with the manufactured pre-strains and the as-built
Ernst moduli held fixed (one exact linear solve), the linearised
eigen-problem, and three 60 s monitoring records at 200 Hz
driven by independently seeded ambient excitation (turbulent wind plus a
traffic stream).  No two records share an excitation realisation, and the
only systematic difference between scenarios is the damage state.

Artefacts:

* ``data/scenario_records/scenario_<sid>_<code>.npz``  - the records
  (3 x 64 x 12000 float32, m/s^2, sensor noise included) with metadata;
* ``results/scenario_inventory.csv``  - one row per scenario;
* ``results/scenario_modal_shifts.csv`` - MAC-matched frequency shifts of
  the seven measured vertical modes;
* ``results/sensor_channels.csv``  - the 64-channel layout;
* ``results/figures/fig06..fig09`` - sensor layout, healthy record,
  severity-frequency response, stay-force redistribution;
* ``results/scenario_report.md`` and the stage-2 block of
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
from sutong_fe.geometry import build_geometry
from sutong_fe.model import FEModel
from sutong_fe.modal import modal_analysis, pairing_report
from sutong_fe.scenarios import SCENARIO_LIST, apply_damage, damage_summary
from sutong_fe.sensors import build_channels
from sutong_fe.simulate import simulate_record, add_sensor_noise

RESULTS = os.path.join(ROOT, "results")
FIGURES = os.path.join(RESULTS, "figures")
RECORDS = os.path.join(ROOT, "data", "scenario_records")

N_MODES_EIGSH = 260      # eigenvalue request; usable set is f <= 12 Hz


# ---------------------------------------------------------------------------
class Check:
    def __init__(self, name, passed, value, tolerance):
        self.name = name
        self.passed = bool(passed)
        self.value = value
        self.tolerance = tolerance


def check_header(checks):
    print("\n== Scenario verification "
          "================================================")
    for c in checks:
        mark = "PASS" if c.passed else "FAIL"
        print(f"  [{mark}] {c.name}: {c.value}  ({c.tolerance})")
    n_fail = sum(1 for c in checks if not c.passed)
    print(f"  -> {len(checks)} checks, {n_fail} failed")
    return n_fail


# ---------------------------------------------------------------------------
def load_scales():
    with open(os.path.join(RESULTS, "calibration_result.json")) as fh:
        return {k: float(v) for k, v in json.load(fh).items()}


def modal_set(model):
    """Eigen-modes of one state, truncated at the superposition cutoff."""
    modes = modal_analysis(model, n_modes=N_MODES_EIGSH)
    return [m for m in modes if m["f"] <= P.EXCITATION["f_modes_max"]]


def build_baseline(scales):
    geo = build_geometry(scales)
    model = FEModel(geo)
    model.shape_find()
    return geo, model, modal_set(model)


def build_scenario_model(scales, scenario, base_eps0, base_Eeq):
    """The one physics pipeline every damage scenario passes through."""
    geo = build_geometry(scales)
    mods, desc = apply_damage(geo, scenario)
    model = FEModel(geo)
    model.eps0 = base_eps0.copy()
    model.E_eq = base_Eeq.copy()
    model.reequilibrate()
    return geo, model, modal_set(model), mods, desc


def mac(w1, w2):
    num = float(np.dot(w1, w2)) ** 2
    den = float(np.dot(w1, w1) * np.dot(w2, w2))
    return num / den if den > 0 else 0.0


def paired_profiles(modes):
    """The seven measured vertical modes with their deck profiles."""
    by_f = {round(m["f"], 6): m for m in modes}
    out = []
    for r in pairing_report(modes):
        if r["f_fe"] is None:
            continue
        m = by_f[round(r["f_fe"], 6)]
        out.append((r["label"], r["f_fe"], m["deck_w"]))
    return out


def match_shifts(healthy_paired, dmg_modes):
    """MAC-match each healthy paired vertical to a damaged mode."""
    out = []
    for label, f_h, w_h in healthy_paired:
        best_f, best_mac = None, 0.0
        for m in dmg_modes:
            if m["kind"] != "vertical":
                continue
            if abs(m["f"] - f_h) / f_h > 0.5:
                continue
            v = mac(w_h, m["deck_w"])
            if v > best_mac:
                best_f, best_mac = m["f"], v
        out.append(dict(
            label=label, f_healthy=f_h, f_damaged=best_f, mac=best_mac,
            shift_pct=(100.0 * (best_f - f_h) / f_h
                       if best_f is not None else None)))
    return out


def simulate_scenario(model, modes, sid, n_records, quick=False):
    """The seeded records of one scenario (kept only transiently)."""
    fs = P.EXCITATION["fs"]
    dur = 20.0 if quick else P.EXCITATION["duration"]
    if quick:
        n_records = min(n_records, 2)
    records, seeds, rms = [], [], []
    for rec in range(n_records):
        rng = np.random.default_rng((P.EXCITATION["seed_base"], sid, rec))
        _, acc = simulate_record(
            model, modes, rng, T_burn=60.0 if quick else None,
            fs_rec=fs, T_rec=dur, chunk_s=10.0)
        acc = add_sensor_noise(acc, rng)
        records.append(acc.astype(np.float32))
        seeds.append((P.EXCITATION["seed_base"], sid, rec))
        rms.append(float(np.sqrt(np.mean(acc ** 2))))
    return records, seeds, rms, fs, int(round(dur * fs))


def save_records(sid, code, records, seeds, fs, n_samples, channels):
    os.makedirs(RECORDS, exist_ok=True)
    path = os.path.join(RECORDS, f"scenario_{sid:02d}_{code}.npz")
    np.savez_compressed(
        path,
        acc=np.stack(records),
        seeds=np.array([list(s) for s in seeds], dtype=np.int64),
        fs=fs, duration_s=n_samples / fs,
        channel_tags=np.array([c["tag"] for c in channels]),
        channel_kinds=np.array([c["kind"] for c in channels]),
        channel_x_m=np.array([c["x"] for c in channels], dtype=np.float64),
        channel_y_m=np.array([c["y"] for c in channels], dtype=np.float64))
    return path


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------
def fig_sensor_layout(geo, channels, path):
    fig, ax = plt.subplots(figsize=(11.5, 4.6))
    for be in geo.beams:
        if be.group == "deck":
            ax.plot([geo.nodes[be.n1].x, geo.nodes[be.n2].x],
                    [geo.nodes[be.n1].y, geo.nodes[be.n2].y],
                    color="0.35", lw=1.4, zorder=2)
    for pr in P.PIERS:
        ax.plot([pr["x"], pr["x"]], [0, P.PIER_TOP_Y],
                color="0.55", lw=3, zorder=1)
    for c in geo.cables:
        ax.plot([c.x_deck, P.PYLON_X[c.pylon]],
                [P.DECK_Y, c.y_anchor], color="0.7", lw=0.5, zorder=1)
    for xp in P.PYLON_X:
        ax.plot([xp, xp], [0, 300.4], color="0.2", lw=5, zorder=3)
    style = dict(
        deck_vertical=("o", "#1f77b4", "deck vertical (43)"),
        deck_longitudinal=("s", "#2ca02c", "deck longitudinal (5)"),
        pylon=("^", "#d62728", "pylon bi-directional (8)"),
        cable_axial=("v", "#ff7f0e", "stay axial (8)"))
    seen = set()
    for ch in channels:
        m, col, lab = style[ch["kind"].replace("-", "_")]
        ax.plot(ch["x"], ch["y"] + (4 if ch["kind"] == "deck-vertical"
                                    else 0), m, color=col, ms=5,
                label=lab if ch["kind"] not in seen else None, zorder=4)
        seen.add(ch["kind"])
    ax.set_xlabel("longitudinal position (m)")
    ax.set_ylabel("elevation (m)")
    ax.set_title("64-channel sensor subset of the monitoring system")
    ax.legend(loc="lower left", fontsize=8, ncol=2)
    ax.set_xlim(-40, 2128)
    ax.set_ylim(-25, 315)
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def fig_healthy_record(acc, modes, path, fs=200.0):
    acc = acc.astype(np.float64)
    t = np.arange(acc.shape[1]) / fs
    fig, axs = plt.subplots(2, 2, figsize=(11.0, 6.4))
    ax = axs[0, 0]
    for k, col in ((5, "#1f77b4"), (21, "#2ca02c"), (37, "#d62728")):
        ax.plot(t[:1200], acc[k, :1200], lw=0.6, color=col, alpha=0.9)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("acceleration (m/s^2)")
    ax.set_title("healthy record, three deck channels (6 s excerpt)")
    ax = axs[0, 1]
    ax.plot(t, acc[21], lw=0.4, color="#1f77b4")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("acceleration (m/s^2)")
    ax.set_title("deck mid-span vertical, full record")
    ax = axs[1, 0]
    f, psd = welch(acc[21], fs=fs, nperseg=8192)
    ax.loglog(f, psd, lw=0.8, color="#1f77b4")
    for m in modes[:9]:
        if m["kind"] == "vertical":
            ax.axvline(m["f"], color="0.6", lw=0.5, ls="--")
    ax.set_xlabel("frequency (Hz)")
    ax.set_ylabel("PSD ((m/s^2)^2/Hz)")
    ax.set_title("mid-span PSD; dashed lines: FE vertical modes")
    ax = axs[1, 1]
    f2, psd2 = welch(acc[48], fs=fs, nperseg=8192)
    ax.loglog(f2, psd2, lw=0.8, color="#2ca02c")
    ax.axvline(0.070, color="0.6", lw=0.5, ls="--")
    ax.set_xlabel("frequency (Hz)")
    ax.set_ylabel("PSD ((m/s^2)^2/Hz)")
    ax.set_title("mid-span longitudinal PSD; dashed: floating mode")
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def fig_severity_shifts(inventory, path):
    fams = sorted({r["family"] for r in inventory if r["family"] != "-"},
                  key=lambda f: (f[0], int(f[1:])))   # F2 before F10
    x = np.arange(len(fams))
    fig, ax = plt.subplots(figsize=(10.5, 4.2))
    offsets = {"light": -0.27, "moderate": 0.0, "severe": 0.27}
    colors = {"light": "#9ecae1", "moderate": "#4292c6", "severe": "#084594"}
    for lev in ("light", "moderate", "severe"):
        vals = [next(r["mean_abs_shift_pct"] for r in inventory
                     if r["family"] == fam and
                     r["severity_label"] == lev) for fam in fams]
        ax.bar(x + offsets[lev], vals, width=0.25, color=colors[lev],
               label=lev)
    ax.set_xticks(x)
    ax.set_xticklabels(fams)
    ax.set_ylabel("mean |frequency shift| of the 7 verticals (%)")
    ax.set_title("damage signal grows monotonically with severity")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def fig_force_redistribution(base_t, dmg_t, geo, path, label="F1 severe"):
    """Stay forces of the low-pylon river fan, healthy vs damaged."""
    riv = sorted((c for c in geo.cables
                  if c.pylon == 0 and c.side == "river"),
                 key=lambda c: c.level)
    ids = [c.eid for c in riv]
    xs = [c.x_deck for c in riv]
    fig, axs = plt.subplots(1, 2, figsize=(10.5, 4.0))
    axs[0].plot(xs, base_t[ids] / 1e6, "o-", ms=3, color="#084594",
                label="healthy")
    axs[0].plot(xs, dmg_t[ids] / 1e6, "v--", ms=4, color="#d62728",
                label=label)
    axs[0].set_xlabel("deck anchor position (m)")
    axs[0].set_ylabel("stay force (MN, both planes)")
    axs[0].legend()
    axs[0].set_title("river fan of the low pylon")
    d = (dmg_t - base_t)[ids] / 1e6
    axs[1].bar(xs, d, width=10, color="#d62728", alpha=0.8)
    axs[1].set_xlabel("deck anchor position (m)")
    axs[1].set_ylabel("force change (MN)")
    axs[1].set_title("redistribution toward the neighbouring stays")
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


# ---------------------------------------------------------------------------
def write_report(path, inventory, shift_rows, checks, quick, runtime):
    by_code = {r["code"]: r for r in inventory}
    lines = [
        "# Sutong Bridge 2-D FE Model - Stage-2 Scenario Report",
        "",
        f"Generated by ``scripts/run_scenarios.py`` in {runtime:.0f} s. "
        "37 scenarios = healthy + 12 damage families x 3 severity levels.",
        "",
        "## 1. Pipeline",
        "",
        "Every scenario passes through the same chain of solvers:",
        "calibrated geometry -> damage applied to element properties ->",
        "post-damage dead-load equilibrium with the manufactured pre-strains",
        "and the as-built Ernst moduli held fixed (one exact linear solve;",
        "the sag-modulus feedback of the 544 m river stays makes any",
        "modulus re-iteration numerically unreliable) ->",
        f"linearised eigen-problem truncated at "
        f"{P.EXCITATION['f_modes_max']:.0f} Hz -> modal-superposition",
        "simulation with uniform 2 % modal damping -> 64 sensor channels +",
        "white sensor noise. Each of the "
        f"{P.EXCITATION['n_records']} records per scenario draws its own",
        "seeded wind field, traffic stream and noise; the only systematic",
        "difference between scenarios is the damage state.",
        "",
        "## 2. Automated checks",
        "",
        "| check | value | limit | result |",
        "|---|---|---|---|",
    ]
    for c in checks:
        lines.append(f"| {c.name} | {c.value} | {c.tolerance} | "
                     f"{'PASS' if c.passed else 'FAIL'} |")
    lines += [
        "",
        "## 3. Frequency response of the seven verticals",
        "",
        "| scenario | mean shift of the 7 paired verticals |"
        " floating (Hz) |",
        "|---|---|---|",
    ]
    for r in inventory:
        lines.append(f"| {r['code']} ({r['severity_pct']} %) | "
                     f"{r['mean_abs_shift_pct']:.3f} % | "
                     f"{r['f_float_hz']:.4f} |")
    if not quick:
        lines += [
            "",
            "Full per-mode MAC-matched shifts: "
            "``results/scenario_modal_shifts.csv``.",
        ]
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
def main():
    quick = "--quick" in sys.argv
    t_start = time.time()

    print("=" * 66)
    print("stage 2: 37 damage scenarios, one consistent physics pipeline")
    print("=" * 66)

    scales = load_scales()
    print(f"calibrated scales: {scales}")

    geo_base, base, base_modes = build_baseline(scales)
    print(f"baseline: {len(base_modes)} modes <= "
          f"{P.EXCITATION['f_modes_max']:.0f} Hz, floating "
          f"{base_modes[0]['f']:.4f} Hz")

    channels, R = build_channels(base)
    with open(os.path.join(RESULTS, "sensor_channels.csv"), "w",
              newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["ch", "tag", "kind", "node", "x_m", "y_m", "direction"])
        for c in channels:
            w.writerow([c["ch"], c["tag"], c["kind"], c["node"],
                        round(c["x"], 1), round(c["y"], 1), c["direction"]])

    healthy_paired = paired_profiles(base_modes)
    f_ref = {lab: f for (lab, f, _) in healthy_paired}

    inventory, shift_rows = [], []
    stat = dict(corr_pairs=[], shapes_ok=True, finite_ok=True,
                rms_all=[], seeds=[])
    healthy_rec0 = None
    f1_severe_tension = None

    for sc in SCENARIO_LIST:
        if quick and sc["sid"] not in (0, 1, 13, 25):
            continue
        t0 = time.time()
        if sc["family"] is None:
            geo, model, modes, mods, desc = geo_base, base, base_modes, \
                [], "healthy state"
        else:
            geo, model, modes, mods, desc = build_scenario_model(
                scales, sc, base.eps0, base.E_eq)
        records, seeds, rms, fs, n_smp = simulate_scenario(
            model, modes, sc["sid"], P.EXCITATION["n_records"], quick)
        save_records(sc["sid"], sc["code"], records, seeds, fs, n_smp,
                     channels)

        # per-scenario statistics while the records are still in memory
        midch = next(i for i, c in enumerate(channels)
                     if abs(c["x"] - 1044) < 1 and
                     c["kind"] == "deck-vertical")
        recs = [r[midch].astype(np.float64) for r in records]
        for a in range(len(recs)):
            for b in range(a + 1, len(recs)):
                stat["corr_pairs"].append(
                    float(np.corrcoef(recs[a], recs[b])[0, 1]))
        for r in records:
            if r.shape[0] != 64 or not np.isfinite(
                    r.astype(np.float64)).all():
                stat["shapes_ok"] = stat["finite_ok"] = False
        stat["rms_all"] += rms
        stat["seeds"] += seeds
        if sc["sid"] == 0:
            healthy_rec0 = records[0].copy()
        if sc["code"] == "F1_severe":
            f1_severe_tension = model.tension.copy()

        shifts = match_shifts(healthy_paired, modes)
        mean_shift = float(np.mean([abs(s["shift_pct"]) for s in shifts]))
        n_cab = sum(1 for (k, _, _, _) in mods if k == "cable")
        n_sec = sum(1 for (k, _, _, _) in mods
                    if k in ("beam", "bearing"))
        inventory.append(dict(
            sid=sc["sid"], code=sc["code"], family=sc["family"] or "-",
            severity_class=sc["severity_class"],
            severity_pct=sc["severity_pct"],
            severity_label=sc["severity_label"],
            description=damage_summary(sc),
            n_damaged_cables=n_cab, n_damaged_sections=n_sec,
            f_float_hz=modes[0]["f"],
            mean_abs_shift_pct=mean_shift,
            max_stress_ratio=float(model.stress_ratios().max()),
            min_tension_MN=float(model.tension.min() / 1e6),
            max_tension_MN=float(model.tension.max() / 1e6),
            equilibrium_residual=model.equilibrium_residual(),
            mean_record_rms_ms2=float(np.mean(rms))))
        for s in shifts:
            shift_rows.append(dict(
                sid=sc["sid"], code=sc["code"], label=s["label"],
                f_healthy=s["f_healthy"], f_damaged=s["f_damaged"],
                mac=round(s["mac"], 4),
                shift_pct=round(s["shift_pct"], 4)))
        print(f"  [{sc['sid']:2d}] {sc['code']:14s} "
              f"mean|df| {mean_shift:6.3f} %  "
              f"sigma/fu max {model.stress_ratios().max():.3f}  "
              f"RMS {np.mean(rms) / 9.81e-3:5.2f} mg  "
              f"[{time.time() - t0:5.1f} s]")
        del records, recs

    # ------------------------------------------------------------------
    # exports
    # ------------------------------------------------------------------
    with open(os.path.join(RESULTS, "scenario_inventory.csv"), "w",
              newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(inventory[0].keys()))
        w.writeheader()
        for row in inventory:
            w.writerow({k: (round(v, 6) if isinstance(v, float) else v)
                        for k, v in row.items()})
    with open(os.path.join(RESULTS, "scenario_modal_shifts.csv"), "w",
              newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(shift_rows[0].keys()))
        w.writeheader()
        for row in shift_rows:
            w.writerow({k: (round(v, 6) if isinstance(v, float) else v)
                        for k, v in row.items()})

    fig_sensor_layout(geo_base, channels,
                      os.path.join(FIGURES, "fig06_sensor_layout.png"))
    fig_healthy_record(healthy_rec0, base_modes,
                       os.path.join(FIGURES, "fig07_healthy_record.png"))
    if not quick:
        fig_severity_shifts(
            inventory, os.path.join(FIGURES, "fig08_severity_shifts.png"))
        fig_force_redistribution(
            base.tension, f1_severe_tension, geo_base,
            os.path.join(FIGURES, "fig09_force_redistribution.png"))
    print("exports written")

    # ------------------------------------------------------------------
    # verification
    # ------------------------------------------------------------------
    checks = []
    n_expected = 37 if not quick else 4
    checks.append(Check("scenarios generated (healthy + 12 x 3)",
                        len(inventory) == n_expected, len(inventory),
                        f"{n_expected}"))
    if not quick:
        labels = [r["severity_label"] for r in inventory]
        comp = [labels.count(lab) for lab in
                ("healthy", "light", "moderate", "severe")]
        checks.append(Check("severity composition healthy/light/mod/severe",
                            comp == [1, 12, 12, 12], comp,
                            "[1, 12, 12, 12]"))

    stage1 = {}
    with open(os.path.join(RESULTS, "modal_calibration.csv")) as fh:
        for row in csv.DictReader(fh):
            stage1[row["label"]] = float(row["f_FE_calibrated_Hz"])
    diffs = [abs(f_ref[lab] - stage1.get(lab, np.nan)) / f_ref[lab]
             for lab in f_ref]
    checks.append(Check(
        "healthy paired frequencies match the stage-1 export",
        bool(stage1) and max(diffs) < 1e-3,
        f"{max(diffs):.2e} max rel diff", "< 1e-3 (CSV is rounded to 4 dp)"))
    checks.append(Check(
        "longitudinal floating frequency of the healthy state",
        abs(base_modes[0]["f"] - 0.0700) < 5e-4,
        f"{base_modes[0]['f']:.4f} Hz", "0.0700 +/- 0.0005 Hz"))

    # The seven measured verticals must be visible as resonances in the
    # simulated ambient records themselves (not only in the eigen table).
    # Each FE mode shape is used as a spatial filter: projecting the 43
    # deck-vertical channels of a full-length healthy record onto the mode
    # shape suppresses the other modes, so the projected signal's PSD must
    # peak at the mode frequency.  The peak (strongest of the three bins
    # around f; a resonance narrower than the 1/60 s Welch bin can split
    # its energy between neighbouring bins) is compared against the local
    # background median.  A dedicated full-length record (seed sid 999)
    # carries the check so that it is identical in quick mode, where the
    # 20 s smoke records cannot resolve the 0.04 Hz mode spacing.
    deck_idx = [i for i, c in enumerate(channels)
                if c["kind"] == "deck-vertical"]
    rng_chk = np.random.default_rng((P.EXCITATION["seed_base"], 999, 0))
    _, acc_chk = simulate_record(base, base_modes, rng_chk)
    acc_chk = add_sensor_noise(acc_chk, rng_chk)
    deck_chk = acc_chk[deck_idx].astype(np.float64)
    by_f = {round(m["f"], 6): m for m in base_modes}
    dof_map = base.dof_map
    rises = []
    for (lab, f_h, _w) in healthy_paired:
        w_ch = np.array([by_f[round(f_h, 6)]["phi_free"][
                             dof_map[3 * channels[i]["node"] + 1]]
                         for i in deck_idx])
        p = deck_chk.T @ w_ch / np.dot(w_ch, w_ch)
        f_psd, psd_p = welch(p, fs=P.EXCITATION["fs"], nperseg=p.shape[0])
        df = f_psd[1] - f_psd[0]
        k = int(round(f_h / df))
        peak = float(np.max(psd_p[max(k - 1, 1):k + 2]))
        band = np.abs(f_psd - f_h) <= 0.15
        band[max(k - 3, 1):k + 4] = False
        rises.append(peak / max(float(np.median(psd_p[band])), 1e-30))
    checks.append(Check(
        "resonant rise of the 7 verticals in a full-length healthy record",
        min(rises) >= 2.0,
        f"{min(rises):.1f}..{max(rises):.1f} x local background",
        ">= 2 x each mode"))

    exact, detail = True, "all listed properties, exact factors"
    for sc in SCENARIO_LIST:
        if sc["family"] is None:
            continue
        if quick and sc["sid"] not in (1, 13, 25):
            continue
        geo_d = build_geometry(scales)
        mods, _ = apply_damage(geo_d, sc)
        listed = {eid for (_, eid, _, _) in mods}
        for (kind, eid, prop, factor) in mods:
            if kind == "cable":
                ratio = geo_d.cables[eid].area / \
                    geo_base.cables[eid].area
            else:
                ratio = geo_d.beams[eid].__dict__[prop] / \
                    geo_base.beams[eid].__dict__[prop]
            if abs(ratio - factor) > 1e-12:
                exact, detail = False, f"{sc['code']} element {eid}"
        for c, cb in zip(geo_d.cables, geo_base.cables):
            if c.eid not in listed and c.area != cb.area:
                exact, detail = False, f"{sc['code']} stray cable change"
        for be, bb in zip(geo_d.beams, geo_base.beams):
            if be.eid not in listed and (be.I != bb.I or be.A != bb.A):
                exact, detail = False, f"{sc['code']} stray section change"
    checks.append(Check("damage application exactness (documented = applied)",
                        exact, detail, "exact"))

    res = max(r["equilibrium_residual"] for r in inventory)
    checks.append(Check("post-damage equilibrium residual", res < 1e-9,
                        f"{res:.1e}", "< 1e-9"))
    tmin = min(r["min_tension_MN"] for r in inventory)
    checks.append(Check("minimum stay force after damage (no slack)",
                        tmin > 0.0, f"{tmin:.2f} MN", "> 0"))
    smax = max(r["max_stress_ratio"] for r in inventory if r["sid"] > 0)
    checks.append(Check(
        "maximum stay stress ratio after damage", smax < 0.55,
        f"{smax:.3f} of fu",
        "< 0.55 (strand loss overloads the remaining wires)"))

    sev_by_code = {r["code"]: r["severity_class"] for r in inventory}
    severe_shifts = [s["shift_pct"] for s in shift_rows
                     if s["sid"] > 0 and sev_by_code[s["code"]] >= 2]
    mean_signed = float(np.mean(severe_shifts)) if severe_shifts else 0.0
    checks.append(Check(
        "mean signed frequency shift of the verticals (severe scenarios)",
        (mean_signed < 0.0) if severe_shifts else True,
        f"{mean_signed:.3f} %" if severe_shifts else "none in quick mode",
        "< 0 (stiffness loss lowers the frequencies)"))
    if not quick:
        sh = {r["code"]: r["mean_abs_shift_pct"] for r in inventory}
        mono = all(sh[f"{f}_light"] < sh[f"{f}_moderate"] <
                  sh[f"{f}_severe"]
                   for f in sorted({r["family"] for r in inventory
                                    if r["family"] != "-"}))
        checks.append(Check(
            "frequency response monotone in severity (all 12 families)",
            mono, "monotone" if mono else "violation found",
            "light < moderate < severe"))

    lo, hi = 1.0e-4, 5.0e-2
    checks.append(Check(
        "record acceleration RMS inside the ambient band",
        lo < min(stat["rms_all"]) and max(stat["rms_all"]) < hi,
        f"{min(stat['rms_all']):.1e}..{max(stat['rms_all']):.1e} m/s^2",
        f"{lo:.0e}..{hi:.0e} (0.01..5 mg)"))
    checks.append(Check(
        "record matrices finite with 64 channels",
        stat["shapes_ok"] and stat["finite_ok"],
        "finite" if stat["finite_ok"] else "NaN found", "all records"))
    # Independence of the excitations: pairwise signed correlation of the
    # mid-span channel between the records of the same scenario.  Ambient
    # records are narrow-band (modal correlation times of tens of seconds),
    # so a SINGLE correlation estimate is intrinsically scattered and its
    # sample mean is tested against its own standard error instead of a
    # fixed bound.  The defect of the earlier dataset - one shared
    # excitation, correlation exactly 1.000 - fails both this test and the
    # near-duplicate test catastrophically.
    corr_pairs = stat["corr_pairs"]
    se = (float(np.std(corr_pairs, ddof=1)) /
          np.sqrt(len(corr_pairs))) if len(corr_pairs) > 1 else 1.0
    checks.append(Check(
        "excitation independence: mean pairwise record correlation",
        len(corr_pairs) > 0 and
        abs(float(np.mean(corr_pairs))) < 2.0 * se,
        f"mean {np.mean(corr_pairs):+.3f} +- {2.0 * se:.3f} (2 SE) "
        f"over {len(corr_pairs)} pairs",
        "|mean| < 2 SE (zero mean at 95 % confidence)"))
    checks.append(Check(
        "no near-duplicate record pair",
        len(corr_pairs) > 0 and max(abs(c) for c in corr_pairs) < 0.95,
        f"max |r| {max(abs(c) for c in corr_pairs):.3f}", "< 0.95"))
    checks.append(Check(
        "all excitation seeds distinct",
        len(set(stat["seeds"])) == len(stat["seeds"]),
        f"{len(set(stat['seeds']))} unique seeds", "no shared seed"))

    sc_d = 1
    sc = next(s for s in SCENARIO_LIST if s["sid"] == sc_d)
    _, model_d, modes_d, _, _ = build_scenario_model(
        scales, sc, base.eps0, base.E_eq)
    accs = []
    for _ in range(2):
        rng = np.random.default_rng((P.EXCITATION["seed_base"], sc_d, 0))
        _, acc = simulate_record(model_d, modes_d, rng, T_burn=60.0,
                                 fs_rec=200.0, T_rec=20.0, chunk_s=10.0)
        accs.append(add_sensor_noise(acc, rng))
    dev = float(np.max(np.abs(accs[0] - accs[1])))
    checks.append(Check("re-simulation of one scenario is deterministic",
                        dev == 0.0, f"max difference {dev:.1e}",
                        "bitwise identical"))

    files = sorted(os.listdir(RECORDS)) if os.path.isdir(RECORDS) else []
    checks.append(Check("record files on disk", len(files) == n_expected,
                        f"{len(files)} files", f"{n_expected}"))
    with np.load(os.path.join(RECORDS, files[0])) as probe:
        checks.append(Check(
            "record file structure (n_rec x 64 x N, float32)",
            probe["acc"].shape[1] == 64 and
            probe["acc"].dtype == np.float32,
            f"{probe['acc'].shape} {probe['acc'].dtype}",
            "(n_rec, 64, N) float32"))

    n_fail = check_header(checks)

    # ------------------------------------------------------------------
    # summary + report
    # ------------------------------------------------------------------
    with open(os.path.join(RESULTS, "summary.json")) as fh:
        summary = json.load(fh)
    summary["stage_2"] = dict(
        n_scenarios=len(inventory),
        n_records_per_scenario=P.EXCITATION["n_records"] if not quick
        else 2,
        n_channels=len(channels), fs_hz=P.EXCITATION["fs"],
        duration_s=P.EXCITATION["duration"] if not quick else 20.0,
        n_modes_baseline=len(base_modes), zeta=P.EXCITATION["zeta"],
        checks_total=len(checks), checks_failed=n_fail,
        runtime_s=round(time.time() - t_start, 1))
    with open(os.path.join(RESULTS, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)

    write_report(os.path.join(RESULTS, "scenario_report.md"),
                 inventory, shift_rows, checks, quick,
                 time.time() - t_start)

    print(f"\nstage 2 {'(quick) ' if quick else ''}complete in "
          f"{time.time() - t_start:.0f} s: "
          f"{len(checks)} checks, {n_fail} failed")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
