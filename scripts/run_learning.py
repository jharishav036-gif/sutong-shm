"""
Stage-4 driver: the learning pipeline of the study.

Run from the repository root:

    python scripts/run_learning.py            # full run, ~40-60 minutes
    python scripts/run_learning.py --quick   # fold 0, seed 0 only
    python scripts/run_learning.py --blocks 3   # at most 3 new protocol
                                                # blocks, then exit
                                                # (resumable checkpoints)

The protocol is checkpointed block by block (the PI-DL protocol, the four
ablations, the classical baselines, the three torch baselines) under
``results/.learning_ckpt``; an interrupted run resumes where it stopped,
``--fresh`` discards the checkpoints, and the checkpoints are removed
again once the final artefacts are written.

The PI-DL network (per-channel CWT-CNN, two-layer sensor-graph GNN,
evidential head) is trained under the scenario-level cross-validation
protocol: 5 GroupKFold folds over the 13 damage groups (the healthy
state plus 12 families), 3 seeds per fold, 15 training runs in total.
The feature standardisation of every fold is fitted on the training
windows only.  The evidential loss carries the Sensoy alpha-tilde
correction; every run trains over the fixed 120-epoch budget of the
protocol and the reported metrics are taken at each run's best epoch
(validation accuracy, ECE tie-break).

The same folds and the same standardised features feed the five
baselines (SVM, Random Forest, 1D-CNN, LSTM, Transformer) and the four
real ablations (no-GNN, no-CWT, no-scalars, no-evidential), each under
the full 15-run protocol.

Artefacts:

* ``results/learning_fold_summary.csv``  - 15 PI-DL runs;
* ``results/learning_confusion.csv``     - pooled confusion matrix;
* ``results/learning_class_distribution.csv`` - actual fold compositions;
* ``results/ablation.csv``              - 15-run ablation table;
* ``results/baseline_comparison.csv``    - 15-run baseline table;
* ``results/learning_hyperparameters.csv``;
* ``results/trained_models/``            - the 15 fold-seed state dicts;
* ``results/figures/fig13..fig19``      - embedding, training curves,
  confusion matrix, ablation bars;
* ``results/learning_report.md`` and the stage-4 block of
  ``results/summary.json``.

Every run trains over the full 120-epoch budget of the protocol (no early
stopping); the reported metrics are evaluated at each run's best epoch
(validation accuracy with an ECE tie-break).  The honest outcome of the
leave-family-out task is reported as measured - near-chance severity
grading on the subtle families and clean binary healthy-vs-damage
detection - and the automated checks assert mechanical correctness
(no leakage, identical protocol across models, determinism), not
outcome superiority.
"""

import csv
import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sutong_fe import params as P
from sutong_fe import learning as LR

RESULTS = os.path.join(ROOT, "results")
FIGURES = os.path.join(RESULTS, "figures")
FEATDIR = os.path.join(ROOT, "data", "features")
MODELS = os.path.join(RESULTS, "trained_models")

CLASS_NAMES = ["healthy", "light", "moderate", "severe"]


# ---------------------------------------------------------------------------
class Check:
    def __init__(self, name, passed, value, tolerance):
        self.name = name
        self.passed = bool(passed)
        self.value = value
        self.tolerance = tolerance


def check_header(checks):
    print("\n== Learning verification "
          "================================================")
    for c in checks:
        mark = "PASS" if c.passed else "FAIL"
        print(f"  [{mark}] {c.name}: {c.value}  ({c.tolerance})")
    n_fail = sum(1 for c in checks if not c.passed)
    print(f"  -> {len(checks)} checks, {n_fail} failed")
    return n_fail


def load_features():
    d = np.load(os.path.join(FEATDIR, "features_windows.npz"))
    return (d["X"], d["sid"], d["family"], d["sev_class"],
            d["record"], d["window"], list(d["code"]))


# ---------------------------------------------------------------------------
def make_ablation_model(W, variant):
    """The PI-DL variants of the ablation study (real module removal)."""
    kw = dict(use_gnn=True, use_cwt=True, use_scalars=True)
    if variant in ("full", "no_evidential"):
        pass
    elif variant == "no_gnn":
        kw["use_gnn"] = False
    elif variant == "no_cwt":
        kw["use_cwt"] = False
    elif variant == "no_scalars":
        kw["use_scalars"] = False
    else:
        raise ValueError(variant)
    return LR.PIDL(W, **kw)


def run_protocol(W, Xs, y, groups, variant="full", evidential=True,
                 quick=False, collect_states=False):
    """One 15-run protocol (or fold-0 / seed-0 when quick).

    Returns (runs, pooled_confusion) where each run dict carries the
    fold / seed / sizes / metrics.
    """
    folds = LR.group_folds(groups, P.LEARNING["n_folds"])
    seeds = P.LEARNING["seeds"] if not quick else (P.LEARNING["seeds"][0],)
    fold_iter = enumerate(folds) if not quick else [(0, folds[0])]
    runs = []
    confusion = np.zeros((4, 4), dtype=int)
    states = []
    histories = {}
    for fold_id, (tr, va) in fold_iter:
        mu, sd = LR.fold_scaler(Xs, tr)
        Z = LR.apply_scaler(Xs, mu, sd)
        cwt, scal = LR.make_model_inputs(Z)
        y = np.asarray(y)
        for seed in seeds:
            run_seed = 1000 * (fold_id + 1) + seed
            torch.manual_seed(run_seed)     # seeds the weight init too
            model = make_ablation_model(W, variant)
            hist, state, best_ep = LR.train_pidl(
                model, cwt[tr], scal[tr], y[tr], cwt[va], scal[va], y[va],
                seed=run_seed, evidential=evidential)
            model.load_state_dict(state)
            pred, conf, unc, ece, acc = LR.evaluate_pidl(
                model, cwt[va], scal[va], y[va])
            for t, p in zip(y[va], pred):
                confusion[t, p] += 1
            runs.append(dict(fold=fold_id, seed=seed, variant=variant,
                             n_train=len(tr), n_val=len(va),
                             best_epoch=best_ep, acc=acc, ece=ece,
                             u=unc))
            states.append((fold_id, seed, model.n_parameters(),
                           {k: v.clone() for k, v in state.items()}))
            histories[(fold_id, seed)] = hist
            print(f"    {variant}{'/CE' if not evidential else ''} "
                  f"fold {fold_id} seed {seed}: acc {acc:.3f} "
                  f"ece {ece:.3f} u {unc:.3f} (ep {best_ep})")
    if collect_states:
        return runs, confusion, states, histories
    return runs, confusion


def run_torch_baseline(kind, Xs, y, groups, quick=False):
    """15-run protocol of a torch baseline on the token matrix."""
    Xt = LR.baseline_input_matrix(Xs, "tokens")
    y = np.asarray(y)
    folds = LR.group_folds(groups, P.LEARNING["n_folds"])
    seeds = P.LEARNING["seeds"] if not quick else (P.LEARNING["seeds"][0],)
    fold_iter = enumerate(folds) if not quick else [(0, folds[0])]
    runs = []
    for fold_id, (tr, va) in fold_iter:
        mu, sd = LR.fold_scaler(Xt, tr)
        Z = LR.apply_scaler(Xt, mu, sd)
        for seed in seeds:
            run_seed = 1000 * (fold_id + 1) + seed
            torch.manual_seed(run_seed)
            torch_model = dict(
                cnn=LR.OneDCNN(n_in=384), lstm=LR.LSTMBaseline(),
                transformer=LR.TransformerBaseline())[kind]
            best = LR.train_torch_baseline(
                torch_model, Z[tr], y[tr], Z[va], y[va],
                seed=run_seed, kind=kind)
            runs.append(dict(fold=fold_id, seed=seed, kind=kind,
                             n_train=len(tr), n_val=len(va),
                             best_epoch=best["epoch"], acc=best["acc"]))
            print(f"    {kind} fold {fold_id} seed {seed}: "
                  f"acc {best['acc']:.3f}")
    return runs


def run_classical_baseline(kind, Xs, y, groups, quick=False):
    Xt = LR.baseline_input_matrix(Xs, "flat")
    y = np.asarray(y)
    folds = LR.group_folds(groups, P.LEARNING["n_folds"])
    seeds = P.LEARNING["seeds"] if not quick else (P.LEARNING["seeds"][0],)
    fold_iter = enumerate(folds) if not quick else [(0, folds[0])]
    runs = []
    for fold_id, (tr, va) in fold_iter:
        mu, sd = LR.fold_scaler(Xt, tr)
        Z = LR.apply_scaler(Xt, mu, sd)
        for seed in seeds:
            pred = LR.run_svm_rf(kind, Z[tr], y[tr], Z[va], seed)
            acc = float((pred == y[va]).mean())
            runs.append(dict(fold=fold_id, seed=seed, kind=kind,
                             n_train=len(tr), n_val=len(va), acc=acc))
            print(f"    {kind} fold {fold_id} seed {seed}: acc {acc:.3f}")
    return runs


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------
def fig_embedding(emb, y, path):
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.8),
                             constrained_layout=True)
    colors = {0: "#2ca02c", 1: "#1f77b4", 2: "#ff7f0e", 3: "#d62728"}
    labels = {0: "healthy", 1: "light", 2: "moderate", 3: "severe"}
    p = PCA(n_components=2, random_state=0).fit_transform(emb)
    t = TSNE(n_components=2, perplexity=25, random_state=0,
             init="pca").fit_transform(emb)
    for ax, proj, name in ((axes[0], t, "t-SNE"), (axes[1], p, "PCA")):
        for c in (0, 1, 2, 3):
            m = y == c
            ax.scatter(proj[m, 0], proj[m, 1], s=10, alpha=0.7,
                       color=colors[c], label=labels[c])
        ax.set_title(f"{name} of the graph embedding\n"
                     "(fold 0, seed 0 model)")
        ax.grid(alpha=0.3)
    axes[0].set_xlabel("dim 1"); axes[0].set_ylabel("dim 2")
    axes[1].set_xlabel("PC 1"); axes[1].set_ylabel("PC 2")
    axes[0].legend(fontsize=8, loc="best")
    fig.savefig(path, dpi=170)
    plt.close(fig)


def fig_training_curves(hists, path):
    """Training curves, mean +/- std over the runs (paper Figure 15).

    Panels: training/validation loss, training/validation accuracy,
    validation ECE, training/validation uncertainty u.
    """
    keys = [("train_loss", "train"), ("val_loss", "val"),
            ("train_acc", "train"), ("val_acc", "val"),
            ("val_ece", "val"), ("train_u", "train"),
            ("val_u", "val")]
    n_ep = max(len(h) for h in hists)
    series = {}
    for key, split in keys:
        grid = np.full((len(hists), n_ep), np.nan)
        for i, h in enumerate(hists):
            v = np.array([hh[key] for hh in h])
            grid[i, : len(v)] = v
        series[key] = (np.nanmean(grid, axis=0), np.nanstd(grid, axis=0))
    fig, axes = plt.subplots(2, 2, figsize=(10.8, 7.4),
                             constrained_layout=True)
    style = dict(train=dict(color="#1f77b4", label="training"),
                 val=dict(color="#d62728", label="validation"))
    ax_map = [("loss", ["train_loss", "val_loss"]),
              ("accuracy", ["train_acc", "val_acc"]),
              ("ECE (validation)", ["val_ece"]),
              ("mean uncertainty u", ["train_u", "val_u"])]
    ep = np.arange(1, n_ep + 1)
    for ax, (title, kk) in zip(axes.ravel(), ax_map):
        for key in kk:
            split = next(s for k, s in keys if k == key)
            m, sd = series[key]
            ax.plot(ep, m, lw=1.5, **style[split])
            ax.fill_between(ep, m - sd, m + sd, alpha=0.18,
                            color=style[split]["color"])
        ax.set_xlabel("epoch")
        ax.set_title(f"{title} (mean ± std, {len(hists)} runs)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc="best")
    fig.suptitle("PI-DL training over the 15-run cross-validation "
                 "protocol")
    fig.savefig(path, dpi=170)
    plt.close(fig)


def fig_confusion(cm, path):
    fig, ax = plt.subplots(figsize=(6.2, 5.4), constrained_layout=True)
    im = ax.imshow(cm, cmap="Blues")
    tot = cm.sum()
    for i in range(4):
        for j in range(4):
            v = cm[i, j]
            ax.text(j, i, f"{v}\n({100 * v / max(cm[i].sum(), 1):.0f}%)",
                    ha="center", va="center", fontsize=9,
                    color="white" if v > 0.55 * cm.max() else "black")
    ax.set_xticks(range(4)); ax.set_yticks(range(4))
    ax.set_xticklabels(CLASS_NAMES)
    ax.set_yticklabels(CLASS_NAMES)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(f"PI-DL pooled confusion\n"
                 f"15 runs, {tot} validation windows")
    fig.colorbar(im, ax=ax)
    fig.savefig(path, dpi=170)
    plt.close(fig)


def fig_ablation(table, path):
    names = [r["variant"] for r in table]
    disp = [n.replace("_", " ") for n in names]
    disp = [n if len(n) <= 16 else n.replace(" (", "\n(") for n in disp]
    acc = np.array([r["acc_mean"] for r in table])
    std = np.array([r["acc_std"] for r in table])
    fig, ax = plt.subplots(figsize=(8.8, 4.6), constrained_layout=True)
    x = np.arange(len(names))
    ax.bar(x, acc, yerr=std, capsize=4, color="#1f77b4", alpha=0.85)
    for xi, a, s in zip(x, acc, std):
        ax.text(xi, a + s + 1.0, f"{a:.1f}", ha="center", fontsize=9)
    ax.set_ylim(30, float((acc + std).max()) + 6)
    ax.set_xticks(x)
    ax.set_xticklabels(disp, fontsize=9)
    ax.set_ylabel("validation accuracy (%), 15-run mean +/- std")
    ax.set_title("Ablations of the PI-DL modules (real module removal)")
    ax.grid(alpha=0.3, axis="y")
    fig.savefig(path, dpi=170)
    plt.close(fig)


# ---------------------------------------------------------------------------
def write_csv(path, header, rows):
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)


def load_summary():
    path = os.path.join(RESULTS, "summary.json")
    if os.path.exists(path):
        with open(path) as fh:
            return json.load(fh)
    return {}


CKPT = os.path.join(RESULTS, ".learning_ckpt")
N_BLOCKS = 9      # pidl + 4 ablations + classical + 3 torch baselines


def _save_ckpt(name, obj):
    os.makedirs(CKPT, exist_ok=True)
    with open(os.path.join(CKPT, name + ".json"), "w") as fh:
        json.dump(obj, fh)


def _load_ckpt(name):
    path = os.path.join(CKPT, name + ".json")
    if os.path.exists(path):
        with open(path) as fh:
            return json.load(fh)
    return None


def _blocks_arg(argv):
    """--blocks N: complete at most N new protocol blocks per invocation."""
    if "--blocks" in argv:
        i = argv.index("--blocks")
        try:
            return int(argv[i + 1])
        except (IndexError, ValueError):
            pass
    return 10 ** 6


def _budget_note(done):
    n_ckpt = 0
    if os.path.isdir(CKPT):
        n_ckpt = len([f for f in os.listdir(CKPT)
                      if f.endswith(".json")])
    print(f"\n[resume] block budget reached after {done} new block(s); "
          f"{N_BLOCKS - n_ckpt} block(s) still to run - re-run the script "
          f"to continue from the checkpoints")


def main():
    quick = "--quick" in sys.argv
    max_blocks = _blocks_arg(sys.argv)
    t0 = time.time()
    os.makedirs(MODELS, exist_ok=True)
    os.makedirs(FIGURES, exist_ok=True)
    if "--fresh" in sys.argv:
        import shutil
        shutil.rmtree(CKPT, ignore_errors=True)

    X, sid, family, y, rec, win, code = load_features()
    Xs = LR.log_transform(X)
    y = np.asarray(y)
    groups = np.asarray(family)

    # sensor kernel of the graph
    d = np.load(os.path.join(ROOT, "data", "scenario_records",
                             "scenario_00_H.npz"))
    W, sigma_d = LR.sensor_kernel(
        np.stack([d["channel_x_m"], d["channel_y_m"]], axis=1))

    print(f"== Stage 4: learning pipeline "
          f"(sigma_d = {sigma_d:.0f} m) ===============")

    done = 0

    def out_of_budget():
        return (not quick) and done >= max_blocks

    # --- block 1: the PI-DL protocol ---------------------------------------
    pidl = _load_ckpt("pidl") if not quick else None
    states = None
    if pidl is not None:
        runs = pidl["runs"]
        confusion = np.array(pidl["confusion"], dtype=int)
        n_params = int(pidl["n_params"])
        histories = dict(pidl["histories"])
        print(f"[resume] PI-DL protocol loaded from checkpoint "
              f"({len(runs)} runs)")
    else:
        runs, confusion, states, histories = run_protocol(
            W, Xs, y, groups, variant="full", evidential=True, quick=quick,
            collect_states=True)
        n_params = int(states[0][2])
        if not quick:
            for fold_id, seed, _, state in states:
                torch_like = {k: v.numpy() for k, v in state.items()}
                np.savez_compressed(
                    os.path.join(MODELS, f"pidl_f{fold_id}_s{seed}.npz"),
                    **torch_like)
            _save_ckpt("pidl", dict(
                runs=runs, confusion=confusion.tolist(),
                n_params=n_params,
                histories={f"f{f}_s{s}": h
                           for (f, s), h in histories.items()}))
            print(f"[checkpoint] PI-DL protocol complete "
                  f"({len(runs)} runs, 15 model files exported)")
            done += 1
            if out_of_budget():
                _budget_note(done)
                return 0

    accs = np.array([r["acc"] for r in runs])
    us = np.array([r["u"] for r in runs])
    eces = np.array([r["ece"] for r in runs])
    u_id_mean = float(us.mean())

    # fold composition table (class distribution)
    folds = LR.group_folds(groups, P.LEARNING["n_folds"])
    comp_rows = []
    for fold_id, (tr, va) in enumerate(folds):
        for split, idx in (("train", tr), ("val", va)):
            counts = [int((y[idx] == c).sum()) for c in range(4)]
            comp_rows.append([fold_id, split, len(idx)] + counts)

    # --- blocks 2-5: the ablations ------------------------------------------
    ablation_rows = [dict(variant="full (PI-DL)", acc_mean=100 * accs.mean(),
                          acc_std=100 * accs.std(), u_mean=u_id_mean,
                          n_params=n_params)]
    for variant, evid in (("no_gnn", True), ("no_cwt", True),
                          ("no_scalars", True), ("no_evidential", False)):
        ck = _load_ckpt(f"abl_{variant}") if not quick else None
        if ck is not None:
            r = ck["runs"]
            print(f"[resume] ablation {variant} loaded from checkpoint")
        else:
            r, _ = run_protocol(W, Xs, y, groups, variant=variant,
                                evidential=evid, quick=quick)
            if not quick:
                _save_ckpt(f"abl_{variant}", dict(runs=r))
                done += 1
                if out_of_budget():
                    _budget_note(done)
                    return 0
        a = np.array([q["acc"] for q in r])
        ablation_rows.append(dict(
            variant=variant + (" (softmax CE)" if not evid else ""),
            acc_mean=100 * a.mean(), acc_std=100 * a.std(),
            u_mean=float(np.mean([q["u"] for q in r if np.isfinite(q["u"])]))
            if evid else float("nan"),
            n_params=LR.PIDL(W, **dict(
                use_gnn=variant != "no_gnn", use_cwt=variant != "no_cwt",
                use_scalars=variant != "no_scalars")).n_parameters()))

    # --- block 6: the classical baselines ------------------------------------
    baseline_rows = []
    baseline_runs_all = []
    ck = _load_ckpt("base_classical") if not quick else None
    if ck is not None:
        classical = {"svm": ck["svm"], "rf": ck["rf"]}
        print("[resume] classical baselines loaded from checkpoint")
    else:
        classical = {k: run_classical_baseline(k, Xs, y, groups,
                                               quick=quick)
                     for k in ("svm", "rf")}
        if not quick:
            _save_ckpt("base_classical", classical)
            done += 1
            if out_of_budget():
                _budget_note(done)
                return 0
    for kind in ("svm", "rf"):
        r = classical[kind]
        baseline_runs_all.append(r)
        a = np.array([q["acc"] for q in r])
        baseline_rows.append(dict(kind=kind, acc_mean=100 * a.mean(),
                                  acc_std=100 * a.std()))

    # --- blocks 7-9: the torch baselines --------------------------------------
    for kind in ("cnn", "lstm", "transformer"):
        ck = _load_ckpt(f"base_{kind}") if not quick else None
        if ck is not None:
            r = ck["runs"]
            print(f"[resume] {kind} baseline loaded from checkpoint")
        else:
            r = run_torch_baseline(kind, Xs, y, groups, quick=quick)
            if not quick:
                _save_ckpt(f"base_{kind}", dict(runs=r))
                done += 1
                if out_of_budget():
                    _budget_note(done)
                    return 0
        baseline_runs_all.append(r)
        a = np.array([q["acc"] for q in r])
        baseline_rows.append(dict(kind=kind, acc_mean=100 * a.mean(),
                                  acc_std=100 * a.std()))

    # --- figures ----------------------------------------------------------
    # embeddings from the fold-0 / seed-0 model
    mu, sd = LR.fold_scaler(Xs, folds[0][0])
    Z = LR.apply_scaler(Xs, mu, sd)
    cwt_all, scal_all = LR.make_model_inputs(Z)
    emb_model = make_ablation_model(W, "full")
    if states is not None:          # fresh in-memory run
        emb_model.load_state_dict(states[0][3])
    else:                           # resumed run: reload the exported model
        f0 = np.load(os.path.join(MODELS, "pidl_f0_s0.npz"))
        emb_model.load_state_dict(
            {k: torch.as_tensor(f0[k]) for k in f0.files})
    emb = LR.graph_embeddings(emb_model, cwt_all, scal_all)
    fig_embedding(emb, y, os.path.join(FIGURES,
                                       "fig13_embedding_space.png"))
    fig_training_curves(list(histories.values()),
                        os.path.join(FIGURES, "fig17_training_curves.png"))
    fig_confusion(confusion, os.path.join(FIGURES, "fig18_confusion.png"))
    fig_ablation(ablation_rows, os.path.join(FIGURES,
                                             "fig19_ablation.png"))

    # --- fold scalers (belong to the exported models) ----------------------
    if not quick:
        scalers = []
        for fold_id, (tr, _) in enumerate(folds):
            mu, sd = LR.fold_scaler(Xs, tr)
            scalers.append((mu, sd))
        np.savez_compressed(
            os.path.join(MODELS, "fold_scalers.npz"),
            mu=np.stack([m for m, _ in scalers]),
            sd=np.stack([s for _, s in scalers]),
            sigma_d=np.array(sigma_d),
            W=W.astype(np.float64))

    # --- tables -----------------------------------------------------------
    write_csv(os.path.join(RESULTS, "learning_fold_summary.csv"),
              ["fold", "seed", "n_train", "n_val", "best_epoch",
               "val_acc", "val_ece", "val_u"],
              [[r["fold"], r["seed"], r["n_train"], r["n_val"],
                r["best_epoch"], round(r["acc"], 4), round(r["ece"], 4),
                round(r["u"], 4)] for r in runs])
    write_csv(os.path.join(RESULTS, "learning_confusion.csv"),
              ["true\\pred"] + CLASS_NAMES + ["total"],
              [[CLASS_NAMES[i]] + list(confusion[i])
               + [int(confusion[i].sum())] for i in range(4)])
    write_csv(os.path.join(RESULTS, "learning_class_distribution.csv"),
              ["fold", "split", "n"] + CLASS_NAMES, comp_rows)
    write_csv(os.path.join(RESULTS, "ablation.csv"),
              ["variant", "acc_mean_pct", "acc_std_pct", "u_mean",
               "n_params"],
              [[r["variant"], round(r["acc_mean"], 2),
                round(r["acc_std"], 2),
                round(r["u_mean"], 4) if np.isfinite(r["u_mean"]) else "",
                r["n_params"]] for r in ablation_rows])
    write_csv(os.path.join(RESULTS, "baseline_comparison.csv"),
              ["model", "acc_mean_pct", "acc_std_pct"],
              [["SVM (RBF)" if r["kind"] == "svm"
                else "Random Forest" if r["kind"] == "rf"
                else "1D-CNN" if r["kind"] == "cnn"
                else "LSTM" if r["kind"] == "lstm" else "Transformer",
                round(r["acc_mean"], 2), round(r["acc_std"], 2)]
               for r in baseline_rows])
    write_csv(os.path.join(RESULTS, "learning_hyperparameters.csv"),
              ["hyperparameter", "value", "note"],
              [["n_classes", 4, "severity classes"],
               ["n_features", 387, "320 CWT + 64 Hilbert + 3 scalars"],
               ["n_window_samples", len(y), "37 x 3 records x 3 windows"],
               ["gnn_sigma_d_m", round(sigma_d, 1),
                "median pairwise sensor distance"],
               ["n_folds", 5, "GroupKFold over 13 damage groups"],
               ["n_seeds", 3, "seeds 0/1/2"],
               ["n_runs", len(runs), "5 folds x 3 seeds"],
               ["batch_size", P.LEARNING["batch_size"], ""],
               ["epochs", P.LEARNING["epochs"],
                "fixed budget, no early stopping; best-epoch selection by "
                "val accuracy, ECE tie-break"],
               ["model_selection", "val accuracy, ECE tie-break",
                "ECE-only selection degenerates to the uniform solution"],
               ["lr", P.LEARNING["lr"], "Adam"],
               ["weight_decay", P.LEARNING["weight_decay"], ""],
               ["scheduler", "StepLR(40, 0.5)", ""],
               ["lambda_kl", P.LEARNING["lambda_kl"],
                "KL weight cap; annealed as min(1, t/10) Sensoy schedule, "
                "alpha-tilde correction inside"],
               ["evidence_cap", P.LEARNING["evidence_cap"],
                "numerical guard"]])

    # --- checks -----------------------------------------------------------
    checks = []
    n_runs_exp = 15 if not quick else 1
    checks.append(Check("protocol completes 15 runs", len(runs) == n_runs_exp,
                        f"{len(runs)}", f"{n_runs_exp}"))
    scen_val_counts = {}
    for fold_id, (tr, va) in enumerate(folds):
        s_tr = set(sid[tr]); s_va = set(sid[va])
        overlap = s_tr & s_va
        scen_val_counts.setdefault(fold_id, set()).update(s_va)
        checks.append(Check(
            f"fold {fold_id}: no scenario in both train and val",
            len(overlap) == 0, f"{len(overlap)} shared", "0"))
    seen = [s for vs in scen_val_counts.values() for s in vs]
    checks.append(Check("each scenario in exactly one validation fold",
                        sorted(seen) == sorted(set(sid.tolist()))
                        and len(seen) == len(set(seen)),
                        f"{len(seen)} val placements over "
                        f"{len(set(sid.tolist()))} scenarios",
                        "each of 37 exactly once"))
    # scaler leakage: standardised validation columns are not zero-mean
    mu, sd = LR.fold_scaler(Xs, folds[0][0])
    Zva = LR.apply_scaler(Xs[folds[0][1]], mu, sd)
    drift = float(np.abs(Zva.mean(axis=0)).mean())
    checks.append(Check(
        "fold scaler fitted on training windows only",
        drift > 1e-3,
        f"val column mean {drift:.3f} (non-zero)",
        "> 0.001 (val keeps its own statistics)"))
    # the alpha-tilde fix must grow evidence on the training folds:
    # final-epoch mean u on train below the uniform-saturation level
    # an uncorrected constant-lambda loss produces on every split
    u_final = float(np.mean([h[-1]["train_u"] for h in histories.values()]))
    checks.append(Check(
        "evidence grows on the training folds (final-epoch u clear of "
        "uniform saturation)", u_final < 0.9, f"train u {u_final:.3f}",
        "< 0.9 (an uncorrected constant-lambda loss pins u at 0.911)"))
    # every baseline under the identical fold / seed protocol
    pidl_fs = sorted((r["fold"], r["seed"]) for r in runs)
    same_protocol = all(
        sorted((r["fold"], r["seed"]) for r in br) == pidl_fs
        for br in baseline_runs_all)
    checks.append(Check(
        "every baseline evaluated under the identical fold/seed protocol",
        same_protocol, f"{len(pidl_fs)} runs per model",
        "same (fold, seed) set as PI-DL"))
    # ablation structural differences
    full_par = n_params
    no_gnn_par = ablation_rows[1]["n_params"]
    checks.append(Check("no-GNN ablation removes message passing",
                        no_gnn_par != full_par,
                        f"{full_par} vs {no_gnn_par} parameters",
                        "different parameter count"))
    # determinism: retrain fold 0 / seed 0 exactly as run_protocol did
    tr, va = folds[0]
    mu, sd = LR.fold_scaler(Xs, tr)
    Z = LR.apply_scaler(Xs, mu, sd)
    cwt, scal = LR.make_model_inputs(Z)
    torch.manual_seed(1000)               # seeds the weight init too
    m2 = make_ablation_model(W, "full")
    m2_hist, m2_state, m2_ep = LR.train_pidl(
        m2, cwt[tr], scal[tr], y[tr], cwt[va], scal[va], y[va],
        seed=1000, evidential=True)
    ref_acc = runs[0]["acc"]
    m2.load_state_dict(m2_state)
    _, _, _, _, m2_acc = LR.evaluate_pidl(m2, cwt[va], scal[va], y[va])
    checks.append(Check("deterministic retraining of fold 0 / seed 0",
                        abs(m2_acc - ref_acc) < 1e-9,
                        f"max diff {abs(m2_acc - ref_acc):.1e}",
                        "identical metrics"))

    n_fail = check_header(checks)

    # --- report -----------------------------------------------------------
    per_class = [confusion[i, i] / max(confusion[i].sum(), 1)
                 for i in range(4)]
    with open(os.path.join(RESULTS, "learning_report.md"), "w") as fh:
        fh.write("# Sutong Bridge 2-D FE Model - Stage-4 Learning "
                 "Report\n\n")
        fh.write(f"Generated by ``scripts/run_learning.py`` in "
                 f"{time.time() - t0:.0f} s.\n\n")
        fh.write("## 1. Protocol\n\nScenario-level GroupKFold (5 folds "
                 "over the 13 damage groups, healthy state included), "
                 f"3 seeds per fold, {len(runs)} runs. The evidential "
                 "loss carries the Sensoy alpha-tilde correction; feature "
                 "standardisation is fitted on the training windows of "
                 f"each fold. Sensor kernel scale sigma_d = {sigma_d:.0f} "
                 "m (median pairwise sensor distance).\n\n")
        fh.write("## 2. Results\n\n")
        fh.write(f"| quantity | value |\n|---|---|\n")
        fh.write(f"| PI-DL accuracy | {100 * accs.mean():.1f} % "
                 f"+/- {100 * accs.std():.1f} % (15 runs) |\n")
        fh.write(f"| mean validation ECE | {eces.mean():.3f} |\n")
        fh.write(f"| mean validation uncertainty | {u_id_mean:.3f} |\n")
        fh.write(f"| per-class recall (H/L/M/S) | "
                 f"{', '.join(f'{v * 100:.0f} %' for v in per_class)} |\n")
        fh.write("\n## 3. Pooled confusion matrix (15 runs)\n\n")
        fh.write("| true \\ pred | " + " | ".join(CLASS_NAMES) + " |\n")
        fh.write("|---|" + "---|" * 4 + "\n")
        for i in range(4):
            fh.write(f"| {CLASS_NAMES[i]} | "
                     + " | ".join(str(v) for v in confusion[i]) + " |\n")
        fh.write("\n## 4. Baselines (same folds, same features)\n\n")
        fh.write("| model | accuracy (15 runs) |\n|---|---|\n")
        for r in baseline_rows:
            name = {"svm": "SVM (RBF)", "rf": "Random Forest",
                    "cnn": "1D-CNN", "lstm": "LSTM",
                    "transformer": "Transformer"}[r["kind"]]
            fh.write(f"| {name} | {r['acc_mean']:.1f} % +/- "
                     f"{r['acc_std']:.1f} % |\n")
        fh.write("\n## 5. Ablations (real module removal, 15 runs)\n\n")
        fh.write("| variant | accuracy | uncertainty | parameters |\n"
                 "|---|---|---|---|\n")
        for r in ablation_rows:
            u = f"{r['u_mean']:.3f}" if np.isfinite(r["u_mean"]) else "-"
            fh.write(f"| {r['variant']} | {r['acc_mean']:.1f} % +/- "
                     f"{r['acc_std']:.1f} % | {u} | {r['n_params']} |\n")
        fh.write("\n## 6. Automated checks\n\n")
        fh.write("| check | value | limit | result |\n|---|---|---|---|\n")
        for c in checks:
            mark = "PASS" if c.passed else "FAIL"
            fh.write(f"| {c.name} | {c.value} | {c.tolerance} | "
                     f"{mark} |\n")

    summary = load_summary()
    summary["stage_4"] = dict(
        n_window_samples=int(len(y)),
        n_runs=int(len(runs)),
        pidl_acc_pct=round(100 * float(accs.mean()), 1),
        pidl_acc_std_pct=round(100 * float(accs.std()), 1),
        val_ece_mean=round(float(eces.mean()), 3),
        val_u_mean=round(u_id_mean, 3),
        gnn_sigma_d_m=round(float(sigma_d), 1),
        checks_total=len(checks),
        checks_failed=n_fail,
        runtime_s=round(time.time() - t0, 1))
    with open(os.path.join(RESULTS, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)

    # --- checkpoint cleanup --------------------------------------------------
    if not quick:
        import shutil
        shutil.rmtree(CKPT, ignore_errors=True)

    print(f"\nstage 4 done in {time.time() - t0:.0f} s "
          f"({n_fail} failed checks)")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
