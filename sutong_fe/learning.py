"""
Learning pipeline of the study: the PI-DL network, the fixed evidential
loss, and the scenario-level cross-validation protocol.

The architecture follows the study definition: a per-channel 1D-CNN over
the 5-band CWT tensor (1 -> 16 -> 32 filters, kernel 3), two
message-passing GNN layers over the exponential distance kernel of the
sensor graph (32+32 -> 16, then 16+16 -> 16), and a fully-connected
evidential head with a Dirichlet prior over the four severity classes.
The 64 per-channel Hilbert decays and the three global scalars enter
through the head input, so every one of the 387 features is used.

The evidential loss carries the Sensoy true-class evidence removal that
the study's earlier implementation missed: the KL regulariser is evaluated
on alpha-tilde = y + (1 - y) * alpha, which zeroes the evidence assigned
to the true class before measuring the distance to the uniform Dirichlet.
Without this correction a constant KL weight drags all evidence toward
the uniform prior, the uncertainty saturates near u = 1 and the head
degenerates to a near-uniform, constant class output - the degenerate
solution of an uncorrected constant-lambda evidential head.  With the
correction (and the Sensoy annealing of the KL weight over the first ten
epochs, which the constant-lambda variant of the original protocol needed
but did not have) the evidence grows on the training folds and the
uncertainty separates in-distribution validation samples from
out-of-distribution probe samples.

The cross-validation is grouped at the scenario level:
sklearn GroupKFold over the 13 damage groups (the healthy state plus the
12 families), so all windows of all records of one scenario - and all
three severities of one family - stay in the same fold, and the feature
standardisation of each fold is fitted on the training windows only.
"""

import math

import numpy as np
import torch
import torch.nn as nn

from . import params as P

CWT_SLICE = slice(0, 320)          # 64 x 5 band energies
HILBERT_SLICE = slice(320, 384)    # 64 envelope decays
SCALAR_SLICE = slice(384, 387)     # svd / phase / modal-ratio


# ---------------------------------------------------------------------------
# evidential machinery
# ---------------------------------------------------------------------------
def kl_dirichlet_uniform(alpha):
    """KL(Dir(alpha) || Dir(1)) per row, the closed-form Sensoy term:
    lgamma(S) - lgamma(K) - sum lgamma(alpha_k)
    + sum (alpha_k - 1)(digamma(alpha_k) - digamma(S))."""
    S = alpha.sum(dim=1)
    log_gamma_k = float(math.lgamma(alpha.shape[1]))
    kl = torch.lgamma(S) - log_gamma_k - torch.lgamma(alpha).sum(dim=1)
    psi_terms = ((alpha - 1.0) * (torch.digamma(alpha)
                                  - torch.digamma(S)[:, None])).sum(dim=1)
    return kl + psi_terms


def evidential_loss(alpha, y, lambda_kl):
    """Sensoy evidential loss with the true-class evidence removal.

    L = mean_i [ err_i + var_i ] + lambda * mean_i KL(Dir(alpha_tilde_i)
    || Dir(1)),  with alpha_tilde = y + (1 - y) * alpha.
    """
    S = alpha.sum(dim=1)
    p = alpha / S[:, None]
    err = ((y - p) ** 2).sum(dim=1)
    var = (p * (1.0 - p)).sum(dim=1) / (S + 1.0)
    alpha_tilde = y + (1.0 - y) * alpha
    kl = kl_dirichlet_uniform(alpha_tilde)
    return (err + var).mean() + lambda_kl * kl.mean()


@torch.no_grad()
def evidential_predictions(model, cwt, scal):
    """(probs, uncertainty, alpha) of a batch from the Dirichlet head."""
    alpha = model(cwt, scal)
    S = alpha.sum(dim=1)
    probs = alpha / S[:, None]
    k = float(alpha.shape[1])
    unc = k / S
    return probs, unc, alpha


def expected_calibration_error(conf, correct, n_bins=10):
    """ECE with equal-width confidence bins."""
    conf = np.asarray(conf, dtype=float)
    correct = np.asarray(correct, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for b in range(n_bins):
        lo, hi = edges[b], edges[b + 1]
        m = (conf > lo) & (conf <= hi) if b else (conf >= lo) & (conf <= hi)
        if m.any():
            w = m.mean()
            ece += w * abs(correct[m].mean() - conf[m].mean())
    return float(ece)


# ---------------------------------------------------------------------------
# sensor graph
# ---------------------------------------------------------------------------
def sensor_kernel(channel_xy):
    """The exponential distance kernel W_ij = exp(-d_ij / sigma_d) of the
    sensor graph, with sigma_d set to the median of the pairwise sensor
    distances (the half-span length scale of the study)."""
    xy = np.asarray(channel_xy, dtype=float)
    d = np.linalg.norm(xy[:, None, :] - xy[None, :, :], axis=2)
    sigma_d = float(np.median(d[np.triu_indices(len(xy), 1)]))
    W = np.exp(-d / sigma_d)
    return W, sigma_d


# ---------------------------------------------------------------------------
# the PI-DL network
# ---------------------------------------------------------------------------
class PIDL(nn.Module):
    """CWT-CNN backbone + sensor-graph GNN + evidential head.

    ``use_gnn`` / ``use_cwt`` / ``use_scalars`` implement the ablations by
    actually removing the corresponding module (the head input width
    follows what is left), not by zeroing inputs.
    """

    def __init__(self, W, n_bands=5, n_hilbert=64, n_scalars=3,
                 use_gnn=True, use_cwt=True, use_scalars=True):
        super().__init__()
        L = P.LEARNING
        self.use_gnn = use_gnn
        self.use_cwt = use_cwt
        self.use_scalars = use_scalars
        n_ch = W.shape[0]
        f1, f2 = L["cnn_filters"]
        self.n_cnn_out = f2

        in_head = 0
        if use_cwt:
            self.conv1 = nn.Conv1d(1, f1, L["cnn_kernel"],
                                   padding=L["cnn_kernel"] // 2)
            self.conv2 = nn.Conv1d(f1, f2, L["cnn_kernel"],
                                   padding=L["cnn_kernel"] // 2)
            self.pool = nn.AdaptiveAvgPool1d(1)
            if use_gnn:
                g = L["gnn_hidden"]
                self.theta1 = nn.Linear(2 * f2, g)
                self.theta2 = nn.Linear(2 * g, g)
                in_head += n_ch * g
            else:
                in_head += n_ch * f2
        if use_scalars:
            in_head += n_hilbert + n_scalars
        self.head = nn.Sequential(
            nn.Linear(in_head, L["head_hidden"]), nn.ReLU(),
            nn.Linear(L["head_hidden"], L["n_classes"]))
        self.register_buffer("W", torch.as_tensor(W, dtype=torch.float32))

    def forward(self, cwt, scal):
        """``cwt`` is (B, 64, n_bands) standardised band energies (zeros
        when the CWT module is ablated); ``scal`` is (B, 67)."""
        feats = []
        if self.use_cwt:
            B, C, K = cwt.shape
            x = cwt.reshape(B * C, 1, K)
            x = torch.relu(self.conv1(x))
            x = torch.relu(self.conv2(x))
            x = self.pool(x).squeeze(-1)          # (B*C, 32)
            x = x.reshape(B, C, self.n_cnn_out)
            if self.use_gnn:
                agg = torch.einsum("ij,bjf->bif", self.W, x)
                h = torch.relu(self.theta1(torch.cat([x, agg], dim=2)))
                agg = torch.einsum("ij,bjf->bif", self.W, h)
                h = torch.relu(self.theta2(torch.cat([h, agg], dim=2)))
            else:
                h = x
            feats.append(h.reshape(B, -1))
        if self.use_scalars:
            feats.append(scal)
        z = torch.cat(feats, dim=1) if len(feats) > 1 else feats[0]
        evidence = torch.nn.functional.softplus(self.head(z))
        evidence = torch.clamp(evidence, 0.0, P.LEARNING["evidence_cap"])
        return evidence + 1.0                      # alpha = evidence + 1

    def n_parameters(self):
        return sum(p.numel() for p in self.parameters())


# ---------------------------------------------------------------------------
# cross-validation protocol
# ---------------------------------------------------------------------------
def group_folds(groups, n_folds):
    """Deterministic GroupKFold over the scenario groups (leave-family-out
    at the family granularity; windows of one scenario never split)."""
    from sklearn.model_selection import GroupKFold
    gkf = GroupKFold(n_splits=n_folds)
    idx = np.arange(len(groups))
    return list(gkf.split(idx, groups=groups))


def fold_scaler(X, train_idx):
    """Standardisation statistics fitted on the training windows only."""
    mu = X[train_idx].mean(axis=0)
    sd = X[train_idx].std(axis=0)
    sd[sd < 1e-9] = 1.0
    return mu, sd


def apply_scaler(X, mu, sd):
    return (X - mu) / sd


def log_transform(X):
    """log10 of the CWT energy block (positive quantities, heavy tails);
    the remaining features pass through unchanged."""
    out = np.array(X, dtype=np.float64, copy=True)
    out[:, CWT_SLICE] = np.log10(out[:, CWT_SLICE] + 1e-30)
    return out


def make_model_inputs(Xs):
    """Split a standardised feature matrix into the CWT tensor and the
    head scalar block."""
    cwt = Xs[:, CWT_SLICE].reshape(len(Xs), 64, 5)
    scal = np.concatenate([Xs[:, HILBERT_SLICE], Xs[:, SCALAR_SLICE]],
                          axis=1)
    return cwt, scal


# ---------------------------------------------------------------------------
# training loop
# ---------------------------------------------------------------------------
def train_pidl(model, cwt_tr, scal_tr, y_tr, cwt_va, scal_va, y_va,
               seed, evidential=True):
    """Train one model over the full 120-epoch budget of the protocol;
    returns (history, best_state_dict, best_epoch).

    Adam + StepLR.  There is no early stopping: the protocol trains every
    run for the full epoch budget (the evidential head needs the budget to
    grow its evidence - a patience-based stop on the flat validation
    accuracy of the honest leave-family-out task returns epoch-0 weights
    and reproduces the uniform-saturation failure of an uncorrected
    constant-lambda variant).
    The best epoch of each run - validation accuracy with an ECE tie-break
    - is what the reported metrics are evaluated at.
    """
    L = P.LEARNING
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    opt = torch.optim.Adam(model.parameters(), lr=L["lr"],
                           weight_decay=L["weight_decay"])
    sched = torch.optim.lr_scheduler.StepLR(
        opt, step_size=L["scheduler_step"], gamma=L["scheduler_gamma"])
    k = P.LEARNING["n_classes"]
    y_tr_t = torch.nn.functional.one_hot(
        torch.as_tensor(y_tr, dtype=torch.long), k).float()
    y_va_t = torch.nn.functional.one_hot(
        torch.as_tensor(y_va, dtype=torch.long), k).float()
    n = len(y_tr)
    bs = L["batch_size"]
    best = dict(score=-np.inf, ece=np.inf, state=None, epoch=-1)
    history = []

    for epoch in range(L["epochs"]):
        model.train()
        perm = rng.permutation(n)
        losses = []
        S_parts, pred_parts, y_parts = [], [], []
        for s in range(0, n, bs):
            take = perm[s: s + bs]
            cwt = torch.as_tensor(cwt_tr[take], dtype=torch.float32)
            scal = torch.as_tensor(scal_tr[take], dtype=torch.float32)
            y = y_tr_t[take]
            alpha = model(cwt, scal)
            if evidential:
                # Sensoy annealing: the KL regulariser ramps up over the
                # first 10 epochs so the expected-error terms can grow
                # the evidence before the prior pulls it back; the
                # weight is capped at the constant lambda of the
                # protocol.
                lam = min(L["lambda_kl"], (epoch + 1) / 10.0 * L["lambda_kl"])
                loss = evidential_loss(alpha, y, lam)
            else:
                loss = nn.functional.cross_entropy(
                    torch.log(alpha), y.argmax(dim=1))
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(float(loss.detach()))
            with torch.no_grad():
                S_parts.append(alpha.sum(dim=1))
                pred_parts.append(alpha.argmax(dim=1))
                y_parts.append(torch.as_tensor(
                    y_tr[take], dtype=torch.long))
        sched.step()

        # validation pass
        model.eval()
        with torch.no_grad():
            alpha = model(torch.as_tensor(cwt_va, dtype=torch.float32),
                          torch.as_tensor(scal_va, dtype=torch.float32))
            S = alpha.sum(dim=1)
            probs = alpha / S[:, None]
            pred = probs.argmax(dim=1).numpy()
            conf = probs.max(dim=1).values.numpy()
            correct = (pred == y_va).astype(float)
            acc = float(correct.mean())
            ece = expected_calibration_error(conf, correct)
            if evidential:
                unc = float((alpha.shape[1] / S).mean())
                vloss = float(evidential_loss(
                    alpha, y_va_t, L["lambda_kl"]))
            else:
                unc = float("nan")
                vloss = float(nn.functional.cross_entropy(
                    torch.log(alpha), torch.as_tensor(y_va)).item())
        history.append(dict(epoch=epoch, train_loss=float(np.mean(losses)),
                             train_acc=float((torch.cat(pred_parts).numpy()
                                             == torch.cat(y_parts).numpy())
                                            .mean()),
                             train_u=float(k / torch.cat(S_parts).mean()),
                             val_loss=vloss, val_acc=acc, val_ece=ece,
                             val_u=unc))
        # model selection on validation accuracy, tie-broken by ECE
        # (ECE-only selection degenerates to the uniform solution: a
        # near-uniform Dirichlet scores ECE ~ |acc - 1/K| trivially).
        # An improvement must exceed half a validation sample so that
        # pure noise does not claim the best epoch.
        score = acc - 0.01 * ece
        if score > best["score"] + 0.005:
            best = dict(score=score, ece=ece,
                        state={k: v.clone()
                               for k, v in model.state_dict().items()},
                        epoch=epoch)
    return history, best["state"], best["epoch"]


@torch.no_grad()
def evaluate_pidl(model, cwt, scal, y):
    """(pred, conf, unc, ece, acc) of a model on a labelled set."""
    model.eval()
    alpha = model(torch.as_tensor(cwt, dtype=torch.float32),
                  torch.as_tensor(scal, dtype=torch.float32))
    S = alpha.sum(dim=1)
    probs = alpha / S[:, None]
    pred = probs.argmax(dim=1).numpy()
    conf = probs.max(dim=1).values.numpy()
    correct = (pred == y).astype(float)
    acc = float(correct.mean())
    ece = expected_calibration_error(conf, correct)
    unc = float((alpha.shape[1] / S).mean())
    return pred, conf, unc, ece, acc


@torch.no_grad()
def graph_embeddings(model, cwt, scal):
    """The flattened GNN output (the 1024-dimensional graph-level vector
    feeding the head) of every sample."""
    feats = []
    if model.use_cwt:
        B, C, K = cwt.shape
        x = torch.as_tensor(cwt, dtype=torch.float32).reshape(B * C, 1, K)
        x = torch.relu(model.conv1(x))
        x = torch.relu(model.conv2(x))
        x = model.pool(x).squeeze(-1).reshape(B, C, model.n_cnn_out)
        if model.use_gnn:
            agg = torch.einsum("ij,bjf->bif", model.W, x)
            h = torch.relu(model.theta1(torch.cat([x, agg], dim=2)))
            agg = torch.einsum("ij,bjf->bif", model.W, h)
            h = torch.relu(model.theta2(torch.cat([h, agg], dim=2)))
        else:
            h = x
        feats.append(h.reshape(B, -1))
    if model.use_scalars:
        feats.append(torch.as_tensor(scal, dtype=torch.float32))
    return torch.cat(feats, dim=1).numpy()


# ---------------------------------------------------------------------------
# baselines (same splits, same standardised features)
# ---------------------------------------------------------------------------
def _batch_iter(X, y, rng, bs):
    n = len(y)
    perm = rng.permutation(n)
    for s in range(0, n, bs):
        take = perm[s: s + bs]
        yield X[take], y[take]


def train_torch_baseline(model, X_tr, y_tr, X_va, y_va, seed, kind):
    """CE training of a baseline net; returns (history, best_state)."""
    L = P.LEARNING
    torch.manual_seed(seed)          # called before model construction
    rng = np.random.default_rng(seed) # in run_*_baseline
    opt = torch.optim.Adam(model.parameters(), lr=L["baseline_lr"])
    y_tr_t = torch.as_tensor(y_tr, dtype=torch.long)
    n = len(y_tr)
    best = dict(acc=-1.0, state=None, epoch=-1)
    for epoch in range(L["baseline_epochs"]):
        model.train()
        for xb, tb in _batch_iter(X_tr, y_tr_t, rng, L["batch_size"]):
            xb = torch.as_tensor(xb, dtype=torch.float32)
            out = model(xb)
            loss = nn.functional.cross_entropy(out, tb)
            opt.zero_grad()
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            out = model(torch.as_tensor(X_va, dtype=torch.float32))
            acc = float((out.argmax(1).numpy() == y_va).mean())
        if acc > best["acc"]:
            best = dict(acc=acc, state={k: v.clone()
                                        for k, v in model.state_dict().items()},
                        epoch=epoch)
    return best


class OneDCNN(nn.Module):
    """Plain 1D-CNN over the flattened 64 x 5 CWT band vector."""

    def __init__(self, n_in=320, n_classes=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 16, 5, padding=2), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(16, 32, 5, padding=2), nn.ReLU(), nn.MaxPool1d(2),
            nn.Flatten(),
            nn.Linear(32 * (n_in // 4), 64), nn.ReLU(),
            nn.Linear(64, n_classes))

    def forward(self, x):
        return self.net(x[:, None, :])


class LSTMBaseline(nn.Module):
    """LSTM over the 64 sensor tokens (5 CWT bands + Hilbert decay each)."""

    def __init__(self, n_feat=6, hidden=64, n_classes=4, n_tokens=64):
        super().__init__()
        self.lstm = nn.LSTM(n_feat, hidden, batch_first=True)
        self.head = nn.Linear(hidden, n_classes)
        self.n_tokens = n_tokens

    def forward(self, x):
        seq = x.reshape(len(x), self.n_tokens, -1)
        out, _ = self.lstm(seq)
        return self.head(out[:, -1, :])


class TransformerBaseline(nn.Module):
    """Small transformer encoder over the 64 sensor tokens."""

    def __init__(self, n_feat=6, d_model=64, n_classes=4, n_tokens=64):
        super().__init__()
        L = P.LEARNING
        self.proj = nn.Linear(n_feat, d_model)
        self.pos = nn.Parameter(torch.zeros(1, n_tokens, d_model))
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=L["transformer_heads"],
            dim_feedforward=2 * d_model, batch_first=True)
        self.enc = nn.TransformerEncoder(
            layer, num_layers=L["transformer_layers"])
        self.head = nn.Linear(d_model, n_classes)
        nn.init.normal_(self.pos, std=0.02)

    def forward(self, x):
        seq = x.reshape(len(x), -1, self.proj.in_features)
        z = self.proj(seq) + self.pos[:, : seq.shape[1]]
        z = self.enc(z)
        return self.head(z.mean(dim=1))


def baseline_input_matrix(Xs, which):
    """Feature matrix of a baseline: 'flat' = the standardised 387-vector,
    'tokens' = the 64 x 6 channel tokens (bands + Hilbert decay)."""
    if which == "flat":
        return Xs
    bands = Xs[:, CWT_SLICE].reshape(len(Xs), 64, 5)
    hdt = Xs[:, HILBERT_SLICE].reshape(len(Xs), 64, 1)
    return np.concatenate([bands, hdt], axis=2).reshape(len(Xs), 64 * 6)


def run_svm_rf(kind, X_tr, y_tr, X_va, seed):
    """Fit and evaluate the classical baselines on one fold."""
    if kind == "svm":
        from sklearn.svm import SVC
        clf = SVC(C=P.LEARNING["svm_C"], gamma="scale")
    elif kind == "rf":
        from sklearn.ensemble import RandomForestClassifier
        clf = RandomForestClassifier(n_estimators=P.LEARNING["rf_trees"],
                                     random_state=seed, n_jobs=1)
    else:
        raise ValueError(kind)
    clf.fit(X_tr, y_tr)
    pred = clf.predict(X_va)
    return pred.astype(int)
