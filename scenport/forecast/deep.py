"""Neural scenario forecasters: LSTM, DeepAR-style GRU, transformer encoder.

All three are global models trained across the universe on per-asset standardised
returns, with a probabilistic head and recursive sampling over the horizon. Only
module that imports torch.
"""

from __future__ import annotations

import logging
import math

import numpy as np
import pandas as pd

from .base import ScenarioForecaster, ScenarioSet, register

log = logging.getLogger(__name__)

CONTEXT = 64          # days of history fed to the encoder
N_FEATURES = 2        # standardised return and its absolute value
EPS = 1e-6


def _torch():
    import torch

    return torch


def pick_device(prefer: str = "auto"):
    torch = _torch()
    if prefer not in ("auto", "cuda", "cpu"):
        raise ValueError(f"unknown device preference {prefer!r}")
    if prefer == "cpu" or (prefer == "auto" and not torch.cuda.is_available()):
        return torch.device("cpu")
    return torch.device("cuda")


# ---------------------------------------------------------------------------
# heads


def _gaussian_nll(params, target):
    torch = _torch()
    mu, log_sigma = params.unbind(-1)
    sigma = torch.nn.functional.softplus(log_sigma) + EPS
    return 0.5 * math.log(2 * math.pi) + torch.log(sigma) + 0.5 * ((target - mu) / sigma) ** 2


def _gaussian_sample(params, rng: np.random.Generator):
    torch = _torch()
    mu, log_sigma = params.unbind(-1)
    sigma = torch.nn.functional.softplus(log_sigma) + EPS
    noise = torch.as_tensor(
        rng.standard_normal(tuple(mu.shape)), dtype=mu.dtype, device=mu.device
    )
    return mu + sigma * noise


def _student_nll(params, target):
    torch = _torch()
    mu, log_sigma, log_nu = params.unbind(-1)
    sigma = torch.nn.functional.softplus(log_sigma) + EPS
    nu = 2.0 + torch.nn.functional.softplus(log_nu)       # finite variance
    z = (target - mu) / sigma
    return -(
        torch.lgamma((nu + 1) / 2)
        - torch.lgamma(nu / 2)
        - 0.5 * torch.log(math.pi * nu)
        - torch.log(sigma)
        - (nu + 1) / 2 * torch.log1p(z**2 / nu)
    )


def _student_sample(params, rng: np.random.Generator):
    """t = mu + sigma * Z / sqrt(chi2_nu / nu), as a normal variance mixture.

    The chi-squared draw is taken in numpy rather than through
    ``torch.distributions``, which offers no way to pass a generator and would
    make the sampling irreproducible across runs.
    """
    torch = _torch()
    mu, log_sigma, log_nu = params.unbind(-1)
    sigma = torch.nn.functional.softplus(log_sigma) + EPS
    nu = 2.0 + torch.nn.functional.softplus(log_nu)         # keeps the variance finite

    nu_np = nu.detach().cpu().numpy().astype(float)
    scale = np.sqrt(nu_np / np.maximum(rng.chisquare(nu_np), 1e-12))
    normal = rng.standard_normal(nu_np.shape)
    shock = torch.as_tensor(normal * scale, dtype=mu.dtype, device=mu.device)
    return mu + sigma * shock


HEADS = {
    "gaussian": (2, _gaussian_nll, _gaussian_sample),
    "student": (3, _student_nll, _student_sample),
}


# ---------------------------------------------------------------------------
# encoders


def _build_network(kind: str, hidden: int, layers: int, dropout: float, n_out: int):
    torch = _torch()
    nn = torch.nn

    class Recurrent(nn.Module):
        def __init__(self, cell):
            super().__init__()
            self.rnn = cell(
                N_FEATURES, hidden, num_layers=layers, batch_first=True,
                dropout=dropout if layers > 1 else 0.0,
            )
            self.drop = nn.Dropout(dropout)
            self.head = nn.Linear(hidden, n_out)

        def forward(self, x):
            out, _ = self.rnn(x)
            return self.head(self.drop(out[:, -1, :]))

    class Encoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.project = nn.Linear(N_FEATURES, hidden)
            self.register_buffer("pos", _positional_encoding(CONTEXT, hidden))
            layer = nn.TransformerEncoderLayer(
                d_model=hidden,
                nhead=4,
                dim_feedforward=4 * hidden,
                dropout=dropout,
                batch_first=True,
                norm_first=True,
            )
            self.body = nn.TransformerEncoder(layer, num_layers=layers)
            self.drop = nn.Dropout(dropout)
            self.head = nn.Linear(hidden, n_out)

        def forward(self, x):
            h = self.project(x) + self.pos[: x.shape[1]].unsqueeze(0)
            h = self.body(h)
            return self.head(self.drop(h[:, -1, :]))

    if kind == "lstm":
        return Recurrent(nn.LSTM)
    if kind == "gru":
        return Recurrent(nn.GRU)
    if kind == "transformer":
        return Encoder()
    raise ValueError(f"unknown encoder {kind!r}")


def _positional_encoding(length: int, dim: int):
    torch = _torch()
    pos = torch.arange(length).unsqueeze(1).float()
    div = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
    pe = torch.zeros(length, dim)
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe


# ---------------------------------------------------------------------------


def _windows(z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Sliding context windows and their next-step targets."""
    if len(z) <= CONTEXT:
        return np.empty((0, CONTEXT)), np.empty(0)
    view = np.lib.stride_tricks.sliding_window_view(z, CONTEXT)[:-1]
    return np.ascontiguousarray(view), z[CONTEXT:]


def _stack_features(window: np.ndarray):
    """(..., CONTEXT) -> (..., CONTEXT, N_FEATURES)."""
    return np.stack([window, np.abs(window)], axis=-1)


class DeepScenarios(ScenarioForecaster):
    """Shared training and sampling machinery. Subclasses fix encoder and head."""

    requires_gpu = True
    encoder = "lstm"
    head = "gaussian"
    mc_dropout = False

    def simulate(self, history, horizon, n_paths, rng, origin=None, progress=None):
        torch = _torch()
        assets = list(history.columns)
        device = pick_device(self.params.get("device", "auto"))
        seed = int(rng.integers(0, 2**31 - 1))
        torch.manual_seed(seed)

        series = {}
        for asset in assets:
            prices = history[asset].dropna().to_numpy(dtype=float)
            logret = np.diff(np.log(prices))
            logret = logret[np.isfinite(logret)]
            if len(logret) <= CONTEXT + 20:
                continue
            mu, sd = float(logret.mean()), float(logret.std())
            series[asset] = (logret, mu, max(sd, EPS))

        if not series:
            raise RuntimeError("no asset has enough history to train a deep forecaster")

        model = self._fit(series, device, seed)

        out = np.zeros((n_paths, horizon, len(assets)))
        for j, asset in enumerate(assets):
            if progress is not None:
                progress(asset)
            if asset not in series:
                log.warning("%s has too little history for %s; emitting a flat path",
                            asset, self.name)
                continue
            logret, mu, sd = series[asset]
            steps = self._sample(model, logret, mu, sd, horizon, n_paths, device, seed + j)
            out[:, :, j] = np.exp(np.cumsum(steps, axis=1)) - 1.0

        return ScenarioSet(
            paths=out,
            assets=assets,
            origin=pd.Timestamp(origin if origin is not None else history.index[-1]),
            forecaster=self.name,
            meta={"params": dict(self.params), "device": str(device)},
        )

    def simulate_one(self, history, horizon, n_paths, rng):     # pragma: no cover
        raise NotImplementedError("deep forecasters are global; use simulate()")

    # -- internals ----------------------------------------------------------

    def _fit(self, series: dict, device, seed: int):
        torch = _torch()

        # Hold out the tail of each asset's windows rather than a random slice of
        # the pooled set: a random split would put a window and its neighbour on
        # opposite sides, and they overlap in 63 of 64 observations.
        val_frac = float(self.params.get("val_fraction", 0.15))
        tr_X, tr_y, va_X, va_y = [], [], [], []
        for logret, mu, sd in series.values():
            z = (logret - mu) / sd
            win, tgt = _windows(z)
            if not len(tgt):
                continue
            cut = max(int(len(tgt) * (1.0 - val_frac)), 1)
            feats = _stack_features(win)
            tr_X.append(feats[:cut])
            tr_y.append(tgt[:cut])
            va_X.append(feats[cut:])
            va_y.append(tgt[cut:])

        n_out, nll, _ = HEADS[self.head]
        model = _build_network(
            self.encoder,
            hidden=int(self.params.get("hidden", 64)),
            layers=int(self.params.get("layers", 2)),
            dropout=float(self.params.get("dropout", 0.15)),
            n_out=n_out,
        ).to(device)

        def to_device(parts):
            return torch.tensor(np.concatenate(parts), dtype=torch.float32).to(device)

        X_tr, y_tr = to_device(tr_X), to_device(tr_y)
        X_va, y_va = to_device(va_X), to_device(va_y)

        opt = torch.optim.AdamW(model.parameters(), lr=float(self.params.get("lr", 1e-3)),
                                weight_decay=1e-4)
        batch = int(self.params.get("batch_size", 512))
        epochs = int(self.params.get("epochs", 60))
        patience = int(self.params.get("patience", 10))

        best_loss, best_state, stale = math.inf, None, 0
        for epoch in range(epochs):
            model.train()
            order = torch.randperm(len(X_tr), device=device)
            for start in range(0, len(order), batch):
                idx = order[start : start + batch]
                opt.zero_grad(set_to_none=True)
                loss = nll(model(X_tr[idx]), y_tr[idx]).mean()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()

            model.eval()
            with torch.no_grad():
                val = float(nll(model(X_va), y_va).mean())
            if val < best_loss - 1e-4:
                best_loss, stale = val, 0
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            else:
                stale += 1
                if stale >= patience:
                    log.info(
                        "%s: early stop at epoch %d (val nll %.4f)",
                        self.name, epoch, best_loss,
                    )
                    break

        if best_state is not None:
            model.load_state_dict(best_state)
        model.eval()
        return model

    def _sample(self, model, logret, mu, sd, horizon, n_paths, device, seed) -> np.ndarray:
        torch = _torch()
        _, _, sampler = HEADS[self.head]

        if self.mc_dropout:
            # Gal and Ghahramani: leave dropout on at inference so the paths carry
            # epistemic uncertainty on top of the head's aleatoric spread.
            for module in model.modules():
                if isinstance(module, torch.nn.Dropout):
                    module.train()

        rng = np.random.default_rng(int(seed))
        torch.manual_seed(int(seed))                 # dropout masks, when they are live
        z = (logret[-CONTEXT:] - mu) / sd
        buffer = torch.tensor(
            np.tile(_stack_features(z), (n_paths, 1, 1)), dtype=torch.float32, device=device
        )

        steps = torch.empty((n_paths, horizon), device=device)
        with torch.no_grad():
            for h in range(horizon):
                draw = sampler(model(buffer), rng)
                steps[:, h] = draw
                nxt = torch.stack([draw, draw.abs()], dim=-1).unsqueeze(1)
                buffer = torch.cat([buffer[:, 1:, :], nxt], dim=1)

        return steps.cpu().numpy() * sd + mu


@register
class LstmScenarios(DeepScenarios):
    name = "lstm"
    encoder = "lstm"
    head = "gaussian"
    mc_dropout = True


@register
class DeepAr(DeepScenarios):
    name = "deepar"
    encoder = "gru"
    head = "student"


@register
class TransformerScenarios(DeepScenarios):
    name = "transformer"
    encoder = "transformer"
    head = "student"
