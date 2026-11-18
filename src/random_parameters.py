"""Random-parameter ordered logit for the primary severity specification.

The fixed-parameter primary is the M1 ordered logit of `severity.py`. This module asks
whether any of its coefficients varies across riders, by simulated maximum likelihood:

    beta_nk = b_k + delta_k' z_n + sigma_k * exp(omega_k' w_n) * nu_nk,  nu_nk ~ N(0, 1)

for each random coefficient k, with the remaining coefficients fixed, and

    P(y_n = j) = E_nu[ Lambda(c_j - x_n' beta_n) - Lambda(c_{j-1} - x_n' beta_n) ].

The expectation is simulated with scrambled Halton draws. The procedure, its thresholds,
the context variables, the draw counts and the seeds are module constants and are
written into the result file before any estimate, so the selection rule is visibly fixed
before its outcome.

Three properties this module is built around:

1. **The likelihood reproduces the fixed ordered logit when nothing is random**, and the
   analytic gradient is checked against central differences at a point where every
   random-parameter term is active. Both checks run first and stop the script on failure.
   At sigma = 0 the sigma component of the gradient is zero by symmetry, so a gradient
   check there would prove nothing about it.
2. **sigma = 0 is always a stationary point**, because the normal mixing distribution is
   symmetric in the sign of sigma. Every fit therefore starts from several non-zero
   values of sigma and keeps the best log-likelihood; |sigma| is reported.
3. **Common random numbers.** One Halton block per (draw count, seed) is generated once
   and shared by every model compared under that setting, so likelihood-ratio statistics
   do not mix simulation noise from different draws into the test.
"""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import optimize, stats
from scipy.special import expit
from scipy.stats import qmc

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import severity as sv  # noqa: E402

DATA = ROOT / "data"
OUT = DATA / "random_parameters.json"

# --------------------------------------------------------------------------
# Pre-stated procedure
# --------------------------------------------------------------------------

DRAWS_ESTIMATION = 500
DRAWS_STABILITY = (200, 1000)
HALTON_SEEDS = (20261005, 20261006, 20261007)   # the first is the estimation seed
SIGMA_STARTS = (0.1, 0.5, 1.0)
ALPHA = 0.05
MIN_CELL = 10          # rows with x_k != 0 needed on each side of a context split
KR_DRAWS_RP = 500
AME_DRAWS = 500        # Halton draws used to integrate the probabilities in the AMEs

# Same set as the spatial split in transferability.spatial_test.
LARGE_CITIES = ("Austin", "Houston", "San Antonio", "Dallas", "Fort Worth", "El Paso")
CONTEXT = ("nighttime", "large_city", "speed_ge45")

KEY_AME_COVARIATES = ("age_unknown", "coord_missing", "poverty_share",
                      "sidewalk_transition", "failure_to_yield", "signal_violation",
                      "vehicle_turning_across", "lane_positioning")

POWER_COVARIATES = ("light_dark_or_unknown", "sidewalk_transition", "poverty_share")
POWER_SIGMA_GRID = (0.5, 1.0, 1.5, 2.0, 3.0)  # binary: raw units; continuous: per SD of x
POWER_REPS = 40
POWER_DRAWS = 200
POWER_SEED = 20261008
N_WORKERS = 8

PROCEDURE = {
    "model": ("random-parameter ordered logit (three classes O < BC < KA), simulated "
              "maximum likelihood, normally distributed random coefficients, all other "
              "coefficients fixed, logistic error"),
    "specification": "M1 (the primary ordered logit of severity.py), 35 covariates",
    "candidate_selection": (
        "Each of the 35 M1 covariates (structured covariates, officer-coded "
        "contributing factors and narrative cues) is made random one at a time, the "
        "others fixed. A candidate is retained if (i) the Wald test of its standard "
        "deviation is significant at 5 percent and (ii) the likelihood-ratio test "
        "against the fixed-parameter M1 is significant at 5 percent using the "
        "chi-bar-squared reference (an equal mixture of chi-squared(0) and "
        "chi-squared(1), because sigma = 0 lies on the boundary of the parameter "
        "space). The naive chi-squared(1) p-value is reported beside it. The joint "
        "model then makes every retained candidate random at once."),
    "heterogeneity": (
        "For each retained random parameter, heterogeneity in the mean "
        "(delta_k' z_n) and in the variance (sigma_k * exp(omega_k' w_n)) is tested one "
        "context variable at a time by likelihood ratio against the joint random-"
        "parameter model; context variables significant at 5 percent enter a final "
        "model together. A context variable is not tested for a covariate if, among "
        "rows where the covariate is non-zero, fewer than MIN_CELL rows fall on either "
        "side of the split, or (for the mean shifter only) if the interaction x_k * z "
        "is collinear with the design."),
    "context_variables": {
        "nighttime": "nighttime indicator of the model frame",
        "large_city": ("crash in one of " + ", ".join(LARGE_CITIES)
                       + " (City_ID); every other value, including 'Not Listed', is 0"),
        "speed_ge45": ("posted speed 45 mph or higher, as in the model frame; the "
                       "'unknown' speed category is coded 0, as are 30 mph or less and "
                       "35 to 40 mph"),
    },
    "min_cell": MIN_CELL,
    "alpha": ALPHA,
    "mixing_distribution": "normal",
    "draws": {
        "type": "scrambled Halton (scipy.stats.qmc.Halton, scramble=True), mapped to "
                "the standard normal by the inverse CDF",
        "estimation": DRAWS_ESTIMATION,
        "stability_checks": list(DRAWS_STABILITY),
        "seeds": list(HALTON_SEEDS),
        "seed_role": ("a seed changes only the random scramble of the Halton sequence; "
                      "the first seed is used for estimation"),
        "unit": ("one draw set per rider observation (522 rows); the frame holds 2 "
                 "crashes with two riders each, which are treated as independent rows "
                 "as in the fixed-parameter model"),
        "common_random_numbers": ("one block per (draw count, seed) is shared by "
                                  "every model compared under that setting"),
    },
    "sigma_starting_values": list(SIGMA_STARTS),
    "optimizer": ("scipy BFGS with analytic gradient, gtol 1e-6; the covariance is the "
                  "inverse of the Hessian obtained by central differences of the "
                  "analytic gradient"),
    "threshold_parameterization": "first cutpoint then log-increments, as in severity.py",
    "average_marginal_effects": {
        "integration_draws": AME_DRAWS,
        "krinsky_robb_draws": KR_DRAWS_RP,
        "krinsky_robb_seed": sv.KR_SEED,
        "rule": ("probabilities integrated over the Halton draws per row, then averaged "
                 "over rows; sibling levels of a categorical are set to 0; a covariate "
                 "that is also a mean or variance shifter is changed in those terms too"),
        "candidate_screen": ("point AMEs only, for the key covariates and the "
                             "candidate itself"),
        "fixed_model_intervals": "severity.average_marginal_effects, 1000 draws",
    },
    "power_simulation": "run only if no candidate is retained",
}


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------

def load_frame() -> pd.DataFrame:
    frame = pd.read_csv(DATA / "model_frame.csv")
    frame["large_city"] = frame["City_ID"].isin(LARGE_CITIES).astype(float)
    return frame


def halton_normal(n: int, r: int, d: int, seed: int) -> np.ndarray:
    """(n, r, d) standard-normal draws; rider i takes points i*r .. (i+1)*r - 1."""
    if d == 0:
        return np.zeros((n, r, 0))
    u = qmc.Halton(d=d, scramble=True, seed=seed).random(n * r)
    u = np.clip(u, 1e-10, 1 - 1e-10)
    return stats.norm.ppf(u).reshape(n, r, d)


# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------

class RPOrderedLogit:
    """Ordered logit with normal random coefficients and mean/variance shifters.

    theta = [beta (K), delta (sum of mean-shifter counts), sigma (d),
             omega (sum of variance-shifter counts), cut_1, log increments]
    """

    def __init__(self, data: pd.DataFrame, columns: list[str], random: list[str],
                 mean_shift: dict[str, list[str]] | None = None,
                 var_shift: dict[str, list[str]] | None = None,
                 draws: np.ndarray | None = None, J: int = 3):
        self.columns = list(columns)
        self.random = list(random)
        self.mean_shift = {k: list((mean_shift or {}).get(k, [])) for k in random}
        self.var_shift = {k: list((var_shift or {}).get(k, [])) for k in random}
        self.J = J
        self.K = len(columns)
        self.d = len(random)
        self.n_delta = sum(len(v) for v in self.mean_shift.values())
        self.n_omega = sum(len(v) for v in self.var_shift.values())
        self.n_params = self.K + self.n_delta + self.d + self.n_omega + (J - 1)
        self.set_data(data, draws)

    def set_data(self, data: pd.DataFrame, draws: np.ndarray | None = None) -> None:
        self.X = data[self.columns].astype(float).to_numpy()
        self.y = (pd.Categorical(data["sev3"], categories=list(sv.SEV_ORDER),
                                 ordered=True).codes
                  if "sev3" in data else np.zeros(len(data), dtype=int))
        self.N = len(data)
        self.xr = [self.X[:, self.columns.index(k)] for k in self.random]
        self.Z = [data[self.mean_shift[k]].astype(float).to_numpy() for k in self.random]
        self.W = [data[self.var_shift[k]].astype(float).to_numpy() for k in self.random]
        if draws is not None:
            self.nu = draws[:, :, :self.d]

    def names(self) -> list[str]:
        out = list(self.columns)
        out += [f"mean_shift[{k}|{z}]" for k in self.random for z in self.mean_shift[k]]
        out += [f"sd[{k}]" for k in self.random]
        out += [f"var_shift[{k}|{w}]" for k in self.random for w in self.var_shift[k]]
        out += ["cut_1"] + [f"log_cut_increment_{j}" for j in range(1, self.J - 1)]
        return out

    def unpack(self, theta):
        i = self.K
        beta = theta[:i]
        delta = []
        for k in self.random:
            m = len(self.mean_shift[k]); delta.append(theta[i:i + m]); i += m
        sigma = theta[i:i + self.d]; i += self.d
        omega = []
        for k in self.random:
            m = len(self.var_shift[k]); omega.append(theta[i:i + m]); i += m
        return beta, delta, sigma, omega, theta[i:]

    def _scale(self, sigma, omega):
        """Per-row standard deviation of each random coefficient, (d, N)."""
        return [sigma[j] * np.exp(self.W[j] @ omega[j]) if self.W[j].shape[1]
                else np.full(self.N, sigma[j]) for j in range(self.d)]

    def eta(self, theta):
        beta, delta, sigma, omega, _ = self.unpack(theta)
        base = self.X @ beta
        for j in range(self.d):
            if self.Z[j].shape[1]:
                base = base + self.xr[j] * (self.Z[j] @ delta[j])
        eta = np.repeat(base[:, None], self.nu.shape[1], axis=1)
        s = self._scale(sigma, omega)
        for j in range(self.d):
            eta = eta + (self.xr[j] * s[j])[:, None] * self.nu[:, :, j]
        return eta, s

    def probs(self, theta) -> np.ndarray:
        """(N, J) class probabilities, integrated over the draws."""
        eta, _ = self.eta(theta)
        cut = sv._cut_from_raw(self.unpack(theta)[4])
        cum = [expit(c - eta) for c in cut]
        p = [cum[0]] + [cum[j] - cum[j - 1] for j in range(1, self.J - 1)] + [1 - cum[-1]]
        return np.stack([q.mean(axis=1) for q in p], axis=1)

    def negll_grad(self, theta):
        beta, delta, sigma, omega, raw_cut = self.unpack(theta)
        cut = sv._cut_from_raw(raw_cut)
        eta, s = self.eta(theta)
        R = eta.shape[1]
        y = self.y
        ext = np.concatenate([[-np.inf], cut, [np.inf]])
        hi = ext[y + 1][:, None] - eta
        lo = ext[y][:, None] - eta
        Fh, Fl = expit(hi), expit(lo)
        fh, fl = Fh * (1 - Fh), Fl * (1 - Fl)
        Pnr = Fh - Fl
        Pn = np.maximum(Pnr.mean(axis=1), 1e-300)
        ll = float(np.log(Pn).sum())

        w = 1.0 / (Pn * R)                         # d log Pn / d Pnr
        g = -(fh - fl)                             # d Pnr / d eta
        A = (g.sum(axis=1)) * w                    # (N,)
        grad = [self.X.T @ A]
        gd, gs, go = [], [], []
        for j in range(self.d):
            xj = self.xr[j]
            if self.Z[j].shape[1]:
                gd.append(self.Z[j].T @ (xj * A))
            B = (g * self.nu[:, :, j]).sum(axis=1) * w          # (N,)
            e = s[j] / sigma[j] if sigma[j] != 0 else (
                np.exp(self.W[j] @ omega[j]) if self.W[j].shape[1] else np.ones(self.N))
            gs.append(float((xj * e * B).sum()))
            if self.W[j].shape[1]:
                go.append(self.W[j].T @ (xj * s[j] * B))
        grad += gd + [np.array(gs)] + go

        # Cutpoints: d log P / d c_j, then the chain rule to the raw parameterization.
        dc = np.zeros(self.J - 1)
        up = fh.sum(axis=1) * w
        dn = fl.sum(axis=1) * w
        for j in range(self.J - 1):
            dc[j] = up[y == j].sum() - dn[y == j + 1].sum()
        draw = np.empty(self.J - 1)
        draw[0] = dc.sum()
        for i in range(1, self.J - 1):
            draw[i] = np.exp(raw_cut[i]) * dc[i:].sum()
        grad.append(draw)
        return -ll, -np.concatenate([np.atleast_1d(np.asarray(v, float)) for v in grad])

    def negll(self, theta):
        return self.negll_grad(theta)[0]

    def grad(self, theta):
        return self.negll_grad(theta)[1]


def hessian_from_gradient(model: RPOrderedLogit, theta: np.ndarray,
                          eps: float = 1e-5) -> np.ndarray:
    n = len(theta)
    H = np.empty((n, n))
    for i in range(n):
        tp, tm = theta.copy(), theta.copy()
        tp[i] += eps; tm[i] -= eps
        H[:, i] = (model.grad(tp) - model.grad(tm)) / (2 * eps)
    return (H + H.T) / 2


def fit(model: RPOrderedLogit, start: np.ndarray, with_cov: bool = True) -> dict:
    t0 = time.time()
    res = optimize.minimize(model.negll_grad, start, jac=True, method="BFGS",
                            options={"maxiter": 5000, "gtol": 1e-6})
    theta = res.x
    g = model.grad(theta)
    out = {"theta": theta, "loglik": float(-res.fun), "success": bool(res.success),
           "message": str(res.message), "n_iter": int(res.nit),
           "max_abs_gradient": float(np.max(np.abs(g))), "seconds": time.time() - t0}
    if with_cov:
        H = hessian_from_gradient(model, theta)
        ev = np.linalg.eigvalsh(H)
        out["hessian_min_eig"] = float(ev.min())
        out["hessian_max_eig"] = float(ev.max())
        out["hessian_pd"] = bool(ev.min() > 0)
        out["hessian_condition"] = float(ev.max() / ev.min()) if ev.min() > 0 else None
        out["cov"] = np.linalg.pinv(H)
    out["converged"] = bool(out["max_abs_gradient"] < 1e-3
                            and (not with_cov or out["hessian_pd"]))
    return out


def best_of_starts(model: RPOrderedLogit, base: np.ndarray,
                   starts=SIGMA_STARTS, with_cov: bool = True) -> dict:
    """Fit from each sigma start (all random coefficients start at the same value)."""
    runs = []
    for s0 in starts:
        th = base.copy()
        i0 = model.K + model.n_delta
        th[i0:i0 + model.d] = s0
        runs.append((s0, fit(model, th, with_cov=False)))
    s_best, best = max(runs, key=lambda r: r[1]["loglik"])
    final = fit(model, best["theta"], with_cov=with_cov) if with_cov else best
    final["n_iter_search"] = best["n_iter"]
    lls = [r[1]["loglik"] for r in runs]
    final["starts"] = [{"sigma_start": s, "loglik": round(r["loglik"], 4),
                        "abs_sigma": [round(abs(v), 4) for v in
                                      model.unpack(r["theta"])[2]],
                        "max_abs_gradient": round(r["max_abs_gradient"], 7)}
                       for s, r in runs]
    final["starts_agree_loglik_within_0.01"] = bool(max(lls) - min(lls) < 0.01)
    return final


def base_vector(model: RPOrderedLogit, fixed_theta: np.ndarray,
                fixed_columns: list[str], extra: dict | None = None) -> np.ndarray:
    """Start vector from a fixed fit: slopes and cutpoints copied, the rest zero."""
    th = np.zeros(model.n_params)
    pos = {c: i for i, c in enumerate(fixed_columns)}
    th[:model.K] = [fixed_theta[pos[c]] for c in model.columns]
    th[-(model.J - 1):] = fixed_theta[-(model.J - 1):]
    for name, val in (extra or {}).items():
        th[model.names().index(name)] = val
    return th


# --------------------------------------------------------------------------
# Reporting helpers
# --------------------------------------------------------------------------

def _r(x, nd=4):
    return None if x is None or (isinstance(x, float) and not np.isfinite(x)) else round(float(x), nd)


SMALLEST_CLASS = 77   # overwritten from the frame in main()


def metrics(loglik: float, k: int, n: int, null_ll: float) -> dict:
    return {"loglik": _r(loglik, 2), "loglik_4dp": _r(loglik, 4), "n_parameters": k,
            "aic": _r(-2 * loglik + 2 * k, 1), "bic": _r(-2 * loglik + k * np.log(n), 1),
            "rho2": _r(1 - loglik / null_ll), "adjusted_rho2": _r(1 - (loglik - k) / null_ll),
            "events_per_parameter_smallest_class": _r(SMALLEST_CLASS / k, 3)}


def lr_boundary(ll_big: float, ll_small: float, q: int, interior: int = 0) -> dict:
    """LR with the naive chi2(q + interior) and the chi-bar-squared mixture.

    q parameters (standard deviations) lie on the boundary under the null and `interior`
    parameters (shifters) do not: p = sum_i C(q,i)/2^q P(chi2(i + interior) > LR).
    """
    lr = max(2 * (ll_big - ll_small), 0.0)
    from math import comb

    def sf(df):
        return stats.chi2.sf(lr, df) if df else float(lr <= 0)
    p_naive = float(stats.chi2.sf(lr, q + interior))
    p_mix = float(sum(comb(q, i) / 2 ** q * sf(i + interior) for i in range(q + 1)))
    return {"lr_chi2": _r(lr, 3), "df": q + interior, "df_on_boundary": q,
            "p_naive_chi2": _r(p_naive, 4), "p_chi_bar_squared": _r(p_mix, 4)}


def lr_plain(ll_big: float, ll_small: float, q: int) -> dict:
    lr = max(2 * (ll_big - ll_small), 0.0)
    return {"lr_chi2": _r(lr, 3), "df": q, "p": _r(float(stats.chi2.sf(lr, q)), 4)}


def param_table(model: RPOrderedLogit, res: dict) -> dict:
    se = np.sqrt(np.clip(np.diag(res["cov"]), 0, None))
    out = {}
    for name, b, s in zip(model.names(), res["theta"], se):
        if name.startswith("sd["):
            b = abs(b)
        z = b / s if s > 0 else float("nan")
        out[name] = {"est": _r(b, 3), "se": _r(s, 3), "z": _r(z, 2),
                     "p": _r(float(2 * stats.norm.sf(abs(z))), 3) if s > 0 else None}
    return out


def convergence(res: dict) -> dict:
    return {"optimizer_success_flag": res["success"], "message": res["message"],
            "n_iter_search": res.get("n_iter_search", res["n_iter"]),
            "n_iter_final_polish": res["n_iter"],
            "max_abs_gradient": _r(res["max_abs_gradient"], 8),
            "hessian_positive_definite": res.get("hessian_pd"),
            "hessian_min_eigenvalue": _r(res.get("hessian_min_eig"), 6),
            "hessian_condition_number": _r(res.get("hessian_condition"), 1),
            "converged": res["converged"]}


# --------------------------------------------------------------------------
# Average marginal effects integrated over the draws
# --------------------------------------------------------------------------

def sibling_map(frame: pd.DataFrame, cols: list[str]) -> dict[str, list[str]]:
    groups = sv.categorical_groups(frame, cols)
    return {c: [m for m in mem if m != c] for mem in groups.values() for c in mem}


def rp_ames(model_spec: dict, theta: np.ndarray, frame: pd.DataFrame,
            columns_to_report: list[str], draws: np.ndarray,
            sib: dict[str, list[str]] | None = None) -> np.ndarray:
    """Point AMEs, (len(columns_to_report), J), rebuilding every derived array per row.

    The counterfactual frame is passed back through the model, so a toggled covariate
    that is also a mean or variance shifter moves its interaction and its scale too.
    """
    cols = model_spec["columns"]
    sib = sibling_map(frame, cols) if sib is None else sib
    m = RPOrderedLogit(frame, **model_spec, draws=draws)
    out = np.zeros((len(columns_to_report), 3))
    for i, c in enumerate(columns_to_report):
        x = frame[c].to_numpy(float)
        if set(np.unique(x)) <= {0.0, 1.0}:
            f1, f0 = frame.copy(), frame.copy()
            f1[c] = 1.0
            for s in sib.get(c, []):
                f1[s] = 0.0
            f0[c] = 0.0
            m.set_data(f1); p1 = m.probs(theta)
            m.set_data(f0); p0 = m.probs(theta)
            out[i] = (p1 - p0).mean(axis=0)
        else:
            h = max(1e-4, 0.01 * float(np.std(x)))
            fp, fm = frame.copy(), frame.copy()
            fp[c] = x + h; fm[c] = x - h
            m.set_data(fp); pp = m.probs(theta)
            m.set_data(fm); pm = m.probs(theta)
            out[i] = ((pp - pm) / (2 * h)).mean(axis=0)
    m.set_data(frame)
    return out


def rp_ames_kr(model_spec, res, frame, cols, draws, n_kr=KR_DRAWS_RP, seed=sv.KR_SEED,
               pool=None, draw_setting=None):
    """Krinsky-Robb intervals over the full parameter vector, sigma and shifters included.

    A drawn sigma of either sign gives the same model, so sign changes across draws are
    harmless. With `pool`, the draws are spread over the workers, which rebuild the same
    Halton block from `draw_setting` = (draws, seed).
    """
    sib = sibling_map(frame, model_spec["columns"])
    point = rp_ames(model_spec, res["theta"], frame, cols, draws, sib)
    rng = np.random.default_rng(seed)
    cov = (res["cov"] + res["cov"].T) / 2
    sample = rng.multivariate_normal(res["theta"], cov, size=n_kr, method="svd")
    if pool is None:
        drawn = np.stack([rp_ames(model_spec, t, frame, cols, draws, sib) for t in sample])
    else:
        r, s = draw_setting
        drawn = np.stack(pool.map(
            _kr_one, [(model_spec, t, cols, r, s, draws.shape[2]) for t in sample],
            chunksize=8))
    out = {}
    for i, c in enumerate(cols):
        out[c] = {}
        for j, lvl in enumerate(sv.SEV_ORDER):
            lo, hi = np.percentile(drawn[:, i, j], [2.5, 97.5])
            out[c][lvl] = {"ame": _r(point[i, j], 5), "ci95": [_r(lo, 5), _r(hi, 5)],
                           "excludes_zero": bool(lo > 0 or hi < 0)}
    return out


# --------------------------------------------------------------------------
# Workers (module level for Windows spawn)
# --------------------------------------------------------------------------

_G: dict = {}


def _init_worker(frame_path: str, cols: list[str], fixed_theta: np.ndarray):
    frame = load_frame()
    _G.update(frame=frame, cols=cols, fixed_theta=fixed_theta, draws={})


def _draws(r: int, seed: int, d: int = 1):
    key = (r, seed, d)
    if key not in _G["draws"]:
        _G["draws"][key] = halton_normal(len(_G["frame"]), r, d, seed)
    return _G["draws"][key]


def _kr_one(args):
    spec, theta, cols, r, seed, d = args
    if "sib" not in _G:
        _G["sib"] = sibling_map(_G["frame"], spec["columns"])
    return rp_ames(spec, theta, _G["frame"], cols, _draws(r, seed, d), _G["sib"])


def _screen_one(args):
    cand, r, seed, starts, with_cov, with_ames = args
    frame, cols, ft = _G["frame"], _G["cols"], _G["fixed_theta"]
    m = RPOrderedLogit(frame, cols, [cand], draws=_draws(r, seed))
    res = best_of_starts(m, base_vector(m, ft, cols), starts=starts, with_cov=with_cov)
    out = {"candidate": cand, "draws": r, "seed": seed, "loglik": res["loglik"],
           "theta": res["theta"], "starts": res["starts"],
           "starts_agree": res["starts_agree_loglik_within_0.01"],
           "max_abs_gradient": res["max_abs_gradient"], "success": res["success"],
           "n_iter": res["n_iter"], "message": res["message"]}
    if with_cov:
        i = m.names().index(f"sd[{cand}]")
        out.update(sd=abs(res["theta"][i]), sd_se=float(np.sqrt(max(res["cov"][i, i], 0))),
                   hessian_pd=res["hessian_pd"], hessian_condition=res["hessian_condition"],
                   hessian_min_eig=res["hessian_min_eig"], converged=res["converged"],
                   cov=res["cov"])
    if with_ames:
        # Point AMEs for the key covariates and the candidate itself.
        rep = list(dict.fromkeys(list(KEY_AME_COVARIATES) + [cand]))
        spec = {"columns": cols, "random": [cand]}
        out["ames"] = rp_ames(spec, res["theta"], frame, rep, _draws(r, seed))
        out["ame_columns"] = rep
    if not with_cov:
        i = m.names().index(f"sd[{cand}]")
        out["sd"] = abs(res["theta"][i])
    return out


def _power_one(args):
    cov_name, sigma_true, rep, r, seed = args
    frame, cols, ft = _G["frame"], _G["cols"], _G["fixed_theta"]
    rng = np.random.default_rng([seed, cols.index(cov_name),
                                 int(round(sigma_true * 100)), rep])
    X = frame[cols].to_numpy(float)
    beta, cut = ft[:len(cols)], sv._cut_from_raw(ft[len(cols):])
    k = cols.index(cov_name)
    s = sigma_true / (float(frame[cov_name].std()) if cov_name == "poverty_share" else 1.0)
    eta = X @ beta + X[:, k] * s * rng.standard_normal(len(X))
    lat = eta + rng.logistic(size=len(X))
    y = np.searchsorted(cut, lat)
    sim = frame.copy()
    sim["sev3"] = np.array(sv.SEV_ORDER)[y]
    if np.bincount(y, minlength=3).min() == 0:
        return {"cov": cov_name, "sigma": sigma_true, "rep": rep, "ok": False}
    try:
        m0 = RPOrderedLogit(sim, cols, [], draws=_draws(r, seed))
        f0 = fit(m0, ft.copy(), with_cov=False)
        m1 = RPOrderedLogit(sim, cols, [cov_name], draws=_draws(r, seed))
        f1 = best_of_starts(m1, base_vector(m1, f0["theta"], cols), starts=(0.5, 1.5),
                            with_cov=True)
        i = m1.names().index(f"sd[{cov_name}]")
        sd, se = abs(f1["theta"][i]), float(np.sqrt(max(f1["cov"][i, i], 0)))
        wald_p = float(2 * stats.norm.sf(sd / se)) if se > 0 else 1.0
        lr = lr_boundary(f1["loglik"], f0["loglik"], 1)
        return {"cov": cov_name, "sigma": sigma_true, "rep": rep, "ok": True,
                "sd_hat": sd, "wald_p": wald_p, "lr_p_mix": lr["p_chi_bar_squared"],
                "retained": bool(wald_p < ALPHA and lr["p_chi_bar_squared"] < ALPHA)}
    except Exception as exc:                                       # noqa: BLE001
        return {"cov": cov_name, "sigma": sigma_true, "rep": rep, "ok": False,
                "error": type(exc).__name__}


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def checks(frame, cols, fixed_ol) -> dict:
    """Hard gates: the fixed special case and the analytic gradient."""
    from statsmodels.miscmodels.ordinal_model import OrderedModel

    m0 = RPOrderedLogit(frame, cols, [], draws=halton_normal(len(frame), 1, 0, 1))
    f0 = fit(m0, np.asarray(fixed_ol.raw, float).copy(), with_cov=False)
    # statsmodels stops BFGS on the per-observation gradient, so its optimum can sit
    # about 1e-3 from the maximum on a weakly identified coefficient while this module
    # runs to gtol 1e-6 on the summed gradient. The reference is therefore continued
    # from its own optimum with a tight tolerance before the comparison, so the gate
    # tests the likelihood rather than two stopping rules. The thresholds are unchanged;
    # the unpolished gaps and the likelihoods at identical parameters are recorded.
    Xd, yd = sv._design(frame, cols)
    ref = OrderedModel(yd, Xd, distr="logit").fit(
        start_params=np.asarray(fixed_ol.raw, float), method="bfgs", maxiter=5000,
        gtol=1e-9, disp=False)
    ref_raw = np.asarray(ref.params, float)
    ll_gap = abs(f0["loglik"] - float(ref.llf))
    beta_gap = float(np.max(np.abs(f0["theta"][:len(cols)] - ref_raw[:len(cols)])))
    ll_gap_raw = abs(f0["loglik"] - fixed_ol.loglik)
    beta_gap_raw = float(np.max(np.abs(f0["theta"][:len(cols)] - fixed_ol.beta)))
    same_point = abs(-m0.negll(np.asarray(fixed_ol.raw, float)) - fixed_ol.loglik)

    # Gradient check with two random coefficients, mean and variance shifters, sigma != 0.
    rng = np.random.default_rng(1)
    dr = halton_normal(len(frame), 50, 2, HALTON_SEEDS[0])
    mg = RPOrderedLogit(frame, cols, ["light_dark_or_unknown", "poverty_share"],
                        mean_shift={"light_dark_or_unknown": ["large_city"],
                                    "poverty_share": ["nighttime"]},
                        var_shift={"light_dark_or_unknown": ["speed_ge45"],
                                   "poverty_share": ["large_city"]}, draws=dr)
    th = base_vector(mg, fixed_ol.raw, cols)
    th += rng.normal(0, 0.1, len(th))
    i = mg.K + mg.n_delta
    th[i:i + 2] = [0.7, -1.2]
    an = mg.grad(th)
    num = np.empty_like(th)
    for j in range(len(th)):
        tp, tm = th.copy(), th.copy()
        tp[j] += 1e-6; tm[j] -= 1e-6
        num[j] = (mg.negll(tp) - mg.negll(tm)) / 2e-6
    rel = float(np.max(np.abs(an - num)) / max(1.0, np.max(np.abs(num))))
    out = {
        "fixed_special_case": {
            "loglik_statsmodels_ordered_logit": _r(fixed_ol.loglik, 6),
            "loglik_statsmodels_ordered_logit_polished": _r(float(ref.llf), 6),
            "loglik_this_module_no_random": _r(f0["loglik"], 6),
            "abs_gap": float(f"{ll_gap:.3g}"), "max_abs_beta_gap": float(f"{beta_gap:.3g}"),
            "abs_gap_vs_unpolished": float(f"{ll_gap_raw:.3g}"),
            "max_abs_beta_gap_vs_unpolished": float(f"{beta_gap_raw:.3g}"),
            "loglik_abs_gap_at_identical_parameters": float(f"{same_point:.3g}"),
            "max_abs_gradient_this_module_at_unpolished_reference": float(
                f"{np.max(np.abs(m0.grad(np.asarray(fixed_ol.raw, float)))):.3g}"),
            "max_abs_gradient_this_module_at_polished_reference": float(
                f"{np.max(np.abs(m0.grad(ref_raw))):.3g}"),
            "max_abs_gradient_this_module_at_own_optimum": float(
                f"{f0['max_abs_gradient']:.3g}"),
            "reference": ("statsmodels OrderedModel continued from the severity.py M1 "
                          "optimum with BFGS gtol 1e-9; gaps against the unpolished M1 "
                          "are recorded beside it"),
            "passes": bool(ll_gap < 1e-3 and beta_gap < 1e-3)},
        "analytic_gradient": {
            "point": ("two random coefficients, one mean shifter and one variance shifter "
                      "each, sigma = 0.7 and -1.2, 50 draws, other parameters perturbed"),
            "max_rel_error_vs_central_difference": float(f"{rel:.3g}"),
            "passes": bool(rel < 1e-4)},
    }
    return out, f0


def context_admissible(frame, cols, k) -> dict:
    """Which context variables can shift the mean or variance of coefficient k."""
    x = frame[k].to_numpy(float)
    nz = x != 0
    X = frame[cols].to_numpy(float)
    rank = np.linalg.matrix_rank(np.column_stack([np.ones(len(X)), X]))
    out = {}
    for z in CONTEXT:
        zz = frame[z].to_numpy(float)
        n1, n0 = int((nz & (zz == 1)).sum()), int((nz & (zz == 0)).sum())
        reason = None
        if z == k:
            reason = "same variable as the random coefficient"
        elif min(n1, n0) < MIN_CELL:
            reason = (f"among rows with {k} non-zero, {n1} have {z}=1 and {n0} have "
                      f"{z}=0; fewer than {MIN_CELL} on one side")
        # A mean shifter adds the column x_k * z, which can duplicate a column already in
        # the design (every nighttime row is also dark or unknown light). A variance
        # shifter adds no column; it needs only the split among rows where x_k != 0.
        mean_reason = reason
        if reason is None:
            r2 = np.linalg.matrix_rank(np.column_stack([np.ones(len(X)), X, x * zz]))
            if r2 <= rank:
                mean_reason = "interaction x_k * z is collinear with the design"
        out[z] = {"mean_admissible": mean_reason is None, "mean_reason": mean_reason,
                  "variance_admissible": reason is None, "variance_reason": reason,
                  "n_nonzero_with_z1": n1, "n_nonzero_with_z0": n0}
    return out


def main() -> None:
    from multiprocessing import Pool

    global SMALLEST_CLASS
    t_start = time.time()
    frame = load_frame()
    SMALLEST_CLASS = int(frame["sev3"].value_counts().min())
    cols = sv.ladder(frame)["M1"]
    null_ll = sv.null_loglikelihood(frame)
    fixed_ol = sv.fit_ordered_logit(frame, cols, "M1")
    result: dict = {
        "generated": datetime.now().isoformat(timespec="seconds"),
        "script": "src/random_parameters.py",
        "inputs": ["data/model_frame.csv", "data/screening.json",
                   "data/severity.json"],
        "procedure": PROCEDURE,
        "primary_fixed_model": "M1 ordered logit (severity.py, numbers.json severity.rungs.M1)",
        "n": len(frame),
        "class_counts": frame["sev3"].value_counts().reindex(sv.SEV_ORDER).astype(int).to_dict(),
        "covariates": cols,
        "large_city_n": int(frame["large_city"].sum()),
        "large_cities": list(LARGE_CITIES),
        "context_variable_counts": {z: int(frame[z].sum()) for z in CONTEXT},
    }

    gate, f0 = checks(frame, cols, fixed_ol)
    result["verification"] = gate
    print(json.dumps(gate, indent=1))
    if not (gate["fixed_special_case"]["passes"] and gate["analytic_gradient"]["passes"]):
        OUT.write_text(json.dumps(result, indent=2, default=float), encoding="utf-8")
        raise SystemExit("verification failed; see the result file")

    ft = f0["theta"]
    ll_fixed = f0["loglik"]
    k_fixed = len(cols) + 2
    m_fixed = RPOrderedLogit(frame, cols, [], draws=halton_normal(len(frame), 1, 0, 1))
    fixed_full = fit(m_fixed, ft.copy(), with_cov=True)
    result["fixed_model"] = {"fit": metrics(ll_fixed, k_fixed, len(frame), null_ll),
                             "convergence": convergence(fixed_full)}

    def save():
        result["runtime_seconds"] = round(time.time() - t_start, 1)
        OUT.write_text(json.dumps(result, indent=2, default=float), encoding="utf-8")

    pool = Pool(N_WORKERS, initializer=_init_worker,
                initargs=(str(DATA / "model_frame.csv"), cols, ft))

    # ---- Candidate screen at the estimation setting -------------------------------
    print("screening candidates ...", flush=True)
    seed0 = HALTON_SEEDS[0]
    screen = pool.map(_screen_one, [(c, DRAWS_ESTIMATION, seed0, SIGMA_STARTS, True, True)
                                    for c in cols])
    fixed_ames = sv.average_marginal_effects(fixed_ol, frame, draws=200)

    cand_out = {}
    for s in screen:
        c = s["candidate"]
        wald_p = float(2 * stats.norm.sf(s["sd"] / s["sd_se"])) if s["sd_se"] > 0 else None
        lr = lr_boundary(s["loglik"], ll_fixed, 1)
        k = k_fixed + 1
        retained = bool(wald_p is not None and wald_p < ALPHA
                        and lr["p_chi_bar_squared"] < ALPHA)
        cand_out[c] = {
            "n_nonzero": int((frame[c] != 0).sum()),
            "sd": _r(s["sd"], 3), "sd_se": _r(s["sd_se"], 3),
            "sd_z": _r(s["sd"] / s["sd_se"], 2) if s["sd_se"] > 0 else None,
            "sd_wald_p": _r(wald_p, 4),
            "mean": _r(s["theta"][cols.index(c)], 3),
            "lr_vs_fixed": lr,
            "fit": metrics(s["loglik"], k, len(frame), null_ll),
            "retained": retained,
            "retained_if_naive_chi2": bool(wald_p is not None and wald_p < ALPHA
                                           and lr["p_naive_chi2"] < ALPHA),
            "convergence": {"max_abs_gradient": _r(s["max_abs_gradient"], 8),
                            "optimizer_success_flag": s["success"],
                            "n_iter": s["n_iter"],
                            "hessian_positive_definite": s["hessian_pd"],
                            "hessian_condition_number": _r(s["hessian_condition"], 1),
                            "converged": s["converged"],
                            "sigma_starts": s["starts"],
                            "starts_agree_loglik_within_0.01": s["starts_agree"]},
            "ame_change_vs_fixed": {
                col: {lvl: {"fixed": fixed_ames[col][lvl]["ame"],
                            "random_parameter": _r(s["ames"][i, j], 5),
                            "change": _r(s["ames"][i, j] - fixed_ames[col][lvl]["ame"], 5)}
                      for j, lvl in enumerate(sv.SEV_ORDER)}
                for i, col in enumerate(s["ame_columns"])},
        }
    result["candidates"] = cand_out
    retained = [c for c in cols if cand_out[c]["retained"]]
    pvals = sorted((cand_out[c]["lr_vs_fixed"]["p_chi_bar_squared"], c) for c in cols)
    m = len(pvals)
    holm = []
    for i, (p, c) in enumerate(pvals):
        if p * (m - i) < ALPHA:
            holm.append(c)
        else:
            break
    bh = []
    for i, (p, c) in enumerate(pvals):
        if p <= ALPHA * (i + 1) / m:
            bh = [c2 for _, c2 in pvals[:i + 1]]
    result["screen_summary"] = {
        "n_candidates": len(cols),
        "retained": retained,
        "n_retained": len(retained),
        "retained_if_naive_chi2": [c for c in cols if cand_out[c]["retained_if_naive_chi2"]],
        "expected_false_retentions_at_alpha": _r(ALPHA * len(cols), 2),
        "lr_significant_after_holm": holm,
        "lr_significant_after_benjamini_hochberg": bh,
        "all_converged": all(cand_out[c]["convergence"]["converged"] for c in cols),
        "all_sigma_starts_agree": all(
            cand_out[c]["convergence"]["starts_agree_loglik_within_0.01"] for c in cols),
        "largest_sd": max(((cand_out[c]["sd"], c) for c in cols))[::-1],
        "smallest_lr_p": list(pvals[0][::-1]),
        "note": ("The multiplicity adjustments are context only; the pre-stated rule "
                 "decides retention."),
    }
    print(json.dumps(result["screen_summary"], indent=1), flush=True)
    save()

    # ---- Stability of the screen across draw counts and seeds ----------------------
    print("stability runs ...", flush=True)
    settings = ([(r, seed0) for r in DRAWS_STABILITY]
                + [(DRAWS_ESTIMATION, s) for s in HALTON_SEEDS[1:]])
    stab = {}
    for r, seed in settings:
        ll0 = ll_fixed   # the fixed model does not depend on the draws
        runs = pool.map(_screen_one, [(c, r, seed, SIGMA_STARTS, True, False) for c in cols])
        key = f"draws{r}_seed{seed}"
        stab[key] = {}
        for s in runs:
            c = s["candidate"]
            wald_p = float(2 * stats.norm.sf(s["sd"] / s["sd_se"])) if s["sd_se"] > 0 else 1.0
            lr = lr_boundary(s["loglik"], ll0, 1)
            stab[key][c] = {"loglik": _r(s["loglik"], 4), "sd": _r(s["sd"], 3),
                            "sd_se": _r(s["sd_se"], 3), "sd_wald_p": _r(wald_p, 4),
                            "lr_p_chi_bar_squared": lr["p_chi_bar_squared"],
                            "retained": bool(wald_p < ALPHA
                                             and lr["p_chi_bar_squared"] < ALPHA),
                            "converged": s["converged"]}
        print(key, "retained:", [c for c in cols if stab[key][c]["retained"]], flush=True)
    flips = {}
    for key, d in stab.items():
        f = [c for c in cols if d[c]["retained"] != cand_out[c]["retained"]]
        if f:
            flips[key] = f
    max_ll_dev = max(abs(stab[k][c]["loglik"] - cand_out[c]["fit"]["loglik_4dp"])
                     for k in stab for c in cols)
    max_sd_dev = max(abs(stab[k][c]["sd"] - cand_out[c]["sd"]) for k in stab for c in cols)
    result["stability"] = {
        "settings": stab,
        "retention_flips_vs_estimation_setting": flips,
        "any_flip": bool(flips),
        "max_abs_loglik_deviation": _r(max_ll_dev, 4),
        "max_abs_sd_deviation": _r(max_sd_dev, 3),
    }
    print(json.dumps({k: v for k, v in result["stability"].items() if k != "settings"}),
          flush=True)
    save()

    # ---- Joint model and heterogeneity ---------------------------------------------
    final_spec = None
    if retained:
        d = len(retained)
        draws_j = halton_normal(len(frame), DRAWS_ESTIMATION, d, seed0)
        mj = RPOrderedLogit(frame, cols, retained, draws=draws_j)
        rj = best_of_starts(mj, base_vector(mj, ft, cols))
        joint = {"random": retained, "fit": metrics(rj["loglik"], mj.n_params,
                                                    len(frame), null_ll),
                 "lr_vs_fixed": lr_boundary(rj["loglik"], ll_fixed, d),
                 "parameters": param_table(mj, rj), "convergence": convergence(rj),
                 "sigma_starts": rj["starts"]}
        jd = {}
        sd_idx = [mj.names().index(f"sd[{k}]") for k in retained]
        for r_ in (200, 500, 1000):
            for seed in HALTON_SEEDS:
                if r_ != DRAWS_ESTIMATION and seed != seed0:
                    continue
                dd = halton_normal(len(frame), r_, d, seed)
                ms = RPOrderedLogit(frame, cols, retained, draws=dd)
                rs = fit(ms, rj["theta"].copy(), with_cov=True)
                jd[f"draws{r_}_seed{seed}"] = {
                    "loglik": _r(rs["loglik"], 4),
                    "sd": {k: _r(abs(rs["theta"][i]), 3) for k, i in zip(retained, sd_idx)},
                    "max_abs_gradient": _r(rs["max_abs_gradient"], 8),
                    "hessian_positive_definite": rs["hessian_pd"]}
        joint["stability"] = jd
        sd_tab = {k: joint["parameters"][f"sd[{k}]"] for k in retained}
        crit = stats.norm.isf(ALPHA / 2)

        def sig(k):
            z = sd_tab[k]["z"]
            return z is not None and abs(z) > crit
        joint["summary"] = {
            "sd_significant_in_joint_at_5pct": [k for k in retained if sig(k)],
            "sd_not_significant_in_joint": [k for k in retained if not sig(k)],
            "sd_range_across_draws_and_seeds": {
                k: [min(v["sd"][k] for v in jd.values()),
                    max(v["sd"][k] for v in jd.values())] for k in retained},
            "loglik_range_across_draws_and_seeds": [
                min(v["loglik"] for v in jd.values()),
                max(v["loglik"] for v in jd.values())],
            "sigma_starts_loglik_spread": _r(
                max(r["loglik"] for r in rj["starts"])
                - min(r["loglik"] for r in rj["starts"]), 4),
        }
        result["joint_model"] = joint
        final_spec, final_res, final_draws = ({"columns": cols, "random": retained},
                                              rj, draws_j)
        save()

        het = {}
        mean_keep, var_keep = {}, {}
        for k in retained:
            adm = context_admissible(frame, cols, k)
            het[k] = {"admissibility": adm, "mean": {}, "variance": {}}
            for z in CONTEXT:
                for kind, arg in (("mean", "mean_shift"), ("variance", "var_shift")):
                    if not adm[z][f"{kind}_admissible"]:
                        continue
                    mh = RPOrderedLogit(frame, cols, retained, **{arg: {k: [z]}},
                                        draws=draws_j)
                    base = np.zeros(mh.n_params)
                    for nm, v in zip(mj.names(), rj["theta"]):
                        base[mh.names().index(nm)] = v
                    rh = fit(mh, base, with_cov=True)
                    nm = (f"mean_shift[{k}|{z}]" if kind == "mean"
                          else f"var_shift[{k}|{z}]")
                    lr = lr_plain(rh["loglik"], rj["loglik"], 1)
                    tab = param_table(mh, rh)
                    het[k][kind][z] = {"estimate": tab[nm], "lr_vs_joint": lr,
                                       "loglik": _r(rh["loglik"], 4),
                                       "convergence": convergence(rh),
                                       "significant": bool(lr["p"] < ALPHA
                                                           and rh["converged"])}
                    if lr["p"] < ALPHA and rh["converged"]:
                        (mean_keep if kind == "mean" else var_keep).setdefault(k, []).append(z)
        n_tests = sum(len(het[k][kind]) for k in het for kind in ("mean", "variance"))
        result["heterogeneity"] = {
            "tests": het, "retained_mean_shifters": mean_keep,
            "retained_variance_shifters": var_keep,
            "n_tests": n_tests,
            "n_tests_not_converged": sum(
                1 for k in het for kind in ("mean", "variance")
                for v in het[k][kind].values() if not v["convergence"]["converged"]),
            "expected_false_positives_at_alpha": _r(ALPHA * n_tests, 2),
            "note": ("A test whose model did not converge is never counted as "
                     "significant. This guard was added after a first run of the "
                     "heterogeneity step; in that run the two non-converged tests had "
                     "p = 0.78 and 0.30, so the guard changed no decision.")}
        if mean_keep or var_keep:
            mf = RPOrderedLogit(frame, cols, retained, mean_shift=mean_keep,
                                var_shift=var_keep, draws=draws_j)
            base = np.zeros(mf.n_params)
            for nm, v in zip(mj.names(), rj["theta"]):
                base[mf.names().index(nm)] = v
            shift_idx = [i for i, nm in enumerate(mf.names())
                         if nm.startswith(("mean_shift[", "var_shift["))]
            # Three starts for the shifters: zero (the joint optimum), +0.5 and -0.5.
            runs = []
            for s0 in (0.0, 0.5, -0.5):
                st = base.copy()
                st[shift_idx] = s0
                runs.append((s0, fit(mf, st, with_cov=False)))
            _, best = max(runs, key=lambda r: r[1]["loglik"])
            rf = fit(mf, best["theta"], with_cov=True)
            rf["n_iter_search"] = best["n_iter"]
            fstab = {}
            sd_names = [f"sd[{k}]" for k in retained]
            sh_names = [mf.names()[i] for i in shift_idx]
            for r_ in (200, 500, 1000):
                for seed in HALTON_SEEDS:
                    if r_ != DRAWS_ESTIMATION and seed != seed0:
                        continue
                    ms = RPOrderedLogit(frame, cols, retained, mean_shift=mean_keep,
                                        var_shift=var_keep,
                                        draws=halton_normal(len(frame), r_, d, seed))
                    rs = fit(ms, rf["theta"].copy(), with_cov=True)
                    nm_all = ms.names()
                    fstab[f"draws{r_}_seed{seed}"] = {
                        "loglik": _r(rs["loglik"], 4),
                        "sd": {n: _r(abs(rs["theta"][nm_all.index(n)]), 3)
                               for n in sd_names},
                        "shifters": {n: _r(rs["theta"][nm_all.index(n)], 3)
                                     for n in sh_names},
                        "max_abs_gradient": _r(rs["max_abs_gradient"], 8),
                        "hessian_positive_definite": rs["hessian_pd"]}
            result["heterogeneity"]["final_model"] = {
                "fit": metrics(rf["loglik"], mf.n_params, len(frame), null_ll),
                "lr_vs_joint": lr_plain(rf["loglik"], rj["loglik"],
                                        mf.n_params - mj.n_params),
                "lr_vs_fixed": lr_boundary(rf["loglik"], ll_fixed, d,
                                           interior=mf.n_params - k_fixed - d),
                "parameters": param_table(mf, rf), "convergence": convergence(rf),
                "shifter_starts": [{"shifter_start": s, "loglik": _r(r["loglik"], 4)}
                                   for s, r in runs],
                "stability": fstab,
                "stability_summary": {
                    "loglik_range": [min(v["loglik"] for v in fstab.values()),
                                     max(v["loglik"] for v in fstab.values())],
                    "sd_range": {n: [min(v["sd"][n] for v in fstab.values()),
                                     max(v["sd"][n] for v in fstab.values())]
                                 for n in sd_names},
                    "shifter_range": {n: [min(v["shifters"][n] for v in fstab.values()),
                                          max(v["shifters"][n] for v in fstab.values())]
                                      for n in sh_names},
                    "all_hessians_positive_definite": all(
                        v["hessian_positive_definite"] for v in fstab.values())}}
            final_spec, final_res = ({"columns": cols, "random": retained,
                                      "mean_shift": mean_keep, "var_shift": var_keep}, rf)
        save()
    else:
        result["joint_model"] = {"estimated": False,
                                 "reason": "no candidate met the pre-stated retention rule"}
        result["heterogeneity"] = {
            "estimated": False,
            "reason": ("heterogeneity in means and variances is defined for retained "
                       "random parameters only, and none was retained"),
            "admissibility_if_any_were_retained": {
                c: context_admissible(frame, cols, c) for c in cols}}
        save()

    # ---- Comparison with the fixed-parameter primary --------------------------------
    comp = {"fixed_M1": result["fixed_model"]["fit"]}
    for c in cols:
        comp[f"single_random[{c}]"] = {**cand_out[c]["fit"],
                                       "lr_vs_fixed": cand_out[c]["lr_vs_fixed"]}
    if retained:
        comp["joint_random_parameter_model"] = {
            **result["joint_model"]["fit"],
            "lr_vs_fixed": result["joint_model"]["lr_vs_fixed"]}
    # Partial proportional-odds reference rows, from the fits already in
    # severity.json (log-likelihood recovered from the LR against M1).
    sev = json.loads((DATA / "severity.json").read_text(encoding="utf-8"))
    for rule, v in sev.get("multiplicity_sensitivity", {}).items():
        if isinstance(v, dict) and v.get("n_freed"):
            lr = v["lr_vs_ordered_logit"]
            comp[f"reference_ppo_{rule}"] = {
                **metrics(ll_fixed + lr["lr_chi2"] / 2, k_fixed + lr["df"], len(frame),
                          null_ll),
                "freed": v["freed"], "lr_vs_fixed": lr,
                "source": "severity.json multiplicity_sensitivity"}
    if final_spec is not None:
        mfin = RPOrderedLogit(frame, **final_spec, draws=final_draws)
        comp["final_random_parameter_model"] = {
            **metrics(final_res["loglik"], mfin.n_params, len(frame), null_ll),
            "lr_vs_fixed": lr_boundary(final_res["loglik"], ll_fixed, mfin.d,
                                       interior=mfin.n_params - k_fixed - mfin.d)}
        print("Krinsky-Robb AMEs for the final random-parameter model ...", flush=True)
        ames = rp_ames_kr(final_spec, final_res, frame, cols,
                          halton_normal(len(frame), AME_DRAWS, len(final_spec["random"]),
                                        seed0),
                          pool=pool, draw_setting=(AME_DRAWS, seed0))
        fixed_kr = sv.average_marginal_effects(fixed_ol, frame)
        result["final_model_ames"] = {
            c: {lvl: {"fixed": fixed_kr[c][lvl], "random_parameter": ames[c][lvl],
                      "change": _r(ames[c][lvl]["ame"] - fixed_kr[c][lvl]["ame"], 5)}
                for lvl in sv.SEV_ORDER} for c in cols}
    if retained:
        brant = sev["proportional_odds_tests"]["covariates_failing_at_5pct"]
        share = frame["sev3"].value_counts(normalize=True)
        result["interpretation"] = {
            "brant_failing_covariates": brant,
            "retained_random_also_failing_brant": [c for c in retained if c in brant],
            "retained_random_not_failing_brant": [c for c in retained if c not in brant],
            "brant_failing_not_retained_random": [c for c in brant if c not in retained],
            "outcome_shares_overall": {k: _r(share[k], 3) for k in sv.SEV_ORDER},
            "outcome_shares_where_covariate_is_1": {
                c: {k: _r((frame.loc[frame[c] == 1, "sev3"] == k).mean(), 3)
                    for k in sv.SEV_ORDER} for c in retained},
            "retained_with_both_tails_above_overall": [
                c for c in retained
                if all((frame.loc[frame[c] == 1, "sev3"] == k).mean() > share[k]
                       for k in ("O", "KA"))],
            "retained_with_middle_class_below_overall": [
                c for c in retained
                if (frame.loc[frame[c] == 1, "sev3"] == "BC").mean() < share["BC"]],
            "retained_o_heavy_not_two_tailed": [
                c for c in retained
                if (frame.loc[frame[c] == 1, "sev3"] == "O").mean() > share["O"]
                and not (frame.loc[frame[c] == 1, "sev3"] == "KA").mean() > share["KA"]],
            "rows_with_both_age_and_gender_unknown": int(
                ((frame["age_unknown"] == 1) & (frame["gender_unknown"] == 1)).sum()),
            "note": ("A normal random coefficient on a binary indicator widens the latent "
                     "distribution of the rows where the indicator is 1, which moves "
                     "probability out of the middle class. Every retained covariate has "
                     "a BC share below the overall share, and every one also fails the "
                     "proportional-odds test, so the retained standard deviations and "
                     "the Brant failures describe largely one feature of the data. For "
                     "{n2} covariates both outer classes are above their overall shares; "
                     "{no} ({names}) are instead O-heavy, and age_unknown and "
                     "gender_unknown share {both} rows."),
        }
        it = result["interpretation"]
        it["note"] = it["note"].format(
            n2=len(it["retained_with_both_tails_above_overall"]),
            no=len(it["retained_o_heavy_not_two_tailed"]),
            names=", ".join(it["retained_o_heavy_not_two_tailed"]),
            both=it["rows_with_both_age_and_gender_unknown"])
    result["comparison"] = comp
    aics = {k: v["aic"] for k, v in comp.items()}
    bics = {k: v["bic"] for k, v in comp.items()}
    result["comparison_summary"] = {
        "best_aic": min(aics, key=aics.get), "best_bic": min(bics, key=bics.get),
        "n_single_random_with_lower_aic_than_fixed": sum(
            1 for k, v in aics.items() if k.startswith("single") and v < aics["fixed_M1"]),
        "n_single_random_with_lower_bic_than_fixed": sum(
            1 for k, v in bics.items() if k.startswith("single") and v < bics["fixed_M1"]),
        "max_abs_ame_change_KA_key_covariates_any_candidate": _r(max(
            abs(cand_out[c]["ame_change_vs_fixed"][col]["KA"]["change"])
            for c in cols for col in KEY_AME_COVARIATES), 5),
        "fixed_vs_joint_and_final_by_bic": {
            k: bics[k] for k in ("fixed_M1", "joint_random_parameter_model",
                                 "final_random_parameter_model") if k in bics},
        "fixed_vs_joint_and_final_by_aic": {
            k: aics[k] for k in ("fixed_M1", "joint_random_parameter_model",
                                 "final_random_parameter_model") if k in aics},
    }
    if "final_model_ames" in result:
        fa = result["final_model_ames"]
        result["comparison_summary"]["final_model_ame_KA_key_covariates"] = {
            c: {"fixed": fa[c]["KA"]["fixed"]["ame"],
                "random_parameter": fa[c]["KA"]["random_parameter"]["ame"],
                "change": fa[c]["KA"]["change"],
                "fixed_excludes_zero": fa[c]["KA"]["fixed"]["excludes_zero"],
                "random_parameter_excludes_zero":
                    fa[c]["KA"]["random_parameter"]["excludes_zero"]}
            for c in KEY_AME_COVARIATES}
        result["comparison_summary"]["ka_excludes_zero_fixed"] = [
            c for c in cols if fa[c]["KA"]["fixed"]["excludes_zero"]]
        result["comparison_summary"]["ka_excludes_zero_random_parameter"] = [
            c for c in cols if fa[c]["KA"]["random_parameter"]["excludes_zero"]]
    save()

    # ---- Detectable SD (only when nothing is retained) -----------------------------
    if not retained:
        print("power simulation ...", flush=True)
        jobs = [(cv, s, rep, POWER_DRAWS, POWER_SEED) for cv in POWER_COVARIATES
                for s in POWER_SIGMA_GRID for rep in range(POWER_REPS)]
        sims = pool.map(_power_one, jobs, chunksize=4)
        pw = {}
        for cv in POWER_COVARIATES:
            pw[cv] = {"n_nonzero": int((frame[cv] != 0).sum()),
                      "sigma_units": ("per standard deviation of x (sd "
                                      f"{frame[cv].std():.4f})" if cv == "poverty_share"
                                      else "raw coefficient units"),
                      "fixed_coefficient": _r(ft[cols.index(cv)], 3), "grid": {}}
            for s in POWER_SIGMA_GRID:
                rows = [x for x in sims if x["cov"] == cv and x["sigma"] == s and x["ok"]]
                n_ok = len(rows)
                pr = np.mean([x["retained"] for x in rows]) if rows else float("nan")
                pw[cv]["grid"][str(s)] = {
                    "n_ok": n_ok, "n_failed": POWER_REPS - n_ok,
                    "power": _r(pr, 3),
                    "mc_se": _r(np.sqrt(pr * (1 - pr) / n_ok), 3) if n_ok else None,
                    "median_sd_hat": _r(np.median([x["sd_hat"] for x in rows]), 3)
                    if rows else None}
            ok = [s for s in POWER_SIGMA_GRID if (pw[cv]["grid"][str(s)]["power"] or 0) >= 0.8]
            pw[cv]["smallest_sigma_with_power_ge_0.8"] = ok[0] if ok else None
        result["detectable_sd"] = {
            "design": (f"Outcomes simulated from the fixed M1 estimates with a normal "
                       f"random coefficient of standard deviation sigma planted on one "
                       f"covariate; {POWER_REPS} replications per grid point; each "
                       f"replication refits the fixed and the one-random-parameter model "
                       f"({POWER_DRAWS} Halton draws, seed {POWER_SEED}, sigma starts 0.5 "
                       f"and 1.5) and applies the pre-stated retention rule (Wald and "
                       f"chi-bar-squared LR, both at 5 percent)."),
            "results": pw}
        save()
    pool.close()
    pool.join()
    print(f"written to {OUT}")


if __name__ == "__main__":
    main()
