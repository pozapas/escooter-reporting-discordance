"""Alternative ordinal specifications and small-sample checks for the primary M1 model.

Five analyses, each written under its own top-level key of
`data/ordinal_alternatives.json`:

* `ordinal_family`: M1 as ordered probit, adjacent-category logit, the partial
  proportional-odds model that frees the Brant-failing covariates, and the generalized
  ordered logit with every coefficient threshold-specific;
* `random_thresholds`: ordered logit with a normal random term on the second
  threshold increment, and on both thresholds, by simulated maximum likelihood;
* `shrinkage`: a Jeffreys-prior penalized (Firth-type) ordered logit and a
  Bayesian ordered logit with normal(0, 1) slope priors;
* `separation`: coefficient and standard-error diagnostics, zero cells and a
  linear-programming separation check for M0, M0c and M1;
* `cue_block_power`: simulated power of the 5-df cue-block likelihood-ratio
  test as a function of a common planted cue effect.

Every model is reduced to a function `probs(theta, X)` over the full M1 design, so one
Krinsky-Robb routine produces the average marginal effects for all of them. That routine
is checked against `severity.average_marginal_effects` on the M1 ordered logit before it is
used for anything else. Cumulative models with threshold-specific slopes can imply a
negative middle-class probability; the rows where that happens are counted at the
optimum, in the counterfactual rows and across the parameter draws, because
`severity._ppo_negll` clips such probabilities and would otherwise hide them. Any such
model is refitted subject to a non-negative middle-class probability on every row
(`*_noncrossing`), so that its log-likelihood, information criteria and likelihood-ratio
statistic refer to a proper probability model.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

# One BLAS thread per process: the pool workers each load numpy, and a full-width
# OpenBLAS buffer per worker exhausts the commit limit on this machine.
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import optimize, stats  # noqa: E402
from scipy.special import expit  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import severity as sv  # noqa: E402
from src import random_parameters as rp  # noqa: E402

DATA = ROOT / "data"
OUT = DATA / "ordinal_alternatives.json"
J = len(sv.SEV_ORDER)

KEY = list(rp.KEY_AME_COVARIATES)        # poverty_share, coord_missing, age_unknown, cues
RT_DRAWS = 500
RT_DRAWS_CHECK = (200, 1000)
RT_SEED = rp.HALTON_SEEDS[0]
RT_SIGMA_STARTS = (0.1, 0.5, 1.0, 2.0)
RT_PROFILE_GRID = (0.0, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0)

BAYES_CHAINS = 4
BAYES_TUNE = 1000
BAYES_DRAWS = 1000
BAYES_SEED = 20261009
BAYES_CUT_SD = 5.0
BAYES_AME_DRAWS = 1000

POWER_BETAS = (0.0, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.75, 1.0)
POWER_BETAS_EXTENSION = (1.25, 1.5)
POWER_REPS = 200
POWER_SEED = 20261010
POWER_ALPHA = 0.05
N_WORKERS = 6

_r = rp._r


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------

def load() -> tuple[pd.DataFrame, list[str], np.ndarray, np.ndarray]:
    frame = pd.read_csv(DATA / "model_frame.csv")
    cols = sv.ladder(frame)["M1"]
    X, y = sv._design(frame, cols)
    return frame, cols, X, y


def brant_sets() -> dict[str, list[str]]:
    sev = json.loads((DATA / "severity.json").read_text(encoding="utf-8"))
    po = sev["proportional_odds_tests"]
    sets = sv.multiplicity_sets(po["per_covariate"])
    return {"uncorrected": list(po["covariates_failing_at_5pct"]),
            "holm": list(sets["holm"])}


# --------------------------------------------------------------------------
# Probability functions over the full M1 design
# --------------------------------------------------------------------------

def _p_from_cum(cum: np.ndarray) -> np.ndarray:
    return np.column_stack([cum[:, 0], cum[:, 1] - cum[:, 0], 1.0 - cum[:, 1]])


def cumulative_probs(theta, X, ip, jf, link="logit"):
    """Cumulative-link probabilities in the `fit_ppo` layout.

    theta = [beta over ip, gamma_1 over jf, gamma_2 over jf, cut_1, log increment], with
    P(Y <= j) = F(cut_j - X_ip beta - X_jf gamma_j). Returns the (N, J) probabilities,
    unclipped, and the number of rows whose cumulative probabilities cross.
    """
    kp, kf = len(ip), len(jf)
    beta = theta[:kp]
    gamma = theta[kp:kp + kf * (J - 1)].reshape(J - 1, kf)
    cut = sv._cut_from_raw(theta[kp + kf * (J - 1):])
    F = expit if link == "logit" else stats.norm.cdf
    base = X[:, ip] @ beta if kp else np.zeros(len(X))
    cum = np.column_stack([F(cut[j] - base - (X[:, jf] @ gamma[j] if kf else 0.0))
                           for j in range(J - 1)])
    # A row exactly on the non-crossing boundary differs only by rounding.
    return _p_from_cum(cum), int((cum[:, 0] > cum[:, 1] + 1e-10).sum())


def adjacent_probs(theta, X):
    """Adjacent-category logit: log P(j+1)/P(j) = alpha_j + x'beta."""
    K = X.shape[1]
    xb = X @ theta[:K]
    a1, a2 = theta[K], theta[K + 1]
    U = np.column_stack([np.zeros(len(X)), a1 + xb, a1 + a2 + 2 * xb])
    U -= U.max(axis=1, keepdims=True)
    e = np.exp(U)
    return e / e.sum(axis=1, keepdims=True), 0


def loglik_from_probs(p: np.ndarray, y: np.ndarray, clip: bool = True) -> float:
    q = p[np.arange(len(y)), y]
    if clip:
        q = np.clip(q, 1e-12, 1.0)
    return float(np.log(q).sum())


# --------------------------------------------------------------------------
# Average marginal effects with Krinsky-Robb (or posterior) draws
# --------------------------------------------------------------------------

def _counterfactuals(X, frame, cols, report):
    sib = rp.sibling_map(frame, cols)
    out = []
    for c in report:
        k = cols.index(c)
        x = X[:, k]
        if set(np.unique(x)) <= {0.0, 1.0}:
            X1, X0 = X.copy(), X.copy()
            X1[:, k] = 1.0
            for s in sib.get(c, []):
                X1[:, cols.index(s)] = 0.0
            X0[:, k] = 0.0
            out.append((X1, X0, 1.0, "discrete change"))
        else:
            h = max(1e-4, 0.01 * float(np.std(x)))
            Xp, Xm = X.copy(), X.copy()
            Xp[:, k] += h
            Xm[:, k] -= h
            out.append((Xp, Xm, 2 * h, "derivative"))
    return out


def ame_draws(prob_fn, theta_point, theta_draws, X, frame, cols, report=KEY,
              point_from_draws: bool = False) -> dict:
    """AMEs for `report`, with 95% percentile intervals over `theta_draws`.

    Sibling levels of a categorical are set to 0 as in `severity._ame_once`. Crossing
    counts are the number of (row, counterfactual) evaluations with a negative
    middle-class probability, at the point estimate and summed over draws.
    """
    cf = _counterfactuals(X, frame, cols, report)

    def once(t):
        eff = np.zeros((len(report), J))
        cross = 0
        for i, (Xa, Xb, d, _) in enumerate(cf):
            pa, ca = prob_fn(t, Xa)
            pb, cb = prob_fn(t, Xb)
            eff[i] = (pa - pb).mean(axis=0) / d
            cross += ca + cb
        return eff, cross

    point, cross_point = once(theta_point)
    drawn = np.empty((len(theta_draws), len(report), J))
    cross_draws = np.zeros(len(theta_draws), dtype=int)
    for d, t in enumerate(theta_draws):
        drawn[d], cross_draws[d] = once(t)
    if point_from_draws:
        point = drawn.mean(axis=0)
    out = {}
    for i, c in enumerate(report):
        out[c] = {"type": cf[i][3]}
        for j, lvl in enumerate(sv.SEV_ORDER):
            lo, hi = np.percentile(drawn[:, i, j], [2.5, 97.5])
            out[c][lvl] = {"ame": _r(point[i, j], 5), "ci95": [_r(lo, 5), _r(hi, 5)],
                           "excludes_zero": bool(lo > 0 or hi < 0)}
    out["_crossing"] = {"counterfactual_rows_crossing_at_estimate": int(cross_point),
                        "draws_with_any_crossing": int((cross_draws > 0).sum()),
                        "n_draws": int(len(theta_draws))}
    return out


def kr_sample(theta, cov, draws=sv.KR_DRAWS, seed=sv.KR_SEED):
    rng = np.random.default_rng(seed)
    cov = (np.asarray(cov) + np.asarray(cov).T) / 2
    return rng.multivariate_normal(np.asarray(theta), cov, size=draws, method="svd")


def ka_summary(ames: dict) -> dict:
    return {c: {"ame": ames[c]["KA"]["ame"], "ci95": ames[c]["KA"]["ci95"],
                "excludes_zero": ames[c]["KA"]["excludes_zero"]} for c in KEY}


# --------------------------------------------------------------------------
# Generic estimation helpers
# --------------------------------------------------------------------------

def hessian_from_grad(grad, theta, eps=1e-5):
    n = len(theta)
    H = np.empty((n, n))
    for i in range(n):
        tp, tm = theta.copy(), theta.copy()
        tp[i] += eps
        tm[i] -= eps
        H[:, i] = (grad(tp) - grad(tm)) / (2 * eps)
    return (H + H.T) / 2


def central_grad(f, theta, eps=1e-6):
    g = np.empty(len(theta))
    for i in range(len(theta)):
        tp, tm = theta.copy(), theta.copy()
        tp[i] += eps
        tm[i] -= eps
        g[i] = (f(tp) - f(tm)) / (2 * eps)
    return g


def hessian_report(H: np.ndarray) -> dict:
    ev = np.linalg.eigvalsh(H)
    return {"hessian_positive_definite": bool(ev.min() > 0),
            "hessian_min_eigenvalue": _r(ev.min(), 6),
            "hessian_condition_number": _r(ev.max() / ev.min(), 1) if ev.min() > 0 else None}


def info_criteria(ll: float, k: int, n: int, null_ll: float) -> dict:
    return {"loglik": _r(ll, 2), "loglik_4dp": _r(ll, 4), "n_parameters": int(k),
            "aic": _r(-2 * ll + 2 * k, 1), "bic": _r(-2 * ll + k * np.log(n), 1),
            "adjusted_rho2": _r(1 - (ll - k) / null_ll, 4)}


def coef_rows(names, theta, se, nd=3) -> dict:
    out = {}
    for nm, b, s in zip(names, theta, se):
        z = b / s if s > 0 else float("nan")
        out[nm] = {"coef": _r(b, nd), "se": _r(s, nd), "z": _r(z, 2),
                   "p": _r(float(2 * stats.norm.sf(abs(z))), 4) if s > 0 else None,
                   "ci95": [_r(b - 1.959964 * s, nd), _r(b + 1.959964 * s, nd)],
                   "excludes_zero": bool(s > 0 and abs(z) > 1.959964)}
    return out


# --------------------------------------------------------------------------
# Cross-validation on the splits of severity.cross_validated_log_loss
# --------------------------------------------------------------------------

def cv_splits(X, y):
    from sklearn.model_selection import RepeatedStratifiedKFold
    sp = RepeatedStratifiedKFold(n_splits=sv.CV_FOLDS, n_repeats=sv.CV_REPEATS,
                                 random_state=sv.CV_SEED)
    return list(sp.split(X, y))


def cv_summary(rows: list[dict]) -> dict:
    good = [r["log_loss"] for r in rows if r["ok"]]
    return {"repeats": sv.CV_REPEATS, "folds": sv.CV_FOLDS, "seed": sv.CV_SEED,
            "n_folds_total": len(rows),
            "n_folds_failed": sum(1 for r in rows if not r["ok"]),
            "n_test_rows_crossing": int(sum(r.get("crossing", 0) for r in rows)),
            "mean_log_loss": _r(np.mean(good), 3) if good else None,
            "mean_log_loss_5dp": _r(np.mean(good), 5) if good else None,
            "sd_across_folds": _r(np.std(good, ddof=1), 3) if len(good) > 1 else None}


def _fold_loss(p, y_te):
    p = np.clip(p, 1e-12, 1.0)
    return -float(np.log(p[np.arange(len(y_te)), y_te]).mean())


_CV: dict = {}


def _cv_init():
    frame, cols, X, y = load()
    _CV.update(frame=frame, cols=cols, X=X, y=y)


def _cv_fold(args):
    """One fold for the logit, probit, adjacent-category and threshold-specific fits."""
    kind, free, tr, te, i = args
    frame, cols, X, y = _CV["frame"], _CV["cols"], _CV["X"], _CV["y"]
    row = {"repeat": i // sv.CV_FOLDS + 1, "fold": i % sv.CV_FOLDS + 1,
           "n_test": int(len(te))}
    try:
        cross = 0
        if kind in ("logit", "probit"):
            from statsmodels.miscmodels.ordinal_model import OrderedModel
            res = OrderedModel(y[tr], X[tr], distr=kind).fit(method="bfgs", maxiter=2000,
                                                             disp=False)
            p, cross = cumulative_probs(np.asarray(res.params), X[te],
                                        list(range(len(cols))), [], link=kind)
        elif kind == "adjacent":
            th = fit_adjacent(X[tr], y[tr])["theta"]
            p, _ = adjacent_probs(th, X[te])
        elif kind == "ppo_noncrossing":
            prop = [c for c in cols if c not in free]
            ip, jf = [cols.index(c) for c in prop], [cols.index(c) for c in free]
            m1_tr = sv.fit_ordered_logit(frame.iloc[tr], cols, "cv")
            nc = fit_ppo_noncrossing(X[tr], y[tr], ip, jf, ppo_start(m1_tr, ip, jf))
            p, cross = cumulative_probs(nc["theta"], X[te], ip, jf)
        else:
            fit = sv.fit_ppo(frame.iloc[tr], cols, free=list(free), name="cv",
                             compute_cov=False)
            prop = [c for c in cols if c not in free]
            p, cross = cumulative_probs(fit.raw, X[te], [cols.index(c) for c in prop],
                                        [cols.index(c) for c in free])
        row.update(log_loss=_r(_fold_loss(p, y[te]), 5), ok=True, crossing=int(cross))
    except Exception as exc:                                       # noqa: BLE001
        row.update(log_loss=None, ok=False, error=type(exc).__name__)
    return row


def run_cv(pool, kind, X, y, free=()):
    jobs = [(kind, tuple(free), tr, te, i) for i, (tr, te) in enumerate(cv_splits(X, y))]
    return cv_summary(pool.map(_cv_fold, jobs))


# --------------------------------------------------------------------------
# Threshold-specific slopes without crossing
# --------------------------------------------------------------------------

def ppo_start(ol: sv.OrdinalFit, ip, jf) -> np.ndarray:
    """Feasible start: the ordered-logit slopes, each freed slope repeated per threshold."""
    return np.concatenate([ol.beta[ip], np.tile(ol.beta[jf], J - 1),
                           np.asarray(ol.raw)[len(ol.beta):]])


def fit_ppo_noncrossing(X, y, ip, jf, start) -> dict:
    """`severity._ppo_negll` maximized subject to P(BC) >= 0 on every row.

    The constraint is lin_2 - lin_1 = exp(r_2) - X_f (gamma_2 - gamma_1) >= 0 per row,
    which keeps the cumulative probabilities ordered, so every class probability is a
    probability and the log-likelihood is a proper one. SLSQP with the analytic
    constraint Jacobian and a central-difference objective gradient.
    """
    Xp, Xf = X[:, ip], X[:, jf]
    kp, kf = len(ip), len(jf)
    obj = lambda t: sv._ppo_negll(t, Xp, Xf, y, J)       # noqa: E731

    def cons(t):
        g = t[kp:kp + kf * (J - 1)].reshape(J - 1, kf)
        return np.exp(t[-1]) - Xf @ (g[1] - g[0])

    def cons_jac(t):
        Jm = np.zeros((len(X), len(t)))
        Jm[:, kp:kp + kf] = Xf
        Jm[:, kp + kf:kp + 2 * kf] = -Xf
        Jm[:, -1] = np.exp(t[-1])
        return Jm

    res = optimize.minimize(obj, start, jac=lambda t: central_grad(obj, t, 1e-6),
                            method="SLSQP",
                            constraints=[{"type": "ineq", "fun": cons, "jac": cons_jac}],
                            options={"maxiter": 2000, "ftol": 1e-12})
    th = res.x
    c = cons(th)
    g = central_grad(obj, th, 1e-6)
    # KKT stationarity: the objective gradient minus the active-constraint combination.
    act = np.where(c < 1e-6)[0]
    if len(act):
        A = cons_jac(th)[act]
        lam, *_ = np.linalg.lstsq(A.T, g, rcond=None)
        kkt = g - A.T @ lam
        lam_min = float(lam.min())
    else:
        kkt, lam_min = g, None
    return {"theta": th, "loglik": -float(res.fun), "success": bool(res.success),
            "message": str(res.message), "n_iter": int(res.nit),
            "n_active_constraints": int(len(act)),
            "min_constraint_value": float(c.min()),
            "max_abs_kkt_residual": float(np.max(np.abs(kkt))),
            "min_active_multiplier": lam_min, "obj": obj}


# --------------------------------------------------------------------------
# Adjacent-category logit
# --------------------------------------------------------------------------

def _adj_parts(theta, X, y):
    N, K = X.shape
    p, _ = adjacent_probs(theta, X)
    T = np.zeros((N, J, K + 2))
    for j in range(J):
        T[:, j, :K] = j * X
        T[:, j, K] = float(j >= 1)
        T[:, j, K + 1] = float(j == 2)
    ll = float(np.log(np.clip(p[np.arange(N), y], 1e-300, None)).sum())
    E = np.einsum("nj,njp->np", p, T)
    grad = (T[np.arange(N), y] - E).sum(axis=0)
    hess = -(np.einsum("nj,njp,njq->pq", p, T, T) - E.T @ E)
    return ll, grad, hess


def fit_adjacent(X, y, start=None) -> dict:
    K = X.shape[1]
    if start is None:
        start = np.zeros(K + 2)
    f = lambda t: -_adj_parts(t, X, y)[0]           # noqa: E731
    g = lambda t: -_adj_parts(t, X, y)[1]           # noqa: E731
    res = optimize.minimize(f, start, jac=g, method="BFGS",
                            options={"maxiter": 10000, "gtol": 1e-8})
    th = res.x
    # Newton polish on the exact Hessian (the log-likelihood is concave).
    for _ in range(20):
        ll, gr, H = _adj_parts(th, X, y)
        step = np.linalg.solve(-H, gr)
        th = th + step
        if np.max(np.abs(step)) < 1e-10:
            break
    ll, gr, H = _adj_parts(th, X, y)
    return {"theta": th, "loglik": ll, "max_abs_gradient": float(np.max(np.abs(gr))),
            "neg_hessian": -H, "bfgs_message": str(res.message),
            "bfgs_success": bool(res.success), "n_iter": int(res.nit)}


# --------------------------------------------------------------------------
# Item 1: the ordinal family
# --------------------------------------------------------------------------

def item_ordinal_family(frame, cols, X, y, pool, null_ll, m1) -> dict:
    from statsmodels.miscmodels.ordinal_model import OrderedModel
    n, K = X.shape
    allidx = list(range(K))
    out: dict = {"design": {
        "covariates": cols, "n": n,
        "ame_covariates": KEY,
        "krinsky_robb": {"draws": sv.KR_DRAWS, "seed": sv.KR_SEED,
                         "rule": ("full parameter vector drawn from the joint normal "
                                  "approximation on the estimation scale; sibling "
                                  "levels set to 0; 2.5 and 97.5 percentiles")},
        "cross_validation": ("10 x 5 repeated stratified folds, seed "
                             f"{sv.CV_SEED}, the splits of severity.cross_validated_"
                             "log_loss; fold log loss is the mean negative log "
                             "probability of the observed class, probabilities clipped "
                             "at 1e-12"),
    }}

    # -- Gate: the generic AME routine reproduces severity.average_marginal_effects
    ref = sv.average_marginal_effects(m1, frame)
    gen = ame_draws(lambda t, Z: cumulative_probs(t, Z, allidx, []), m1.raw,
                    kr_sample(m1.raw, m1.cov), X, frame, cols)
    gap = max(abs(gen[c][l]["ame"] - ref[c][l]["ame"]) + abs(gen[c][l]["ci95"][0]
              - ref[c][l]["ci95"][0]) + abs(gen[c][l]["ci95"][1] - ref[c][l]["ci95"][1])
              for c in KEY for l in sv.SEV_ORDER)
    out["verification"] = {"generic_ame_vs_severity_ame_max_abs_gap": float(f"{gap:.3g}"),
                           "passes": bool(gap < 1e-4)}
    if gap >= 1e-4:
        raise SystemExit("generic AME routine does not reproduce severity.py")

    models: dict = {}

    # -- Ordered logit (the primary), for reference
    t0 = time.time()
    cv_logit = run_cv(pool, "logit", X, y)
    models["ordered_logit_M1"] = {
        "role": "primary (severity.py M1)",
        "fit": info_criteria(m1.loglik, m1.n_params, n, null_ll),
        "cross_validated_log_loss": cv_logit,
        "ka_ames": ka_summary(gen), "ames": {c: gen[c] for c in KEY},
        "convergence": {"converged": m1.converged}}
    print(f"ordered logit: CV {cv_logit['mean_log_loss_5dp']} ({time.time()-t0:.0f}s)",
          flush=True)

    # -- (a) Ordered probit
    mod = OrderedModel(y, X, distr="probit")
    res = mod.fit(method="bfgs", maxiter=5000, disp=False, gtol=1e-8)
    th = np.asarray(res.params)
    score = np.asarray(mod.score(th))
    Hneg = -np.asarray(mod.hessian(th))
    cov = np.linalg.inv(Hneg)
    se = np.sqrt(np.diag(cov))
    ames = ame_draws(lambda t, Z: cumulative_probs(t, Z, allidx, [], link="probit"),
                     th, kr_sample(th, cov), X, frame, cols)
    models["ordered_probit"] = {
        "fit": info_criteria(float(res.llf), K + 2, n, null_ll),
        "cutpoints": [_r(c, 3) for c in sv._cut_from_raw(th[K:])],
        "coefficients": coef_rows(cols, th[:K], se[:K]),
        "cross_validated_log_loss": run_cv(pool, "probit", X, y),
        "ka_ames": ka_summary(ames), "ames": {c: ames[c] for c in KEY},
        "convergence": {"optimizer_converged_flag": bool(res.mle_retvals["converged"]),
                        "max_abs_gradient": _r(np.max(np.abs(score)), 8),
                        **hessian_report(Hneg),
                        "converged": bool(np.max(np.abs(score)) < 1e-3
                                          and np.linalg.eigvalsh(Hneg).min() > 0)},
        "note": "slopes are on the probit scale; the AMEs are directly comparable"}
    print("probit done", flush=True)

    # -- (b) Adjacent-category logit
    null_fit = fit_adjacent(np.zeros((n, 0)), y)
    adj = fit_adjacent(X, y)
    # Gradient check of the analytic score at a perturbed point
    rng = np.random.default_rng(3)
    tp = adj["theta"] + rng.normal(0, 0.1, K + 2)
    num = central_grad(lambda t: _adj_parts(t, X, y)[0], tp)
    rel = float(np.max(np.abs(num - _adj_parts(tp, X, y)[1]))
                / max(1.0, np.max(np.abs(num))))
    th = adj["theta"]
    cov = np.linalg.inv(adj["neg_hessian"])
    se = np.sqrt(np.diag(cov))
    ames = ame_draws(adjacent_probs, th, kr_sample(th, cov), X, frame, cols)
    models["adjacent_category_logit"] = {
        "specification": ("log P(Y = j+1) / P(Y = j) = alpha_j + x'beta for j = O->BC "
                          "and BC->KA, one slope vector shared by both adjacent pairs; "
                          "a positive slope moves probability toward the more severe "
                          "class. Custom maximum likelihood with the analytic score and "
                          "the exact Hessian."),
        "verification": {
            "intercept_only_loglik": _r(null_fit["loglik"], 6),
            "null_loglik_from_class_shares": _r(null_ll, 6),
            "intercept_only_reproduces_null": bool(abs(null_fit["loglik"] - null_ll) < 1e-6),
            "analytic_score_max_rel_error": float(f"{rel:.3g}"),
            "passes": bool(abs(null_fit["loglik"] - null_ll) < 1e-6 and rel < 1e-5)},
        "fit": info_criteria(adj["loglik"], K + 2, n, null_ll),
        "intercepts": {"alpha_O_to_BC": _r(th[K], 3), "alpha_BC_to_KA": _r(th[K + 1], 3)},
        "coefficients": coef_rows(cols, th[:K], se[:K]),
        "cross_validated_log_loss": run_cv(pool, "adjacent", X, y),
        "ka_ames": ka_summary(ames), "ames": {c: ames[c] for c in KEY},
        "convergence": {"bfgs_success_flag": adj["bfgs_success"],
                        "max_abs_gradient": _r(adj["max_abs_gradient"], 8),
                        **hessian_report(adj["neg_hessian"]),
                        "converged": bool(adj["max_abs_gradient"] < 1e-3
                                          and np.linalg.eigvalsh(adj["neg_hessian"]).min() > 0)}}
    print("adjacent done", flush=True)

    # -- (c) Partial proportional odds and (d) generalized ordered logit
    sets = brant_sets()
    variants = [("partial_proportional_odds_brant", sets["uncorrected"],
                 "covariates failing the per-covariate LR proportional-odds test at 5 "
                 "percent, uncorrected (the severity.py rule; numbers.json "
                 "severity.proportional_odds.failing)"),
                ("partial_proportional_odds_holm", sets["holm"],
                 "covariates failing after Holm correction (numbers.json "
                 "severity.multiplicity.holm), the set BIC prefers to ordered logit"),
                ("generalized_ordered_logit", list(cols),
                 "every coefficient threshold-specific (no proportional-odds constraint)")]
    for key, free, why in variants:
        t0 = time.time()
        fit = sv.fit_ppo(frame, cols, free=list(free), name=key, compute_cov=False)
        prop = [c for c in cols if c not in free]
        ip, jf = [cols.index(c) for c in prop], [cols.index(c) for c in free]
        kp, kf = len(ip), len(jf)
        Xp, Xf = X[:, ip], X[:, jf]
        obj = lambda t: sv._ppo_negll(t, Xp, Xf, y, J)  # noqa: E731
        H = sv._numeric_hessian(obj, fit.raw)
        hr = hessian_report(H)
        cov = np.linalg.inv(H) if hr["hessian_positive_definite"] else np.linalg.pinv(H)
        se = np.sqrt(np.clip(np.diag(cov), 0, None))
        pfun = lambda t, Z, ip=ip, jf=jf: cumulative_probs(t, Z, ip, jf)  # noqa: E731
        p_hat, cross_hat = pfun(fit.raw, X)
        ll_unclipped = loglik_from_probs(p_hat, y, clip=False) if (p_hat[np.arange(n), y] > 0).all() else None
        names = (prop + [f"{c}|cut1" for c in free] + [f"{c}|cut2" for c in free]
                 + ["cut_1", "log_cut_increment"])
        beta_all = fit.raw[:kp + kf * (J - 1)]
        se_all = se[:kp + kf * (J - 1)]
        max_coef = float(np.max(np.abs(beta_all)))
        max_se = float(np.max(se_all))
        freed = {}
        for i, c in enumerate(free):
            g1, g2 = fit.raw[kp + i], fit.raw[kp + kf + i]
            s1, s2 = se[kp + i], se[kp + kf + i]
            # Wald test of equality of the two threshold-specific slopes
            a = np.zeros(len(fit.raw))
            a[kp + i], a[kp + kf + i] = 1.0, -1.0
            vd = float(a @ cov @ a)
            wd = (g1 - g2) ** 2 / vd if vd > 0 else float("nan")
            freed[c] = {
                "cut1_BC_or_KA_vs_O": coef_rows(["g"], [g1], [s1])["g"],
                "cut2_KA_vs_O_or_BC": coef_rows(["g"], [g2], [s2])["g"],
                "difference_wald_chi2": _r(wd, 3),
                "difference_wald_p": _r(float(stats.chi2.sf(wd, 1)), 4) if vd > 0 else None,
                "ordered_logit_M1_coef": _r(m1.beta[cols.index(c)], 3)}
        entry = {
            "freed_covariates": list(free), "n_freed": len(free), "why": why,
            "fit": info_criteria(fit.loglik, fit.n_params, n, null_ll),
            "lr_vs_ordered_logit_M1": sv.nested_lr_test(m1, fit),
            "cutpoints": [_r(c, 3) for c in fit.cut],
            "threshold_specific_coefficients": freed,
            "proportional_coefficients": coef_rows(prop, fit.raw[:kp], se[:kp]),
            "crossing": {"rows_with_negative_middle_probability_at_estimate": int(cross_hat),
                         "loglik_without_clipping": _r(ll_unclipped, 4)
                         if ll_unclipped is not None else None,
                         "likelihood_is_proper": bool(cross_hat == 0)},
            "estimability": {"max_abs_coefficient": _r(max_coef, 3),
                             "max_standard_error": _r(max_se, 3),
                             "any_coefficient_or_se_above_10": bool(max_coef > 10 or max_se > 10)},
            "convergence": {"optimizer_message": fit.extras["optimizer_message"],
                            "optimizer_success_flag": fit.extras["optimizer_success_flag"],
                            "n_iter": fit.extras["n_iter"],
                            "max_abs_gradient": _r(fit.extras["max_abs_gradient_at_optimum"], 8),
                            **hr,
                            "converged": bool(fit.converged and hr["hessian_positive_definite"])},
        }
        ames = ame_draws(pfun, fit.raw, kr_sample(fit.raw, cov), X, frame, cols)
        entry["crossing"]["ame_counterfactual_rows_crossing_at_estimate"] = \
            ames["_crossing"]["counterfactual_rows_crossing_at_estimate"]
        entry["crossing"]["kr_draws_with_any_crossing"] = ames["_crossing"]["draws_with_any_crossing"]
        entry["ka_ames"] = ka_summary(ames)
        entry["ames"] = {c: ames[c] for c in KEY}
        entry["cross_validated_log_loss"] = run_cv(pool, "ppo", X, y, free=free)
        est = entry["estimability"]
        entry["estimable"] = bool(entry["convergence"]["converged"]
                                  and not est["any_coefficient_or_se_above_10"]
                                  and cross_hat == 0)
        if not entry["estimable"]:
            why_not = []
            if not entry["convergence"]["converged"]:
                why_not.append("not converged (gradient or Hessian)")
            if est["any_coefficient_or_se_above_10"]:
                why_not.append("a coefficient or standard error above 10")
            if cross_hat:
                why_not.append(f"{cross_hat} rows with crossing cumulative probabilities")
            entry["not_estimable_because"] = why_not
        models[key] = entry

        if cross_hat:
            t0 = time.time()
            nc = fit_ppo_noncrossing(X, y, ip, jf, ppo_start(m1, ip, jf))
            Hn = sv._numeric_hessian(nc["obj"], nc["theta"])
            hrn = hessian_report(Hn)
            covn = np.linalg.inv(Hn) if hrn["hessian_positive_definite"] else np.linalg.pinv(Hn)
            sen = np.sqrt(np.clip(np.diag(covn), 0, None))
            _, cross_nc = pfun(nc["theta"], X)
            k_nc = fit.n_params
            lr_nc = max(2 * (nc["loglik"] - m1.loglik), 0.0)
            freed_nc = {}
            for i, c in enumerate(free):
                g1, g2 = nc["theta"][kp + i], nc["theta"][kp + kf + i]
                freed_nc[c] = {
                    "cut1_BC_or_KA_vs_O": coef_rows(["g"], [g1], [sen[kp + i]])["g"],
                    "cut2_KA_vs_O_or_BC": coef_rows(["g"], [g2], [sen[kp + kf + i]])["g"]}
            ames_nc = ame_draws(pfun, nc["theta"], kr_sample(nc["theta"], covn), X, frame,
                                cols)
            conv_nc = bool(nc["success"] and nc["max_abs_kkt_residual"] < 1e-3
                           and nc["min_constraint_value"] > -1e-8)
            bic_nc = -2 * nc["loglik"] + k_nc * np.log(n)
            models[key + "_noncrossing"] = {
                "freed_covariates": list(free), "n_freed": len(free),
                "why": ("the same model maximized subject to a non-negative middle-class "
                        "probability on every row, so the likelihood is a proper one; the "
                        f"unconstrained fit has {cross_hat} rows whose cumulative "
                        "probabilities cross"),
                "fit": info_criteria(nc["loglik"], k_nc, n, null_ll),
                "lr_vs_ordered_logit_M1": {
                    "lr_chi2": _r(lr_nc, 4), "df": k_nc - m1.n_params,
                    "p": _r(float(stats.chi2.sf(lr_nc, k_nc - m1.n_params)), 6),
                    "note": ("naive chi-squared reference; with active inequality "
                             "constraints the reference distribution is not exactly "
                             "chi-squared")},
                "bic_minus_ordered_logit_M1": _r(bic_nc - (-2 * m1.loglik
                                                           + m1.n_params * np.log(n)), 1),
                "loglik_gap_to_unconstrained": _r(fit.loglik - nc["loglik"], 4),
                "cutpoints": [_r(c, 3) for c in
                              sv._cut_from_raw(nc["theta"][kp + kf * (J - 1):])],
                "threshold_specific_coefficients": freed_nc,
                "crossing": {
                    "rows_with_negative_middle_probability_at_estimate": int(cross_nc),
                    "rows_on_the_constraint_boundary": nc["n_active_constraints"],
                    "ame_counterfactual_rows_crossing_at_estimate":
                        ames_nc["_crossing"]["counterfactual_rows_crossing_at_estimate"],
                    "kr_draws_with_any_crossing":
                        ames_nc["_crossing"]["draws_with_any_crossing"]},
                "standard_errors": ("inverse numerical Hessian of the log-likelihood at the "
                                    "constrained optimum; with rows on the boundary these "
                                    "are approximate"),
                "convergence": {"optimizer_success_flag": nc["success"],
                                "message": nc["message"], "n_iter": nc["n_iter"],
                                "max_abs_kkt_residual": _r(nc["max_abs_kkt_residual"], 8),
                                "min_active_multiplier": _r(nc["min_active_multiplier"], 6),
                                "min_constraint_value": _r(nc["min_constraint_value"], 10),
                                **hrn, "converged": conv_nc},
                "ka_ames": ka_summary(ames_nc), "ames": {c: ames_nc[c] for c in KEY},
                "cross_validated_log_loss": run_cv(pool, "ppo_noncrossing", X, y, free=free),
                "estimable": bool(conv_nc and hrn["hessian_positive_definite"]
                                  and max(np.max(np.abs(nc["theta"][:kp + kf * (J - 1)])),
                                          np.max(sen[:kp + kf * (J - 1)])) <= 10
                                  and cross_nc == 0),
            }
            print(f"{key}_noncrossing: ll {nc['loglik']:.2f}, active "
                  f"{nc['n_active_constraints']}, kkt {nc['max_abs_kkt_residual']:.2e} "
                  f"({time.time()-t0:.0f}s)", flush=True)
        print(f"{key}: ll {fit.loglik:.2f}, crossing {cross_hat}, "
              f"maxcoef {max_coef:.2f}, maxse {max_se:.2f} ({time.time()-t0:.0f}s)", flush=True)

    if "partial_proportional_odds_brant" in models:
        ref_ll = json.loads((DATA / "severity.json").read_text(
            encoding="utf-8"))["ppo"]["fit"]["loglik"]
        models["partial_proportional_odds_brant"]["verification"] = {
            "loglik_severity": ref_ll,
            "loglik_refit": _r(models["partial_proportional_odds_brant"]["fit"]["loglik_4dp"], 4),
            "reproduces": bool(abs(ref_ll - models["partial_proportional_odds_brant"]["fit"]["loglik_4dp"]) < 1e-3)}

    # The Brant decision in severity.py rests on 34 single-covariate partial
    # proportional-odds fits made with the clipped likelihood. Count crossing rows in each.
    t0 = time.time()
    single = {}
    for c in cols:
        prop = [x for x in cols if x != c]
        ip, jf = [cols.index(x) for x in prop], [cols.index(c)]
        f1 = sv.fit_ppo(frame, cols, free=[c], name=f"po_{c}", compute_cov=False)
        _, cr = cumulative_probs(f1.raw, X, ip, jf)
        row = {"rows_crossing": int(cr), "lr_chi2_clipped": _r(max(2 * (f1.loglik - m1.loglik), 0), 4)}
        if cr:
            nc = fit_ppo_noncrossing(X, y, ip, jf, ppo_start(m1, ip, jf))
            lr = max(2 * (nc["loglik"] - m1.loglik), 0.0)
            row.update(lr_chi2_noncrossing=_r(lr, 4),
                       lr_p_noncrossing=_r(float(stats.chi2.sf(lr, 1)), 4),
                       noncrossing_converged=bool(nc["success"]
                                                  and nc["max_abs_kkt_residual"] < 1e-3))
        single[c] = row
    failing = brant_sets()["uncorrected"]
    out["brant_single_covariate_crossing"] = {
        "per_covariate": single,
        "n_fits_with_crossing": sum(1 for v in single.values() if v["rows_crossing"]),
        "failing_set_changes": sorted(
            c for c, v in single.items() if v["rows_crossing"]
            and (v["lr_p_noncrossing"] < 0.05) != (c in failing)),
    }
    print(f"single-covariate crossing check ({time.time()-t0:.0f}s): "
          f"{out['brant_single_covariate_crossing']['n_fits_with_crossing']} fits cross",
          flush=True)

    # Monte Carlo stability of the primary's Krinsky-Robb intervals for the key covariates.
    seeds = [sv.KR_SEED + i for i in range(1, 4)]
    chk = {s: sv.average_marginal_effects(m1, frame, seed=s) for s in seeds}
    out["primary_kr_seed_check"] = {
        "seeds": [sv.KR_SEED] + seeds, "draws": sv.KR_DRAWS,
        "ka_ci95_by_seed": {c: {str(sv.KR_SEED): gen[c]["KA"]["ci95"],
                                **{str(s): chk[s][c]["KA"]["ci95"] for s in seeds}}
                            for c in ("poverty_share", "coord_missing", "age_unknown")},
        "ka_excludes_zero_by_seed": {c: {str(sv.KR_SEED): gen[c]["KA"]["excludes_zero"],
                                         **{str(s): chk[s][c]["KA"]["excludes_zero"]
                                            for s in seeds}}
                                     for c in ("poverty_share", "coord_missing", "age_unknown")},
    }

    out["models"] = models
    prim = {c for c in KEY if gen[c]["KA"]["excludes_zero"]}
    proper = [k for k, v in models.items()
              if v.get("crossing", {}).get("rows_with_negative_middle_probability_at_estimate", 0) == 0]
    out["summary"] = {
        "ka_excludes_zero_by_model": {k: [c for c in KEY if v["ka_ames"][c]["excludes_zero"]]
                                      for k, v in models.items()},
        "differs_from_primary": {k: sorted(set(c for c in KEY if v["ka_ames"][c]["excludes_zero"])
                                           ^ prim) for k, v in models.items()},
        "aic": {k: v["fit"]["aic"] for k, v in models.items()},
        "bic": {k: v["fit"]["bic"] for k, v in models.items()},
        "cv_log_loss": {k: v["cross_validated_log_loss"]["mean_log_loss"]
                        for k, v in models.items()},
        "best_aic": min(models, key=lambda k: models[k]["fit"]["aic"]),
        "best_bic": min(models, key=lambda k: models[k]["fit"]["bic"]),
        "best_cv_log_loss": min(models, key=lambda k:
                                models[k]["cross_validated_log_loss"]["mean_log_loss_5dp"]),
        "proper_likelihood_models": proper,
        "best_aic_proper": min(proper, key=lambda k: models[k]["fit"]["aic"]),
        "best_bic_proper": min(proper, key=lambda k: models[k]["fit"]["bic"]),
        "best_cv_log_loss_proper": min(proper, key=lambda k: models[k][
            "cross_validated_log_loss"]["mean_log_loss_5dp"]),
        "all_converged": {k: v["convergence"]["converged"] for k, v in models.items()},
        "estimable": {k: v.get("estimable", True) for k, v in models.items()},
    }
    return out


# --------------------------------------------------------------------------
# Item 2: random thresholds
# --------------------------------------------------------------------------

class RandomThresholds:
    """Ordered logit with normal random terms in the thresholds.

    cut_1n = c_1 + s_shift * nu_shift,  cut_2n = cut_1n + exp(psi + s_gap * nu_gap),
    theta = [beta (K), c_1, psi, (s_shift), s_gap]. With s_shift = s_gap = 0 this is the
    fixed ordered logit in the severity.py parameterization (psi is the log increment).
    The log-normal increment keeps the thresholds ordered for every draw.
    """

    def __init__(self, X, y, nu_gap, nu_shift=None):
        self.X, self.y = X, y
        self.K = X.shape[1]
        self.ng = nu_gap
        self.ns = nu_shift
        self.shift = nu_shift is not None
        self.n_params = self.K + 3 + int(self.shift)

    def names(self, cols):
        return list(cols) + ["cut_1", "log_increment_mean"] + (
            ["sd_shift_both_thresholds"] if self.shift else []) + ["sd_log_increment"]

    def unpack(self, t):
        K = self.K
        if self.shift:
            return t[:K], t[K], t[K + 1], t[K + 2], t[K + 3]
        return t[:K], t[K], t[K + 1], 0.0, t[K + 2]

    def negll_grad(self, t):
        beta, c1, psi, ss, sg = self.unpack(t)
        eta = self.X @ beta
        y = self.y
        A = c1 - eta[:, None] + (ss * self.ns if self.shift else 0.0)
        A = np.broadcast_to(A, self.ng.shape)
        g = np.exp(psi + sg * self.ng)
        B = A + g
        FA, FB = expit(A), expit(B)
        fA, fB = FA * (1 - FA), FB * (1 - FB)
        y0, y1, y2 = (y == 0)[:, None], (y == 1)[:, None], (y == 2)[:, None]
        P = np.where(y0, FA, np.where(y1, FB - FA, expit(-B)))
        a = np.where(y0, fA, np.where(y1, -fA, 0.0))
        b = np.where(y1, fB, np.where(y2, -fB, 0.0))
        Pbar = np.maximum(P.mean(axis=1), 1e-300)
        w = 1.0 / Pbar
        ab = (a + b).mean(axis=1) * w
        grad = [-self.X.T @ ab, [ab.sum()], [((b * g).mean(axis=1) * w).sum()]]
        if self.shift:
            grad.append([(((a + b) * self.ns).mean(axis=1) * w).sum()])
        grad.append([((b * g * self.ng).mean(axis=1) * w).sum()])
        ll = float(np.log(Pbar).sum())
        return -ll, -np.concatenate([np.atleast_1d(np.asarray(v, float)) for v in grad])

    def negll(self, t):
        return self.negll_grad(t)[0]

    def grad(self, t):
        return self.negll_grad(t)[1]


def _rt_fit(model, start, fixed_idx=None, fixed_val=None):
    """BFGS on the analytic gradient, optionally with one parameter held fixed."""
    if fixed_idx is None:
        res = optimize.minimize(model.negll_grad, start, jac=True, method="BFGS",
                                options={"maxiter": 10000, "gtol": 1e-6})
        return res.x, -res.fun, res
    free = [i for i in range(len(start)) if i != fixed_idx]

    def fg(z):
        t = np.empty(len(start))
        t[free] = z
        t[fixed_idx] = fixed_val
        f, g = model.negll_grad(t)
        return f, g[free]
    res = optimize.minimize(fg, start[free], jac=True, method="BFGS",
                            options={"maxiter": 10000, "gtol": 1e-6})
    t = np.empty(len(start))
    t[free] = res.x
    t[fixed_idx] = fixed_val
    return t, -res.fun, res


def _rt_best(model, base, sd_idx):
    runs = []
    for s0 in RT_SIGMA_STARTS:
        st = base.copy()
        st[sd_idx] = s0
        th, ll, res = _rt_fit(model, st)
        runs.append((s0, th, ll, res))
    s_best, th, ll, res = max(runs, key=lambda r: r[2])
    th, ll, res = _rt_fit(model, th)
    g = model.grad(th)
    H = hessian_from_grad(model.grad, th)
    hr = hessian_report(H)
    cov = np.linalg.inv(H) if hr["hessian_positive_definite"] else np.linalg.pinv(H)
    lls = [r[2] for r in runs]
    return {"theta": th, "loglik": ll, "cov": cov, "H": H,
            "convergence": {"optimizer_success_flag": bool(res.success),
                            "message": str(res.message),
                            "max_abs_gradient": _r(np.max(np.abs(g)), 8), **hr,
                            "converged": bool(np.max(np.abs(g)) < 1e-3
                                              and hr["hessian_positive_definite"]),
                            "sigma_starts": [{"sd_start": s, "loglik": _r(l, 4),
                                              "abs_sd": [_r(abs(t[i]), 4) for i in sd_idx]}
                                             for s, t, l, _ in runs],
                            "starts_agree_loglik_within_0.01": bool(max(lls) - min(lls) < 0.01)}}


def item_random_thresholds(frame, cols, X, y, null_ll, m1) -> dict:
    n, K = X.shape
    out: dict = {"design": {
        "model": ("ordered logit with cut_1n = c_1 + s_shift * nu_1 and "
                  "cut_2n = cut_1n + exp(psi + s_gap * nu_2), nu ~ N(0, 1) per rider; "
                  "variant 'increment' has s_shift = 0, variant 'both' estimates both. "
                  "The random term on cut_1 moves both thresholds together; with no "
                  "covariate in the thresholds it is a random intercept and is "
                  "identified only through the shape of the logistic-normal mixture."),
        "estimation": ("simulated maximum likelihood, scrambled Halton draws "
                       "(random_parameters.halton_normal), analytic gradient, BFGS, "
                       "covariance from central differences of the gradient"),
        "draws": RT_DRAWS, "draws_check": list(RT_DRAWS_CHECK), "halton_seed": RT_SEED,
        "sd_starting_values": list(RT_SIGMA_STARTS),
        "lr_reference": ("chi-bar-squared: equal mixture of chi2(0) and chi2(1) for one "
                         "SD; 1/4, 1/2, 1/4 mixture of chi2(0), chi2(1), chi2(2) for two "
                         "(random_parameters.lr_boundary)")}}
    ll_fixed = m1.loglik
    k_fixed = m1.n_params

    # Gates: the fixed special case and the analytic gradient
    d500 = rp.halton_normal(n, RT_DRAWS, 2, RT_SEED)
    m_inc = RandomThresholds(X, y, d500[:, :, 1])
    base = np.concatenate([m1.raw, [0.0]])
    ll0 = -m_inc.negll(base)
    m_both = RandomThresholds(X, y, d500[:, :, 1], d500[:, :, 0])
    tp = np.concatenate([m1.raw, [0.6, 0.8]]) + np.random.default_rng(5).normal(0, 0.05, K + 4)
    an = m_both.grad(tp)
    num = central_grad(m_both.negll, tp)
    rel = float(np.max(np.abs(an - num)) / max(1.0, np.max(np.abs(num))))
    out["verification"] = {
        "loglik_M1": _r(ll_fixed, 6), "loglik_this_model_at_sd_zero": _r(ll0, 6),
        "fixed_special_case_passes": bool(abs(ll0 - ll_fixed) < 1e-6),
        "analytic_gradient_max_rel_error": float(f"{rel:.3g}"),
        "gradient_passes": bool(rel < 1e-5)}
    if not (out["verification"]["fixed_special_case_passes"]
            and out["verification"]["gradient_passes"]):
        out["aborted"] = "verification failed"
        return out

    variants = {}
    for key, q in (("increment", 1), ("both", 2)):
        res_by = {}
        for r in (RT_DRAWS,) + RT_DRAWS_CHECK:
            dd = d500 if r == RT_DRAWS else rp.halton_normal(n, r, 2, RT_SEED)
            m = (RandomThresholds(X, y, dd[:, :, 1]) if q == 1
                 else RandomThresholds(X, y, dd[:, :, 1], dd[:, :, 0]))
            b0 = np.concatenate([m1.raw, np.zeros(q)])
            sd_idx = list(range(K + 2, K + 2 + q))
            fr = _rt_best(m, b0, sd_idx)
            res_by[r] = (m, fr)
            print(f"random thresholds {key} draws {r}: ll {fr['loglik']:.4f} "
                  f"sd {[round(abs(fr['theta'][i]), 4) for i in sd_idx]}", flush=True)
        m, fr = res_by[RT_DRAWS]
        names = m.names(cols)
        se = np.sqrt(np.clip(np.diag(fr["cov"]), 0, None))
        sd_tab = {}
        for i in range(K + 2, K + 2 + q):
            s, sse = abs(fr["theta"][i]), se[i]
            sd_tab[names[i]] = {"sd": _r(s, 4), "se": _r(sse, 4),
                                "z": _r(s / sse, 2) if sse > 0 else None,
                                "wald_p": _r(float(2 * stats.norm.sf(s / sse)), 4) if sse > 0 else None}
        lr = rp.lr_boundary(fr["loglik"], ll_fixed, q)
        variants[key] = {
            "random_sds": sd_tab,
            "fit": info_criteria(fr["loglik"], k_fixed + q, n, null_ll),
            "fixed_M1_fit": info_criteria(ll_fixed, k_fixed, n, null_ll),
            "lr_vs_fixed_M1": lr,
            "bic_change_vs_M1": _r((-2 * fr["loglik"] + (k_fixed + q) * np.log(n))
                                   - (-2 * ll_fixed + k_fixed * np.log(n)), 1),
            "cut_1": _r(fr["theta"][K], 3), "log_increment_mean": _r(fr["theta"][K + 1], 3),
            "max_abs_slope_change_vs_M1": _r(np.max(np.abs(fr["theta"][:K] - m1.beta)), 4),
            "convergence": fr["convergence"],
            "draw_stability": {
                f"draws{r}": {"loglik": _r(res_by[r][1]["loglik"], 4),
                              "sd": [_r(abs(res_by[r][1]["theta"][i]), 4)
                                     for i in range(K + 2, K + 2 + q)],
                              "lr_p_chi_bar_squared": rp.lr_boundary(
                                  res_by[r][1]["loglik"], ll_fixed, q)["p_chi_bar_squared"],
                              "converged": res_by[r][1]["convergence"]["converged"]}
                for r in (RT_DRAWS,) + RT_DRAWS_CHECK}}

    # Profile log-likelihood over the increment SD (variant 'increment', 500 draws)
    m = RandomThresholds(X, y, d500[:, :, 1])
    prof = {}
    start = np.concatenate([m1.raw, [0.0]])
    for s in RT_PROFILE_GRID:
        th, ll, res = _rt_fit(m, start, fixed_idx=K + 2, fixed_val=s)
        prof[str(s)] = {"loglik": _r(ll, 4), "lr_vs_M1": _r(2 * (ll - ll_fixed), 4),
                        "log_increment_mean": _r(th[K + 1], 3)}
    variants["increment"]["profile_loglik_over_sd"] = prof
    out["variants"] = variants
    inc = variants["increment"]
    out["summary"] = {
        "sd_log_increment_increment_variant": inc["random_sds"]["sd_log_increment"],
        "lr_p_chi_bar_squared": {k: v["lr_vs_fixed_M1"]["p_chi_bar_squared"]
                                 for k, v in variants.items()},
        "bic_change_vs_M1": {k: v["bic_change_vs_M1"] for k, v in variants.items()},
        "converged": {k: v["convergence"]["converged"] for k, v in variants.items()},
        "profile_loglik_range": [min(v["loglik"] for v in prof.values()),
                                 max(v["loglik"] for v in prof.values())],
    }
    return out


# --------------------------------------------------------------------------
# Item 3: shrinkage
# --------------------------------------------------------------------------

def _ol_info(theta_raw, X, y):
    """Log-likelihood and expected information on (beta, cut_1, cut_2)."""
    K = X.shape[1]
    beta, cut = theta_raw[:K], sv._cut_from_raw(theta_raw[K:])
    eta = X @ beta
    F1, F2 = expit(cut[0] - eta), expit(cut[1] - eta)
    f1, f2 = F1 * (1 - F1), F2 * (1 - F2)
    P = np.column_stack([F1, F2 - F1, 1 - F2])
    N = len(y)
    D = np.zeros((N, J, K + 2))
    D[:, 0, :K] = -f1[:, None] * X
    D[:, 1, :K] = -(f2 - f1)[:, None] * X
    D[:, 2, :K] = f2[:, None] * X
    D[:, 0, K], D[:, 1, K] = f1, -f1
    D[:, 1, K + 1], D[:, 2, K + 1] = f2, -f2
    W = (D / np.sqrt(np.maximum(P, 1e-300))[:, :, None]).reshape(-1, K + 2)
    info = W.T @ W
    ll = float(np.log(np.maximum(P[np.arange(N), y], 1e-300)).sum())
    return ll, info, P, D


def _firth_obj(theta_raw, X, y):
    ll, info, _, _ = _ol_info(theta_raw, X, y)
    sign, logdet = np.linalg.slogdet(info)
    return -(ll + 0.5 * logdet) if sign > 0 else 1e10


def fit_firth(X, y, start):
    K = X.shape[1]
    f = lambda t: _firth_obj(t, X, y)                  # noqa: E731
    g = lambda t: central_grad(f, t, eps=1e-6)         # noqa: E731
    res = optimize.minimize(f, start, jac=g, method="BFGS",
                            options={"maxiter": 10000, "gtol": 1e-6})
    th = res.x
    gr = g(th)
    Hpen = hessian_from_grad(g, th, eps=1e-4)
    ll, info, _, _ = _ol_info(th, X, y)
    cut = sv._cut_from_raw(th[K:])
    cov_nat = np.linalg.inv(info)
    # Delta method to the raw scale (cut_1, log(cut_2 - cut_1)) used by the AME draws.
    A = np.eye(K + 2)
    gap = cut[1] - cut[0]
    A[K + 1, K], A[K + 1, K + 1] = -1.0 / gap, 1.0 / gap
    cov_raw = A @ cov_nat @ A.T
    return {"theta": th, "loglik": ll, "penalized_loglik": -float(res.fun),
            "cov_nat": cov_nat, "cov_raw": cov_raw, "cut": cut,
            "convergence": {"optimizer_success_flag": bool(res.success),
                            "message": str(res.message), "n_iter": int(res.nit),
                            "max_abs_gradient_penalized": _r(np.max(np.abs(gr)), 8),
                            **hessian_report(Hpen),
                            "converged": bool(np.max(np.abs(gr)) < 1e-3
                                              and np.linalg.eigvalsh(Hpen).min() > 0)}}


def fit_bayes(X, y, prior_sd, init_cut, label):
    import arviz as az
    import nutpie
    import pymc as pm
    import pytensor.tensor as pt
    from src import audit_model as am

    K = X.shape[1]
    with pm.Model() as model:
        beta = pm.Normal("beta", mu=0.0, sigma=np.asarray(prior_sd, float), shape=K)
        cut = pm.Normal("cut", mu=0.0, sigma=BAYES_CUT_SD, shape=J - 1,
                        transform=pm.distributions.transforms.ordered,
                        initval=np.asarray(init_cut, float))
        eta = pt.dot(pt.as_tensor_variable(X), beta)
        pm.OrderedLogistic("y", eta=eta, cutpoints=cut, observed=y)
    t0 = time.time()
    compiled = nutpie.compile_pymc_model(model)
    idata = nutpie.sample(compiled, draws=BAYES_DRAWS, tune=BAYES_TUNE, chains=BAYES_CHAINS,
                          seed=BAYES_SEED, progress_bar=False, target_accept=0.9)
    diag = am.diagnostics(idata, ["beta", "cut"])
    diag["seconds"] = round(time.time() - t0, 1)
    b = idata.posterior["beta"].to_numpy().reshape(-1, K)
    c = idata.posterior["cut"].to_numpy().reshape(-1, J - 1)
    print(f"bayes {label}: {diag}", flush=True)
    return b, c, diag, az


def item_shrinkage(frame, cols, X, y, null_ll, m1) -> dict:
    n, K = X.shape
    allidx = list(range(K))
    pfun = lambda t, Z: cumulative_probs(t, Z, allidx, [])  # noqa: E731
    out: dict = {}
    se_ml = np.sqrt(np.diag(m1.cov)[:K])

    # ---- (a) Jeffreys-prior penalized ordered logit -------------------------------
    ll_chk, info_chk, P_chk, D_chk = _ol_info(m1.raw, X, y)
    # finite-difference check of dP/d(beta, cut) at the M1 estimate
    nat = np.concatenate([m1.beta, m1.cut])

    def p_nat(v):
        e = X @ v[:K]
        F1, F2 = expit(v[K] - e), expit(v[K + 1] - e)
        return np.column_stack([F1, F2 - F1, 1 - F2])
    Dnum = np.empty_like(D_chk)
    for i in range(K + 2):
        vp, vm = nat.copy(), nat.copy()
        vp[i] += 1e-6
        vm[i] -= 1e-6
        Dnum[:, :, i] = (p_nat(vp) - p_nat(vm)) / 2e-6
    d_err = float(np.max(np.abs(Dnum - D_chk)))
    fi = fit_firth(X, y, m1.raw.copy())
    se_f = np.sqrt(np.diag(fi["cov_nat"])[:K])
    ames_f = ame_draws(pfun, fi["theta"], kr_sample(fi["theta"], fi["cov_raw"]), X, frame, cols)
    out["firth"] = {
        "method": ("penalized likelihood l(theta) + 0.5 log det I(theta), I the expected "
                   "information of the cumulative logit on (beta, cut_1, cut_2) "
                   "(Jeffreys-prior penalty, Firth 1993). The cumulative logit is not a "
                   "canonical link, so this penalized estimator is Firth-type but is not "
                   "claimed to equal the adjusted-score mean-bias reduction of Kosmidis "
                   "(2014). Standard errors are the inverse expected information at the "
                   "penalized estimate; AME intervals use 1000 Krinsky-Robb draws."),
        "verification": {"loglik_at_M1_estimate": _r(ll_chk, 6), "loglik_M1": _r(m1.loglik, 6),
                         "loglik_matches": bool(abs(ll_chk - m1.loglik) < 1e-6),
                         "dP_analytic_vs_finite_difference_max_abs_error": float(f"{d_err:.3g}"),
                         "passes": bool(abs(ll_chk - m1.loglik) < 1e-6 and d_err < 1e-6)},
        "fit": info_criteria(fi["loglik"], K + 2, n, null_ll),
        "penalized_loglik": _r(fi["penalized_loglik"], 4),
        "cutpoints": [_r(c, 3) for c in fi["cut"]],
        "coefficients": coef_rows(cols, fi["theta"][:K], se_f),
        "key_coefficients_vs_M1": {
            c: {"M1": _r(m1.beta[cols.index(c)], 3), "M1_se": _r(se_ml[cols.index(c)], 3),
                "firth": _r(fi["theta"][cols.index(c)], 3), "firth_se": _r(se_f[cols.index(c)], 3),
                "shrinkage_ratio": _r(fi["theta"][cols.index(c)] / m1.beta[cols.index(c)], 3)}
            for c in KEY},
        "max_abs_coefficient": _r(np.max(np.abs(fi["theta"][:K])), 3),
        "max_abs_coefficient_M1": _r(np.max(np.abs(m1.beta)), 3),
        "mean_abs_shrinkage_of_slopes": _r(np.mean(np.abs(fi["theta"][:K]))
                                           / np.mean(np.abs(m1.beta)), 3),
        "convergence": fi["convergence"],
        "ka_ames": ka_summary(ames_f), "ames": {c: ames_f[c] for c in KEY}}
    print("firth done", flush=True)

    # ---- (b) Bayesian ordered logit ----------------------------------------------
    sd_x = X.std(axis=0)
    cont = [c for c in cols if not set(np.unique(frame[c])) <= {0.0, 1.0}]
    priors = {
        "raw_coding": (np.ones(K),
                       "normal(0, 1) on every slope in the coding of the model frame "
                       "(binary 0/1; poverty_share a 0-1 share; road density in km per "
                       "100 m buffer)"),
        "continuous_per_2sd": (np.array([1.0 / (2 * sd_x[i]) if c in cont else 1.0
                                         for i, c in enumerate(cols)]),
                               "normal(0, 1) on every slope with the continuous covariates "
                               "(" + ", ".join(cont) + ") measured per two standard "
                               "deviations (Gelman et al. 2008), so a binary and a "
                               "continuous slope get a prior of comparable strength"),
    }
    bayes = {}
    rng = np.random.default_rng(BAYES_SEED)
    for key, (psd, desc) in priors.items():
        b, c, diag, az = fit_bayes(X, y, psd, m1.cut, key)
        theta_draws = np.column_stack([b, c[:, :1], np.log(np.maximum(c[:, 1:] - c[:, :1],
                                                                       1e-12))])
        idx = rng.choice(len(theta_draws), size=min(BAYES_AME_DRAWS, len(theta_draws)),
                         replace=False)
        sub = theta_draws[idx]
        ames_b = ame_draws(pfun, theta_draws.mean(axis=0), sub, X, frame, cols,
                           point_from_draws=True)
        pm_theta = np.concatenate([b.mean(axis=0), [c[:, 0].mean()],
                                   [np.log(c[:, 1].mean() - c[:, 0].mean())]])
        ll_pm = loglik_from_probs(pfun(pm_theta, X)[0], y)
        sd_post = b.std(axis=0)
        bayes[key] = {
            "prior": desc,
            "prior_sd_by_covariate": {c_: _r(psd[i], 3) for i, c_ in enumerate(cols)
                                      if psd[i] != 1.0} or "1 for every slope",
            "cutpoint_prior": f"normal(0, {BAYES_CUT_SD}) with the ordered transform",
            "sampler": {"engine": "nutpie (NUTS)", "chains": BAYES_CHAINS, "tune": BAYES_TUNE,
                        "draws_per_chain": BAYES_DRAWS, "seed": BAYES_SEED,
                        "target_accept": 0.9},
            "diagnostics": diag,
            "loglik_at_posterior_mean": _r(ll_pm, 2),
            "loglik_M1": _r(m1.loglik, 2),
            "cutpoints_posterior_mean": [_r(v, 3) for v in c.mean(axis=0)],
            "key_coefficients": {
                c_: {"posterior_mean": _r(b[:, cols.index(c_)].mean(), 3),
                     "posterior_sd": _r(sd_post[cols.index(c_)], 3),
                     "ci95": [_r(np.percentile(b[:, cols.index(c_)], 2.5), 3),
                              _r(np.percentile(b[:, cols.index(c_)], 97.5), 3)],
                     "M1": _r(m1.beta[cols.index(c_)], 3),
                     "M1_se": _r(se_ml[cols.index(c_)], 3),
                     "prior_to_likelihood_precision_ratio":
                         _r((se_ml[cols.index(c_)] / psd[cols.index(c_)]) ** 2, 3)}
                for c_ in KEY},
            "sign_agreement_with_M1": int(np.sum(np.sign(b.mean(axis=0)) == np.sign(m1.beta))),
            "max_abs_posterior_mean": _r(np.max(np.abs(b.mean(axis=0))), 3),
            "ame_draws": int(len(sub)),
            "ame_point": "posterior mean of the AME",
            "ka_ames": ka_summary(ames_b), "ames": {c_: ames_b[c_] for c_ in KEY}}
    out["bayesian"] = bayes

    ref = sv.average_marginal_effects(m1, frame)
    prim = {c for c in KEY if ref[c]["KA"]["excludes_zero"]}
    rows = {"firth": out["firth"]["ka_ames"]}
    rows.update({f"bayes_{k}": v["ka_ames"] for k, v in bayes.items()})
    out["summary"] = {
        "primary_ka_excludes_zero": sorted(prim),
        "ka_excludes_zero_by_model": {k: [c for c in KEY if v[c]["excludes_zero"]]
                                      for k, v in rows.items()},
        "differs_from_primary": {k: sorted({c for c in KEY if v[c]["excludes_zero"]} ^ prim)
                                 for k, v in rows.items()},
        "bayes_healthy": {k: v["diagnostics"]["healthy"] for k, v in bayes.items()},
        "firth_converged": out["firth"]["convergence"]["converged"]}
    return out


# --------------------------------------------------------------------------
# Item 4: separation
# --------------------------------------------------------------------------

def lp_separation(X, y) -> dict:
    """Linear-programming check for (quasi-)complete separation.

    Ordinal: is there (beta, a_1 <= a_2), beta != 0, with x'beta <= a_1 for O rows,
    a_1 <= x'beta <= a_2 for BC rows and x'beta >= a_2 for KA rows? Maximize the sum of
    the margins under a box constraint; a positive optimum means such a direction
    exists. With both classes present and a full-rank design the only zero-optimum
    solution is beta = 0. The same check is run for each binary cumulative split
    (the Konis 2007 formulation with an intercept).
    """
    from scipy.optimize import linprog
    N, K = X.shape
    rank_ok = bool(np.linalg.matrix_rank(np.column_stack([np.ones(N), X])) == K + 1)

    def solve(rows):
        A = np.array([r for r in rows])
        # maximize sum(A v)  s.t.  A v >= 0,  -1 <= v <= 1
        res = linprog(-A.sum(axis=0), A_ub=-A, b_ub=np.zeros(len(A)),
                      bounds=[(-1, 1)] * A.shape[1], method="highs")
        return float(-res.fun) if res.status == 0 else None, res.status

    rows = []
    for xn, yn in zip(X, y):
        if yn == 0:
            rows.append(np.concatenate([-xn, [1.0, 0.0]]))
        elif yn == 1:
            rows.append(np.concatenate([xn, [-1.0, 0.0]]))
            rows.append(np.concatenate([-xn, [0.0, 1.0]]))
        else:
            rows.append(np.concatenate([xn, [0.0, -1.0]]))
    opt, st = solve(rows)
    out = {"design_full_rank_with_intercept": rank_ok,
           "ordinal": {"lp_optimum": _r(opt, 8), "lp_status": st,
                       "separation": bool(opt is not None and opt > 1e-7)}}
    for j, lab in ((0, "split_O_vs_BC_KA"), (1, "split_O_BC_vs_KA")):
        s = np.where(y > j, 1.0, -1.0)
        o, st = solve(list(s[:, None] * np.column_stack([np.ones(N), X])))
        out[lab] = {"lp_optimum": _r(o, 8), "lp_status": st,
                    "separation": bool(o is not None and o > 1e-7)}
    out["any_separation"] = bool(out["ordinal"]["separation"]
                                 or out["split_O_vs_BC_KA"]["separation"]
                                 or out["split_O_BC_vs_KA"]["separation"])
    return out


def item_separation(frame) -> dict:
    nums = json.loads((DATA / "numbers.json").read_text(encoding="utf-8"))
    spec = sv.ladder(frame)
    out = {"method": ("severity.separation_diagnostics on each ordered-logit rung (flag if "
                      "any |coef| or SE exceeds 10), zero cells in each binary covariate "
                      "by outcome-class table, and a linear-programming separation check "
                      "(ordinal and per cumulative split).")}
    for name, cols in spec.items():
        fit = sv.fit_ordered_logit(frame, cols, name)
        X, y = sv._design(frame, cols)
        diag = sv.separation_diagnostics(fit)
        se = np.sqrt(np.clip(np.diag(fit.cov)[:len(fit.beta)], 0, None))
        zero = {}
        for i, c in enumerate(cols):
            x = X[:, i]
            if set(np.unique(x)) <= {0.0, 1.0}:
                tab = {lvl: int(((x == 1) & (y == j)).sum()) for j, lvl in enumerate(sv.SEV_ORDER)}
                if min(tab.values()) == 0:
                    zero[c] = tab
        smallest = min(((int(((X[:, i] == 1) & (y == j)).sum()), c, lvl)
                        for i, c in enumerate(cols) if set(np.unique(X[:, i])) <= {0.0, 1.0}
                        for j, lvl in enumerate(sv.SEV_ORDER)))
        out[name] = {
            "n_covariates": len(cols),
            "max_abs_coefficient": diag["max_abs_coefficient"],
            "covariate_with_max_abs_coefficient": cols[int(np.argmax(np.abs(fit.beta)))],
            "max_standard_error": diag["max_standard_error"],
            "covariate_with_max_standard_error": cols[int(np.argmax(se))],
            "max_abs_z": diag["max_abs_t"],
            "suggests_separation_coef_or_se_above_10": diag["suggests_separation"],
            "numbers_json_separation_flag": nums["severity"]["rungs"][name].get("separation_flag"),
            "agrees_with_numbers_json": bool(diag["suggests_separation"]
                                             == nums["severity"]["rungs"][name].get("separation_flag")),
            "binary_covariates_with_a_zero_class_cell": zero,
            "smallest_binary_class_cell": {"count": smallest[0], "covariate": smallest[1],
                                           "class": smallest[2]},
            "linear_programming_check": lp_separation(X, y),
            "converged": fit.converged,
        }
        print(f"separation {name}: {out[name]['max_abs_coefficient']} "
              f"{out[name]['max_standard_error']} LP {out[name]['linear_programming_check']['any_separation']}",
              flush=True)
    out["summary"] = {k: {"max_abs_coefficient": out[k]["max_abs_coefficient"],
                          "max_standard_error": out[k]["max_standard_error"],
                          "flag": out[k]["suggests_separation_coef_or_se_above_10"],
                          "lp_separation": out[k]["linear_programming_check"]["any_separation"]}
                      for k in spec}
    return out


# --------------------------------------------------------------------------
# Item 5: power of the cue-block test
# --------------------------------------------------------------------------

_PW: dict = {}


def _pw_init():
    frame, cols, X, y = load()
    spec = sv.ladder(frame)
    c0 = spec["M0c"]
    cues = [c for c in spec["M1"] if c not in c0]
    m0c = sv.fit_ordered_logit(frame, c0, "M0c")
    Xc = frame[cues].to_numpy(float)
    _PW.update(frame=frame, c0=c0, cues=cues, m0c=m0c, X0=frame[c0].to_numpy(float),
               Xcue=Xc, Xcue_c=Xc - Xc.mean(axis=0),
               draws0=rp.halton_normal(len(frame), 1, 0, 1))


def _pw_one(args):
    bi, beta, rep = args
    rng = np.random.default_rng([POWER_SEED, bi, rep])
    m0c = _PW["m0c"]
    X0, Xcc = _PW["X0"], _PW["Xcue_c"]
    eta = X0 @ m0c.beta + beta * Xcc.sum(axis=1)
    lat = eta + rng.logistic(size=len(eta))
    y = np.searchsorted(m0c.cut, lat)
    counts = np.bincount(y, minlength=J)
    row = {"bi": bi, "beta": beta, "rep": rep, "counts": counts.tolist()}
    if counts.min() == 0:
        row.update(ok=False, reason="empty class")
        return row
    sim = _PW["frame"][_PW["c0"] + _PW["cues"]].copy()
    sim["sev3"] = np.array(sv.SEV_ORDER)[y]
    try:
        mA = rp.RPOrderedLogit(sim, _PW["c0"], [], draws=_PW["draws0"])
        fA = rp.fit(mA, np.asarray(m0c.raw, float).copy(), with_cov=False)
        cols1 = _PW["c0"] + _PW["cues"]
        mB = rp.RPOrderedLogit(sim, cols1, [], draws=_PW["draws0"])
        st = np.concatenate([fA["theta"][:len(_PW["c0"])], np.zeros(len(_PW["cues"])),
                             fA["theta"][len(_PW["c0"]):]])
        fB = rp.fit(mB, st, with_cov=False)
        lr = max(2 * (fB["loglik"] - fA["loglik"]), 0.0)
        p = float(stats.chi2.sf(lr, len(_PW["cues"])))
        conv = bool(fA["converged"] and fB["converged"])
        row.update(ok=conv, lr=lr, p=p, reject=bool(p < POWER_ALPHA),
                   converged_M0c=fA["converged"], converged_M1=fB["converged"],
                   mean_cue_coef=float(np.mean(fB["theta"][len(_PW["c0"]):len(cols1)])))
        if not conv:
            row["reason"] = "not converged"
    except Exception as exc:                                       # noqa: BLE001
        row.update(ok=False, reason=type(exc).__name__)
    return row


def item_power(pool, frame) -> dict:
    spec = sv.ladder(frame)
    cues = [c for c in spec["M1"] if c not in spec["M0c"]]
    betas = list(POWER_BETAS)
    rows = pool.map(_pw_one, [(i, b, r) for i, b in enumerate(betas)
                              for r in range(POWER_REPS)], chunksize=10)

    def tab(rows, betas):
        res = {}
        for i, b in enumerate(betas):
            rr = [x for x in rows if x["beta"] == b]
            ok = [x for x in rr if x["ok"]]
            pw = float(np.mean([x["reject"] for x in ok])) if ok else float("nan")
            res[str(b)] = {"odds_ratio": _r(np.exp(b), 3), "n_ok": len(ok),
                           "n_failed": len(rr) - len(ok),
                           "failure_reasons": sorted({x.get("reason", "") for x in rr if not x["ok"]}),
                           "power": _r(pw, 3),
                           "mc_se": _r(np.sqrt(pw * (1 - pw) / len(ok)), 3) if ok else None,
                           "median_lr": _r(np.median([x["lr"] for x in ok]), 3) if ok else None,
                           "mean_estimated_cue_coefficient": _r(np.mean([x["mean_cue_coef"] for x in ok]), 3) if ok else None,
                           "mean_class_counts": dict(zip(sv.SEV_ORDER, [
                               _r(v, 1) for v in np.mean([x["counts"] for x in rr], axis=0)]))}
        return res

    grid = tab(rows, betas)
    if (grid[str(betas[-1])]["power"] or 0) < 0.8:
        ext = list(POWER_BETAS_EXTENSION)
        off = len(betas)
        rows += pool.map(_pw_one, [(off + i, b, r) for i, b in enumerate(ext)
                                   for r in range(POWER_REPS)], chunksize=10)
        betas += ext
        grid = tab(rows, betas)
    pw = [grid[str(b)]["power"] for b in betas]
    b80 = None
    for k in range(1, len(betas)):
        if pw[k - 1] < 0.8 <= pw[k]:
            b80 = betas[k - 1] + (0.8 - pw[k - 1]) * (betas[k] - betas[k - 1]) / (pw[k] - pw[k - 1])
            break
    nums = json.loads((DATA / "numbers.json").read_text(encoding="utf-8"))
    m1 = sv.fit_ordered_logit(frame, spec["M1"], "M1")
    obs = {c: _r(m1.beta[spec["M1"].index(c)], 3) for c in cues}
    counts = "/".join(str(int(v)) for v in
                      frame["sev3"].value_counts().reindex(sv.SEV_ORDER).to_numpy())
    return {
        "design": (f"Truth: the fitted M0c ordered logit ({len(spec['M0c'])} covariates) plus "
                   f"a common planted log-odds effect beta on each of the {len(cues)} admitted "
                   f"cues ({', '.join(cues)}), applied to their observed values centered at "
                   f"the sample mean so the class balance stays near the observed {counts} "
                   f"(centering only moves the thresholds). Outcomes simulated from the "
                   f"latent logistic model for the observed n = {len(frame)} rows and "
                   f"covariates. Each replication fits M0c and M1 by maximum likelihood "
                   f"(random_parameters.RPOrderedLogit with no random coefficient, BFGS, "
                   f"analytic gradient) and applies the 5-df likelihood-ratio test at "
                   f"{POWER_ALPHA}. {POWER_REPS} replications per beta, seed "
                   f"default_rng([{POWER_SEED}, beta index, replication]). A replication "
                   f"counts only if both fits converge (max |gradient| < 1e-3)."),
        "cues": cues, "n_replications_per_beta": POWER_REPS, "seed": POWER_SEED,
        "alpha": POWER_ALPHA, "betas": betas, "grid": grid,
        "beta_at_power_0.8_interpolated": _r(b80, 3),
        "odds_ratio_at_power_0.8": _r(np.exp(b80), 3) if b80 is not None else None,
        "interpolation": "linear between the bracketing grid points",
        "size_at_beta_0": grid["0.0"]["power"],
        "observed": {"cue_block_lr": nums["severity"]["nested_lr"]["M0c_to_M1"],
                     "m1_cue_coefficients": obs,
                     "m1_mean_cue_coefficient": _r(np.mean(list(obs.values())), 3)},
    }


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main() -> None:
    from multiprocessing import Pool

    ap = argparse.ArgumentParser()
    ap.add_argument("--items", nargs="*", default=["1", "2", "3", "4", "5"])
    ap.add_argument("--out", default=None,
                    help="write to this file instead of data/ordinal_alternatives.json")
    args = ap.parse_args()
    out_path = Path(args.out) if args.out else OUT

    t_start = time.time()
    frame, cols, X, y = load()
    null_ll = sv.null_loglikelihood(frame)
    m1 = sv.fit_ordered_logit(frame, cols, "M1")
    result = (json.loads(out_path.read_text(encoding="utf-8")) if out_path.exists() else {})
    result.update({
        "generated": datetime.now().isoformat(timespec="seconds"),
        "script": "src/ordinal_alternatives.py",
        "inputs": ["data/model_frame.csv", "data/screening.json",
                   "data/severity.json", "data/numbers.json"],
        "specification": (f"M1 (severity.ladder), {len(cols)} covariates, n = {len(y)}, "
                          + " < ".join(sv.SEV_ORDER)),
        "primary_reference": {"model": "M1 ordered logit (severity.py)",
                              "loglik": _r(m1.loglik, 2),
                              "ka_excludes_zero": sorted(
                                  c for c, v in json.loads((DATA / "numbers.json").read_text(
                                      encoding="utf-8"))["severity"]["ames_excluding_zero"].items()
                                  if v["KA"]["ci95"][0] > 0 or v["KA"]["ci95"][1] < 0),
                              "source": "numbers.json severity.ames_excluding_zero"},
        "settings": {"kr_draws": sv.KR_DRAWS, "kr_seed": sv.KR_SEED,
                     "cv": {"repeats": sv.CV_REPEATS, "folds": sv.CV_FOLDS, "seed": sv.CV_SEED},
                     "random_thresholds": {"draws": RT_DRAWS, "draws_check": list(RT_DRAWS_CHECK),
                                           "halton_seed": RT_SEED},
                     "bayes": {"chains": BAYES_CHAINS, "tune": BAYES_TUNE, "draws": BAYES_DRAWS,
                               "seed": BAYES_SEED, "cut_prior_sd": BAYES_CUT_SD},
                     "power": {"betas": list(POWER_BETAS), "extension": list(POWER_BETAS_EXTENSION),
                               "reps": POWER_REPS, "seed": POWER_SEED, "alpha": POWER_ALPHA}},
    })

    def save():
        result["runtime_seconds_last_run"] = round(time.time() - t_start, 1)
        out_path.write_text(json.dumps(result, indent=2, default=float), encoding="utf-8")

    if "4" in args.items:
        result["separation"] = item_separation(frame)
        save()
    if "1" in args.items:
        pool = Pool(N_WORKERS, initializer=_cv_init)
        result["ordinal_family"] = item_ordinal_family(frame, cols, X, y, pool, null_ll, m1)
        pool.close()
        pool.join()
        save()
    if "2" in args.items:
        result["random_thresholds"] = item_random_thresholds(frame, cols, X, y, null_ll, m1)
        save()
    if "3" in args.items:
        result["shrinkage"] = item_shrinkage(frame, cols, X, y, null_ll, m1)
        save()
    if "5" in args.items:
        pool = Pool(N_WORKERS, initializer=_pw_init)
        result["cue_block_power"] = item_power(pool, frame)
        pool.close()
        pool.join()
        save()
    print(f"written to {out_path} ({time.time() - t_start:.0f}s)")


if __name__ == "__main__":
    main()
