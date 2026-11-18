"""The ordinal severity models.

The specification ladder of `prespec.md` section 6, estimated in the ordinal family of
section 7:

* ordered logit (proportional odds) via `statsmodels`;
* the Brant test per covariate and omnibus, computed from the binary cumulative logits;
* a partial proportional-odds model by explicit maximum likelihood, freeing only the
  covariates that fail the Brant test;
* average marginal effects for all three outcomes with Krinsky-Robb intervals.

Three properties this module is built around, because each is a way to be quietly
wrong rather than loudly wrong:

1. **The partial proportional-odds likelihood is verified against the ordered-logit
   special case** before it is used for anything. A wrong likelihood and a wrong Brant
   test look identical from the outside, and the Brant test chooses the primary model.
2. **Average marginal effects respect categorical groups.** Setting `speed_ge45` to 1
   sets its sibling levels to 0, so the counterfactual row is a row that could exist.
   Ignoring this is the standard bug in dummy-coded designs and it produces plausible
   numbers that are wrong.
3. **Krinsky-Robb draws include the cutpoints**, drawn from the joint covariance of the
   full parameter vector. Holding cutpoints fixed understates every interval.

Events per parameter and separation diagnostics are reported for every rung, because
with 77 O and 82 KA outcomes the parameter budget is the binding constraint.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import optimize, stats

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

SEV_ORDER = ("O", "BC", "KA")
KR_DRAWS = 1000
KR_SEED = 20260917


# --------------------------------------------------------------------------
# Specification ladder
# --------------------------------------------------------------------------

STRUCTURED_PREFIXES = ("speed", "light", "eth", "gender", "age", "veh", "coll",
                       "intx", "ctrl", "road")
STRUCTURED_EXTRA = ("weekend", "nighttime", "coord_missing", "poverty_share",
                    "bike_facility_nearest", "road_density_km_100m", "urban_area_200k")
CONTRIB_PREFIX = "cf_"
CUES = ("wrong_way", "sidewalk_transition", "driveway_alley", "failure_to_yield",
        "signal_violation", "swerve_loss_control", "dooring",
        "distraction_impairment", "vehicle_turning_across", "lane_positioning")


def included_covariates() -> list[str]:
    """Covariates the frozen screening rules admitted, in a stable order."""
    screening = json.loads((DATA / "screening.json").read_text(encoding="utf-8"))
    return [r["candidate"] for r in screening["candidates"]
            if r["disposition"].startswith("included")]


def ladder(frame: pd.DataFrame) -> dict[str, list[str]]:
    """M0, M0c and M1 as lists of column names, from the screened set."""
    inc = [c for c in included_covariates() if c in frame.columns]
    structured = [c for c in inc
                  if c.split("_")[0] in STRUCTURED_PREFIXES or c in STRUCTURED_EXTRA]
    contrib = [c for c in inc if c.startswith(CONTRIB_PREFIX)]
    cues = [c for c in inc if c in CUES]
    return {"M0": structured, "M0c": structured + contrib,
            "M1": structured + contrib + cues}


# Binary covariates that are deliberately not levels of a dummy-coded categorical, so
# have no siblings to zero. Anything binary that is neither here, nor a cue, nor an
# officer-coded contributing factor, nor a level of a group in `references` is a
# variable nobody has classified, and `categorical_groups` refuses to guess.
STANDALONE_BINARY = frozenset({
    "weekend", "nighttime", "coord_missing",
    # The bike-facility buffers are nested, not levels of one categorical, and only one
    # enters any model at a time, so none of them has a sibling to zero. The pre-specified
    # primary form, presence on the nearest segment, is the fourth of these.
    "bike_facility_nearest", "bike_facility_50m", "bike_facility_100m",
    "bike_facility_200m",
    # Inside a 2020 Census urban area of 200,000 or more (prespec section 2.1): a single
    # indicator against everything outside it, not a level of a dummy-coded categorical.
    "urban_area_200k",
})


def categorical_groups(frame: pd.DataFrame,
                       columns: list[str]) -> dict[str, list[str]]:
    """Map each dummy-coded categorical to its member columns.

    Used so that an average marginal effect for one level sets its sibling levels to
    zero. Getting this wrong is silent: the effect is then computed from rows where two
    mutually exclusive levels of one categorical are both 1, which is a row that cannot
    exist, and the resulting number looks entirely plausible.

    Membership is not inferred from the name prefix alone. `road_density_km_100m` and
    `road_not_city_street` share the prefix `road`, and zeroing a continuous density
    whenever the road-class indicator is set would be exactly the impossible
    counterfactual this function exists to prevent. A column joins a group only if its
    prefix names a categorical in the frozen screening output *and* the column is
    binary, since a continuous variable is never a dummy level.

    Any binary column this leaves unclassified raises, rather than being silently
    treated as standalone.
    """
    screening = json.loads((DATA / "screening.json").read_text(encoding="utf-8"))
    prefixes = set(screening["references"])

    binary = {c for c in columns
              if set(np.unique(frame[c].dropna().to_numpy())) <= {0.0, 1.0}}

    groups: dict[str, list[str]] = {}
    classified: set[str] = set()
    for col in columns:
        if col not in binary:
            continue
        prefix = col.split("_")[0]
        if prefix in prefixes:
            groups.setdefault(prefix, []).append(col)
            classified.add(col)
        elif col in STANDALONE_BINARY or col in CUES or col.startswith(CONTRIB_PREFIX):
            classified.add(col)

    unclassified = sorted(binary - classified)
    if unclassified:
        raise ValueError(
            f"binary covariates that belong to no declared group and are not declared "
            f"standalone: {unclassified}. Add each to STANDALONE_BINARY if it has no "
            f"sibling levels, or make sure its prefix appears in the screening "
            f"references map. Guessing here silently corrupts the marginal effects."
        )
    return groups


# --------------------------------------------------------------------------
# Ordered logit
# --------------------------------------------------------------------------

@dataclass
class OrdinalFit:
    name: str
    columns: list[str]
    beta: np.ndarray            # slopes
    cut: np.ndarray             # increasing cutpoints, length J-1
    raw: np.ndarray             # [beta, free slopes, cut_1, log increments]
    cov: np.ndarray             # covariance of `raw`, NOT of [beta, cut]
    loglik: float
    n: int
    converged: bool
    free_columns: tuple[str, ...] = ()   # covariates with category-specific slopes
    beta_free: np.ndarray | None = None  # shape (J-1, len(free_columns))
    extras: dict = field(default_factory=dict)

    @property
    def n_params(self) -> int:
        """Free parameters: proportional slopes, threshold-specific slopes, cutpoints.

        `self.beta` holds only the proportional slopes — a freed covariate is in
        `beta_free`, never in `beta` — so the freed coefficients are added whole rather
        than net of a base count.
        """
        k = len(self.beta) + len(self.cut)
        if self.beta_free is not None:
            k += int(self.beta_free.size)
        return k


def _cut_from_raw(raw: np.ndarray) -> np.ndarray:
    """Thresholds from the unconstrained parameterization both fits use.

    The first element is the first cutpoint; each later element is the log of the gap
    to the next one, which keeps the cutpoints ordered without a constrained optimizer.
    Krinsky-Robb draws are taken in this space and transformed here, so every draw
    yields ordered thresholds and carries the correct joint uncertainty. Drawing on the
    cutpoint scale with a covariance estimated on this scale would be a category error.
    """
    return np.concatenate([[raw[0]], raw[0] + np.cumsum(np.exp(raw[1:]))])


def _design(frame: pd.DataFrame, columns: list[str],
            allow_drop: bool = False, outcome: str = "sev3",
            order: tuple[str, ...] = SEV_ORDER) -> tuple[np.ndarray, np.ndarray]:
    """Design matrix and coded outcome, with missing rows refused by default.

    Dropping incomplete rows quietly would leave the fit on one sample while
    `events_per_parameter` and `null_loglikelihood` describe another, because both count
    from the full frame. The events-per-parameter table reports the
    parameter budget, so it must describe the rows the model was actually fitted on. The spatial
    layer makes this live: the 78 rider rows with no located crash have no OSM feature,
    so as soon as those columns are attached this would start dropping them.
    """
    X = frame[columns].astype(float).to_numpy()
    y = pd.Categorical(frame[outcome], categories=list(order), ordered=True).codes
    keep = ~np.isnan(X).any(axis=1) & (y >= 0)

    if not keep.all() and not allow_drop:
        bad = [c for c in columns if frame[c].isna().any()]
        raise ValueError(
            f"{int((~keep).sum())} of {len(keep)} rows are incomplete and would be "
            f"dropped from the fit while the reported sample size and the "
            f"events-per-parameter table still describe all {len(keep)}. Columns with "
            f"missing values: {bad or 'none (the outcome is missing)'}. Either give "
            f"these covariates a missingness indicator as prespec section 3.2 does, or "
            f"pass allow_drop=True and report the reduced sample explicitly."
        )
    return X[keep], y[keep]


def fit_ordered_logit(frame: pd.DataFrame, columns: list[str], name: str,
                      outcome: str = "sev3",
                      order: tuple[str, ...] = SEV_ORDER) -> OrdinalFit:
    from statsmodels.miscmodels.ordinal_model import OrderedModel

    X, y = _design(frame, columns, outcome=outcome, order=order)
    model = OrderedModel(y, X, distr="logit")
    res = model.fit(method="bfgs", maxiter=2000, disp=False)
    k = X.shape[1]
    params = np.asarray(res.params)
    beta = params[:k]
    # statsmodels parameterizes thresholds as the first cutpoint then log-increments.
    cut = _cut_from_raw(params[k:])
    return OrdinalFit(
        name=name, columns=list(columns), beta=beta, cut=cut, raw=params,
        cov=np.asarray(res.cov_params()), loglik=float(res.llf), n=int(len(y)),
        converged=bool(res.mle_retvals.get("converged", True)),
        extras={"threshold_parameterization": "first cutpoint then log-increments",
                "outcome": outcome, "order": list(order)},
    )


# --------------------------------------------------------------------------
# Partial proportional odds, by explicit maximum likelihood
# --------------------------------------------------------------------------

def _ppo_negll(theta: np.ndarray, X: np.ndarray, Xf: np.ndarray, y: np.ndarray,
               J: int) -> float:
    """Negative log-likelihood of the partial proportional-odds model.

    Cumulative form: P(Y <= j) = logistic(cut_j - X beta - Xf gamma_j). The
    proportional part `beta` is shared across the J-1 cumulative logits; the free part
    `gamma_j` varies by threshold.
    """
    kp, kf = X.shape[1], Xf.shape[1]
    beta = theta[:kp]
    gamma = theta[kp:kp + kf * (J - 1)].reshape(J - 1, kf)
    cut = _cut_from_raw(theta[kp + kf * (J - 1):])

    eta_common = X @ beta if kp else np.zeros(len(y))
    cum = np.empty((len(y), J - 1))
    for j in range(J - 1):
        lin = cut[j] - eta_common - (Xf @ gamma[j] if kf else 0.0)
        cum[:, j] = 1.0 / (1.0 + np.exp(-lin))

    probs = np.empty((len(y), J))
    probs[:, 0] = cum[:, 0]
    for j in range(1, J - 1):
        probs[:, j] = cum[:, j] - cum[:, j - 1]
    probs[:, J - 1] = 1.0 - cum[:, J - 2]
    probs = np.clip(probs, 1e-12, 1.0)
    return -float(np.log(probs[np.arange(len(y)), y]).sum())


def _numeric_hessian(f, x: np.ndarray, eps: float = 1e-4) -> np.ndarray:
    """Central-difference Hessian of `f` at `x`, symmetrized."""
    n = len(x)
    H = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            xpp, xpm, xmp, xmm = x.copy(), x.copy(), x.copy(), x.copy()
            xpp[i] += eps; xpp[j] += eps
            xpm[i] += eps; xpm[j] -= eps
            xmp[i] -= eps; xmp[j] += eps
            xmm[i] -= eps; xmm[j] -= eps
            H[i, j] = H[j, i] = (f(xpp) - f(xpm) - f(xmp) + f(xmm)) / (4 * eps ** 2)
    return (H + H.T) / 2


def fit_ppo(frame: pd.DataFrame, columns: list[str], free: list[str],
            name: str, compute_cov: bool = True) -> OrdinalFit:
    """Fit the partial proportional-odds model, freeing `free` covariates."""
    prop = [c for c in columns if c not in free]
    X, y = _design(frame, prop) if prop else (np.zeros((len(frame), 0)), None)
    Xf, y2 = _design(frame, free) if free else (np.zeros((len(frame), 0)), None)
    if y is None:
        y = y2
    if len(free) and len(prop):
        both = _design(frame, prop + free)
        X, y = both[0][:, :len(prop)], both[1]
        Xf = both[0][:, len(prop):]
    J = len(SEV_ORDER)

    kp, kf = X.shape[1], Xf.shape[1]
    obj = lambda t: _ppo_negll(t, X, Xf, y, J)  # noqa: E731

    try:
        warm = fit_ordered_logit(frame, prop + free, f"{name}_warmstart")
        wb = dict(zip(prop + free, warm.beta))
        start = np.concatenate([
            np.array([wb[c] for c in prop]) if kp else np.zeros(0),
            np.tile([wb[c] for c in free], J - 1) if kf else np.zeros(0),
            warm.raw[len(warm.beta):],
        ])
    except Exception:
        start = np.concatenate([np.zeros(kp + kf * (J - 1)), np.array([-1.0, 0.0])])

    res = optimize.minimize(obj, start, method="BFGS",
                            options={"maxiter": 5000, "gtol": 1e-6})

    theta = res.x
    beta = theta[:kp]
    gamma = theta[kp:kp + kf * (J - 1)].reshape(J - 1, kf) if kf else None
    cut = _cut_from_raw(theta[kp + kf * (J - 1):])
    # BFGS's `hess_inv` is a secant approximation accumulated along the search path, not
    # an estimate of the information matrix; intervals built from it are unreliable. The
    # observed information is cheap here, so take it directly.
    # The Hessian costs O(p^2) likelihood evaluations. The proportional-odds tests need
    # only the log-likelihood, so they skip it.
    cov = (np.linalg.pinv(_numeric_hessian(obj, theta)) if compute_cov
           else np.full((len(theta), len(theta)), np.nan))

    eps = 1e-5
    grad = np.array([(obj(theta + eps * e) - obj(theta - eps * e)) / (2 * eps)
                     for e in np.eye(len(theta))])
    grad_norm = float(np.max(np.abs(grad)))
    converged = bool(res.success or grad_norm < 1e-3)

    return OrdinalFit(
        name=name, columns=list(prop), beta=beta, cut=cut, raw=theta, cov=np.asarray(cov),
        loglik=float(-res.fun), n=int(len(y)), converged=converged,
        free_columns=tuple(free), beta_free=gamma,
        extras={"optimizer_message": str(res.message), "n_iter": int(res.nit),
                "optimizer_success_flag": bool(res.success),
                "max_abs_gradient_at_optimum": round(grad_norm, 8),
                "warm_started_from_ordered_logit": bool(len(start) and res.nit >= 0)},
    )


def verify_ppo_against_ordered_logit(frame: pd.DataFrame,
                                     columns: list[str]) -> dict:
    """The no-free-covariate PPO must reproduce ordered logit. Check before trusting it.

    Also checks the analytic-free gradient numerically, so a likelihood error cannot
    hide behind a converged optimizer.
    """
    ol = fit_ordered_logit(frame, columns, "verify_ol")
    ppo = fit_ppo(frame, columns, free=[], name="verify_ppo")

    ll_gap = abs(ol.loglik - ppo.loglik)
    beta_gap = float(np.max(np.abs(ol.beta - ppo.beta))) if len(columns) else 0.0
    cut_gap = float(np.max(np.abs(ol.cut - ppo.cut)))

    X, y = _design(frame, columns)
    Xf = np.zeros((len(y), 0))
    theta = ppo.raw
    eps = 1e-5
    num = np.zeros(len(theta))
    base = _ppo_negll(theta, X, Xf, y, len(SEV_ORDER))
    for i in range(len(theta)):
        up = theta.copy(); up[i] += eps
        dn = theta.copy(); dn[i] -= eps
        num[i] = (_ppo_negll(up, X, Xf, y, len(SEV_ORDER))
                  - _ppo_negll(dn, X, Xf, y, len(SEV_ORDER))) / (2 * eps)
    grad_max = float(np.max(np.abs(num)))

    return {
        "loglik_ordered_logit": ol.loglik,
        "loglik_ppo_no_free": ppo.loglik,
        "loglik_abs_gap": ll_gap,
        "max_abs_beta_gap": beta_gap,
        "max_abs_cutpoint_gap": cut_gap,
        "max_abs_numeric_gradient_at_optimum": grad_max,
        "passes": bool(ll_gap < 1e-3 and beta_gap < 1e-3 and cut_gap < 1e-3),
        "base_negll": base,
    }


# --------------------------------------------------------------------------
# Brant test
# --------------------------------------------------------------------------

def proportional_odds_tests(frame: pd.DataFrame, columns: list[str]) -> dict:
    """Test the proportional-odds assumption, per covariate and as an omnibus.

    Two statistics are computed for each covariate.

    The **likelihood-ratio** statistic compares the ordered logit against a partial
    proportional-odds model that frees that one covariate, using the maximum-likelihood
    code verified in `verify_ppo_against_ordered_logit`. The omnibus frees every
    covariate at once. This is the decision rule.

    The **Wald** statistic is the classical Brant form: fit the two binary cumulative
    logits separately and test the equality of their slopes. It is reported alongside
    for comparability with the published literature, but it is not what decides
    anything, because it treats the two binary fits as independent. They are fitted on
    the same rows and are positively correlated, so the correct denominator is
    V11 + V22 - 2*V12 and this one omits the cross term. The denominator is therefore
    too large, the statistic too small, and the test conservative — it under-rejects.
    That bias runs toward keeping ordered logit as primary, which is the conclusion at
    stake, so it must not be the arbiter of its own favoured outcome.

    With three ordered classes there are two cumulative logits, so each covariate's
    statistic carries one degree of freedom and the omnibus carries as many as there are
    covariates.
    """
    import statsmodels.api as sm

    X, y = _design(frame, columns)
    Xc = sm.add_constant(X, has_constant="add")
    J = len(SEV_ORDER)

    fits = [sm.Logit((y > j).astype(int), Xc).fit(disp=False, maxiter=200)
            for j in range(J - 1)]

    ol = fit_ordered_logit(frame, columns, "po_baseline")

    per: dict[str, dict] = {}
    for i, col in enumerate(columns):
        b = np.array([f.params[i + 1] for f in fits])
        v = np.array([f.cov_params()[i + 1, i + 1] for f in fits])
        var = v[0] + v[1]
        wald = float((b[0] - b[1]) ** 2 / var) if var > 0 else float("nan")

        freed = fit_ppo(frame, columns, free=[col], name=f"po_{col}",
                        compute_cov=False)
        lr = float(2 * (freed.loglik - ol.loglik))
        lr = max(lr, 0.0)
        p_lr = float(1 - stats.chi2.cdf(lr, df=1))

        per[col] = {
            "lr_chi2": round(lr, 4),
            "lr_p": round(p_lr, 4),
            "df": 1,
            "fails_at_5pct": bool(p_lr < 0.05),
            "converged": freed.converged,
            "wald_chi2_brant_form": round(wald, 4) if wald == wald else None,
            "wald_p": (round(float(1 - stats.chi2.cdf(wald, df=1)), 4)
                       if wald == wald else None),
            "beta_cumlogit_1": round(float(b[0]), 4),
            "beta_cumlogit_2": round(float(b[1]), 4),
        }

    full = fit_ppo(frame, columns, free=list(columns), name="po_omnibus",
                   compute_cov=False)
    omni = max(float(2 * (full.loglik - ol.loglik)), 0.0)
    df = len(columns)
    p_omni = float(1 - stats.chi2.cdf(omni, df=df))
    wald_sum = float(np.nansum([d["wald_chi2_brant_form"] or np.nan
                                for d in per.values()]))
    failing = [c for c, d in per.items() if d["fails_at_5pct"]]

    return {
        "decision_rule": "likelihood ratio",
        "per_covariate": per,
        "omnibus_lr_chi2": round(omni, 4),
        "omnibus_df": df,
        "omnibus_lr_p": round(p_omni, 6),
        "omnibus_rejects_at_5pct": bool(p_omni < 0.05),
        "omnibus_converged": full.converged,
        "omnibus_wald_chi2_brant_form": round(wald_sum, 4),
        "omnibus_wald_p": round(float(1 - stats.chi2.cdf(wald_sum, df=df)), 6),
        "covariates_failing_at_5pct": failing,
        "note": (
            "prespec section 7 frees only the covariates that fail individually, so an "
            "omnibus rejection with no individual failure would yield a partial "
            "proportional-odds model that frees nothing and is arithmetically identical "
            "to ordered logit. In that case ordered logit is reported as primary and the "
            "omnibus rejection is stated in the text as driven by degrees of freedom "
            "rather than by any identifiable covariate. This rule was fixed in code "
            "before the test was run; see docs/prespec_deviations.md."
        ),
    }


# --------------------------------------------------------------------------
# Probabilities and average marginal effects
# --------------------------------------------------------------------------

def _probs(X: np.ndarray, beta: np.ndarray, cut: np.ndarray) -> np.ndarray:
    J = len(cut) + 1
    eta = X @ beta if X.shape[1] else np.zeros(len(X))
    cum = 1.0 / (1.0 + np.exp(-(cut[None, :] - eta[:, None])))
    p = np.empty((len(X), J))
    p[:, 0] = cum[:, 0]
    for j in range(1, J - 1):
        p[:, j] = cum[:, j] - cum[:, j - 1]
    p[:, J - 1] = 1.0 - cum[:, J - 2]
    return p


def average_marginal_effects(fit: OrdinalFit, frame: pd.DataFrame,
                             draws: int = KR_DRAWS, seed: int = KR_SEED,
                             groups: dict[str, list[str]] | None = None,
                             outcome: str = "sev3",
                             order: tuple[str, ...] = SEV_ORDER) -> dict:
    """AMEs for every covariate and every outcome, with Krinsky-Robb intervals.

    Indicators get a discrete change from 0 to 1 with sibling levels of the same
    categorical group set to 0. Continuous covariates get a numerical derivative.
    Effects are averaged over observations first, then percentiles are taken across
    parameter draws.

    `groups` overrides the categorical grouping, which is otherwise read from the frozen
    screening output. Pass it only for simulated frames that have no screening file; on
    real data, leaving it unset is what keeps the grouping tied to the frozen
    specification rather than to whatever this call happens to believe.
    """
    cols = fit.columns
    X, _ = _design(frame, cols, outcome=outcome, order=order)
    groups = categorical_groups(frame, cols) if groups is None else groups
    sibling: dict[str, list[int]] = {}
    for _, members in groups.items():
        for c in members:
            sibling[c] = [cols.index(m) for m in members if m != c and m in cols]

    is_binary = {c: set(np.unique(X[:, i])) <= {0.0, 1.0} for i, c in enumerate(cols)}

    rng = np.random.default_rng(seed)
    k = len(fit.beta)
    n_raw = k + len(fit.cut)
    # Draw the slopes and the threshold parameters jointly, on the scale the covariance
    # was actually estimated on, then map the thresholds back to cutpoints per draw.
    # Holding the cutpoints at their point estimates would understate every interval.
    mean_vec = np.asarray(fit.raw)[:n_raw]
    cov = np.asarray(fit.cov)[:n_raw, :n_raw]
    cov = (cov + cov.T) / 2
    sample = rng.multivariate_normal(mean_vec, cov, size=draws, method="svd")

    out: dict[str, dict] = {}
    point = _ame_once(X, fit.beta, fit.cut, cols, sibling, is_binary)
    drawn = np.empty((draws, len(cols), len(order)))
    for d in range(draws):
        drawn[d] = _ame_once(X, sample[d, :k], _cut_from_raw(sample[d, k:]),
                             cols, sibling, is_binary)

    for i, col in enumerate(cols):
        entry = {}
        for j, lvl in enumerate(order):
            lo, hi = np.percentile(drawn[:, i, j], [2.5, 97.5])
            entry[lvl] = {
                "ame": round(float(point[i, j]), 5),
                "ci95": [round(float(lo), 5), round(float(hi), 5)],
                "excludes_zero": bool(lo > 0 or hi < 0),
            }
        entry["type"] = "discrete change" if is_binary[col] else "derivative"
        out[col] = entry
    return out


def _ame_once(X, beta, cut, cols, sibling, is_binary) -> np.ndarray:
    eff = np.zeros((len(cols), len(cut) + 1))
    for i, col in enumerate(cols):
        if is_binary[col]:
            X1 = X.copy()
            X1[:, i] = 1.0
            for s in sibling.get(col, []):
                X1[:, s] = 0.0
            X0 = X.copy()
            X0[:, i] = 0.0
            eff[i] = (_probs(X1, beta, cut) - _probs(X0, beta, cut)).mean(axis=0)
        else:
            h = max(1e-4, 0.01 * float(np.std(X[:, i])))
            Xp, Xm = X.copy(), X.copy()
            Xp[:, i] += h
            Xm[:, i] -= h
            eff[i] = ((_probs(Xp, beta, cut) - _probs(Xm, beta, cut)) / (2 * h)).mean(axis=0)
    return eff


# --------------------------------------------------------------------------
# Repeated stratified cross-validation (prespec section 6)
# --------------------------------------------------------------------------

CV_REPEATS = 10
CV_FOLDS = 5
CV_SEED = 20260918


def cross_validated_log_loss(frame: pd.DataFrame, columns: list[str],
                             repeats: int = CV_REPEATS, folds: int = CV_FOLDS,
                             seed: int = CV_SEED, free: list[str] | None = None) -> dict:
    """Out-of-fold multiclass log loss, stratified on the outcome.

    Fold-level values are returned as well as the summary, because prespec section 6
    asks for them: with 77 observations in the smallest class a single fold holds about
    15 of them, so the spread across folds is large and a bare mean would suggest more
    precision than the design supports.

    A fold whose training split will not support the model is recorded as a failure
    rather than skipped. Dropping such folds silently would report the mean over exactly
    the folds where the model happened to behave.

    `free` names covariates given threshold-specific coefficients, so that the
    cross-validated column describes the same model family as the primary specification.
    Leaving it empty fits ordered logit. Comparing a cross-validated ordered logit
    against a partial proportional-odds primary would put the wrong family in the table.
    """
    from sklearn.model_selection import RepeatedStratifiedKFold

    X_all, y_all = _design(frame, columns)
    splitter = RepeatedStratifiedKFold(n_splits=folds, n_repeats=repeats,
                                       random_state=seed)
    fold_rows: list[dict] = []
    for i, (train_idx, test_idx) in enumerate(splitter.split(X_all, y_all)):
        rep, fold = divmod(i, folds)
        try:
            if free:
                fit = fit_ppo(frame.iloc[train_idx], columns, free=list(free),
                              name="cv", compute_cov=False)
                prop = [c for c in columns if c not in free]
                ip = [columns.index(c) for c in prop]
                jf = [columns.index(c) for c in free]
                Xt = X_all[test_idx]
                cum = np.empty((len(test_idx), len(fit.cut)))
                for j in range(len(fit.cut)):
                    lin = (fit.cut[j] - Xt[:, ip] @ fit.beta
                           - Xt[:, jf] @ fit.beta_free[j])
                    cum[:, j] = 1.0 / (1.0 + np.exp(-lin))
                p = np.empty((len(test_idx), len(fit.cut) + 1))
                p[:, 0] = cum[:, 0]
                for j in range(1, len(fit.cut)):
                    p[:, j] = cum[:, j] - cum[:, j - 1]
                p[:, -1] = 1.0 - cum[:, -1]
            else:
                from statsmodels.miscmodels.ordinal_model import OrderedModel
                res = OrderedModel(y_all[train_idx], X_all[train_idx],
                                   distr="logit").fit(method="bfgs", maxiter=2000,
                                                      disp=False)
                k = X_all.shape[1]
                params = np.asarray(res.params)
                p = _probs(X_all[test_idx], params[:k], _cut_from_raw(params[k:]))
            p = np.clip(p, 1e-12, 1.0)
            ll = -float(np.log(p[np.arange(len(test_idx)),
                                 y_all[test_idx]]).mean())
            fold_rows.append({"repeat": rep + 1, "fold": fold + 1,
                              "n_test": int(len(test_idx)),
                              "log_loss": round(ll, 5), "ok": True})
        except Exception as exc:                       # noqa: BLE001
            fold_rows.append({"repeat": rep + 1, "fold": fold + 1,
                              "n_test": int(len(test_idx)),
                              "log_loss": None, "ok": False,
                              "error": type(exc).__name__})

    good = [r["log_loss"] for r in fold_rows if r["ok"]]
    return {
        "repeats": repeats,
        "folds": folds,
        "seed": seed,
        "n_folds_total": len(fold_rows),
        "n_folds_failed": sum(1 for r in fold_rows if not r["ok"]),
        "mean_log_loss": round(float(np.mean(good)), 5) if good else None,
        "sd_across_folds": round(float(np.std(good, ddof=1)), 5) if len(good) > 1 else None,
        "min_log_loss": round(float(np.min(good)), 5) if good else None,
        "max_log_loss": round(float(np.max(good)), 5) if good else None,
        "fold_level": fold_rows,
    }


def multiplicity_sets(per_covariate: dict) -> dict[str, list[str]]:
    """Which covariates are freed under three multiplicity rules.

    The pre-specification names a 5 percent threshold and says nothing about testing
    thirty-three covariates at once. Under the null, one or two would be expected to
    fail by chance. Reporting all three sets means the choice is visible rather than
    made silently, and lets a reader see whether it changes the primary model.
    """
    ps = sorted((v["lr_p"], c) for c, v in per_covariate.items())
    m = len(ps)

    raw = [c for p_, c in ps if p_ < 0.05]

    holm = []
    for i, (p_, c) in enumerate(ps):
        if p_ * (m - i) < 0.05:
            holm.append(c)
        else:
            break

    bh: list[str] = []
    for i, (p_, c) in enumerate(ps):
        if p_ <= 0.05 * (i + 1) / m:
            bh = [c2 for _, c2 in ps[:i + 1]]

    return {"uncorrected": raw, "benjamini_hochberg": bh, "holm": holm}


def primary_model_sensitivity(frame: pd.DataFrame, columns: list[str],
                              ol: OrdinalFit, sets: dict[str, list[str]],
                              null_ll: float) -> dict:
    """Does the multiplicity rule change which model is primary?

    prespec section 7 confirms a partial proportional-odds primary by BIC. If BIC
    prefers ordered logit under every multiplicity rule, the choice of rule does not
    affect the specification and can be reported rather than decided.
    """
    out: dict[str, dict] = {}
    for rule, free in sets.items():
        if not free:
            out[rule] = {"n_freed": 0, "primary": "ordered logit",
                         "reason": "nothing freed"}
            continue
        ppo = fit_ppo(frame, columns, free=list(free), name=f"ppo_{rule}")
        bic = confirm_by_bic(ppo, ol)
        out[rule] = {
            "n_freed": len(free),
            "freed": list(free),
            "lr_vs_ordered_logit": nested_lr_test(ol, ppo),
            "bic_confirmation": bic,
            "primary": ("partial proportional odds" if bic["confirmed"]
                        else "ordered logit"),
        }
    primaries = {v["primary"] for v in out.values()}
    out["agrees_across_rules"] = len(primaries) == 1
    out["primary_if_agreed"] = primaries.pop() if len(primaries) == 1 else None
    return out


def confirm_by_bic(ppo: OrdinalFit, ol: OrdinalFit) -> dict:
    """prespec section 7: a partial proportional-odds primary is confirmed by BIC.

    Freeing a covariate always raises the log-likelihood, so the likelihood-ratio test
    alone cannot say the extra parameters earn their place. BIC can, and it is the
    criterion the pre-specification named.
    """
    bic_ppo = -2 * ppo.loglik + ppo.n_params * np.log(ppo.n)
    bic_ol = -2 * ol.loglik + ol.n_params * np.log(ol.n)
    return {
        "bic_partial_proportional_odds": round(float(bic_ppo), 3),
        "bic_ordered_logit": round(float(bic_ol), 3),
        "difference": round(float(bic_ppo - bic_ol), 3),
        "confirmed": bool(bic_ppo < bic_ol),
        "note": ("A negative difference confirms the partial proportional-odds model as "
                 "primary. A positive difference means the freed coefficients do not pay "
                 "for themselves, and ordered logit is reported as primary with the "
                 "Brant result stated in the text."),
    }


# --------------------------------------------------------------------------
# Ladder diagnostics
# --------------------------------------------------------------------------

def events_per_parameter(frame: pd.DataFrame, columns: list[str],
                         outcome: str = "sev3",
                         order: tuple[str, ...] = SEV_ORDER) -> dict:
    counts = frame[outcome].value_counts().reindex(order).fillna(0).astype(int)
    n_params = len(columns) + len(order) - 1
    return {
        "n_covariates": len(columns),
        "n_parameters": n_params,
        "class_counts": counts.to_dict(),
        "smallest_class": int(counts.min()),
        "events_per_parameter_smallest_class": round(float(counts.min() / n_params), 3),
    }


def separation_diagnostics(fit: OrdinalFit) -> dict:
    se = np.sqrt(np.clip(np.diag(fit.cov)[:len(fit.beta)], 0, None))
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.abs(fit.beta) / se
    return {
        "max_abs_coefficient": round(float(np.max(np.abs(fit.beta))), 4),
        "max_standard_error": round(float(np.nanmax(se)), 4),
        "any_coefficient_above_10": bool(np.max(np.abs(fit.beta)) > 10),
        "any_standard_error_above_10": bool(np.nanmax(se) > 10),
        "max_abs_t": round(float(np.nanmax(ratio)), 3),
        "suggests_separation": bool(np.max(np.abs(fit.beta)) > 10
                                    or np.nanmax(se) > 10),
    }


def nested_lr_test(small: OrdinalFit, large: OrdinalFit) -> dict:
    stat = 2 * (large.loglik - small.loglik)
    df = large.n_params - small.n_params
    p = float(1 - stats.chi2.cdf(stat, df)) if df > 0 else float("nan")
    return {"lr_chi2": round(float(stat), 4), "df": int(df),
            "p": round(p, 6) if p == p else None}


def coefficient_table(fit: OrdinalFit) -> dict:
    """Slopes with standard errors, for the model estimates table.

    `fit.cov` is the covariance of `raw`, whose leading `len(beta)` entries are the
    proportional slopes in column order, so the leading diagonal block is what a slope
    standard error needs and no reparameterization is involved.

    A covariate given threshold-specific slopes lives in `beta_free`, never in `beta`, so
    it is reported once per threshold rather than silently omitted.
    """
    k = len(fit.beta)
    se = np.sqrt(np.diag(np.asarray(fit.cov))[:k])
    out: dict = {"columns": list(fit.columns), "slopes": {}}
    for i, col in enumerate(fit.columns[:k]):
        b, s = float(fit.beta[i]), float(se[i])
        z = b / s if s > 0 else float("nan")
        out["slopes"][col] = {
            "coef": round(b, 6),
            "se": round(s, 6),
            "z": round(z, 4),
            "p": round(float(2 * stats.norm.sf(abs(z))), 6) if s > 0 else None,
            "ci95": [round(b - 1.959964 * s, 6), round(b + 1.959964 * s, 6)],
            "excludes_zero": bool(s > 0 and abs(z) > 1.959964),
        }

    if fit.beta_free is not None and fit.free_columns:
        free = {}
        for j, col in enumerate(fit.free_columns):
            free[col] = [round(float(v), 6) for v in np.asarray(fit.beta_free)[:, j]]
        out["threshold_specific_slopes"] = free

    # Values only. `raw` carries the first cutpoint and the logs of the gaps to the
    # next ones, so a standard error on that scale is not one on this scale.
    out["cutpoints"] = [round(float(c), 6) for c in fit.cut]
    out["n"] = int(fit.n)
    out["loglik"] = round(float(fit.loglik), 4)
    return out


def fit_metrics(fit: OrdinalFit, null_loglik: float) -> dict:
    k = fit.n_params
    rho2 = 1 - fit.loglik / null_loglik
    adj = 1 - (fit.loglik - k) / null_loglik
    return {
        "loglik": round(fit.loglik, 4),
        "n_parameters": k,
        "n": fit.n,
        "rho2": round(float(rho2), 4),
        "adjusted_rho2": round(float(adj), 4),
        "aic": round(float(-2 * fit.loglik + 2 * k), 3),
        "bic": round(float(-2 * fit.loglik + k * np.log(fit.n)), 3),
        "converged": fit.converged,
    }


def null_loglikelihood(frame: pd.DataFrame, outcome: str = "sev3",
                       order: tuple[str, ...] = SEV_ORDER) -> float:
    counts = frame[outcome].value_counts().reindex(order).fillna(0).to_numpy()
    p = counts / counts.sum()
    return float((counts * np.log(p)).sum())


if __name__ == "__main__":
    frame = pd.read_csv(DATA / "model_frame.csv")
    spec = ladder(frame)
    print("Specification ladder (preliminary: the spatial layer and the adjudicated")
    print("labels are not final, so every number below is throwaway).\n")
    for name, cols in spec.items():
        print(f"  {name}: {len(cols)} covariates")
    print()

    print("Verifying the partial proportional-odds likelihood against ordered logit")
    check = verify_ppo_against_ordered_logit(frame, spec["M0"][:6])
    for key in ("loglik_ordered_logit", "loglik_ppo_no_free", "loglik_abs_gap",
                "max_abs_beta_gap", "max_abs_cutpoint_gap",
                "max_abs_numeric_gradient_at_optimum", "passes"):
        print(f"  {key}: {check[key]}")
    if not check["passes"]:
        raise SystemExit("PPO likelihood does not reproduce ordered logit; stop.")
    print()

    null_ll = null_loglikelihood(frame)
    results: dict = {"ppo_verification": check, "null_loglik": null_ll, "rungs": {}}
    fits: dict[str, OrdinalFit] = {}
    for name, cols in spec.items():
        fit = fit_ordered_logit(frame, cols, name)
        fits[name] = fit
        results["rungs"][name] = {
            "fit": fit_metrics(fit, null_ll),
            "events_per_parameter": events_per_parameter(frame, cols),
            "separation": separation_diagnostics(fit),
            "coefficients": coefficient_table(fit),
        }
        cv = cross_validated_log_loss(frame, cols)
        results["rungs"][name]["cross_validation"] = cv
        epp = results["rungs"][name]["events_per_parameter"]
        sep = results["rungs"][name]["separation"]
        print(f"{name}: loglik {fit.loglik:.2f}, {epp['n_parameters']} parameters, "
              f"{epp['events_per_parameter_smallest_class']} events per parameter "
              f"in the smallest class, separation flag {sep['suggests_separation']}")
        print(f"     cross-validated log loss {cv['mean_log_loss']} "
              f"(sd across folds {cv['sd_across_folds']}, "
              f"range {cv['min_log_loss']} to {cv['max_log_loss']}, "
              f"{cv['n_folds_failed']} of {cv['n_folds_total']} folds failed)")

    results["lr_M0_to_M0c"] = nested_lr_test(fits["M0"], fits["M0c"])
    results["lr_M0c_to_M1"] = nested_lr_test(fits["M0c"], fits["M1"])
    print("\nnested LR tests:", json.dumps({k: results[k] for k in
          ("lr_M0_to_M0c", "lr_M0c_to_M1")}))

    brant = proportional_odds_tests(frame, spec["M1"])
    results["proportional_odds_tests"] = brant
    print()
    print(f"proportional odds, omnibus likelihood ratio "
          f"{brant['omnibus_lr_chi2']} on {brant['omnibus_df']} df, "
          f"p {brant['omnibus_lr_p']}, rejects {brant['omnibus_rejects_at_5pct']}")
    print(f"  (classical Brant Wald form, reported for comparability only: "
          f"{brant['omnibus_wald_chi2_brant_form']}, p "
          f"{brant['omnibus_wald_p']})")
    failing = brant["covariates_failing_at_5pct"]
    print(f"covariates failing individually at 5 percent: {failing or 'none'}")

    # prespec.md section 7: ordered logit is primary unless the Brant test finds a
    # covariate whose effect differs across the thresholds. Freeing none of them would
    # reproduce ordered logit exactly, so in that case there is nothing to estimate.
    sets = multiplicity_sets(brant["per_covariate"])
    sens = primary_model_sensitivity(frame, spec["M1"], fits["M1"], sets, null_ll)
    results["multiplicity_sensitivity"] = sens
    print("  freed under each multiplicity rule: " + ", ".join(
        f"{k} {len(v)}" for k, v in sets.items()))
    for rule in ("uncorrected", "benjamini_hochberg", "holm"):
        r = sens[rule]
        if r["n_freed"]:
            print(f"    {rule:<20} {r['n_freed']:>2} freed, BIC difference "
                  f"{r['bic_confirmation']['difference']:+.1f} -> {r['primary']}")
    print(f"  the multiplicity rule changes the primary model: "
          f"{not sens['agrees_across_rules']}")

    if failing:
        ppo = fit_ppo(frame, spec["M1"], free=failing, name="M1_ppo")
        results["ppo"] = {
            "free_columns": list(ppo.free_columns),
            "fit": fit_metrics(ppo, null_ll),
            "vs_ordered_logit": nested_lr_test(fits["M1"], ppo),
            "bic_confirmation": confirm_by_bic(ppo, fits["M1"]),
            "convergence": ppo.extras,
            "coefficients": coefficient_table(ppo),
        }
        confirmed = results["ppo"]["bic_confirmation"]["confirmed"]
        primary = ppo if confirmed else fits["M1"]
        results["primary_model"] = ("partial proportional odds" if confirmed
                                    else "ordered logit (BIC did not confirm the "
                                         "partial proportional-odds model)")
    else:
        primary = fits["M1"]
        results["primary_model"] = "ordered logit"
        results["ppo"] = {
            "estimated": False,
            "reason": ("no covariate fails the Brant test individually, so the partial "
                       "proportional-odds model frees nothing and is identical to "
                       "ordered logit (prespec section 7)"),
        }
    print(f"primary model: {results['primary_model']}")

    print()
    print(f"Average marginal effects, {KR_DRAWS} Krinsky-Robb draws")
    ames = average_marginal_effects(primary, frame)
    results["average_marginal_effects"] = ames
    shown = [c for c in primary.columns if c in CUES] or primary.columns[:6]
    for col in shown:
        ka = ames[col]["KA"]
        flag = " *" if ka["excludes_zero"] else ""
        print(f"  {col:<28} KA {ka['ame']:+.4f} "
              f"[{ka['ci95'][0]:+.4f}, {ka['ci95'][1]:+.4f}]{flag}")
    n_sig = sum(1 for c in primary.columns if ames[c]["KA"]["excludes_zero"])
    print(f"  {n_sig} of {len(primary.columns)} covariates have a KA interval that "
          f"excludes zero")

    (DATA / "severity.json").write_text(
        json.dumps(results, indent=2, default=float), encoding="utf-8")
    print()
    print(f"written to {DATA / 'severity.json'}")
    layers = json.loads((DATA / "screening.json").read_text(encoding="utf-8"))["layers"]
    print("Preliminary. Spatial layers: " + "; ".join(f"{k} {v}" for k, v in layers.items()))
    cue_layer = layers.get("cues", "")
    if "final adjudicated" in cue_layer:
        print("Cue labels are the adjudicated human panel where one exists and the "
              "Route B reference run elsewhere; see data/final_labels_report.json "
              "for the split and its consequences.")
    else:
        print("PRELIMINARY cue labels. No number here goes in the paper until the "
              "coders' adjudicated labels replace them.")
