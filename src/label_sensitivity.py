"""Label-variant sensitivities for the narrative-cue covariates.

The primary severity models enter the cues as the final hard labels: the majority of three
blinded coders on the 187 human-coded narratives and the Route B reference run on the
rest. A predicted label or probability entered as an ordinary covariate is a plug-in
regressor whose first-stage error the second-stage intervals ignore, so the comparison
set is estimated here: calibrated probabilities, uncalibrated
probabilities, hard classifications under a fixed threshold, and manual labels on the
validation subset. This module estimates every one of them against the same
specification, with the same explanatory-gain test the paper reports:

* the nested likelihood-ratio test of the cue block, M1 against M0c, from `severity.py`;
* the paired out-of-fold log-loss difference on the repeated stratified folds of
  `severity.cross_validated_log_loss`;
* cue coefficients and average marginal effects on KA with Krinsky-Robb intervals.

Five things here are decisions rather than mechanics, and each is recorded in the output.

**The specification is held fixed.** M0c and the five cue columns are the ones the frozen
screening admitted on the primary labels. A variant is a change of measurement, not a
change of model, so no variant is re-screened. The event cells each variant produces are
reported so that a reader can see where a cue would have failed the event rule.

**Calibration is built here, because nothing upstream stores it.** The five-run vote shares
are calibrated isotonically, fitted by five-fold cross-fitting on the
human-coded subset. Human-coded narratives receive their out-of-fold value; every other
narrative receives the mean of the five fold maps, none of which saw it. The human subset
is not a simple random sample (stratified validation draw plus the flagged adjudication
set), so the fit carries design weights: flagged narratives were all coded, so they are a
certainty stratum with weight one, and the other validation narratives carry the inverse
of their inclusion probability.

**Hybrid by default.** Where a narrative carries a human label, that label is observed
data; drawing over it would add noise, not propagate uncertainty. The calibrated plug-in
and the label draws therefore replace only the automated rows, which is what makes the
draws a multiple imputation of the labels the primary model actually assumed. The fully
automated plug-in (calibrated probabilities on every row) is reported beside it.

**Draws from the vote share alone attenuate toward the null.** An imputation model that
ignores the outcome produces imputed cues that are independent of severity given the vote
share, so the imputed rows dilute any cue-severity association. That bias favours the
stated null. A second set of draws uses an outcome-informed imputation model (the human
label on the logit of the calibrated probability and the severity class), with its
parameters drawn from their approximate posterior per imputation, so the direction of the
bias can be checked rather than asserted.

**The reduced model's rules were fixed before it was run.** On the 187 human-labelled
riders the full-sample event rule (ten per severity cell) admits no cue. The rule used is
five per cell, applied identically to cues and to binary base covariates; continuous base
covariates are kept. The same model with all five cues and the same model on the same
rows with Route B labels are reported beside it.
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import optimize, stats

import severity as sv
from labels import MECHANISMS

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = DATA / "label_sensitivity.json"

N_DRAWS = 50
DRAW_SEED = 20261005
CAL_FOLDS = 5
CAL_SEED = 20261006
IMPUTE_PRIOR_SD = 2.5          # normal prior on non-intercept imputation coefficients
REDUCED_MIN_EVENTS = 5         # per severity cell, human-labelled subset only
VOTE_THRESHOLD = 0.5           # hard label = more than half of the five runs

INPUTS = {
    "model_frame": "data/model_frame.csv",
    "final_labels": "data/final_labels.csv",
    "reference_labels": "data/reference_labels.csv",
    "fusion": "data/fusion.csv",
    "vote_shares": "data/vote_shares.csv",
    "validation_ids": "data/validation_ids.csv",
    "flagged_ids": "data/flagged_ids.csv",
    "screening": "data/screening.json",
    "severity": "data/severity.json",
    "numbers": "data/numbers.json",
}


# --------------------------------------------------------------------------
# Label tables, one row per narrative
# --------------------------------------------------------------------------

def _crash_table(path: Path, cols: list[str]) -> pd.DataFrame:
    t = pd.read_csv(path).drop_duplicates(subset="Crash_ID").set_index("Crash_ID")
    return t[cols].astype(float)


def with_cues(frame: pd.DataFrame, table: pd.DataFrame,
              rows: pd.Series | None = None) -> pd.DataFrame:
    """Copy of `frame` with the cue columns in `table` mapped in by Crash_ID.

    Labels live at the narrative level and two crashes carry two riders, so mapping by
    Crash_ID is what keeps both riders of one crash on one label. `rows` restricts the
    replacement to a subset of rider rows; the others keep their primary label.
    """
    out = frame.copy()
    mask = np.ones(len(out), bool) if rows is None else rows.to_numpy()
    for c in table.columns:
        mapped = out["Crash_ID"].map(table[c])
        if mapped[mask].isna().any():
            raise ValueError(f"{c}: {int(mapped[mask].isna().sum())} rows have no label")
        out[c] = out[c].astype(float)
        out.loc[mask, c] = mapped[mask].astype(float)
    return out


# --------------------------------------------------------------------------
# Cross-fitted isotonic calibration
# --------------------------------------------------------------------------

def design_weights(human_ids: list[int]) -> pd.Series:
    """Horvitz-Thompson weights for the human-coded narratives.

    Every flagged narrative was coded, either in the validation draw or in the
    adjudication round, so the flagged set is a certainty stratum. The remaining
    validation narratives carry the inverse of their stratified inclusion probability.
    """
    val = pd.read_csv(DATA / "validation_ids.csv").set_index("Crash_ID")
    flagged = set(pd.read_csv(DATA / "flagged_ids.csv")["Crash_ID"].astype(int))
    w = {}
    for cid in human_ids:
        if cid in flagged:
            w[cid] = 1.0
        elif cid in val.index:
            w[cid] = float(1.0 / val.loc[cid, "inclusion_prob"])
        else:
            raise ValueError(f"human-coded narrative {cid} is neither flagged nor sampled")
    return pd.Series(w)


def _ece(p: np.ndarray, y: np.ndarray, w: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    total, err = w.sum(), 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            err += w[m].sum() / total * abs(np.average(p[m], weights=w[m])
                                            - np.average(y[m], weights=w[m]))
    return float(err)


def calibrate(votes: pd.DataFrame, human: pd.DataFrame,
              weights: pd.Series) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Cross-fitted isotonic map from vote share to the human label, per mechanism.

    Returns calibrated probabilities for every narrative, the out-of-fold probabilities
    on the human subset, and the calibration diagnostics. The vote share takes six
    values (0 to 5 of 5 runs), so each fold map is a short monotone step function.
    """
    from sklearn.isotonic import IsotonicRegression
    from sklearn.model_selection import StratifiedKFold

    hid = human.index.to_numpy()
    others = votes.index.difference(human.index)
    cal = pd.DataFrame(index=votes.index, columns=votes.columns, dtype=float)
    oof = pd.DataFrame(index=human.index, columns=votes.columns, dtype=float)
    diag: dict = {}
    grid = np.round(np.arange(0, 1.01, 0.2), 1)

    for c in votes.columns:
        y = human[c].to_numpy().astype(int)
        x = votes.loc[hid, c].to_numpy()
        w = weights.reindex(hid).to_numpy()
        # A rare mechanism cannot be split five ways with a positive in every fold.
        n_splits = min(CAL_FOLDS, int(min(y.sum(), len(y) - y.sum())))
        maps = []
        if n_splits >= 2:
            skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=CAL_SEED)
            for tr, te in skf.split(x, y):
                iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
                iso.fit(x[tr], y[tr], sample_weight=w[tr])
                oof.loc[hid[te], c] = iso.predict(x[te])
                maps.append(iso)
            folds_used = len(maps)
        else:
            # Too few positives to stratify: one map on all human rows, flagged as such.
            iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
            iso.fit(x, y, sample_weight=w)
            oof.loc[hid, c] = iso.predict(x)
            maps.append(iso)
            folds_used = 1
        cal.loc[hid, c] = oof.loc[hid, c]
        cal.loc[others, c] = np.mean([m.predict(votes.loc[others, c].to_numpy())
                                      for m in maps], axis=0)

        po = oof.loc[hid, c].to_numpy().astype(float)
        ones = np.ones_like(w)
        diag[c] = {
            "human_positives": int(y.sum()),
            "folds_used": folds_used,
            "map_mean_over_folds": {f"{g:.1f}": round(float(np.mean(
                [m.predict([g])[0] for m in maps])), 4) for g in grid},
            "oof_brier_weighted": round(float(np.average((po - y) ** 2, weights=w)), 4),
            "oof_brier_unweighted": round(float(np.mean((po - y) ** 2)), 4),
            "raw_vote_share_brier_weighted": round(float(np.average((x - y) ** 2,
                                                                    weights=w)), 4),
            "oof_ece_weighted": round(_ece(po, y, w), 4),
            "oof_ece_unweighted": round(_ece(po, y, ones), 4),
            "raw_vote_share_ece_weighted": round(_ece(x, y, w), 4),
        }
    return cal.astype(float), oof.astype(float), diag


# --------------------------------------------------------------------------
# Marginal effects for the cue columns only
# --------------------------------------------------------------------------

def cue_ames(fit: sv.OrdinalFit, frame: pd.DataFrame, cues: list[str],
             draws: int = sv.KR_DRAWS, seed: int = sv.KR_SEED) -> dict:
    """Discrete-change AMEs (0 to 1) for the cue columns, with Krinsky-Robb intervals.

    The parameter draws are taken exactly as `severity.average_marginal_effects` takes
    them (same seed, same joint draw of slopes and threshold parameters), so on the
    primary labels this reproduces its numbers; that is checked before anything else
    uses it. The difference is that a plug-in probability is also given the 0 to 1
    change rather than a derivative, so every variant reports the same quantity. Cues
    belong to no categorical group, so there are no sibling levels to zero.
    """
    cols = fit.columns
    X, _ = sv._design(frame, cols)
    rng = np.random.default_rng(seed)
    k = len(fit.beta)
    n_raw = k + len(fit.cut)
    cov = np.asarray(fit.cov)[:n_raw, :n_raw]
    cov = (cov + cov.T) / 2
    sample = rng.multivariate_normal(np.asarray(fit.raw)[:n_raw], cov, size=draws,
                                     method="svd")
    idx = [cols.index(c) for c in cues]

    def once(beta, cut):
        eff = np.zeros((len(idx), len(cut) + 1))
        for r, i in enumerate(idx):
            X1, X0 = X.copy(), X.copy()
            X1[:, i], X0[:, i] = 1.0, 0.0
            eff[r] = (sv._probs(X1, beta, cut) - sv._probs(X0, beta, cut)).mean(axis=0)
        return eff

    point = once(fit.beta, fit.cut)
    drawn = np.stack([once(sample[d, :k], sv._cut_from_raw(sample[d, k:]))
                      for d in range(draws)])
    out = {}
    for r, c in enumerate(cues):
        entry = {}
        for j, lvl in enumerate(sv.SEV_ORDER):
            lo, hi = np.percentile(drawn[:, r, j], [2.5, 97.5])
            entry[lvl] = {"ame": round(float(point[r, j]), 5),
                          "ci95": [round(float(lo), 5), round(float(hi), 5)],
                          "excludes_zero": bool(lo > 0 or hi < 0),
                          "_point": float(point[r, j]),
                          "_kr_sd": float(np.std(drawn[:, r, j], ddof=1))}
        entry["type"] = "discrete change 0 to 1"
        out[c] = entry
    return out


def _public(ames: dict) -> dict:
    """Drop the unrounded helper fields before writing."""
    return {c: {k: ({kk: vv for kk, vv in v.items() if not kk.startswith("_")}
                    if isinstance(v, dict) else v) for k, v in e.items()}
            for c, e in ames.items()}


# --------------------------------------------------------------------------
# One variant
# --------------------------------------------------------------------------

def event_cells(frame: pd.DataFrame, cues: list[str]) -> dict:
    out = {}
    for c in cues:
        tab = frame.groupby("sev3")[c].sum().reindex(sv.SEV_ORDER).fillna(0)
        binary = set(np.unique(frame[c])) <= {0.0, 1.0}
        out[c] = {"by_severity": {k: (int(v) if binary else round(float(v), 2))
                                  for k, v in tab.items()},
                  "kind": "count" if binary else "expected count (sum of probabilities)",
                  "smallest_cell": (int(tab.min()) if binary else round(float(tab.min()), 2)),
                  "meets_full_sample_event_rule": bool(tab.min() >= 10)}
    return out


def cv_difference(cv_m1: dict, cv_m0c: dict) -> dict:
    """Paired fold-level difference, M1 minus M0c. Positive means the cues hurt."""
    a = {(r["repeat"], r["fold"]): r["log_loss"] for r in cv_m1["fold_level"] if r["ok"]}
    b = {(r["repeat"], r["fold"]): r["log_loss"] for r in cv_m0c["fold_level"] if r["ok"]}
    keys = sorted(set(a) & set(b))
    d = np.array([a[k] - b[k] for k in keys])
    return {"mean_log_loss_M1": cv_m1["mean_log_loss"],
            "mean_log_loss_M0c": cv_m0c["mean_log_loss"],
            "mean_difference_M1_minus_M0c": round(float(d.mean()), 5),
            "sd_difference_across_folds": round(float(d.std(ddof=1)), 5),
            "share_of_folds_where_cues_lower_log_loss": round(float((d < 0).mean()), 4),
            "n_paired_folds": int(len(d)),
            "n_folds_failed_M1": cv_m1["n_folds_failed"]}


def evaluate(frame_v: pd.DataFrame, base: list[str], cues: list[str],
             m0c: sv.OrdinalFit, cv_m0c: dict | None, do_cv: bool = True) -> dict:
    m1 = sv.fit_ordered_logit(frame_v, base + cues, "M1_variant")
    lr = sv.nested_lr_test(m0c, m1)
    coefs = sv.coefficient_table(m1)["slopes"]
    ames = cue_ames(m1, frame_v, cues)
    out = {
        "n": m1.n,
        "converged": m1.converged,
        "loglik_M1": round(m1.loglik, 4),
        "loglik_M0c": round(m0c.loglik, 4),
        "cue_block_lr": lr,
        "separation": sv.separation_diagnostics(m1),
        "coefficients": {c: coefs[c] for c in cues},
        "ame": _public(ames),
        "events": event_cells(frame_v, cues),
        "printed": {"lr_chi2": round(lr["lr_chi2"], 2), "df": lr["df"],
                    "p": round(lr["p"], 2)},
    }
    if do_cv and cv_m0c is not None:
        cv_m1 = sv.cross_validated_log_loss(frame_v, base + cues)
        out["cross_validation"] = cv_difference(cv_m1, cv_m0c)
        out["printed"]["cv_log_loss_M1"] = round(cv_m1["mean_log_loss"], 3)
        out["printed"]["cv_difference"] = round(
            out["cross_validation"]["mean_difference_M1_minus_M0c"], 3)
    return out, m1, ames


# --------------------------------------------------------------------------
# Rubin's rules
# --------------------------------------------------------------------------

def rubin_scalar(q: np.ndarray, u: np.ndarray, n_com: float) -> dict:
    """Pooled estimate, Barnard-Rubin degrees of freedom, fraction of missing information."""
    m = len(q)
    qbar, W = float(np.mean(q)), float(np.mean(u))
    B = float(np.var(q, ddof=1))
    T = W + (1 + 1 / m) * B
    lam = (1 + 1 / m) * B / T if T > 0 else 0.0
    r = (1 + 1 / m) * B / W if W > 0 else float("inf")
    nu_old = (m - 1) / lam ** 2 if lam > 0 else float("inf")
    nu_obs = (n_com + 1) / (n_com + 3) * n_com * (1 - lam)
    nu = 1 / (1 / nu_old + 1 / nu_obs)
    fmi = (r + 2 / (nu + 3)) / (r + 1)
    se = np.sqrt(T)
    tcrit = stats.t.ppf(0.975, nu)
    p = float(2 * stats.t.sf(abs(qbar) / se, nu)) if se > 0 else None
    return {"estimate": qbar, "se": float(se), "ci95": [qbar - tcrit * se,
                                                        qbar + tcrit * se],
            "p": p, "df_barnard_rubin": float(nu), "within_var": W, "between_var": B,
            "relative_increase_in_variance": float(r),
            "lambda_proportion_of_variance_from_imputation": float(lam),
            "fraction_missing_information": float(fmi)}


def _round_rubin(d: dict, nd: int) -> dict:
    out = {}
    for k, v in d.items():
        if isinstance(v, list):
            out[k] = [round(float(x), nd) for x in v]
        elif isinstance(v, float):
            out[k] = round(v, nd if k in ("estimate", "se", "within_var",
                                          "between_var") else 4)
        else:
            out[k] = v
    out["excludes_zero"] = bool(out["ci95"][0] > 0 or out["ci95"][1] < 0)
    return out


def rubin_d1(Q: np.ndarray, U: np.ndarray) -> dict:
    """Multivariate Wald test of the cue block (Li, Raghunathan and Rubin 1991)."""
    m, k = Q.shape
    qbar = Q.mean(axis=0)
    W = U.mean(axis=0)
    dev = Q - qbar
    B = dev.T @ dev / (m - 1)
    r1 = (1 + 1 / m) * float(np.trace(B @ np.linalg.inv(W))) / k
    D1 = float(qbar @ np.linalg.inv((1 + r1) * W) @ qbar / k)
    t = k * (m - 1)
    if t > 4:
        nu = 4 + (t - 4) * (1 + (1 - 2 / t) / r1) ** 2
    else:
        nu = t * (1 + 1 / k) * (1 + 1 / r1) ** 2 / 2
    p = float(stats.f.sf(D1, k, nu))
    return {"D1_F": round(D1, 4), "df_numerator": k, "df_denominator": round(nu, 2),
            "p": round(p, 6), "average_relative_increase_in_variance": round(r1, 4),
            "printed": {"F": round(D1, 2), "p": round(p, 2)}}


def rubin_d2(d: np.ndarray, k: int) -> dict:
    """Combined likelihood-ratio chi-squares (Li, Meng, Raghunathan and Rubin 1991)."""
    m = len(d)
    r2 = (1 + 1 / m) * float(np.var(np.sqrt(np.clip(d, 0, None)), ddof=1))
    D2 = (float(np.mean(d)) / k - (m + 1) / (m - 1) * r2) / (1 + r2)
    nu = k ** (-3 / m) * (m - 1) * (1 + 1 / r2) ** 2
    p = float(stats.f.sf(max(D2, 0.0), k, nu))
    return {"D2_F": round(D2, 4), "df_numerator": k, "df_denominator": round(nu, 2),
            "p": round(p, 6), "r2": round(r2, 4),
            "note": "a negative D2 is truncated at zero for the p-value"}


def run_draws(name: str, frame: pd.DataFrame, base: list[str], cues: list[str],
              m0c: sv.OrdinalFit, auto_rows: pd.Series, auto_ids: np.ndarray,
              prob_fn, seed: int) -> dict:
    """Refit M1 on N_DRAWS label draws for the automated rows and pool by Rubin's rules.

    `prob_fn(m, rng)` returns a DataFrame of positive-label probabilities indexed by the
    automated Crash_IDs, one column per cue, for imputation `m`.
    """
    rng = np.random.default_rng(seed)
    Qb, Ub, Qa, Ua, lrs, ps, conv, sep, prev = [], [], [], [], [], [], [], [], []
    for m in range(N_DRAWS):
        p = prob_fn(m, rng)
        lab = pd.DataFrame((rng.random(p.shape) < p.to_numpy()).astype(float),
                           index=p.index, columns=p.columns)
        fv = with_cues(frame, lab, rows=auto_rows)
        m1 = sv.fit_ordered_logit(fv, base + cues, f"M1_{name}_{m}")
        idx = [m1.columns.index(c) for c in cues]
        Qb.append(m1.beta[idx])
        Ub.append(np.asarray(m1.cov)[np.ix_(idx, idx)])
        a = cue_ames(m1, fv, cues)
        Qa.append([[a[c][l]["_point"] for l in sv.SEV_ORDER] for c in cues])
        Ua.append([[a[c][l]["_kr_sd"] ** 2 for l in sv.SEV_ORDER] for c in cues])
        lr = sv.nested_lr_test(m0c, m1)
        lrs.append(lr["lr_chi2"])
        ps.append(lr["p"])
        conv.append(m1.converged)
        sep.append(sv.separation_diagnostics(m1)["suggests_separation"])
        prev.append(fv[cues].mean().to_numpy())
    Qb, Ub, Qa, Ua = map(np.asarray, (Qb, Ub, Qa, Ua))
    n_com = len(frame) - (len(base) + len(cues) + len(sv.SEV_ORDER) - 1)

    coefs, ames, fmi = {}, {}, {}
    for r, c in enumerate(cues):
        coefs[c] = _round_rubin(rubin_scalar(Qb[:, r], Ub[:, r, r], n_com), 4)
        ames[c] = {lvl: _round_rubin(rubin_scalar(Qa[:, r, j], Ua[:, r, j], n_com), 5)
                   for j, lvl in enumerate(sv.SEV_ORDER)}
        fmi[c] = {"coefficient": coefs[c]["fraction_missing_information"],
                  "ame_KA": ames[c]["KA"]["fraction_missing_information"]}
    lrs, ps = np.array(lrs), np.array(ps)
    d1 = rubin_d1(Qb, Ub)
    return {
        "n_draws": N_DRAWS,
        "seed": seed,
        "n_complete_data_df": int(n_com),
        "all_converged": bool(all(conv)),
        "n_not_converged": int(sum(1 for c in conv if not c)),
        "n_separation_flag": int(sum(sep)),
        "cue_block_test_D1_wald": d1,
        "cue_block_test_D2_combined_lr": rubin_d2(lrs, len(cues)),
        "per_draw_lr": {"chi2_min": round(float(lrs.min()), 4),
                        "chi2_median": round(float(np.median(lrs)), 4),
                        "chi2_max": round(float(lrs.max()), 4),
                        "p_min": round(float(ps.min()), 6),
                        "p_median": round(float(np.median(ps)), 6),
                        "p_max": round(float(ps.max()), 6),
                        "share_p_below_0.05": round(float((ps < 0.05).mean()), 4),
                        "df": len(cues)},
        "coefficients_pooled": coefs,
        "ame_pooled": ames,
        "fraction_missing_information": fmi,
        "mean_prevalence_over_draws": {c: round(float(v), 4)
                                       for c, v in zip(cues, np.mean(prev, axis=0))},
        "printed": {"D1_p": d1["printed"]["p"],
                    "per_draw_p_range": [round(float(ps.min()), 2),
                                         round(float(ps.max()), 2)]},
        "cross_validation": "not run: 50 refits per draw would cost 2,500 fits; the "
                            "pooled block test answers the question the draws exist for",
    }


# --------------------------------------------------------------------------
# Outcome-informed imputation model
# --------------------------------------------------------------------------

def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 0.01, 0.99)
    return np.log(p / (1 - p))


def fit_imputation_model(x: np.ndarray, sev: np.ndarray, y: np.ndarray,
                         w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Penalized logistic regression of the human label, with its Laplace covariance.

    Design: intercept, logit of the calibrated probability, and indicators for O and KA
    (BC is the reference). A normal prior with scale 2.5 on the non-intercept terms keeps
    the fit finite when a cue has no positive in one severity class among the human rows,
    which happens here. Weights are the design weights rescaled to sum to the number of
    human rows, so they reweight without inflating the information.
    """
    Z = np.column_stack([np.ones(len(x)), x, (sev == "O").astype(float),
                         (sev == "KA").astype(float)])
    w = w * len(w) / w.sum()
    prec = np.array([0.0] + [1 / IMPUTE_PRIOR_SD ** 2] * 3)

    def negpost(b):
        eta = Z @ b
        return float(-(w * (y * eta - np.logaddexp(0, eta))).sum()
                     + 0.5 * (prec * b ** 2).sum())

    def grad(b):
        mu = 1 / (1 + np.exp(-(Z @ b)))
        return -(Z.T @ (w * (y - mu))) + prec * b

    res = optimize.minimize(negpost, np.zeros(Z.shape[1]), jac=grad, method="BFGS")
    mu = 1 / (1 + np.exp(-(Z @ res.x)))
    H = Z.T @ (Z * (w * mu * (1 - mu))[:, None]) + np.diag(prec)
    return res.x, np.linalg.inv(H)


# --------------------------------------------------------------------------
# Reduced model on the human-labelled riders
# --------------------------------------------------------------------------

def reduced_human_model(frame: pd.DataFrame, base: list[str], cues: list[str],
                        reference: pd.DataFrame) -> dict:
    sub = frame[frame["label_source"] == "human"].reset_index(drop=True)
    if len(sub) != 187:
        raise SystemExit(f"expected 187 human-labelled rider rows, found {len(sub)}")

    def min_cell(c):
        return int(sub.groupby("sev3")[c].sum().reindex(sv.SEV_ORDER).fillna(0).min())

    def is_binary(c):
        return set(np.unique(sub[c])) <= {0.0, 1.0}

    base_kept, base_dropped = [], {}
    for c in base:
        if sub[c].std() == 0:
            base_dropped[c] = "no variation in the subset"
        elif is_binary(c) and min_cell(c) < REDUCED_MIN_EVENTS:
            base_dropped[c] = f"smallest severity cell {min_cell(c)}"
        else:
            base_kept.append(c)
    cue_cells = {c: {"by_severity": {k: int(v) for k, v in sub.groupby("sev3")[c].sum()
                                     .reindex(sv.SEV_ORDER).fillna(0).items()},
                     "smallest_cell": min_cell(c),
                     "admitted": bool(min_cell(c) >= REDUCED_MIN_EVENTS)} for c in cues}
    admitted = [c for c in cues if cue_cells[c]["admitted"]]

    null_ll = sv.null_loglikelihood(sub)
    m0 = sv.fit_ordered_logit(sub, base_kept, "reduced_base")

    def block(cols, frame_):
        m1 = sv.fit_ordered_logit(frame_, base_kept + cols, "reduced_cues")
        lr = sv.nested_lr_test(m0, m1)
        coefs = sv.coefficient_table(m1)["slopes"]
        return {"cues": cols,
                "converged": m1.converged,
                "loglik": round(m1.loglik, 4),
                "cue_block_lr": lr,
                "printed": {"lr_chi2": round(lr["lr_chi2"], 2), "df": lr["df"],
                            "p": round(lr["p"], 2)},
                "separation": sv.separation_diagnostics(m1),
                "fit": sv.fit_metrics(m1, null_ll),
                "coefficients": {c: coefs[c] for c in cols},
                "ame": _public(cue_ames(m1, frame_, cols))}

    sub_ref = with_cues(sub, reference[cues])
    out = {
        "rows": int(len(sub)),
        "batch_split": {str(int(k)): int(v) for k, v in
                        sub["label_batch"].value_counts().sort_index().items()},
        "severity_counts": {k: int(v) for k, v in
                            sub["sev3"].value_counts().reindex(sv.SEV_ORDER).items()},
        "rule": (f"binary covariates and cues enter only with at least "
                 f"{REDUCED_MIN_EVENTS} events in every severity class of the subset; "
                 f"continuous covariates are kept. Fixed before estimation."),
        "base_covariates": base_kept,
        "base_covariates_dropped": base_dropped,
        "base_converged": m0.converged,
        "base_loglik": round(m0.loglik, 4),
        "base_separation": sv.separation_diagnostics(m0),
        "events_per_parameter": sv.events_per_parameter(sub, base_kept + admitted),
        "cue_cells_human_labels": cue_cells,
        "cue_cells_route_b_labels": {c: {k: int(v) for k, v in sub_ref.groupby("sev3")[c]
                                          .sum().reindex(sv.SEV_ORDER).fillna(0).items()}
                                     for c in cues},
        "admitted_cues": admitted,
        "human_labels_admitted_cues": block(admitted, sub) if admitted else None,
        "human_labels_all_five_cues": block(list(cues), sub),
        "route_b_labels_same_rows_admitted_cues": (block(admitted, sub_ref)
                                                   if admitted else None),
    }
    return out


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main() -> dict:
    t0 = time.time()
    frame = pd.read_csv(DATA / "model_frame.csv")
    spec = sv.ladder(frame)
    base = spec["M0c"]
    cues = [c for c in spec["M1"] if c in sv.CUES]
    assert spec["M1"] == base + cues and len(cues) == 5, "unexpected ladder"
    mech = [m for m in MECHANISMS if m in frame.columns]

    human_mask = frame["label_source"] == "human"
    auto_mask = ~human_mask
    split = frame.loc[human_mask, "label_batch"].value_counts().to_dict()
    assert human_mask.sum() == 187 and split == {1.0: 150, 2.0: 37}, split

    fl =pd.read_csv(DATA / "final_labels.csv").set_index("Crash_ID")
    human_tab = fl.loc[fl["label_source"] == "human", mech].astype(float)
    reference = _crash_table(DATA / "reference_labels.csv", mech)
    fusion = _crash_table(DATA / "fusion.csv", mech)
    votes = _crash_table(DATA / "vote_shares.csv", mech)
    majority = (votes > VOTE_THRESHOLD).astype(float)
    fusion_equals_reference = bool((fusion.reindex(reference.index) == reference)
                                   .all().all())

    weights = design_weights(list(human_tab.index))
    cal, oof, cal_diag = calibrate(votes, human_tab, weights)
    hybrid_cal = cal.copy()
    hybrid_cal.loc[human_tab.index] = human_tab

    results: dict = {
        "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "script": "src/label_sensitivity.py",
        "inputs": INPUTS,
        "settings": {
            "model": "ordered logit, M0c vs M1 nested likelihood-ratio test "
                     "(severity.fit_ordered_logit, severity.nested_lr_test)",
            "M0c_covariates": len(base),
            "cues": cues,
            "kr_draws": sv.KR_DRAWS, "kr_seed": sv.KR_SEED,
            "cv": {"repeats": sv.CV_REPEATS, "folds": sv.CV_FOLDS, "seed": sv.CV_SEED},
            "label_draws": N_DRAWS, "draw_seed": DRAW_SEED,
            "calibration": {"method": "isotonic, cross-fitted", "folds": CAL_FOLDS,
                            "seed": CAL_SEED,
                            "fit_on": "187 human-coded narratives (150 validation, "
                                      "37 adjudication)",
                            "weights": "flagged narratives 1 (certainty stratum); other "
                                       "validation narratives 1 / inclusion probability",
                            "human_rows_get": "out-of-fold calibrated value",
                            "other_rows_get": "mean of the five fold maps"},
            "vote_share_threshold_for_hard_label": f"> {VOTE_THRESHOLD} (3 or more "
                                                   f"of 5 runs)",
            "imputation_prior_sd": IMPUTE_PRIOR_SD,
            "reduced_model_min_events": REDUCED_MIN_EVENTS,
            "ame_definition": "discrete change from 0 to 1 averaged over riders, for "
                              "binary labels and plug-in probabilities alike",
        },
        "notes": {
            "fusion_labels_identical_to_route_b_reference": fusion_equals_reference,
            "fusion_note": ("the fused hard label keeps Route B wherever the routes "
                            "disagree, so as a label source it equals the reference run; "
                            "it is not estimated as a separate variant"
                            if fusion_equals_reference else
                            "fusion labels differ from the reference run"),
            "calibration_not_stored_upstream": (
                "no isotonic calibration exists in src before this script; the "
                "calibrated probabilities and their out-of-fold diagnostics are built "
                "and saved here for the first time"),
            "draw_independence": ("labels are drawn independently across cues given "
                                  "their probabilities; co-occurrence between cues in the "
                                  "automated rows is not preserved"),
            "attenuation": ("draws from the calibrated probability alone ignore severity, "
                            "so they attenuate any cue-severity association in the "
                            "automated rows toward zero, which favours the null; the "
                            "outcome-informed draws are the check on that"),
            "plugin_cv_note": ("calibrated plug-in values for the automated rows use maps "
                               "fitted on the human rows only, so the outcome never enters "
                               "the calibration; the cross-validated log loss treats them "
                               "as fixed covariates"),
        },
    }

    # ---- (i) primary, and the reproduction check --------------------------
    print("fitting M0c and the primary M1")
    m0c = sv.fit_ordered_logit(frame, base, "M0c")
    cv_m0c = sv.cross_validated_log_loss(frame, base)
    prim, m1p, ames_p = evaluate(frame, base, cues, m0c, cv_m0c)

    sp = json.loads((DATA / "severity.json").read_text(encoding="utf-8"))
    nj = json.loads((DATA / "numbers.json").read_text(encoding="utf-8"))["severity"]
    lr_ref = sp["lr_M0c_to_M1"]
    ame_match = all(
        prim["ame"][c][l]["ame"] == sp["average_marginal_effects"][c][l]["ame"]
        and prim["ame"][c][l]["ci95"] == sp["average_marginal_effects"][c][l]["ci95"]
        for c in cues for l in sv.SEV_ORDER)
    ame_nj_match = all(
        prim["ame"][c][l]["ame"] == nj["ames_all"][c][l]["ame"] for c in cues
        for l in sv.SEV_ORDER)
    repro = {
        "lr_matches_severity": prim["cue_block_lr"] == lr_ref,
        "lr_matches_numbers_json": prim["cue_block_lr"] == nj["nested_lr"]["M0c_to_M1"],
        "ames_match_severity": ame_match,
        "ames_match_numbers_json": ame_nj_match,
        "cv_M1_matches": round(prim["cross_validation"]["mean_log_loss_M1"], 3)
                         == nj["rungs"]["M1"]["cv_log_loss"],
        "cv_M0c_matches": round(cv_m0c["mean_log_loss"], 3)
                          == nj["rungs"]["M0c"]["cv_log_loss"],
        "coef_matches_numbers_json": all(
            round(prim["coefficients"][c]["coef"], 3)
            == nj["rungs"]["M1"]["coefficients"]["slopes"][c]["coef"] for c in cues),
        "reference_values": {"severity_lr": lr_ref,
                             "numbers_json_lr": nj["nested_lr"]["M0c_to_M1"]},
    }
    repro["all_match"] = all(v for k, v in repro.items() if isinstance(v, bool))
    results["primary_reproduction"] = repro
    print("  reproduction:", {k: v for k, v in repro.items() if isinstance(v, bool)})
    if not repro["all_match"]:
        OUT.write_text(json.dumps(results, indent=2, default=float), encoding="utf-8")
        raise SystemExit("primary does not reproduce numbers.json; stopped")

    variants: dict = {"i_final_labels_primary": {
        "description": "final labels: human majority on 187 riders, Route B reference "
                       "run on 335", **prim}}

    # ---- (ii) automated labels without adjudication -----------------------
    spec_v = [
        ("ii_llm_reference_labels",
         "Route B reference run (temperature 0) on every rider, no human labels",
         with_cues(frame, reference[cues])),
        ("ii_b_vote_share_majority_labels",
         "hard label when more than half of the five temperature-0.7 runs are positive, "
         "every rider",
         with_cues(frame, majority[cues])),
        ("iii_plugin_calibrated_hybrid",
         "calibrated probability on the 335 automated riders, human label on the 187",
         with_cues(frame, hybrid_cal[cues])),
        ("iii_b_plugin_calibrated_all_rows",
         "calibrated probability on every rider (out-of-fold on the human-coded ones)",
         with_cues(frame, cal[cues])),
        ("iii_c_plugin_raw_vote_share",
         "uncalibrated five-run vote share on every rider",
         with_cues(frame, votes[cues])),
    ]
    for key, desc, fv in spec_v:
        print(f"fitting {key}")
        res, _, _ = evaluate(fv, base, cues, m0c, cv_m0c)
        variants[key] = {"description": desc, **res}

    # ---- (iv) label draws, combined by Rubin's rules ----------------------
    auto_ids = np.array(sorted(set(frame.loc[auto_mask, "Crash_ID"])))
    cal_auto = cal.loc[auto_ids, cues]

    print(f"(iv) {N_DRAWS} draws from the calibrated probabilities")
    variants["iv_draws_calibrated"] = {
        "description": ("50 Bernoulli draws from the calibrated probability for the 335 "
                        "automated riders, human labels kept on the 187; M1 refitted per "
                        "draw and pooled by Rubin's rules"),
        **run_draws("cal", frame, base, cues, m0c, auto_mask, auto_ids,
                    lambda m, rng: cal_auto, DRAW_SEED)}

    # Outcome-informed imputation model, fitted on the human rows.
    sev_by_crash = frame.drop_duplicates("Crash_ID").set_index("Crash_ID")["sev3"]
    hid = human_tab.index.to_numpy()
    imp_models, imp_report = {}, {}
    for c in cues:
        b, V = fit_imputation_model(_logit(oof.loc[hid, c].to_numpy()),
                                    sev_by_crash.reindex(hid).to_numpy(),
                                    human_tab.loc[hid, c].to_numpy(),
                                    weights.reindex(hid).to_numpy())
        imp_models[c] = (b, V)
        se = np.sqrt(np.diag(V))
        imp_report[c] = {n: {"coef": round(float(b[i]), 4), "se": round(float(se[i]), 4)}
                         for i, n in enumerate(("intercept", "logit_calibrated_p",
                                                "sev_O", "sev_KA"))}
    xa = {c: _logit(cal_auto[c].to_numpy()) for c in cues}
    sa = sev_by_crash.reindex(auto_ids).to_numpy()
    Za = {c: np.column_stack([np.ones(len(auto_ids)), xa[c], (sa == "O").astype(float),
                              (sa == "KA").astype(float)]) for c in cues}

    def informed(m, rng):
        out = {}
        for c in cues:
            b, V = imp_models[c]
            bd = rng.multivariate_normal(b, V)
            out[c] = 1 / (1 + np.exp(-(Za[c] @ bd)))
        return pd.DataFrame(out, index=auto_ids)

    print(f"(iv-b) {N_DRAWS} outcome-informed draws")
    variants["iv_b_draws_outcome_informed"] = {
        "description": ("50 draws for the automated riders from a logistic imputation "
                        "model of the human label on logit(calibrated p) and severity "
                        "class, parameters drawn from their Laplace posterior each time; "
                        "human labels kept on the 187"),
        "imputation_model": imp_report,
        **run_draws("inf", frame, base, cues, m0c, auto_mask, auto_ids, informed,
                    DRAW_SEED + 1)}

    # ---- (v) human-labelled riders only -----------------------------------
    print("(v) reduced model on the human-labelled riders")
    variants["v_human_subset_reduced"] = reduced_human_model(frame, base, cues, reference)
    results["variants"] = variants
    results["calibration"] = cal_diag

    # ---- prevalence table --------------------------------------------------
    def prev(f, rows=None):
        f = f if rows is None else f[rows.to_numpy()]
        return {m: round(float(f[m].mean()), 4) for m in mech}

    hum = frame[human_mask]
    prevalence = {
        "unit": "share of rider rows (n = 522 unless stated); plug-in rows are mean "
                "probabilities",
        "final_labels_primary": prev(frame),
        "llm_reference_labels": prev(with_cues(frame, reference[mech])),
        "vote_share_majority_labels": prev(with_cues(frame, majority[mech])),
        "raw_vote_share_mean": prev(with_cues(frame, votes[mech])),
        "calibrated_all_rows_mean": prev(with_cues(frame, cal[mech])),
        "calibrated_hybrid_mean": prev(with_cues(frame, hybrid_cal[mech])),
        "draws_calibrated_mean": variants["iv_draws_calibrated"][
            "mean_prevalence_over_draws"],
        "draws_outcome_informed_mean": variants["iv_b_draws_outcome_informed"][
            "mean_prevalence_over_draws"],
        "human_subset_human_labels_n187": prev(hum),
        "human_subset_route_b_labels_n187": prev(with_cues(hum, reference[mech])),
        "automated_subset_route_b_labels_n335": prev(frame, auto_mask),
    }
    results["prevalence"] = prevalence

    # ---- summary -----------------------------------------------------------
    full = {k: v["cue_block_lr"]["p"] for k, v in variants.items()
            if "cue_block_lr" in v}
    full["iv_draws_calibrated_D1"] = variants["iv_draws_calibrated"][
        "cue_block_test_D1_wald"]["p"]
    full["iv_b_draws_outcome_informed_D1"] = variants["iv_b_draws_outcome_informed"][
        "cue_block_test_D1_wald"]["p"]
    vr = variants["v_human_subset_reduced"]
    ka_excl = {k: [c for c in cues if v["ame"][c]["KA"]["excludes_zero"]]
               for k, v in variants.items() if "ame" in v}
    ka_excl.update({k: [c for c in cues if v["ame_pooled"][c]["KA"]["excludes_zero"]]
                    for k, v in variants.items() if "ame_pooled" in v})
    results["summary"] = {
        "cue_block_p_full_sample": full,
        "cue_block_p_range_full_sample": [round(min(full.values()), 6),
                                          round(max(full.values()), 6)],
        "cue_block_p_range_printed": [round(min(full.values()), 2),
                                      round(max(full.values()), 2)],
        "any_full_sample_variant_rejects_at_5pct": bool(min(full.values()) < 0.05),
        "human_subset_admitted_cues": vr["admitted_cues"],
        "human_subset_p_admitted_cues": (vr["human_labels_admitted_cues"]["cue_block_lr"]
                                         ["p"] if vr["human_labels_admitted_cues"]
                                         else None),
        "human_subset_p_all_five_cues": vr["human_labels_all_five_cues"]["cue_block_lr"]["p"],
        "cues_with_KA_ame_interval_excluding_zero": ka_excl,
        "runtime_seconds": round(time.time() - t0, 1),
    }
    OUT.write_text(json.dumps(results, indent=2, default=float), encoding="utf-8")
    return results


if __name__ == "__main__":
    res = main()
    s = res["summary"]
    print()
    print("cue-block p by variant:")
    for k, v in s["cue_block_p_full_sample"].items():
        print(f"  {k:<40} {v}")
    print(f"human subset: admitted {s['human_subset_admitted_cues']}, "
          f"p {s['human_subset_p_admitted_cues']}; all five p "
          f"{s['human_subset_p_all_five_cues']}")
    print(f"KA intervals excluding zero: {s['cues_with_KA_ame_interval_excluding_zero']}")
    print(f"written to {OUT}")
