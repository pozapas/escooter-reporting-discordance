"""M2: the primary severity model refitted on the audit's latent class.

M1 is the ordered logit of `severity.py` (BIC did not confirm the partial
proportional-odds model, so ordered logit is primary). M2 keeps its covariates and
estimator and replaces the coded outcome by the latent severity class of the audit.

## The expected-likelihood formulation

Each rider contributes three weighted copies, one per latent class c, and the weight is
the posterior probability of that class given the rider's record:

    w_ic = pi_c p_K(k_i | c) p_N(n_i | c) / sum_c' pi_c' p_K(k_i | c') p_N(n_i | c')

    l_M2(beta, tau) = sum_i sum_c w_ic log P(Y = c | x_i; beta, tau)

so the weights of a rider sum to one, and a rider whose record leaves no doubt
contributes one copy at full weight. `audit_final.py` saves w for the posterior mean and
for 100 posterior draws. M2 is refitted once per draw. The within-draw variance is a
rider-clustered sandwich (the three copies of a rider are one observation, and the
inverse of the weighted Hessian would treat them as three), and the draws are combined
with Rubin's rules, so the spread of the audit posterior enters the standard errors.

The weights are a function of the coded KABCO level, which is also what M1 models, and
of the narrative indicator. Nothing here removes that dependence; it is reported, not
corrected.

## Two alternative weight definitions (the appendix comparison)

(i)  coded outcome, weight = posterior probability of the coded class, w_i = w_i,sev3_i;
(ii) modal reassignment, outcome = argmax_c w_ic, unit weight.

Both use the posterior-mean weights. Their standard errors are rider sandwiches too, and
M1 is reported with the same sandwich beside its published model-based errors, so that a
change in which intervals exclude zero cannot come from the type of standard error.
"""

from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import optimize, special, stats

import severity as sv

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

SEV = sv.SEV_ORDER
Z95 = 1.959964


# --------------------------------------------------------------------------
# Weighted ordered logit, same parameterization as severity.fit_ordered_logit
# --------------------------------------------------------------------------

def _negll_and_scores(theta: np.ndarray, X: np.ndarray, y: np.ndarray,
                      w: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """Weighted negative log-likelihood, its gradient, and per-row scores.

    theta = [beta, first cutpoint, log increments], the raw vector of `OrdinalFit`, so a
    fit from here can be handed to `severity.average_marginal_effects` unchanged.
    """
    k = X.shape[1]
    beta, r = theta[:k], theta[k:]
    cut = sv._cut_from_raw(r)
    J = len(cut) + 1
    eta = X @ beta
    ext = np.concatenate([[-np.inf], cut, [np.inf]])
    a = ext[y + 1] - eta
    b = ext[y] - eta
    Fa, Fb = special.expit(a), special.expit(b)
    fa, fb = Fa * (1 - Fa), Fb * (1 - Fb)
    p = np.clip(Fa - Fb, 1e-300, None)
    ll = float(np.sum(w * np.log(p)))

    s_beta = X * ((fb - fa) / p)[:, None]
    dc = np.zeros((len(y), J - 1))
    up = y < J - 1
    dc[np.where(up)[0], y[up]] += fa[up] / p[up]
    lo = y > 0
    dc[np.where(lo)[0], y[lo] - 1] -= fb[lo] / p[lo]
    s_r = np.empty((len(y), len(r)))
    s_r[:, 0] = dc.sum(axis=1)
    for m in range(1, len(r)):
        s_r[:, m] = np.exp(r[m]) * dc[:, m:].sum(axis=1)
    scores = np.column_stack([s_beta, s_r])
    return -ll, -(w[:, None] * scores).sum(axis=0), scores


def _hessian(theta, X, y, w, eps=1e-5) -> np.ndarray:
    """Hessian of the weighted negative log-likelihood by differencing the gradient."""
    n = len(theta)
    H = np.empty((n, n))
    for i in range(n):
        e = np.zeros(n)
        e[i] = eps
        H[:, i] = (_negll_and_scores(theta + e, X, y, w)[1]
                   - _negll_and_scores(theta - e, X, y, w)[1]) / (2 * eps)
    return (H + H.T) / 2


def fit_weighted(X: np.ndarray, y: np.ndarray, w: np.ndarray, cluster: np.ndarray,
                 start: np.ndarray, columns: list[str], name: str) -> sv.OrdinalFit:
    """Weighted ordered logit with model-based and cluster-sandwich covariances.

    `cov` on the returned fit is the sandwich; the inverse Hessian is kept in extras.
    """
    k = X.shape[1]
    res = optimize.minimize(lambda t: _negll_and_scores(t, X, y, w)[:2], start,
                            jac=True, method="BFGS",
                            options={"maxiter": 10000, "gtol": 1e-7})
    theta = res.x
    negll, grad, scores = _negll_and_scores(theta, X, y, w)
    H = _hessian(theta, X, y, w)
    Hinv = np.linalg.pinv(H)
    U = pd.DataFrame(w[:, None] * scores).groupby(cluster).sum().to_numpy()
    sandwich = Hinv @ (U.T @ U) @ Hinv
    grad_max = float(np.max(np.abs(grad)))
    return sv.OrdinalFit(
        name=name, columns=list(columns), beta=theta[:k],
        cut=sv._cut_from_raw(theta[k:]), raw=theta, cov=sandwich,
        loglik=float(-negll), n=int(len(np.unique(cluster))),
        converged=bool(res.success or grad_max < 1e-4),
        extras={"cov_model_based": Hinv, "max_abs_gradient": grad_max,
                "optimizer_message": str(res.message), "n_iter": int(res.nit),
                "sum_of_weights": float(w.sum())},
    )


def stacked(X: np.ndarray, probs: np.ndarray):
    """Three copies of every rider, one per latent class, weighted by its probability."""
    n = len(X)
    Xs = np.tile(X, (3, 1))
    ys = np.repeat(np.arange(3), n)
    ws = probs.T.reshape(-1)
    cl = np.tile(np.arange(n), 3)
    return Xs, ys, ws, cl


# --------------------------------------------------------------------------
# Rubin's rules
# --------------------------------------------------------------------------

def rubin(thetas: np.ndarray, covs: np.ndarray, n_obs: int) -> dict:
    """Pool m estimates. Barnard-Rubin degrees of freedom, per parameter."""
    m, p = thetas.shape
    qbar = thetas.mean(axis=0)
    ubar = covs.mean(axis=0)
    bmat = np.cov(thetas, rowvar=False, ddof=1)
    T = ubar + (1 + 1 / m) * bmat
    b, u, t = np.diag(bmat), np.diag(ubar), np.diag(T)
    lam = np.clip((1 + 1 / m) * b / t, 1e-12, 1.0)
    r = (1 + 1 / m) * b / u
    nu_old = (m - 1) / lam ** 2
    nu_com = n_obs - p
    nu_obs = (nu_com + 1) / (nu_com + 3) * nu_com * (1 - lam)
    nu = 1 / (1 / nu_old + 1 / nu_obs)
    fmi = (r + 2 / (nu + 3)) / (1 + r)
    return {"qbar": qbar, "T": T, "ubar": ubar, "B": bmat, "lambda": lam,
            "fmi": fmi, "df": nu, "m": m}


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------

def coef_block(columns: list[str], beta: np.ndarray, se: np.ndarray,
               df: np.ndarray | None = None, extra: dict | None = None) -> dict:
    out = {}
    for i, c in enumerate(columns):
        q = stats.t.ppf(0.975, df[i]) if df is not None else Z95
        lo, hi = beta[i] - q * se[i], beta[i] + q * se[i]
        row = {"coef": round(float(beta[i]), 3), "se": round(float(se[i]), 3),
               "ci95": [round(float(lo), 3), round(float(hi), 3)],
               "excludes_zero": bool(lo > 0 or hi < 0)}
        if extra:
            for key, arr in extra.items():
                row[key] = round(float(arr[i]), 3)
        out[c] = row
    return out




def excluding_sets(coefs: dict, ames: dict) -> dict:
    return {
        "coefficients": sorted(c for c, v in coefs.items() if v["excludes_zero"]),
        "ame": {lvl: sorted(c for c, v in ames.items() if v[lvl]["excludes_zero"])
                for lvl in SEV},
    }


def compare_to_m1(m1_coefs: dict, m1_sets: dict, m1_ames: dict, coefs: dict,
                  sets: dict, ames: dict) -> dict:
    gained_c = sorted(set(sets["coefficients"]) - set(m1_sets["coefficients"]))
    lost_c = sorted(set(m1_sets["coefficients"]) - set(sets["coefficients"]))
    flips = sorted(c for c in coefs
                   if np.sign(coefs[c]["coef"]) != np.sign(m1_coefs[c]["coef"])
                   and (coefs[c]["excludes_zero"] or m1_coefs[c]["excludes_zero"]))
    ame_changes = {}
    for lvl in SEV:
        g = sorted(set(sets["ame"][lvl]) - set(m1_sets["ame"][lvl]))
        lo = sorted(set(m1_sets["ame"][lvl]) - set(sets["ame"][lvl]))
        ame_changes[lvl] = {"newly_excluding_zero": g, "no_longer_excluding_zero": lo}
    ame_sign_flips = sorted(
        f"{c}:{lvl}" for c in ames for lvl in SEV
        if np.sign(ames[c][lvl]["ame"]) != np.sign(m1_ames[c][lvl]["ame"])
        and (ames[c][lvl]["excludes_zero"] or m1_ames[c][lvl]["excludes_zero"]))
    return {"coefficients_newly_excluding_zero": gained_c,
            "coefficients_no_longer_excluding_zero": lost_c,
            "sign_flips_among_excluding": flips,
            "ame": ame_changes, "ame_sign_flips_among_excluding": ame_sign_flips,
            "max_abs_coef_change": round(float(max(
                abs(coefs[c]["coef"] - m1_coefs[c]["coef"]) for c in coefs)), 3)}


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main() -> dict:
    t_start = time.perf_counter()
    frame = pd.read_csv(DATA / "model_frame.csv")
    numbers = json.loads((DATA / "numbers.json").read_text(encoding="utf-8"))
    cp = np.load(DATA / "audit_posterior_classprobs.npz")
    if not (cp["rider_row_id"] == frame["rider_row_id"].astype(str).to_numpy()).all():
        raise SystemExit("class-probability file is not in model_frame row order")
    if not (cp["sev3"] == frame["sev3"].astype(str).to_numpy()).all():
        raise SystemExit("class-probability file disagrees with the frame on sev3")

    cols = sv.ladder(frame)["M1"]
    X, y = sv._design(frame, cols)
    n, k = X.shape
    riders = np.arange(n)
    ones = np.ones(n)

    # M1 exactly as severity.py fits it, then checked against numbers.json.
    m1 = sv.fit_ordered_logit(frame, cols, "M1")
    m1_pub = numbers["severity"]["rungs"]["M1"]
    m1_tab = sv.coefficient_table(m1)["slopes"]
    m1_check = {
        "loglik": round(m1.loglik, 4), "loglik_numbers_json": m1_pub["loglik"],
        "max_abs_coef_gap_vs_numbers_json": round(max(
            abs(m1_tab[c]["coef"] - m1_pub["coefficients"]["slopes"][c]["coef"])
            for c in cols), 4),
        "max_abs_se_gap_vs_numbers_json": round(max(
            abs(m1_tab[c]["se"] - m1_pub["coefficients"]["slopes"][c]["se"])
            for c in cols), 4),
    }

    # The weighted likelihood must reproduce M1 when every weight is one on the coded class.
    onehot = np.eye(3)[y]
    Xs, ys, ws, cl = stacked(X, onehot)
    check = fit_weighted(Xs, ys, ws, cl, m1.raw.copy(), cols, "check")
    # statsmodels stops BFGS on the per-observation gradient, so the M1 optimum can sit
    # about 1e-3 from the maximum on a weakly identified coefficient while this fit runs
    # to gtol 1e-7 on the summed gradient. The comparison is made against the statsmodels
    # fit continued from the M1 optimum with a tight tolerance, so the gate tests the
    # likelihood rather than two stopping rules; the thresholds are unchanged and the
    # gaps against the unpolished M1 are recorded beside them.
    from statsmodels.miscmodels.ordinal_model import OrderedModel
    ref = OrderedModel(y, X, distr="logit").fit(start_params=m1.raw.copy(), method="bfgs",
                                                 maxiter=5000, gtol=1e-9, disp=False)
    ref_raw = np.asarray(ref.params, float)
    ref_cut = sv._cut_from_raw(ref_raw[k:])
    verify = {
        "reference": ("statsmodels OrderedModel continued from the severity.py M1 optimum "
                      "with BFGS gtol 1e-9"),
        "loglik_gap": float(abs(check.loglik - float(ref.llf))),
        "max_abs_beta_gap": float(np.max(np.abs(check.beta - ref_raw[:k]))),
        "max_abs_cut_gap": float(np.max(np.abs(check.cut - ref_cut))),
        "loglik_gap_vs_unpolished_m1": float(abs(check.loglik - m1.loglik)),
        "max_abs_beta_gap_vs_unpolished_m1": float(np.max(np.abs(check.beta - m1.beta))),
        "max_abs_cut_gap_vs_unpolished_m1": float(np.max(np.abs(check.cut - m1.cut))),
        "max_abs_model_se_gap": float(np.max(np.abs(
            np.sqrt(np.diag(check.extras["cov_model_based"]))[:k]
            - np.sqrt(np.diag(m1.cov))[:k]))),
    }
    verify["passes"] = bool(verify["loglik_gap"] < 1e-4 and verify["max_abs_beta_gap"] < 1e-3
                            and verify["max_abs_cut_gap"] < 1e-3)
    print("weighted likelihood reproduces M1:", verify)
    if not verify["passes"]:
        raise SystemExit("weighted ordered logit does not reproduce M1; stop")

    # M1 with the rider sandwich, for like-for-like comparison.
    m1_sw = fit_weighted(X, y, ones, riders, m1.raw.copy(), cols, "M1_sandwich")
    se_m1_model = np.sqrt(np.diag(m1.cov))[:k]
    se_m1_sw = np.sqrt(np.diag(m1_sw.cov))[:k]
    m1_coefs_model = coef_block(cols, m1.beta, se_m1_model)
    m1_coefs_sw = coef_block(cols, m1_sw.beta, se_m1_sw)

    t0 = time.perf_counter()
    m1_ames_model = sv.average_marginal_effects(m1, frame)
    ame_secs = time.perf_counter() - t0
    m1_ames_sw = sv.average_marginal_effects(m1_sw, frame)
    pub_ames = numbers["severity"]["ames_all"]
    m1_ame_check = round(max(abs(m1_ames_model[c][l]["ame"] - pub_ames[c][l]["ame"])
                             for c in cols for l in SEV), 6)
    m1_ame_ci_check = round(max(abs(m1_ames_model[c][l]["ci95"][i] - pub_ames[c][l]["ci95"][i])
                                for c in cols for l in SEV for i in (0, 1)), 6)
    m1_sets_model = excluding_sets(m1_coefs_model, m1_ames_model)
    m1_sets_sw = excluding_sets(m1_coefs_sw, m1_ames_sw)

    # Expected likelihood over the 100 posterior draws, Rubin-combined.
    def expected_likelihood(draws: np.ndarray, label: str, with_ames: bool) -> dict:
        thetas, covs, covs_model, lls, conv, grads = [], [], [], [], [], []
        t1 = time.perf_counter()
        for d in range(draws.shape[0]):
            Xs, ys, ws, cl = stacked(X, draws[d])
            f = fit_weighted(Xs, ys, ws, cl, m1.raw.copy(), cols, f"{label}_{d}")
            thetas.append(f.raw)
            covs.append(f.cov)
            covs_model.append(f.extras["cov_model_based"])
            lls.append(f.loglik)
            conv.append(f.converged)
            grads.append(f.extras["max_abs_gradient"])
        fit_secs = time.perf_counter() - t1
        thetas, covs, covs_model = map(np.asarray, (thetas, covs, covs_model))
        pool = rubin(thetas, covs, n)
        pool_model = rubin(thetas, covs_model, n)
        se = np.sqrt(np.diag(pool["T"]))[:k]
        coefs = coef_block(cols, pool["qbar"][:k], se, df=pool["df"][:k],
                           extra={"fraction_missing_information": pool["fmi"][:k],
                                  "between_draw_share_of_variance": pool["lambda"][:k],
                                  "se_within_only": np.sqrt(np.diag(pool["ubar"]))[:k]})
        coefs_model = coef_block(cols, pool_model["qbar"][:k],
                                 np.sqrt(np.diag(pool_model["T"]))[:k],
                                 df=pool_model["df"][:k])
        res = {
            "n_draws": int(draws.shape[0]),
            "all_converged": bool(all(conv)),
            "n_converged": int(sum(conv)),
            "max_abs_gradient_worst_draw": float(max(grads)),
            "expected_loglik_mean_over_draws": round(float(np.mean(lls)), 3),
            "fit_seconds": round(fit_secs, 1),
            "coefficients": coefs,
            "coefficients_model_based_within": coefs_model,
            "cutpoints_pooled": [round(float(c), 3)
                                 for c in sv._cut_from_raw(pool["qbar"][k:])],
            "fmi_summary": {"median": round(float(np.median(pool["fmi"][:k])), 3),
                            "max": round(float(np.max(pool["fmi"][:k])), 3),
                            "argmax": cols[int(np.argmax(pool["fmi"][:k]))]},
        }
        if with_ames:
            pooled = sv.OrdinalFit(name=label, columns=list(cols), beta=pool["qbar"][:k],
                                   cut=sv._cut_from_raw(pool["qbar"][k:]),
                                   raw=pool["qbar"], cov=pool["T"], loglik=float(np.mean(lls)),
                                   n=n, converged=bool(all(conv)))
            res["ames"] = sv.average_marginal_effects(pooled, frame)
            res["excluding_zero"] = excluding_sets(coefs, res["ames"])
        else:
            res["excluding_zero"] = {"coefficients": sorted(
                c for c, v in coefs.items() if v["excludes_zero"])}
        return res

    el = expected_likelihood(cp["moderate_draws"], "M2_EL", with_ames=True)
    print(f"EL: {el['n_converged']}/{el['n_draws']} converged, {el['fit_seconds']} s")

    # Single fit at the posterior-mean weights, as a reference point for the pooled one.
    Xs, ys, ws, cl = stacked(X, cp["moderate_mean"])
    el_mean = fit_weighted(Xs, ys, ws, cl, m1.raw.copy(), cols, "M2_EL_posterior_mean")
    el_mean_coefs = coef_block(cols, el_mean.beta, np.sqrt(np.diag(el_mean.cov))[:k])

    # Sensitivity of the EL weights to the audit specification (coefficients only).
    el_weak = expected_likelihood(cp["weak_draws"], "M2_EL_weak", with_ames=False)
    el_dep = expected_likelihood(cp["dependence_moderate_draws"], "M2_EL_dependence",
                                 with_ames=False)

    # (i) coded outcome, weight = posterior probability of the coded class.
    p_coded = cp["moderate_mean"][np.arange(n), y]
    alt1 = fit_weighted(X, y, p_coded, riders, m1.raw.copy(), cols, "M2_alt_coded_weight")
    alt1_coefs = coef_block(cols, alt1.beta, np.sqrt(np.diag(alt1.cov))[:k])
    alt1_ames = sv.average_marginal_effects(alt1, frame)
    ess = float(p_coded.sum() ** 2 / (p_coded ** 2).sum())

    # (ii) modal reassignment, unit weight.
    modal = cp["moderate_mean"].argmax(axis=1)
    mf = frame.copy()
    mf["sev3_modal"] = np.array(SEV)[modal]
    alt2_sm = sv.fit_ordered_logit(mf, cols, "M2_alt_modal", outcome="sev3_modal")
    alt2 = fit_weighted(X, modal, ones, riders, alt2_sm.raw.copy(), cols,
                        "M2_alt_modal_sandwich")
    alt2_sep = sv.separation_diagnostics(alt2_sm)
    alt2_epp = sv.events_per_parameter(mf, cols, outcome="sev3_modal")
    alt2_coefs = coef_block(cols, alt2.beta, np.sqrt(np.diag(alt2.cov))[:k])
    alt2_coefs_model = coef_block(cols, alt2_sm.beta, np.sqrt(np.diag(alt2_sm.cov))[:k])
    alt2_ames = sv.average_marginal_effects(alt2, frame)

    # Like-for-like comparison against M1 with the rider sandwich, and against published M1.
    variants = {
        "expected_likelihood": (el["coefficients"], el["excluding_zero"], el["ames"]),
        "alt_i_coded_class_weight": (alt1_coefs, excluding_sets(alt1_coefs, alt1_ames),
                                     alt1_ames),
        "alt_ii_modal_reassignment": (alt2_coefs, excluding_sets(alt2_coefs, alt2_ames),
                                      alt2_ames),
    }
    comparison = {
        name: {"vs_M1_sandwich": compare_to_m1(m1_coefs_sw, m1_sets_sw, m1_ames_sw,
                                               *v),
               "vs_M1_published": compare_to_m1(m1_coefs_model, m1_sets_model,
                                                m1_ames_model, *v)}
        for name, v in variants.items()}

    out = {
        "generated": datetime.now().isoformat(timespec="seconds"),
        "script": "src/m2_expected_likelihood.py",
        "inputs": ["data/model_frame.csv", "data/audit_posterior_classprobs.npz",
                   "data/screening.json", "data/numbers.json"],
        "settings": {
            "estimator": "ordered logit (primary M1 of severity.py), weighted maximum "
                         "likelihood in the same raw parameterization",
            "covariates": cols, "n_riders": n,
            "class_probabilities": "moderate prior, dependence fixed at 0 (audit primary)",
            "n_posterior_draws": int(cp["moderate_draws"].shape[0]),
            "within_draw_variance": "rider-clustered sandwich H^-1 (sum_i U_i U_i') H^-1",
            "pooling": "Rubin's rules on the raw parameter vector, Barnard-Rubin df",
            "ame": ("severity.average_marginal_effects, 1000 Krinsky-Robb draws from the "
                    "pooled mean and total variance (EL) or the sandwich (alternatives)"),
            "kr_draws": sv.KR_DRAWS, "kr_seed": sv.KR_SEED,
        },
        "weight_definition": {
            "expected_likelihood": ("w_ic = pi_c p_K(k_i|c) p_N(n_i|c) / sum_c' pi_c' "
                                    "p_K(k_i|c') p_N(n_i|c'); l = sum_i sum_c w_ic log "
                                    "P(Y=c|x_i)"),
            "alt_i": "outcome = coded sev3, weight = w_i,coded",
            "alt_ii": "outcome = argmax_c w_ic, weight = 1",
            "note": ("w depends on the record only through its (KABCO, narrative "
                     "indicator) cell, so there are twenty distinct weight vectors; "
                     "they are listed in audit_final.json. w is a function of the coded "
                     "KABCO level that M1 models."),
        },
        "m1_reproduction": {**m1_check, "ame_max_abs_gap_vs_numbers_json": m1_ame_check,
                            "ame_ci_max_abs_gap_vs_numbers_json": m1_ame_ci_check},
        "weighted_likelihood_verification": verify,
        "M1": {
            "coefficients_model_based": m1_coefs_model,
            "coefficients_sandwich": m1_coefs_sw,
            "ames_model_based": m1_ames_model,
            "ames_sandwich": m1_ames_sw,
            "excluding_zero_model_based": m1_sets_model,
            "excluding_zero_sandwich": m1_sets_sw,
            "loglik": round(m1.loglik, 3),
        },
        "M2_expected_likelihood": el,
        "expected_class_counts": {
            "posterior_mean_weights": {s_: round(float(cp["moderate_mean"][:, i].sum()), 1)
                                       for i, s_ in enumerate(SEV)},
            "mean_over_draws": {s_: round(float(cp["moderate_draws"][:, :, i].sum(axis=1).mean()), 1)
                                for i, s_ in enumerate(SEV)},
            "coded": {s_: int((y == i).sum()) for i, s_ in enumerate(SEV)}},
        "M2_expected_likelihood_at_posterior_mean_weights": {
            "coefficients": el_mean_coefs,
            "excluding_zero": sorted(c for c, v in el_mean_coefs.items()
                                     if v["excludes_zero"]),
            "converged": el_mean.converged},
        "M2_alt_i_coded_class_weight": {
            "coefficients": alt1_coefs, "ames": alt1_ames,
            "excluding_zero": excluding_sets(alt1_coefs, alt1_ames),
            "sum_of_weights": round(float(p_coded.sum()), 2),
            "effective_n": round(ess, 1),
            "weight_summary": {"mean": round(float(p_coded.mean()), 4),
                               "min": round(float(p_coded.min()), 4),
                               "by_coded_class": {s: round(float(p_coded[y == i].mean()), 4)
                                                  for i, s in enumerate(SEV)}},
            "converged": alt1.converged},
        "M2_alt_ii_modal_reassignment": {
            "coefficients": alt2_coefs,
            "coefficients_model_based": alt2_coefs_model,
            "ames": alt2_ames,
            "excluding_zero": excluding_sets(alt2_coefs, alt2_ames),
            "modal_counts": {s: int((modal == i).sum()) for i, s in enumerate(SEV)},
            "events_per_parameter": alt2_epp,
            "separation": alt2_sep,
            "converged": bool(alt2_sm.converged and alt2.converged),
            "loglik": round(alt2_sm.loglik, 3)},
        "M2_expected_likelihood_audit_sensitivity": {
            "weak_prior_weights": {k_: el_weak[k_] for k_ in
                                   ("n_converged", "coefficients", "excluding_zero",
                                    "fmi_summary")},
            "estimated_dependence_weights": {k_: el_dep[k_] for k_ in
                                             ("n_converged", "coefficients",
                                              "excluding_zero", "fmi_summary")},
        },
        "comparison": comparison,
        "run_seconds": {"total": None, "one_ame_call": round(ame_secs, 1),
                        "el_100_fits": el["fit_seconds"]},
    }
    out["run_seconds"]["total"] = round(time.perf_counter() - t_start, 1)
    path = DATA / "m2.json"
    path.write_text(json.dumps(out, indent=2, default=float), encoding="utf-8")
    print(f"written to {path} ({out['run_seconds']['total']} s)")
    return out


if __name__ == "__main__":
    main()
