"""Appendix specifications for the severity models.

Two things `prespec.md` asks for beside the primary:

* section 2 — the four-class ordering (O, C, B, KA) estimated in the primary ordinal
  form. The five-class KABCO ordering is deliberately not estimated; the reason is the
  fatality count, which is reported in the text with its number rather than modelled.
* section 7 — the multinomial logit as a robustness specification, with the
  Hausman-McFadden and Small-Hsiao tests of the independence of irrelevant
  alternatives.

Everything here reuses the estimation code in `severity.py`, whose likelihood is
verified against the ordered-logit special case in `test_severity.py`. Nothing in this
module re-implements a fit.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy import stats

import severity as sv

SEV4_ORDER = ("O", "C", "B", "KA")


# --------------------------------------------------------------------------
# The four-class ordering (prespec section 2)
# --------------------------------------------------------------------------

def four_class_model(frame: pd.DataFrame, columns: list[str]) -> dict:
    """M1 re-estimated on O / C / B / KA in the primary ordinal form.

    Splitting the middle class into C and B adds a cutpoint and spreads the same 522
    records across four cells instead of three, so the events-per-parameter ratio falls
    further. It is reported for that reason as much as any other: a reader comparing the
    two should be able to see what the finer ordering costs.
    """
    fit = sv.fit_ordered_logit(frame, columns, "M1_four_class",
                               outcome="sev4", order=SEV4_ORDER)
    null_ll = sv.null_loglikelihood(frame, outcome="sev4", order=SEV4_ORDER)
    return {
        "order": list(SEV4_ORDER),
        "fit": sv.fit_metrics(fit, null_ll),
        "events_per_parameter": sv.events_per_parameter(
            frame, columns, outcome="sev4", order=SEV4_ORDER),
        "separation": sv.separation_diagnostics(fit),
        "average_marginal_effects": sv.average_marginal_effects(
            fit, frame, outcome="sev4", order=SEV4_ORDER),
        "cutpoints": [round(float(c), 4) for c in fit.cut],
    }


# --------------------------------------------------------------------------
# Multinomial logit and the independence of irrelevant alternatives (section 7)
# --------------------------------------------------------------------------

def _mnl_probs(X: np.ndarray, params: np.ndarray) -> np.ndarray:
    """Outcome probabilities for a multinomial logit with the first outcome as base."""
    eta = np.column_stack([np.zeros(len(X)), X @ params])
    eta -= eta.max(axis=1, keepdims=True)
    p = np.exp(eta)
    return p / p.sum(axis=1, keepdims=True)


def small_hsiao(Xc: np.ndarray, y: np.ndarray, drop: int, remaining: list[int],
                recode: dict, replications: int = 25,
                seed: int = 20260914) -> dict:
    """Small-Hsiao test of independence, over many random splits rather than one.

    The test partitions the sample at random, so a single run reports the split as much
    as the data. On this frame that is not a quibble: across twelve splits the p-value
    for removing BC ranged from 0.0000 to 0.9998 with a median of 0.88, so whichever
    conclusion an author wanted was available by choosing a seed. Reporting one seed
    would have put a spurious rejection in the appendix.

    What is reported instead is the distribution: the proportion of splits that reject,
    and the range. A test that rejects in two splits out of twenty-five is not evidence
    against independence; it is evidence the test has little to say at this sample size.
    """
    import statsmodels.api as sm

    rng = np.random.default_rng(seed)
    ps: list[float] = []
    stats_: list[float] = []
    failures = 0

    for _ in range(replications):
        try:
            half = rng.random(len(y)) < 0.5
            a = sm.MNLogit(y[half], Xc[half]).fit(method="bfgs", maxiter=3000,
                                                  disp=False)
            b = sm.MNLogit(y[~half], Xc[~half]).fit(method="bfgs", maxiter=3000,
                                                    disp=False)
            w = 1 / np.sqrt(2)
            pooled = w * np.asarray(a.params) + (1 - w) * np.asarray(b.params)

            sub = (~half) & (y != drop)
            if sub.sum() < Xc.shape[1] + 5:
                failures += 1
                continue

            keep_cols = [o for o in range(len(sv.SEV_ORDER)) if o in remaining]
            p_pooled = _mnl_probs(Xc[sub], pooled)[:, keep_cols]
            p_pooled = p_pooled / p_pooled.sum(axis=1, keepdims=True)
            cols = [recode[int(v)] for v in y[sub]]
            ll_pooled = float(np.log(np.clip(
                p_pooled[np.arange(int(sub.sum())), cols], 1e-12, 1.0)).sum())

            r2 = sm.MNLogit(np.array(cols), Xc[sub]).fit(method="bfgs", maxiter=3000,
                                                         disp=False)
            stat = float(-2 * (ll_pooled - r2.llf))
            if stat < 0:
                failures += 1
                continue
            pv = float(1 - stats.chi2.cdf(stat, df=Xc.shape[1]))
            if not np.isfinite(stat) or not np.isfinite(pv):
                failures += 1
                continue
            stats_.append(stat)
            ps.append(pv)
        except Exception:                                         # noqa: BLE001
            failures += 1

    if not ps:
        return {"replications": replications, "usable": 0,
                "note": "no split produced a usable statistic"}

    rejects = sum(1 for v in ps if v < 0.05)
    return {
        "replications": replications,
        "usable": len(ps),
        "failed_or_negative": failures,
        "df": int(Xc.shape[1]),
        "reject_fraction_at_5pct": round(rejects / len(ps), 3),
        "n_splits_rejecting": rejects,
        "p_median": round(float(np.median(ps)), 4),
        "p_min": round(float(np.min(ps)), 4),
        "p_max": round(float(np.max(ps)), 4),
        "chi2_median": round(float(np.median(stats_)), 3),
        "conclusion": ("independence rejected in a majority of splits"
                       if rejects > len(ps) / 2 else
                       "independence not rejected in a majority of splits"),
    }


def small_hsiao_calibration(n: int, n_covariates: int,
                            class_shares: tuple[float, ...],
                            replications: int = 20,
                            seed: int = 20260914) -> dict:
    """Is the Small-Hsiao test valid at this sample size and this many covariates?

    The test is asymptotic. Whether the asymptotics have arrived is an empirical
    question, and it can be answered directly: generate data *from* a multinomial logit,
    so independence holds by construction, at the same shape as the real frame, and see
    how often the test rejects. A valid test rejects about 5 percent of the time. If it
    rejects far more, the statistic on the real data carries no information about
    independence, only about dimension, and must not be read as evidence.

    This matters here. The multinomial logit on the real frame carries 68 parameters, and
    the test fits it on random halves of about 261 rows each. That ratio is the thing
    under suspicion, so it is the thing measured.
    """
    import statsmodels.api as sm

    rng = np.random.default_rng(seed)
    X = (rng.random((n, n_covariates)) < 0.3).astype(float)
    B = rng.normal(size=(n_covariates, len(class_shares) - 1)) * 0.3
    intercepts = np.log(np.asarray(class_shares) / class_shares[0])
    eta = np.column_stack([np.zeros(n), X @ B]) + intercepts
    pr = np.exp(eta)
    pr /= pr.sum(axis=1, keepdims=True)
    y = np.array([rng.choice(len(class_shares), p=row) for row in pr])
    Xc = sm.add_constant(X, has_constant="add")

    out: dict[str, dict] = {}
    for drop in range(len(class_shares)):
        remaining = sorted(int(v) for v in np.unique(y[y != drop]))
        if len(remaining) < 2:
            continue
        recode = {v: i for i, v in enumerate(remaining)}
        r = small_hsiao(Xc, y, drop, remaining, recode,
                        replications=replications, seed=seed + drop)
        out[f"drop_{drop}"] = {
            "reject_fraction": r.get("reject_fraction_at_5pct"),
            "n_splits_rejecting": r.get("n_splits_rejecting"),
            "usable": r.get("usable"),
        }

    fractions = [v["reject_fraction"] for v in out.values()
                 if v.get("reject_fraction") is not None]
    worst = max(fractions) if fractions else None
    return {
        "n": n,
        "n_covariates": n_covariates,
        "multinomial_parameters": (n_covariates + 1) * (len(class_shares) - 1),
        "class_shares": list(class_shares),
        "independence_holds_by_construction": True,
        "per_outcome": out,
        "worst_reject_fraction": worst,
        "nominal_level": 0.05,
        "test_is_usable": bool(worst is not None and worst <= 0.15),
        "interpretation": (
            "A nominal 5 percent test should reject in about 1 run in 20 on data where "
            "independence holds by construction. A materially higher rate means the "
            "asymptotics have not arrived at this sample size and covariate count, and "
            "the statistic computed on the real frame measures dimension rather than "
            "independence."
        ),
    }


def multinomial_robustness(frame: pd.DataFrame, columns: list[str],
                           seed: int = 20260914,
                           sh_replications: int = 25) -> dict:
    """Multinomial logit, with both pre-specified tests of independence.

    Each test removes one outcome and asks whether the coefficients that remain move
    more than sampling error allows. They approach it differently: Hausman-McFadden
    compares the full and restricted estimates directly, while Small-Hsiao splits the
    sample and compares log-likelihoods, which avoids the covariance difference that
    makes the first test awkward.

    That awkwardness is worth stating plainly rather than papering over. The
    Hausman-McFadden statistic is a quadratic form in the difference of two covariance
    matrices, and that difference need not be positive definite in a finite sample, so
    the statistic can come out negative. This is a known property of the test, not a
    numerical failure and not something to clip to zero. McFadden's own reading is that
    a negative value is evidence the assumption has not been violated. It is reported as
    computed, with a flag.
    """
    import statsmodels.api as sm

    X, y = sv._design(frame, columns)
    Xc = sm.add_constant(X, has_constant="add")
    J = len(sv.SEV_ORDER)
    rng = np.random.default_rng(seed)

    full = sm.MNLogit(y, Xc).fit(method="bfgs", maxiter=3000, disp=False)
    n_par = int(np.asarray(full.params).size)

    per_outcome: dict[str, dict] = {}
    for drop in range(J):
        label = sv.SEV_ORDER[drop]
        keep = y != drop
        remaining = sorted(int(v) for v in np.unique(y[keep]))
        if len(remaining) < 2:
            per_outcome[label] = {"skipped": "fewer than two outcomes remain"}
            continue

        recode = {v: i for i, v in enumerate(remaining)}
        y_r = np.array([recode[int(v)] for v in y[keep]])

        entry: dict = {"outcome_removed": label,
                       "n_retained": int(keep.sum())}
        try:
            restricted = sm.MNLogit(y_r, Xc[keep]).fit(method="bfgs", maxiter=3000,
                                                       disp=False)
        except Exception as exc:                                  # noqa: BLE001
            per_outcome[label] = {**entry, "error": type(exc).__name__}
            continue

        # The full model has J-1 coefficient blocks against its own base outcome. Only
        # the blocks for outcomes that survive, and that are not the restricted model's
        # new base, are comparable.
        full_blocks = [o for o in range(J) if o != 0]
        shared = [i for i, o in enumerate(full_blocks)
                  if o in remaining and o != remaining[0]]

        if not shared:
            entry["hausman_mcfadden_skipped"] = (
                "no coefficient block is shared once the base outcome is re-chosen")
        else:
            b_f = np.asarray(full.params)[:, shared].ravel(order="F")
            b_r = np.asarray(restricted.params).ravel(order="F")
            n = min(len(b_f), len(b_r))
            diff = b_f[:n] - b_r[:n]
            V_f = np.asarray(full.cov_params())[:n, :n]
            V_r = np.asarray(restricted.cov_params())[:n, :n]
            dV = V_r - V_f
            try:
                hm = float(diff @ np.linalg.pinv(dV) @ diff)
            except Exception:                                     # noqa: BLE001
                hm = float("nan")
            entry["hausman_mcfadden_chi2"] = round(hm, 4) if hm == hm else None
            entry["hausman_mcfadden_df"] = int(n)
            entry["hausman_mcfadden_negative"] = bool(hm == hm and hm < 0)
            entry["hausman_mcfadden_p"] = (
                round(float(1 - stats.chi2.cdf(hm, df=n)), 4)
                if hm == hm and hm >= 0 else None)

        entry["small_hsiao"] = small_hsiao(Xc, y, drop, remaining, recode,
                                           replications=sh_replications, seed=seed)

        per_outcome[label] = entry

    calibration = small_hsiao_calibration(
        n=int(len(y)), n_covariates=len(columns),
        class_shares=tuple(
            float((y == j).mean()) for j in range(J)), seed=seed)

    rejected = sorted(
        k for k, v in per_outcome.items()
        if (v.get("hausman_mcfadden_p") is not None
            and v["hausman_mcfadden_p"] < 0.05)
        or (calibration["test_is_usable"]
            and v.get("small_hsiao", {}).get("reject_fraction_at_5pct", 0) > 0.5))

    return {
        "multinomial_fit": {
            "loglik": round(float(full.llf), 4),
            "n_parameters": n_par,
            "n": int(len(y)),
            "converged": bool(full.mle_retvals.get("converged", True)),
        },
        "iia_tests": per_outcome,
        "small_hsiao_calibration": calibration,
        "outcomes_where_independence_rejected": rejected,
        "seed": seed,
        "note": (
            "A negative Hausman-McFadden statistic is a known finite-sample property of "
            "the test rather than an error, and is reported rather than clipped; "
            "McFadden reads it as evidence that independence has not been violated. "
            "The Small-Hsiao statistic is reported as a distribution over random "
            "splits rather than as a single value, because on this frame one split "
            "gives p = 0.0000 and another p = 0.9998. It is then read only if the "
            "calibration check shows the test holds its nominal level at this sample "
            "size and covariate count; see small_hsiao_calibration. "
            "The multinomial logit is a robustness specification only. It discards the "
            "ordering of the outcome, which is the feature the primary model is chosen "
            "to respect, so it is not a candidate for primary however it compares on "
            "fit."
        ),
    }


if __name__ == "__main__":
    frame = pd.read_csv(sv.DATA / "model_frame.csv")
    cols = sv.ladder(frame)["M1"]

    print("Appendix specifications. Preliminary: the spatial layer is not attached and")
    print("the labels are the fused ones, not the coders' adjudicated ones.")
    print()

    four = four_class_model(frame, cols)
    epp = four["events_per_parameter"]
    print(f"four-class ordering {' < '.join(SEV4_ORDER)}")
    print(f"  class counts {epp['class_counts']}")
    print(f"  {epp['n_parameters']} parameters, "
          f"{epp['events_per_parameter_smallest_class']} events per parameter in the "
          f"smallest class")
    print(f"  log-likelihood {four['fit']['loglik']}, "
          f"adjusted rho-squared {four['fit']['adjusted_rho2']}, "
          f"separation flag {four['separation']['suggests_separation']}")

    print()
    print("multinomial logit, independence of irrelevant alternatives")
    mnl = multinomial_robustness(frame, cols)
    print(f"  multinomial log-likelihood {mnl['multinomial_fit']['loglik']} on "
          f"{mnl['multinomial_fit']['n_parameters']} parameters")
    for label, v in mnl["iia_tests"].items():
        hm = v.get("hausman_mcfadden_chi2")
        neg = " (negative)" if v.get("hausman_mcfadden_negative") else ""
        sh = v.get("small_hsiao", {})
        print(f"  removing {label:<3} Hausman-McFadden {hm}{neg} "
              f"p {v.get('hausman_mcfadden_p')}")
        print(f"          Small-Hsiao rejects in "
              f"{sh.get('n_splits_rejecting')} of {sh.get('usable')} splits, "
              f"median p {sh.get('p_median')}, "
              f"range {sh.get('p_min')} to {sh.get('p_max')}")
    cal = mnl["small_hsiao_calibration"]
    print()
    print("  Small-Hsiao calibration, on data where independence holds by construction")
    print(f"    same shape as the real frame: n {cal['n']}, {cal['n_covariates']} "
          f"covariates, {cal['multinomial_parameters']} multinomial parameters")
    for k, v in cal["per_outcome"].items():
        print(f"    {k}: rejects {v['n_splits_rejecting']} of {v['usable']} splits "
              f"at a nominal 5 percent")
    print(f"    worst rejection rate {cal['worst_reject_fraction']} against a nominal "
          f"0.05 -> test usable: {cal['test_is_usable']}")
    if not cal["test_is_usable"]:
        print("    The Small-Hsiao result on the real frame is therefore NOT read as "
              "evidence about independence.")
    print()
    print(f"  independence rejected for: "
          f"{mnl['outcomes_where_independence_rejected'] or 'no outcome'}")

    out = {"four_class": four, "multinomial_robustness": mnl}
    path = sv.DATA / "severity_appendix.json"
    path.write_text(json.dumps(out, indent=2, default=float), encoding="utf-8")
    print()
    print(f"written to {path} (preliminary)")
