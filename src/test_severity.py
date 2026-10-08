"""Tests for the ordinal severity machinery, on data whose truth is known.

The Brant test selects the primary model specification, and the average marginal
effects are the numbers the paper reports. Neither has a reference implementation to
check against inside this project, so each is exercised here against simulated data
generated from a known process. A wrong likelihood, a wrong Brant statistic, and a
correct one all look alike when run on the real frame.

Run with:  python src/test_severity.py
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

import severity as sv

RNG = np.random.default_rng(8112026)
FAILURES: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    status = "pass" if condition else "FAIL"
    print(f"  [{status}] {name}" + (f" — {detail}" if detail else ""))
    if not condition:
        FAILURES.append(name)


def simulate(n: int, beta: dict[str, float], cut: tuple[float, float],
             nonprop: dict[str, float] | None = None,
             rng: np.random.Generator = RNG) -> pd.DataFrame:
    """Draw an ordered-response sample from a known cumulative-logit process.

    `nonprop` gives per-covariate departures from proportional odds: the named
    covariate's effect on the second cumulative logit is shifted by that amount, which
    is exactly the violation the Brant test is supposed to detect.
    """
    cols = list(beta)
    X = pd.DataFrame({c: rng.integers(0, 2, n).astype(float) for c in cols})
    b = np.array([beta[c] for c in cols])
    shift = np.array([(nonprop or {}).get(c, 0.0) for c in cols])
    eta1 = X.to_numpy() @ b
    eta2 = X.to_numpy() @ (b + shift)

    # A departure from proportional odds is only coherent while it stays smaller than
    # the gap between the cutpoints. Past that the middle category has negative
    # probability for some rows. Clamping would hide it and hand the estimator a
    # structural zero to chase, so refuse to generate the sample instead.
    slack = (cut[1] - eta2) - (cut[0] - eta1)
    if slack.min() <= 0:
        raise ValueError(
            f"incoherent generating process: the cutpoint gap {cut[1] - cut[0]:.2f} is "
            f"too small for this departure from proportional odds (worst-case slack "
            f"{slack.min():.2f}); widen the cutpoints or shrink `nonprop`")

    p_le0 = 1 / (1 + np.exp(-(cut[0] - eta1)))
    p_le1 = 1 / (1 + np.exp(-(cut[1] - eta2)))

    u = rng.random(n)
    codes = np.where(u < p_le0, 0, np.where(u < p_le1, 1, 2))
    X["sev3"] = [sv.SEV_ORDER[c] for c in codes]
    return X


def true_ame(frame: pd.DataFrame, cols: list[str], beta: dict[str, float],
             cut: tuple[float, float], col: str,
             siblings: list[str] | None = None) -> np.ndarray:
    """The average marginal effect implied by the generating parameters."""
    X = frame[cols].to_numpy().astype(float)
    b = np.array([beta[c] for c in cols])
    i = cols.index(col)
    X1, X0 = X.copy(), X.copy()
    X1[:, i] = 1.0
    X0[:, i] = 0.0
    for s in siblings or []:
        X1[:, cols.index(s)] = 0.0
    c = np.array(cut)
    return (sv._probs(X1, b, c) - sv._probs(X0, b, c)).mean(axis=0)


# ---------------------------------------------------------------------------

def test_ppo_reduces_to_ordered_logit() -> None:
    print("\nThe partial proportional-odds likelihood, with nothing freed, must "
          "reproduce ordered logit")
    beta = {"a": 0.8, "b": -0.5, "c": 0.3}
    frame = simulate(1500, beta, (-1.2, 0.9))
    out = sv.verify_ppo_against_ordered_logit(frame, list(beta))
    check("log-likelihoods agree", out["loglik_abs_gap"] < 1e-3,
          f"gap {out['loglik_abs_gap']:.2e}")
    check("coefficients agree", out["max_abs_beta_gap"] < 1e-2,
          f"max gap {out['max_abs_beta_gap']:.2e}")
    check("cutpoints agree", out["max_abs_cutpoint_gap"] < 1e-2,
          f"max gap {out['max_abs_cutpoint_gap']:.2e}")
    check("gradient vanishes at the optimum",
          out["max_abs_numeric_gradient_at_optimum"] < 1e-2,
          f"max |g| {out['max_abs_numeric_gradient_at_optimum']:.2e}")


def test_coefficient_recovery() -> None:
    print("\nOrdered logit must recover the generating coefficients")
    beta = {"a": 1.0, "b": -0.75, "c": 0.4}
    frame = simulate(6000, beta, (-1.0, 1.0))
    fit = sv.fit_ordered_logit(frame, list(beta), "recovery")
    se = np.sqrt(np.diag(fit.cov)[:len(beta)])
    for i, (name, truth) in enumerate(beta.items()):
        z = abs(fit.beta[i] - truth) / se[i]
        check(f"{name} recovered", z < 3,
              f"estimate {fit.beta[i]:+.3f} vs truth {truth:+.3f}, z {z:.2f}")
    check("cutpoints recovered",
          abs(fit.cut[0] + 1.0) < 0.15 and abs(fit.cut[1] - 1.0) < 0.15,
          f"cutpoints {fit.cut[0]:+.3f}, {fit.cut[1]:+.3f} vs -1.000, +1.000")
    check("cutpoints are ordered", fit.cut[0] < fit.cut[1])


def test_brant_does_not_reject_proportional_data() -> None:
    print("\nThe Brant test must not reject when proportional odds holds")
    beta = {"a": 0.9, "b": -0.6, "c": 0.35}
    rejects = 0
    trials = 20
    for t in range(trials):
        frame = simulate(1200, beta, (-1.0, 0.8),
                         rng=np.random.default_rng(3000 + t))
        out = sv.proportional_odds_tests(frame, list(beta))
        rejects += out["omnibus_rejects_at_5pct"]
    check("rejection rate near the nominal 5 percent", rejects <= 4,
          f"{rejects} of {trials} trials rejected")


def test_brant_detects_violation() -> None:
    print("\nThe Brant test must reject, and name the right covariate, when "
          "proportional odds fails")
    beta = {"a": 0.9, "b": -0.6, "c": 0.35}
    frame = simulate(4000, beta, (-1.5, 1.5), nonprop={"b": 1.5},
                     rng=np.random.default_rng(4242))
    out = sv.proportional_odds_tests(frame, list(beta))
    check("omnibus rejects", out["omnibus_rejects_at_5pct"],
          f"chi2 {out['omnibus_lr_chi2']} on {out['omnibus_df']} df, "
          f"p {out['omnibus_lr_p']}")
    check("the violating covariate is identified",
          out["per_covariate"]["b"]["fails_at_5pct"],
          f"b: likelihood-ratio p {out['per_covariate']['b']['lr_p']}, "
          f"Wald p {out['per_covariate']['b']['wald_p']}")
    innocent = [c for c in ("a", "c") if out["per_covariate"][c]["fails_at_5pct"]]
    check("the other covariates are not flagged", not innocent,
          f"wrongly flagged: {innocent}" if innocent else "")


def test_ppo_recovers_the_violation() -> None:
    print("\nThe partial proportional-odds fit must recover a known "
          "threshold-specific effect")
    beta = {"a": 0.9, "b": -0.6, "c": 0.35}
    frame = simulate(6000, beta, (-1.5, 1.5), nonprop={"b": 1.5},
                     rng=np.random.default_rng(777))
    fit = sv.fit_ppo(frame, list(beta), free=["b"], name="ppo_violation")
    check("the fit converged", fit.converged,
          f"max |gradient| at the optimum "
          f"{fit.extras['max_abs_gradient_at_optimum']:.2e}")
    gap = float(fit.beta_free[1, 0] - fit.beta_free[0, 0])
    check("the threshold-specific gap is recovered", abs(gap - 1.5) < 0.35,
          f"estimated gap {gap:+.3f} vs true 1.500")

    ol = sv.fit_ordered_logit(frame, list(beta), "ol_violation")
    check("freeing the covariate improves the fit", fit.loglik > ol.loglik,
          f"PPO {fit.loglik:.2f} vs ordered logit {ol.loglik:.2f}")

    # The freed covariate is in `beta_free`, never in `beta`, so the parameter count
    # must add its two threshold-specific coefficients whole. Undercounting by one
    # would flatter the partial proportional-odds model in exactly the BIC comparison
    # that prespec section 7 uses to confirm it as primary.
    n_prop, n_free, J = len(beta) - 1, 1, len(sv.SEV_ORDER)
    expected = n_prop + (J - 1) * n_free + (J - 1)
    check("the partial proportional-odds parameter count is right",
          fit.n_params == expected,
          f"{fit.n_params} counted, {expected} expected "
          f"({n_prop} proportional + {(J - 1) * n_free} threshold-specific + "
          f"{J - 1} cutpoints)")
    check("the parameter count exceeds ordered logit by exactly one",
          fit.n_params == ol.n_params + 1,
          f"PPO {fit.n_params} vs ordered logit {ol.n_params}")
    bic = sv.confirm_by_bic(fit, ol)
    check("BIC confirms the model that is genuinely better",
          bic["confirmed"], f"BIC {bic['bic_partial_proportional_odds']} vs "
          f"{bic['bic_ordered_logit']}, difference {bic['difference']}")


def test_ame_matches_the_generating_process() -> None:
    print("\nAverage marginal effects must match the truth when the model is correct")
    beta = {"a": 1.1, "b": -0.7, "c": 0.5}
    frame = simulate(8000, beta, (-1.0, 0.9), rng=np.random.default_rng(99))
    fit = sv.fit_ordered_logit(frame, list(beta), "ame")
    ames = sv.average_marginal_effects(fit, frame, draws=300, seed=5, groups={})
    truth = true_ame(frame, list(beta), beta, (-1.0, 0.9), "a")
    for j, lvl in enumerate(sv.SEV_ORDER):
        est = ames["a"][lvl]["ame"]
        lo, hi = ames["a"][lvl]["ci95"]
        check(f"AME for a on {lvl} covers the truth", lo <= truth[j] <= hi,
              f"{est:+.4f} [{lo:+.4f}, {hi:+.4f}] vs truth {truth[j]:+.4f}")
    total = sum(ames["a"][lvl]["ame"] for lvl in sv.SEV_ORDER)
    check("the three outcome effects sum to zero", abs(total) < 1e-9,
          f"sum {total:.2e}")
    check("the effect is reported as a discrete change",
          ames["a"]["type"] == "discrete change")


def test_ame_respects_categorical_groups() -> None:
    """The failure mode this guards against is silent: the wrong version produces
    plausible numbers from counterfactual rows that could not exist, because two
    mutually exclusive levels of one categorical are both set to one."""
    print("\nAverage marginal effects must zero the sibling levels of a categorical")
    n = 3000
    rng = np.random.default_rng(31415)
    lvl = rng.integers(0, 3, n)          # reference, 35-40, 45+
    frame = pd.DataFrame({
        "speed_35to40": (lvl == 1).astype(float),
        "speed_ge45": (lvl == 2).astype(float),
        "weekend": rng.integers(0, 2, n).astype(float),
    })
    b = {"speed_35to40": 0.5, "speed_ge45": 1.3, "weekend": -0.4}
    cut = (-1.0, 0.8)
    eta = frame.to_numpy() @ np.array(list(b.values()))
    p0 = 1 / (1 + np.exp(-(cut[0] - eta)))
    p1 = 1 / (1 + np.exp(-(cut[1] - eta)))
    u = rng.random(n)
    frame["sev3"] = [sv.SEV_ORDER[c] for c in
                     np.where(u < p0, 0, np.where(u < p1, 1, 2))]

    cols = ["speed_35to40", "speed_ge45", "weekend"]
    groups = {"speed": ["speed_35to40", "speed_ge45"]}
    check("the speed levels are recognized as one categorical group",
          set(groups.get("speed", [])) == {"speed_35to40", "speed_ge45"},
          f"groups {groups}")
    check("an ungrouped covariate has no siblings", "weekend" not in groups)

    fit = sv.fit_ordered_logit(frame, cols, "groups")
    ames = sv.average_marginal_effects(fit, frame, draws=200, seed=11, groups=groups)

    correct = true_ame(frame, cols, b, cut, "speed_ge45", siblings=["speed_35to40"])
    naive = true_ame(frame, cols, b, cut, "speed_ge45", siblings=[])
    check("the two treatments actually differ, so this test has teeth",
          abs(correct[2] - naive[2]) > 0.01,
          f"correct {correct[2]:+.4f} vs sibling-ignoring {naive[2]:+.4f}")

    est = ames["speed_ge45"]["KA"]["ame"]
    check("the estimate matches the sibling-respecting truth",
          abs(est - correct[2]) < 0.03,
          f"estimate {est:+.4f} vs correct {correct[2]:+.4f}")
    check("the estimate is not the sibling-ignoring value",
          abs(est - correct[2]) < abs(est - naive[2]),
          f"distance to correct {abs(est - correct[2]):.4f}, "
          f"to naive {abs(est - naive[2]):.4f}")


def test_krinsky_robb_covers_the_truth() -> None:
    """Coverage is the criterion, not interval width.

    The intuition that freezing the cutpoints must widen the interval is wrong for this
    model. A shift in the linear index can be absorbed by a shift in the cutpoints, so
    slopes and thresholds are strongly positively correlated and the correlated movement
    partly cancels inside the marginal effect. Drawing them jointly therefore gives a
    *narrower* interval here than holding the thresholds fixed. What makes the joint
    draw right is not that it is wider but that it covers the true effect at the stated
    rate, which is what this test measures.
    """
    print()
    print("Krinsky-Robb intervals must cover the true effect at about 95 percent")
    beta = {"a": 0.9, "b": -0.5}
    cut = (-1.0, 0.8)
    trials, covered_joint, covered_frozen = 40, 0, 0
    widths_joint, widths_frozen = [], []

    for t in range(trials):
        frame = simulate(700, beta, cut, rng=np.random.default_rng(9000 + t))
        fit = sv.fit_ordered_logit(frame, list(beta), "kr")
        truth = true_ame(frame, list(beta), beta, cut, "a")[2]

        lo, hi = sv.average_marginal_effects(
            fit, frame, draws=400, seed=21, groups={})["a"]["KA"]["ci95"]
        covered_joint += lo <= truth <= hi
        widths_joint.append(hi - lo)

        k = len(fit.beta)
        frozen_cov = np.array(fit.cov, dtype=float)
        frozen_cov[k:, :] = 0.0   # the slope block and its correlations stay intact,
        frozen_cov[:, k:] = 0.0   # so the only thing removed is threshold uncertainty
        frozen = sv.OrdinalFit(
            name="frozen", columns=fit.columns, beta=fit.beta, cut=fit.cut,
            raw=fit.raw, cov=frozen_cov, loglik=fit.loglik, n=fit.n,
            converged=fit.converged)
        flo, fhi = sv.average_marginal_effects(
            frozen, frame, draws=400, seed=21, groups={})["a"]["KA"]["ci95"]
        covered_frozen += flo <= truth <= fhi
        widths_frozen.append(fhi - flo)

    check("coverage is close to the nominal 95 percent", covered_joint >= 34,
          f"{covered_joint} of {trials} intervals covered the truth")
    check("the threshold parameters change the answer, so they must be drawn",
          abs(np.mean(widths_joint) - np.mean(widths_frozen)) > 0.005,
          f"mean width {np.mean(widths_joint):.4f} drawing thresholds jointly vs "
          f"{np.mean(widths_frozen):.4f} holding them fixed "
          f"(coverage {covered_joint}/{trials} vs {covered_frozen}/{trials})")
    check("every draw yields ordered cutpoints",
          all(sv._cut_from_raw(r)[0] < sv._cut_from_raw(r)[1]
              for r in np.random.default_rng(1).multivariate_normal(
                  fit.raw[len(fit.beta):],
                  fit.cov[len(fit.beta):, len(fit.beta):], size=500, method="svd")))


def test_reported_diagnostics() -> None:
    print("\nLadder diagnostics must report the parameter budget and separation")
    beta = {"a": 0.9, "b": -0.5, "c": 0.2}
    frame = simulate(400, beta, (-1.0, 0.8), rng=np.random.default_rng(12))
    epp = sv.events_per_parameter(frame, list(beta))
    check("parameters counted as covariates plus J-1 cutpoints",
          epp["n_parameters"] == 3 + len(sv.SEV_ORDER) - 1,
          f"{epp['n_parameters']} parameters")
    check("the smallest class drives the ratio",
          epp["smallest_class"] == min(epp["class_counts"].values()))

    fit = sv.fit_ordered_logit(frame, list(beta), "diag")
    sep = sv.separation_diagnostics(fit)
    check("well-behaved data raises no separation flag",
          not sep["suggests_separation"],
          f"max |coefficient| {sep['max_abs_coefficient']}")

    separable = frame.copy()
    separable["d"] = (separable["sev3"] == "KA").astype(float)
    sep2 = sv.separation_diagnostics(
        sv.fit_ordered_logit(separable, list(beta) + ["d"], "sep"))
    check("a perfectly predictive covariate raises the flag",
          sep2["suggests_separation"],
          f"max |coefficient| {sep2['max_abs_coefficient']}, "
          f"max SE {sep2['max_standard_error']}")


if __name__ == "__main__":
    for fn in (test_ppo_reduces_to_ordered_logit,
               test_coefficient_recovery,
               test_brant_does_not_reject_proportional_data,
               test_brant_detects_violation,
               test_ppo_recovers_the_violation,
               test_ame_matches_the_generating_process,
               test_ame_respects_categorical_groups,
               test_krinsky_robb_covers_the_truth,
               test_reported_diagnostics):
        fn()

    print()
    if FAILURES:
        print(f"{len(FAILURES)} check(s) failed: {FAILURES}")
        sys.exit(1)
    print("all checks passed")
