"""The KABCO-narrative reporting-discordance audit.

A latent severity class Y in {O, BC, KA} is measured twice by the same police record:
once by the officer's KABCO code, once by the officer's narrative through the four-level
indicator of `audit_indicator.py`. Neither measurement is injury. The model asks how the
two disagree, and nothing in it licenses calling either one correct.

## Identification, counted before fitting

The observed data are a 5 by 4 table: KABCO by narrative indicator. That is 20 cells,
19 free once the total is fixed. The parameters must fit inside that.

| block | parameterization | free parameters |
|---|---|---|
| latent prevalence | 3 classes | 2 |
| KABCO measurement | ordered, 4 cutpoints + 2 latent-class shifts | 6 |
| narrative measurement | 3 rows x 4 columns, Dirichlet, rows sum to 1 | 9 |
| | **total** | **17** |

Seventeen against nineteen. Identified, with two degrees of freedom to spare, and only
because the KABCO measurement carries an adjacent-category ordered
structure. An unconstrained 3 by 5 confusion matrix would take 12 parameters
instead of 6, giving 23 against 19 — **under-identified before a single covariate is
added**. The ordered structure is load-bearing for identification, not a stylistic
choice, and a sampler fitting the unconstrained version would still report healthy
R-hat, adequate effective sample size and no divergences while wandering a ridge.

Once the reporting-context shifters enter, identification rests entirely on covariate
variation, and the margin above is gone. This is the concrete content of the phrase
"partial identification" in the limitations paragraph, and it is why the covariate model
is fitted only after the no-covariate model is shown to reproduce a table computed by
counting.

## What this module does, in order

1. Fit the no-covariate model: prevalences and the two measurement blocks.
2. Check its implied 5 by 4 table against the raw cross-tabulation. That table involves
   no priors and no model, so it is the ground truth for the measurement layer. If the
   intercept-only model cannot reproduce it, the parameterization is wrong, and that is
   found here rather than inside a 48-parameter fit.
3. Confirm the latent classes have not permuted. Labels are arbitrary in a latent-class
   model; if the class called KA is not the one carrying the K and A codes, every
   shifter reads backwards and no convergence diagnostic notices.
4. Only then add the shifters.

Recoverability is checked separately in `audit_recovery.py`: simulate from the model with
known confusion matrices and known shifters at this sample size, fit, and see whether the
truth comes back. If it does not, the finding is that the design cannot detect an effect
of that size at n = 522 — a different and stronger statement than an interval covering
zero.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

SEV3 = ("O", "BC", "KA")
KABCO = ("O", "C", "B", "A", "K")
N_LEVELS = ("no cue", "O-consistent", "BC-consistent", "KA-consistent")

DRAWS = 1000
TUNE = 2000
CHAINS = 4
SEED = 20260914

# The two prior settings. "Moderate" concentrates the
# narrative measurement rows toward the diagonal; "weak" is close to uniform, so a
# parameter that moves under one and not the other is being driven by the prior.
# `class_sd` and `shift_sd` are deliberately different. The latent-class shifts have to
# span the whole severity range — separating a no-injury class from a fatal one takes
# several units on the log-odds scale — while a reporting-context shifter is a nudge to
# how a given severity gets coded and should be small. An earlier version used one sigma
# for both, at 1.0. That put the prior mean of the KA class shift at 1.60 against a true
# 3.2 in the recovery study, and the model made up the difference by inflating the
# reporting shifters: a planted 0.25, 0.5 and 1.0 came back as 0.45, 0.75 and 1.70, an
# upward bias near 1.7x at every size. A prior centred at zero cannot inflate an
# estimate; a prior that cannot reach the truth elsewhere in the model can.
PRIOR_SETS = {
    "moderate": {"n_diag": 4.0, "n_off": 1.0, "shift_sd": 1.0, "class_sd": 3.0},
    "weak": {"n_diag": 1.5, "n_off": 1.0, "shift_sd": 2.5, "class_sd": 3.0},
}


def observed_table(indicator: pd.DataFrame) -> np.ndarray:
    """The 5 by 4 contingency table the model must reproduce."""
    ct = pd.crosstab(indicator["sev_kabco"], indicator["narrative_indicator"])
    return ct.reindex(index=list(KABCO), columns=list(N_LEVELS)).fillna(0).to_numpy()


# Each KABCO code and each narrative level implies a position on the
# severity scale; CONCORDANCE marks the cells where the two agree. A dependence term
# raises the probability of those cells beyond what the latent class alone explains,
# which is what correlated reporting looks like: the officer who writes "no injury" in
# the narrative is the same officer who codes O on the form, so the two are not
# independent given the underlying severity.
_KABCO_POS = {"O": 0, "C": 1, "B": 1, "A": 2, "K": 2}
_N_POS = {"no cue": None, "O-consistent": 0, "BC-consistent": 1, "KA-consistent": 2}
CONCORDANCE = np.array([
    [0.0 if _N_POS[n] is None else float(_KABCO_POS[k] == _N_POS[n])
     for n in N_LEVELS]
    for k in KABCO])


def build_no_covariate_model(table: np.ndarray, priors: dict,
                             dependence: float | None = 0.0):
    """Prevalences and the two measurement blocks, with no reporting context.

    The KABCO block is an ordered model: four shared cutpoints and a shift per latent
    class, with the O class fixed at zero for identification. That is what "adjacent
    category structure" means here, and the identification count in the module
    docstring depends on it.

    `dependence` is the conditional-dependence term: a log-scale boost
    applied to cells where the KABCO code and the narrative indicator agree. Zero is the
    conditionally-independent model the audit is specified on. Passing a number fixes it
    there; passing None estimates it, which is outside the pre-specification and is used
    only to ask what the data want.
    """
    import pymc as pm
    import pytensor.tensor as pt

    counts = table.reshape(-1)

    with pm.Model() as model:
        prevalence = pm.Dirichlet("prevalence", a=np.ones(3))

        # KABCO measurement: ordered, shared cutpoints, one shift per latent class.
        #
        # Both orderings were first imposed as cumulative sums of exponentials, which is
        # the obvious trick and samples badly: exp() of a parameter with a wide prior
        # spans orders of magnitude, so the posterior develops a funnel. Under the weak
        # priors that produced an r-hat of 1.17, a bulk effective sample size of 17, and
        # 85 divergences — a chain exploring almost nothing. The geometry, not the model,
        # was at fault, so the ordering is now imposed by PyMC's ordered transform for
        # the cutpoints and by half-normal increments for the class shifts. Both are
        # well scaled and neither changes what the model says.
        cutpoints = pm.Normal(
            "cutpoints", mu=np.linspace(-1.5, 2.0, 4), sigma=2.0, shape=4,
            transform=pm.distributions.transforms.ordered,
            initval=np.linspace(-1.5, 2.0, 4))

        step_bc = pm.HalfNormal("kabco_step_bc", sigma=priors["class_sd"])
        step_ka = pm.HalfNormal("kabco_step_ka", sigma=priors["class_sd"])
        shift = pm.Deterministic(
            "kabco_shift",
            pt.stack([pt.zeros(()), step_bc, step_bc + step_ka]))

        # P(KABCO = k | Y = y) from the ordered cumulative form.
        eta = cutpoints[None, :] - shift[:, None]
        cum = pm.math.sigmoid(eta)
        kabco_p = pt.concatenate(
            [cum[:, :1],
             cum[:, 1:] - cum[:, :-1],
             1.0 - cum[:, -1:]], axis=1)
        kabco_p = pm.Deterministic("kabco_p", kabco_p)

        # Narrative measurement: one Dirichlet row per latent class, concentrated on the
        # column that class should produce.
        alpha = np.full((3, 4), priors["n_off"])
        for row, col in enumerate((1, 2, 3)):     # O->O-consistent, BC->BC, KA->KA
            alpha[row, col] = priors["n_diag"]
            alpha[row, 0] = priors["n_diag"]      # "no cue" is common in every class
        narrative_p = pm.Dirichlet("narrative_p", a=alpha, shape=(3, 4))

        # Marginal cell probability: sum over the latent class.
        cell = pt.sum(
            prevalence[:, None, None] * kabco_p[:, :, None] * narrative_p[:, None, :],
            axis=0)

        if dependence is None:
            lam = pm.Normal("dependence", mu=0.0, sigma=1.0)
        else:
            lam = pt.constant(float(dependence))
        cell = cell * pt.exp(lam * pt.constant(CONCORDANCE))
        cell = pm.Deterministic("cell", cell / pt.sum(cell))

        pm.Multinomial("obs", n=int(counts.sum()), p=cell.reshape((-1,)),
                       observed=counts)
    return model


def sample(model, draws: int = DRAWS, tune: int = TUNE, chains: int = CHAINS,
           seed: int = SEED, target_accept: float = 0.95):
    import nutpie
    compiled = nutpie.compile_pymc_model(model)
    return nutpie.sample(compiled, draws=draws, tune=tune, chains=chains,
                         seed=seed, progress_bar=False,
                         target_accept=target_accept)


def diagnostics(idata, var_names: list[str]) -> dict:
    """R-hat, effective sample size and divergences."""
    import arviz as az
    summary = az.summary(idata, var_names=var_names)
    div = 0
    if hasattr(idata, "sample_stats") and "diverging" in idata.sample_stats:
        div = int(idata.sample_stats["diverging"].to_numpy().sum())
    return {
        "max_r_hat": round(float(summary["r_hat"].max()), 4),
        "min_ess_bulk": int(summary["ess_bulk"].min()),
        "min_ess_tail": int(summary["ess_tail"].min()),
        "divergences": div,
        "n_parameters_summarized": int(len(summary)),
        "healthy": bool(summary["r_hat"].max() < 1.01
                        and summary["ess_bulk"].min() > 400
                        and div == 0),
    }


def posterior_predictive_table(idata, n: int) -> np.ndarray:
    """The 5 by 4 table the fitted model implies, as posterior mean counts."""
    cell = idata.posterior["cell"].to_numpy()
    return cell.reshape(-1, 5, 4).mean(axis=0) * n


def check_against_raw(implied: np.ndarray, observed: np.ndarray) -> dict:
    """Does the measurement layer reproduce the table computed by counting?

    The raw cross-tabulation uses no priors and no latent class. If the intercept-only
    model cannot match it, the parameterization is wrong, and that must be caught before
    any shifter is interpreted.
    """
    resid = implied - observed
    with np.errstate(divide="ignore", invalid="ignore"):
        pearson = np.where(observed > 0, resid / np.sqrt(np.maximum(observed, 1)), 0.0)
    return {
        "max_abs_count_error": round(float(np.max(np.abs(resid))), 2),
        "total_abs_count_error": round(float(np.sum(np.abs(resid))), 2),
        "max_abs_pearson_residual": round(float(np.max(np.abs(pearson))), 3),
        "observed": observed.astype(int).tolist(),
        "implied": np.round(implied, 1).tolist(),
        "reproduces_raw_table": bool(np.max(np.abs(pearson)) < 2.0),
        "note": ("A Pearson residual above about 2 in any cell means the measurement "
                 "layer cannot represent the observed table, which is a "
                 "parameterization problem and not a finding."),
    }


def check_class_alignment(idata) -> dict:
    """Have the latent labels permuted?

    Latent class labels are arbitrary. If the class called KA is not the one producing
    the K and A codes, every shifter reads backwards, and R-hat cannot see it because all
    chains can agree on the same permuted mode.
    """
    kabco_p = idata.posterior["kabco_p"].to_numpy().reshape(-1, 3, 5).mean(axis=0)
    severe = kabco_p[:, 3:].sum(axis=1)      # P(A or K | class)
    none = kabco_p[:, 0]                     # P(O | class)
    # The full measurement matrix, P(coded severity | latent class), collapsed to the
    # three classes the paper models. This is the quantity Figure 4 shows
    # under each prior, so it is kept after the alignment check.
    collapsed = np.column_stack([kabco_p[:, 0],                 # O
                                 kabco_p[:, 1:3].sum(axis=1),   # B and C -> BC
                                 severe])                       # A and K -> KA
    return {
        "kabco_measurement": {
            SEV3[i]: {SEV3[j]: round(float(collapsed[i, j]), 4) for j in range(3)}
            for i in range(3)},
        "kabco_measurement_five_level": {
            SEV3[i]: {lvl: round(float(kabco_p[i, j]), 4)
                      for j, lvl in enumerate(("O", "C", "B", "A", "K"))}
            for i in range(3)},
        "p_severe_given_class": {SEV3[i]: round(float(severe[i]), 4) for i in range(3)},
        "p_no_injury_given_class": {SEV3[i]: round(float(none[i]), 4)
                                    for i in range(3)},
        "severity_ordering_holds": bool(severe[0] < severe[1] < severe[2]),
        "class_with_most_severe_mass": SEV3[int(np.argmax(severe))],
        "aligned": bool(np.argmax(severe) == 2 and np.argmax(none) == 0),
    }


def run_no_covariate(prior_name: str = "moderate",
                     dependence: float | None = 0.0) -> dict:
    indicator = pd.read_csv(DATA / "narrative_indicator.csv")
    table = observed_table(indicator)
    priors = PRIOR_SETS[prior_name]

    model = build_no_covariate_model(table, priors, dependence=dependence)
    idata = sample(model)

    diag = diagnostics(idata, ["prevalence", "cutpoints", "kabco_shift",
                               "narrative_p"])
    implied = posterior_predictive_table(idata, int(table.sum()))
    fitcheck = check_against_raw(implied, table)
    align = check_class_alignment(idata)

    prev = idata.posterior["prevalence"].to_numpy().reshape(-1, 3)
    narr = idata.posterior["narrative_p"].to_numpy().reshape(-1, 3, 4)

    dep_post = None
    if dependence is None and "dependence" in idata.posterior:
        v = idata.posterior["dependence"].to_numpy().ravel()
        dep_post = {"mean": round(float(v.mean()), 4),
                    "ci95": [round(float(np.percentile(v, 2.5)), 4),
                             round(float(np.percentile(v, 97.5)), 4)],
                    "excludes_zero": bool(np.percentile(v, 2.5) > 0
                                          or np.percentile(v, 97.5) < 0)}

    return {
        "prior_set": prior_name,
        "priors": priors,
        "dependence_fixed_at": dependence,
        "dependence_posterior": dep_post,
        "n": int(table.sum()),
        "diagnostics": diag,
        "fit_against_raw_table": fitcheck,
        "class_alignment": align,
        "latent_prevalence": {
            SEV3[i]: {"mean": round(float(prev[:, i].mean()), 4),
                      "ci95": [round(float(np.percentile(prev[:, i], 2.5)), 4),
                               round(float(np.percentile(prev[:, i], 97.5)), 4)]}
            for i in range(3)},
        "narrative_measurement": {
            SEV3[i]: {N_LEVELS[j]: round(float(narr[:, i, j].mean()), 4)
                      for j in range(4)}
            for i in range(3)},
        "labels_are_preliminary": True,
    }


if __name__ == "__main__":
    print("Discordance audit, stage 1: no reporting-context covariates.")
    print("PRELIMINARY — built on the reference run, not the adjudicated labels.")
    print()

    out = {}
    for prior_name in ("moderate", "weak"):
        res = run_no_covariate(prior_name)
        out[prior_name] = res
        d, f, a = res["diagnostics"], res["fit_against_raw_table"], res["class_alignment"]
        print(f"--- {prior_name} priors ---")
        print(f"  r-hat max {d['max_r_hat']}, bulk ESS min {d['min_ess_bulk']}, "
              f"divergences {d['divergences']}, healthy {d['healthy']}")
        print(f"  reproduces the raw table: {f['reproduces_raw_table']} "
              f"(largest Pearson residual {f['max_abs_pearson_residual']}, "
              f"largest count error {f['max_abs_count_error']})")
        print(f"  classes aligned: {a['aligned']}, severity ordering holds "
              f"{a['severity_ordering_holds']}")
        print(f"  P(A or K | class): {a['p_severe_given_class']}")
        print("  latent prevalence:")
        for k, v in res["latent_prevalence"].items():
            print(f"    {k:<3} {v['mean']:.3f} [{v['ci95'][0]:.3f}, {v['ci95'][1]:.3f}]")
        print("  narrative measurement rows:")
        for cls, row in res["narrative_measurement"].items():
            cells = "  ".join(f"{k} {v:.3f}" for k, v in row.items())
            print(f"    {cls:<3} {cells}")
        print()

    path = DATA / "audit_stage1.json"
    path.write_text(json.dumps(out, indent=2, default=float), encoding="utf-8")
    print(f"written to {path}")
