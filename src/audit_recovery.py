"""Can this design recover a reporting-context shifter at all, at n = 522?

The audit design calls for shifter vectors on all four KABCO transitions against a
twelve-variable reporting-context vector: 48 parameters, estimated from 522 records through two
imperfect measurements of a latent class nobody observes. Whether that is possible is an
empirical question, and it is answerable before looking at a single real estimate.

The method is the one that caught the false Small-Hsiao rejection. Simulate from the
model with **known** shifters at the real sample size, real covariate prevalences and
real class shares; fit; and see whether the truth comes back. Two things can then be
said precisely rather than vaguely:

* if a known effect of a given size is recovered, an interval covering zero on the real
  data means no effect of that size is present;
* if a known effect is *not* recovered, an interval covering zero means the design
  cannot see an effect of that size, which is a statement about the study and not about
  e-scooter crashes.

Those are different claims and only one of them is usually supported. The distinction is
the same one that had to be corrected in the narrative-cue write-up, and it is cheaper to
settle here than to argue about afterwards.

The shifters act on the KABCO measurement: the four cutpoints separating O|C, C|B, B|A
and A|K each move with the reporting context, conditional on the latent severity class.
That is what "shifters on all four transitions" means here — reporting context
changes how a given underlying severity gets coded, not what the severity is.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import audit_model as am

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

# The reporting-context vector, with the prevalences observed in the frame.
# `urbanized` comes from Rural_Fl rather than Rural_Urban_Type_ID, which is missing for
# 415 of 522 records and cannot carry a coefficient.
CONTEXT = (
    ("eth_black", 0.241), ("eth_hispanic", 0.243), ("eth_other_or_unknown", 0.142),
    ("gender_female", 0.239), ("gender_unknown", 0.050),
    ("age_under25", 0.490), ("age_45plus", 0.123), ("age_unknown", 0.065),
    ("nighttime", 0.153), ("urbanized", 0.879),
)
CONTINUOUS = ("narrative_length_z",)

N_TRANSITIONS = 4


def simulate(n: int, true_delta: np.ndarray, seed: int,
             prevalence=(0.14, 0.79, 0.07),
             cutpoints=(-1.2, 0.2, 1.6, 2.8),
             class_shift=(0.0, 1.4, 3.2)) -> dict:
    """Draw a dataset from the audit model with known shifters.

    `true_delta` has one row per transition and one column per context variable, so it
    is the object the fit has to recover.
    """
    rng = np.random.default_rng(seed)
    names = [c for c, _ in CONTEXT] + list(CONTINUOUS)

    cols = {c: (rng.random(n) < p).astype(float) for c, p in CONTEXT}
    for c in CONTINUOUS:
        cols[c] = rng.normal(size=n)
    X = np.column_stack([cols[c] for c in names])

    y = rng.choice(3, size=n, p=np.asarray(prevalence))

    # Cutpoints move with the reporting context; the latent class shifts the index.
    cut = np.asarray(cutpoints)[None, :] + X @ true_delta.T          # (n, 4)

    # A shifter moves one cutpoint and not its neighbours, so a large enough effect
    # reorders them for some records and the category between them takes negative
    # probability. Clipping and renormalizing would hide that and quietly make the
    # generating process something other than the model being fitted — which is exactly
    # how a recovery study comes to measure nothing. An earlier version did precisely
    # that and "failed to recover" a planted 2.0 as 4.28, a mismatch rather than a
    # finding. Refuse to generate instead.
    slack = np.diff(cut, axis=1).min() if cut.shape[1] > 1 else np.inf
    if slack <= 0:
        raise ValueError(
            f"incoherent generating process: an effect of this size reorders the "
            f"cutpoints for at least one record (smallest gap {slack:.3f}). The planted "
            f"effect must stay smaller than the gap between adjacent cutpoints, here "
            f"{min(np.diff(cutpoints)):.2f}.")

    eta = cut - np.asarray(class_shift)[y][:, None]
    cum = 1.0 / (1.0 + np.exp(-eta))
    p_kabco = np.column_stack([cum[:, :1], np.diff(cum, axis=1), 1.0 - cum[:, -1:]])
    kabco = np.array([rng.choice(5, p=row / row.sum()) for row in p_kabco])

    narrative_rows = np.array([
        [0.47, 0.45, 0.05, 0.03],
        [0.74, 0.01, 0.24, 0.01],
        [0.29, 0.03, 0.08, 0.60],
    ])
    narr = np.array([rng.choice(4, p=narrative_rows[c]) for c in y])

    return {"X": X, "names": names, "kabco": kabco, "narrative": narr,
            "latent": y, "true_delta": true_delta}


def build_shifter_model(X: np.ndarray, kabco: np.ndarray, narrative: np.ndarray,
                        priors: dict):
    """Individual-level audit model with shifters on all four KABCO transitions.

    The latent class is summed out rather than sampled, so there are no discrete
    parameters and the sampler sees a smooth posterior.
    """
    import pymc as pm
    import pytensor.tensor as pt

    n, k = X.shape

    with pm.Model() as model:
        prevalence = pm.Dirichlet("prevalence", a=np.ones(3))

        cutpoints = pm.Normal(
            "cutpoints", mu=np.linspace(-1.2, 2.8, 4), sigma=2.0, shape=4,
            transform=pm.distributions.transforms.ordered,
            initval=np.linspace(-1.2, 2.8, 4))

        # The class shifts span the severity range; the reporting shifters below do not.
        # They must not share a prior scale — see PRIOR_SETS in audit_model.
        step_bc = pm.HalfNormal("kabco_step_bc", sigma=priors["class_sd"])
        step_ka = pm.HalfNormal("kabco_step_ka", sigma=priors["class_sd"])
        class_shift = pt.stack([pt.zeros(()), step_bc, step_bc + step_ka])

        # Started at zero, which is the no-reporting-effect point and always valid.
        delta = pm.Normal("delta", mu=0.0, sigma=priors["shift_sd"],
                          shape=(N_TRANSITIONS, k),
                          initval=np.zeros((N_TRANSITIONS, k)))

        moved = cutpoints[None, :] + pt.dot(X, delta.T)              # (n, 4)
        eta = moved[:, None, :] - class_shift[None, :, None]         # (n, 3, 4)
        cum = pm.math.sigmoid(eta)
        p_kabco = pt.concatenate(
            [cum[:, :, :1], cum[:, :, 1:] - cum[:, :, :-1], 1.0 - cum[:, :, -1:]],
            axis=2)                                                  # (n, 3, 5)
        # The cutpoints are ordered, but a per-transition shift can reorder them for an
        # individual record, and the category probability is then a negative difference.
        # Its logarithm is undefined, which is what made every initialization point fail
        # on the first attempt. Flooring keeps the likelihood finite and makes the
        # invalid region costly rather than fatal, so the sampler walks away from it
        # instead of dying at the edge.
        p_kabco = pt.clip(p_kabco, 1e-9, 1.0)

        alpha = np.full((3, 4), priors["n_off"])
        for row, col in enumerate((1, 2, 3)):
            alpha[row, col] = priors["n_diag"]
            alpha[row, 0] = priors["n_diag"]
        narrative_p = pm.Dirichlet("narrative_p", a=alpha, shape=(3, 4))

        idx = np.arange(n)
        lik_kabco = p_kabco[idx, :, kabco]                           # (n, 3)
        lik_narr = narrative_p[:, narrative].T                       # (n, 3)
        joint = prevalence[None, :] * lik_kabco * lik_narr
        pm.Potential("obs", pt.sum(pt.log(pt.clip(pt.sum(joint, axis=1),
                                                  1e-300, np.inf))))

    return model


def recovery_check(effect_size: float, seed: int = 20260914,
                   n: int = 522, draws: int = 600, tune: int = 1200) -> dict:
    """Plant a known shifter on one transition and one variable, then try to find it.

    Only one entry of the 48 is non-zero. That is the most favourable case there is: if
    a single isolated effect of this size cannot be recovered, forty-eight simultaneous
    ones certainly cannot.
    """
    names = [c for c, _ in CONTEXT] + list(CONTINUOUS)
    k = len(names)
    true_delta = np.zeros((N_TRANSITIONS, k))
    target = (1, names.index("eth_black"))        # the C|B transition
    true_delta[target] = effect_size

    data = simulate(n, true_delta, seed)
    model = build_shifter_model(data["X"], data["kabco"], data["narrative"],
                                am.PRIOR_SETS["moderate"])
    idata = am.sample(model, draws=draws, tune=tune, chains=4, seed=seed)

    import arviz as az
    summary = az.summary(idata, var_names=["delta"])
    post = idata.posterior["delta"].to_numpy().reshape(-1, N_TRANSITIONS, k)
    planted = post[:, target[0], target[1]]
    lo, hi = np.percentile(planted, [2.5, 97.5])

    zeros = [(t, j) for t in range(N_TRANSITIONS) for j in range(k)
             if (t, j) != target]
    false_positives = sum(
        1 for t, j in zeros
        if np.percentile(post[:, t, j], 2.5) > 0
        or np.percentile(post[:, t, j], 97.5) < 0)

    div = 0
    if hasattr(idata, "sample_stats") and "diverging" in idata.sample_stats:
        div = int(idata.sample_stats["diverging"].to_numpy().sum())

    return {
        "effect_size": effect_size,
        "n": n,
        "planted_at": {"transition": int(target[0]), "variable": names[target[1]]},
        "posterior_mean": round(float(planted.mean()), 4),
        "ci95": [round(float(lo), 4), round(float(hi), 4)],
        "covers_truth": bool(lo <= effect_size <= hi),
        "excludes_zero": bool(lo > 0 or hi < 0),
        "recovered": bool((lo <= effect_size <= hi) and (lo > 0 or hi < 0)),
        "interval_width": round(float(hi - lo), 4),
        "false_positives_among_47_true_zeros": false_positives,
        "diagnostics": {
            "max_r_hat": round(float(summary["r_hat"].max()), 4),
            "min_ess_bulk": int(summary["ess_bulk"].min()),
            "divergences": div,
        },
    }


if __name__ == "__main__":
    print("Can the audit design recover a known reporting-context shifter at n = 522?")
    print("One of the 48 shifters is non-zero; the other 47 are exactly zero.")
    print("The cutpoints sit 1.4 apart, so effects above about 1.0 cannot be")
    print("planted coherently and are not tried.")
    print()

    results = []
    # The cutpoints sit 1.4 apart, so an effect much above 1.0 reorders them and the
    # generating process stops being the model. Those sizes are refused rather than
    # fudged, which bounds what this study can ask.
    for size in (0.25, 0.5, 1.0):
        r = recovery_check(size)
        results.append(r)
        d = r["diagnostics"]
        print(f"planted effect {size:>4}  ->  posterior {r['posterior_mean']:+.3f} "
              f"[{r['ci95'][0]:+.3f}, {r['ci95'][1]:+.3f}]  width {r['interval_width']:.2f}")
        print(f"   covers the truth {r['covers_truth']}, excludes zero "
              f"{r['excludes_zero']}, RECOVERED {r['recovered']}")
        print(f"   false positives among the 47 true zeros: "
              f"{r['false_positives_among_47_true_zeros']}")
        print(f"   r-hat {d['max_r_hat']}, bulk ESS {d['min_ess_bulk']}, "
              f"divergences {d['divergences']}")
        print()

    smallest = next((r["effect_size"] for r in results if r["recovered"]), None)
    print(f"smallest planted effect recovered: "
          f"{smallest if smallest is not None else 'none of those tried'}")
    path = DATA / "audit_recovery.json"
    path.write_text(json.dumps(results, indent=2, default=float), encoding="utf-8")
    print(f"written to {path}")
