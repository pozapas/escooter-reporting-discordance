"""The discordance audit on the final adjudicated labels, with saved posteriors.

`audit_model.py` defines the audit and is not changed here. This script reruns its
primary specification (dependence fixed at zero, moderate and weak priors) on the final
narrative indicator, and this time keeps what the earlier stage-1 run threw away:

* the per-rider posterior latent-class probabilities, as posterior means and as 100
  posterior draws, keyed by `rider_row_id` of `model_frame.csv`, for M2;
* the conditional-dependence term estimated rather than fixed (the `dependence=None`
  path of `build_no_covariate_model`), with its own diagnostics;
* the narrative-length sensitivity: narrative measurement rows that move with
  standardized narrative length.

## Why the class probabilities take only twenty distinct values

With no reporting-context covariates, the posterior probability that rider i belongs to
latent class c depends on the record only through its cell of the 5 by 4 table:

    w_ic = pi_c p_K(k_i | c) p_N(n_i | c) / sum_c' pi_c' p_K(k_i | c') p_N(n_i | c')

The dependence term multiplies a whole cell by exp(lambda * concordance), the same factor
for every latent class, so it cancels from this ratio. It still changes the weights, but
only through the other parameters it moves. The twenty cell-level vectors are written
out in full, because which posterior enters M2 as a weight bears directly on how M2
is read.

## The narrative-length model

The stage-1 audit has an asymmetry: the KABCO side was allowed to move with reporting context
while the narrative measurement was held fixed, although narrative length is the most
direct candidate for changing what a narrative can say. Here each narrative row is
logit-linear in standardized length, with "no cue" as the reference column:

    p_N(n | c, z) = softmax_n( log p0_N(n | c) + gamma_cn z ),   gamma_c,no cue = 0

so gamma = 0 is the primary model and the Dirichlet prior applies to the rows at the mean
length. Length is also related to the coded severity itself (fatal narratives are the
longest), so if the latent prevalence were held fixed in length, that association would
have nowhere to go but the narrative slopes. The latent prevalence is therefore made
multinomial-logit in length as well, and the version without that is reported beside it.
"""

from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

import audit_indicator as ai
import audit_model as am

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

SEV3 = am.SEV3
KABCO = am.KABCO
N_LEVELS = am.N_LEVELS

N_POSTERIOR_DRAWS = 100          # M2 is refitted over this many draws
IMPLIED_TABLE_DRAWS = 400        # thinned draws used for the length-model implied table
FIXED_DEPENDENCE_GRID = (0.5, 1.0)
# Every sensitivity fit (estimated dependence, narrative length) is also run with longer
# chains, and the long run is the one to report. Those posteriors are flatter than the
# primary one, and at the pipeline settings two runs with the same seed differ in the
# second decimal (nutpie does not reproduce them bit for bit, while the primary fit is
# reproduced exactly), and some fall short of the health rule. Both runs are kept.
LONG_DRAWS = 4000
LONG_TUNE = 4000


# --------------------------------------------------------------------------
# Inputs
# --------------------------------------------------------------------------

def load_inputs() -> tuple[pd.DataFrame, dict]:
    """The model frame with the narrative indicator attached by position.

    `model_frame.csv` has 522 rider rows but 520 crashes, so a merge on Crash_ID would
    cross-join the two multi-rider crashes. The indicator file was written in frame order;
    that is asserted row by row and the indicator is attached by position. The indicator
    is also rebuilt from `final_labels.csv` and compared, so a stale CSV cannot pass.
    """
    frame = pd.read_csv(DATA / "model_frame.csv")
    ind = pd.read_csv(DATA / "narrative_indicator.csv")

    if len(frame) != len(ind):
        raise SystemExit("model_frame.csv and narrative_indicator.csv differ in length")
    for col in ("Crash_ID", "sev3", "sev_kabco"):
        if not (frame[col].astype(str).to_numpy() == ind[col].astype(str).to_numpy()).all():
            raise SystemExit(f"row order differs between the frame and the indicator on {col}")
    if not frame["rider_row_id"].is_unique:
        raise SystemExit("rider_row_id is not unique")

    labels = pd.read_csv(DATA / "final_labels.csv")
    labels["rebuilt"] = np.asarray(ai.build_indicator(labels)).astype(str)
    if not labels["Crash_ID"].is_unique:
        raise SystemExit("final_labels.csv has repeated Crash_ID")
    rebuilt = frame[["Crash_ID"]].merge(labels[["Crash_ID", "rebuilt"]],
                                        on="Crash_ID", how="left")["rebuilt"]
    agree = int((rebuilt.to_numpy() == ind["narrative_indicator"].to_numpy()).sum())

    frame = frame.copy()
    frame["narrative_indicator"] = ind["narrative_indicator"].to_numpy()
    check = {
        "rows": int(len(frame)),
        "distinct_crashes": int(frame["Crash_ID"].nunique()),
        "row_order_matches": True,
        "indicator_rebuilt_from_final_labels_agrees": f"{agree}/{len(frame)}",
        "indicator_is_current": bool(agree == len(frame)),
    }
    if not check["indicator_is_current"]:
        raise SystemExit("narrative_indicator.csv is stale against final_labels.csv")
    return frame, check


def _leaves(d, prefix=""):
    if isinstance(d, dict):
        for k, v in d.items():
            yield from _leaves(v, f"{prefix}.{k}" if prefix else str(k))
    elif isinstance(d, list):
        for i, v in enumerate(d):
            yield from _leaves(v, f"{prefix}[{i}]")
    else:
        yield prefix, d


def compare(new: dict, old: dict, tol: float = 0.0) -> dict:
    """Leaf-by-leaf comparison of two result blocks."""
    a = dict(_leaves(new))
    b = dict(_leaves(old))
    diffs, max_abs = [], 0.0
    for k in sorted(set(a) | set(b)):
        va, vb = a.get(k), b.get(k)
        if isinstance(va, (int, float)) and isinstance(vb, (int, float)) \
                and not isinstance(va, bool) and not isinstance(vb, bool):
            gap = abs(float(va) - float(vb))
            max_abs = max(max_abs, gap)
            if gap > tol:
                diffs.append({"key": k, "new": va, "old": vb})
        elif va != vb:
            diffs.append({"key": k, "new": va, "old": vb})
    return {"n_compared": len(set(a) | set(b)), "n_differences": len(diffs),
            "max_abs_numeric_difference": round(max_abs, 6),
            "identical": not diffs, "differences": diffs[:40]}


# --------------------------------------------------------------------------
# The no-covariate model, as audit_model defines it
# --------------------------------------------------------------------------

def _flat(idata, name: str) -> np.ndarray:
    v = idata.posterior[name].to_numpy()
    return v.reshape((-1,) + v.shape[2:])


def summarize_stage1(idata, table: np.ndarray, prior_name: str,
                     dependence: float | None) -> dict:
    """The same summary `audit_model.run_no_covariate` returns, from a kept idata."""
    var_names = ["prevalence", "cutpoints", "kabco_shift", "narrative_p"]
    if dependence is None:
        var_names.append("dependence")
    diag = am.diagnostics(idata, var_names)
    implied = am.posterior_predictive_table(idata, int(table.sum()))
    prev = _flat(idata, "prevalence")
    narr = _flat(idata, "narrative_p")

    dep_post = None
    if dependence is None:
        v = _flat(idata, "dependence")
        import arviz as az
        s = az.summary(idata, var_names=["dependence"])
        dep_post = {
            "mean": round(float(v.mean()), 4),
            "sd": round(float(v.std(ddof=1)), 4),
            "ci95": [round(float(np.percentile(v, 2.5)), 4),
                     round(float(np.percentile(v, 97.5)), 4)],
            "excludes_zero": bool(np.percentile(v, 2.5) > 0
                                  or np.percentile(v, 97.5) < 0),
            "r_hat": round(float(s["r_hat"].iloc[0]), 4),
            "ess_bulk": int(s["ess_bulk"].iloc[0]),
            "ess_tail": int(s["ess_tail"].iloc[0]),
            "prior": "Normal(0, 1)",
        }

    return {
        "prior_set": prior_name,
        "priors": am.PRIOR_SETS[prior_name],
        "dependence_fixed_at": dependence,
        "dependence_posterior": dep_post,
        "n": int(table.sum()),
        "diagnostics": diag,
        "fit_against_raw_table": am.check_against_raw(implied, table),
        "class_alignment": am.check_class_alignment(idata),
        "latent_prevalence": {
            SEV3[i]: {"mean": round(float(prev[:, i].mean()), 4),
                      "ci95": [round(float(np.percentile(prev[:, i], 2.5)), 4),
                               round(float(np.percentile(prev[:, i], 97.5)), 4)]}
            for i in range(3)},
        "narrative_measurement": {
            SEV3[i]: {N_LEVELS[j]: round(float(narr[:, i, j].mean()), 4)
                      for j in range(4)}
            for i in range(3)},
    }


def fit_stage1(table: np.ndarray, prior_name: str, dependence: float | None,
               draws: int = am.DRAWS, tune: int = am.TUNE) -> tuple[dict, object, float]:
    model = am.build_no_covariate_model(table, am.PRIOR_SETS[prior_name],
                                        dependence=dependence)
    t0 = time.perf_counter()
    idata = am.sample(model, draws=draws, tune=tune)
    secs = time.perf_counter() - t0
    res = summarize_stage1(idata, table, prior_name, dependence)
    res["run_seconds"] = round(secs, 1)
    res["draws_per_chain"], res["tune"] = draws, tune
    return res, idata, secs


# --------------------------------------------------------------------------
# Posterior latent-class probabilities
# --------------------------------------------------------------------------

def cell_class_probs(prev: np.ndarray, kabco_p: np.ndarray,
                     narr_p: np.ndarray) -> np.ndarray:
    """P(latent class | KABCO cell, narrative cell) per draw, shape (S, 5, 4, 3)."""
    joint = (prev[:, None, None, :]
             * np.transpose(kabco_p, (0, 2, 1))[:, :, None, :]
             * np.transpose(narr_p, (0, 2, 1))[:, None, :, :])
    return joint / joint.sum(axis=3, keepdims=True)


def thin_indices(total: int, k: int = N_POSTERIOR_DRAWS) -> np.ndarray:
    """k draws evenly spaced across all chains, not the head of the first chain."""
    return np.unique(np.linspace(0, total - 1, k).round().astype(int))


def class_probabilities(idata, kabco_idx: np.ndarray, narr_idx: np.ndarray) -> dict:
    prev = _flat(idata, "prevalence")
    kabco_p = _flat(idata, "kabco_p")
    narr_p = _flat(idata, "narrative_p")
    cells = cell_class_probs(prev, kabco_p, narr_p)               # (S, 5, 4, 3)
    per_rider = cells[:, kabco_idx, narr_idx, :]                  # (S, n, 3)
    idx = thin_indices(per_rider.shape[0])
    # With the dependence term free, `prevalence` is no longer the marginal class share,
    # because exp(lambda * concordance) is applied after the mixture and renormalized over
    # the whole table. The share implied for these riders is reported beside it.
    share = per_rider.mean(axis=1)                                # (S, 3)
    implied_share = {SEV3[c]: {"mean": round(float(share[:, c].mean()), 4),
                               "ci95": [round(float(np.percentile(share[:, c], 2.5)), 4),
                                        round(float(np.percentile(share[:, c], 97.5)), 4)]}
                     for c in range(3)}
    return {"cells": cells, "mean": per_rider.mean(axis=0),
            "implied_class_share": implied_share,
            "draws": per_rider[idx], "draw_indices": idx,
            "n_total_draws": int(per_rider.shape[0])}


def cell_table(cells: np.ndarray, observed: np.ndarray) -> list[dict]:
    """The twenty cell-level weight vectors, with posterior intervals."""
    rows = []
    for k, kl in enumerate(KABCO):
        for j, nl in enumerate(N_LEVELS):
            v = cells[:, k, j, :]
            rows.append({
                "kabco": kl, "narrative": nl, "n_riders": int(observed[k, j]),
                "posterior_mean": {SEV3[c]: round(float(v[:, c].mean()), 4)
                                   for c in range(3)},
                "ci95": {SEV3[c]: [round(float(np.percentile(v[:, c], 2.5)), 4),
                                   round(float(np.percentile(v[:, c], 97.5)), 4)]
                         for c in range(3)},
                "modal_class": SEV3[int(np.argmax(v.mean(axis=0)))],
            })
    return rows


def reassignment_summary(mean: np.ndarray, sev3: np.ndarray) -> dict:
    coded = np.array([SEV3.index(s) for s in sev3])
    modal = mean.argmax(axis=1)
    p_coded = mean[np.arange(len(coded)), coded]
    cross = pd.crosstab(pd.Categorical([SEV3[c] for c in coded], categories=SEV3),
                        pd.Categorical([SEV3[c] for c in modal], categories=SEV3),
                        dropna=False)
    return {
        "coded_counts": {s: int((coded == i).sum()) for i, s in enumerate(SEV3)},
        "modal_counts": {s: int((modal == i).sum()) for i, s in enumerate(SEV3)},
        "riders_whose_modal_class_differs_from_coded": int((modal != coded).sum()),
        "share_modal_differs": round(float((modal != coded).mean()), 3),
        "coded_by_modal": {r: {c: int(cross.loc[r, c]) for c in SEV3} for r in SEV3},
        "posterior_prob_of_coded_class": {
            "mean": round(float(p_coded.mean()), 4),
            "min": round(float(p_coded.min()), 4),
            "median": round(float(np.median(p_coded)), 4),
            "by_coded_class": {s: round(float(p_coded[coded == i].mean()), 4)
                               for i, s in enumerate(SEV3)}},
    }


# --------------------------------------------------------------------------
# Narrative-length sensitivity
# --------------------------------------------------------------------------

def build_length_model(kabco: np.ndarray, narr: np.ndarray, z: np.ndarray,
                       priors: dict, class_specific: bool = True,
                       prevalence_slopes: bool = True):
    """Individual-level audit model whose narrative rows depend on narrative length.

    The KABCO block, the Dirichlet rows (now the rows at mean length) and the class
    prevalence are as in `audit_model.build_no_covariate_model`. The latent class is
    summed out. `class_specific` gives each latent class its own three length slopes
    (nine in all); otherwise the three slopes are shared by every class, which changes
    how often each narrative level appears with length but not how well it separates
    the classes. Slopes take the reporting-shifter prior scale.
    """
    import pymc as pm
    import pytensor.tensor as pt

    n = len(kabco)
    onehot = np.eye(4)[narr]                                         # (n, 4)

    with pm.Model() as model:
        prevalence = pm.Dirichlet("prevalence", a=np.ones(3))

        cutpoints = pm.Normal(
            "cutpoints", mu=np.linspace(-1.5, 2.0, 4), sigma=2.0, shape=4,
            transform=pm.distributions.transforms.ordered,
            initval=np.linspace(-1.5, 2.0, 4))
        step_bc = pm.HalfNormal("kabco_step_bc", sigma=priors["class_sd"])
        step_ka = pm.HalfNormal("kabco_step_ka", sigma=priors["class_sd"])
        shift = pm.Deterministic(
            "kabco_shift", pt.stack([pt.zeros(()), step_bc, step_bc + step_ka]))
        eta = cutpoints[None, :] - shift[:, None]
        cum = pm.math.sigmoid(eta)
        kabco_p = pm.Deterministic("kabco_p", pt.concatenate(
            [cum[:, :1], cum[:, 1:] - cum[:, :-1], 1.0 - cum[:, -1:]], axis=1))

        alpha = np.full((3, 4), priors["n_off"])
        for row, col in enumerate((1, 2, 3)):
            alpha[row, col] = priors["n_diag"]
            alpha[row, 0] = priors["n_diag"]
        narrative_p = pm.Dirichlet("narrative_p", a=alpha, shape=(3, 4))

        if class_specific:
            g = pm.Normal("narr_length_slope", mu=0.0, sigma=priors["shift_sd"],
                          shape=(3, 3), initval=np.zeros((3, 3)))
            gamma = pt.concatenate([pt.zeros((3, 1)), g], axis=1)    # (3, 4)
        else:
            g = pm.Normal("narr_length_slope", mu=0.0, sigma=priors["shift_sd"],
                          shape=3, initval=np.zeros(3))
            gamma = pt.concatenate([pt.zeros(1), g])[None, :] * pt.ones((3, 1))

        logit_n = pt.log(narrative_p)[None, :, :] + z[:, None, None] * gamma[None, :, :]
        narr_i = pt.special.softmax(logit_n, axis=2)                 # (n, 3, 4)
        lik_narr = pt.sum(narr_i * onehot[:, None, :], axis=2)      # (n, 3)

        if prevalence_slopes:
            b = pm.Normal("prev_length_slope", mu=0.0, sigma=priors["shift_sd"],
                          shape=2, initval=np.zeros(2))
            logit_p = (pt.log(prevalence)[None, :]
                       + z[:, None] * pt.concatenate([pt.zeros(1), b])[None, :])
            prev_i = pt.special.softmax(logit_p, axis=1)             # (n, 3)
        else:
            prev_i = pt.ones((n, 1)) * prevalence[None, :]

        lik_kabco = kabco_p[:, kabco].T                              # (n, 3)
        joint = prev_i * lik_kabco * lik_narr
        pm.Potential("obs", pt.sum(pt.log(pt.clip(pt.sum(joint, axis=1),
                                                  1e-300, np.inf))))
    return model


def _softmax(x: np.ndarray, axis: int) -> np.ndarray:
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def length_model_quantities(idata, z: np.ndarray, kabco_idx: np.ndarray,
                            narr_idx: np.ndarray, class_specific: bool,
                            prevalence_slopes: bool) -> dict:
    """Implied 5 by 4 table and per-rider class probabilities from posterior draws."""
    prev = _flat(idata, "prevalence")
    kabco_p = _flat(idata, "kabco_p")
    narr_p = _flat(idata, "narrative_p")
    g = _flat(idata, "narr_length_slope")
    S = prev.shape[0]
    idx = np.unique(np.linspace(0, S - 1, IMPLIED_TABLE_DRAWS).round().astype(int))

    implied = np.zeros((5, 4))
    post_mean = np.zeros((len(z), 3))
    for s in idx:
        gamma = (np.concatenate([np.zeros((3, 1)), g[s]], axis=1) if class_specific
                 else np.tile(np.concatenate([[0.0], g[s]]), (3, 1)))
        narr_i = _softmax(np.log(narr_p[s])[None] + z[:, None, None] * gamma[None], 2)
        if prevalence_slopes:
            b = _flat(idata, "prev_length_slope")[s]
            prev_i = _softmax(np.log(prev[s])[None] + z[:, None]
                              * np.concatenate([[0.0], b])[None], 1)
        else:
            prev_i = np.tile(prev[s], (len(z), 1))
        # P(k, n | z_i) summed over riders.
        implied += np.einsum("ic,ck,icn->kn", prev_i, kabco_p[s], narr_i)
        joint = prev_i * kabco_p[s][:, kabco_idx].T * narr_i[np.arange(len(z)), :, narr_idx]
        post_mean += joint / joint.sum(axis=1, keepdims=True)
    return {"implied": implied / len(idx), "class_prob_mean": post_mean / len(idx),
            "n_draws_used": int(len(idx))}


def fit_length_model(frame: pd.DataFrame, table: np.ndarray, kabco_idx, narr_idx,
                     class_specific: bool, prevalence_slopes: bool,
                     prior_name: str = "moderate", draws: int = am.DRAWS,
                     tune: int = am.TUNE) -> tuple[dict, np.ndarray]:
    z = frame["narrative_chars_z"].astype(float).to_numpy()
    priors = am.PRIOR_SETS[prior_name]
    model = build_length_model(kabco_idx, narr_idx, z, priors,
                               class_specific=class_specific,
                               prevalence_slopes=prevalence_slopes)
    t0 = time.perf_counter()
    try:
        idata = am.sample(model, draws=draws, tune=tune)
    except Exception as exc:                                # recorded, not hidden
        return {"estimated": False, "error": repr(exc),
                "run_seconds": round(time.perf_counter() - t0, 1)}, None
    secs = time.perf_counter() - t0

    var_names = ["prevalence", "cutpoints", "kabco_shift", "narrative_p",
                 "narr_length_slope"]
    if prevalence_slopes:
        var_names.append("prev_length_slope")
    diag = am.diagnostics(idata, var_names)
    q = length_model_quantities(idata, z, kabco_idx, narr_idx,
                                class_specific, prevalence_slopes)
    fitcheck = am.check_against_raw(q["implied"], table)
    align = am.check_class_alignment(idata)

    g = _flat(idata, "narr_length_slope")
    slopes = {}
    if class_specific:
        for c in range(3):
            for j in range(1, 4):
                v = g[:, c, j - 1]
                slopes[f"{SEV3[c]}|{N_LEVELS[j]}"] = _post(v)
    else:
        for j in range(1, 4):
            slopes[f"all|{N_LEVELS[j]}"] = _post(g[:, j - 1])
    prev_slopes = None
    if prevalence_slopes:
        b = _flat(idata, "prev_length_slope")
        prev_slopes = {SEV3[c + 1]: _post(b[:, c]) for c in range(2)}

    prev = _flat(idata, "prevalence")
    narr = _flat(idata, "narrative_p")
    res = {
        "estimated": True,
        "prior_set": prior_name,
        "class_specific_narrative_slopes": class_specific,
        "prevalence_depends_on_length": prevalence_slopes,
        "n_length_parameters": (9 if class_specific else 3) + (2 if prevalence_slopes else 0),
        "length_variable": "narrative_chars_z",
        "run_seconds": round(secs, 1),
        "draws_per_chain": draws, "tune": tune,
        "diagnostics": diag,
        "fit_against_raw_table": fitcheck,
        "class_alignment": {k: align[k] for k in
                            ("kabco_measurement", "p_severe_given_class",
                             "severity_ordering_holds", "aligned")},
        "narrative_length_slopes": slopes,
        "n_narrative_slopes_excluding_zero": int(sum(v["excludes_zero"]
                                                     for v in slopes.values())),
        "prevalence_length_slopes": prev_slopes,
        "latent_prevalence_at_mean_length": {
            SEV3[i]: {"mean": round(float(prev[:, i].mean()), 4),
                      "ci95": [round(float(np.percentile(prev[:, i], 2.5)), 4),
                               round(float(np.percentile(prev[:, i], 97.5)), 4)]}
            for i in range(3)},
        "narrative_measurement_at_mean_length": {
            SEV3[i]: {N_LEVELS[j]: round(float(narr[:, i, j].mean()), 4)
                      for j in range(4)} for i in range(3)},
        "implied_table_draws": q["n_draws_used"],
    }
    return res, q["class_prob_mean"]


def _post(v: np.ndarray) -> dict:
    lo, hi = np.percentile(v, [2.5, 97.5])
    return {"mean": round(float(v.mean()), 4),
            "ci95": [round(float(lo), 4), round(float(hi), 4)],
            "excludes_zero": bool(lo > 0 or hi < 0)}


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main() -> dict:
    started = time.perf_counter()
    frame, input_check = load_inputs()
    table = am.observed_table(frame)
    kabco_idx = np.array([KABCO.index(k) for k in frame["sev_kabco"]])
    narr_idx = np.array([N_LEVELS.index(n) for n in frame["narrative_indicator"]])

    numbers = json.loads((DATA / "numbers.json").read_text(encoding="utf-8"))
    old_stage1 = json.loads((DATA / "audit_stage1.json").read_text(encoding="utf-8"))

    # Cross-tabulation against numbers.json.
    ct_kabco = ai.cross_tabulation(frame, "sev_kabco", ai.KABCO_ORDER)
    ct_sev3 = ai.cross_tabulation(frame, "sev3", ai.SEV3_ORDER)
    crosstab_check = {
        "cross_tab_kabco": compare(ct_kabco, numbers["audit"]["cross_tab_kabco"]),
        "cross_tab_sev3": compare(ct_sev3, numbers["audit"]["cross_tab_sev3"]),
    }
    print("cross-tab matches numbers.json:",
          crosstab_check["cross_tab_kabco"]["identical"],
          crosstab_check["cross_tab_sev3"]["identical"])

    out: dict = {
        "generated": datetime.now().isoformat(timespec="seconds"),
        "script": "src/audit_final.py",
        "inputs": ["data/model_frame.csv", "data/narrative_indicator.csv",
                   "data/final_labels.csv", "data/numbers.json",
                   "data/audit_stage1.json"],
        "outputs": ["data/audit_final.json",
                    "data/audit_posterior_classprobs.npz"],
        "settings": {"draws": am.DRAWS, "tune": am.TUNE, "chains": am.CHAINS,
                     "seed": am.SEED, "target_accept": 0.95, "sampler": "nutpie",
                     "prior_sets": am.PRIOR_SETS,
                     "n_posterior_draws_saved": N_POSTERIOR_DRAWS,
                     "draw_thinning": "evenly spaced over all chains and draws",
                     "sensitivity_fits": {"draws": LONG_DRAWS, "tune": LONG_TUNE,
                                          "when": "every sensitivity fit; the long "
                                                  "run is the reported one"}},
        "input_check": input_check,
        "observed_table": {"rows": list(KABCO), "columns": list(N_LEVELS),
                           "counts": table.astype(int).tolist()},
        "crosstab_vs_numbers_json": crosstab_check,
    }

    # 1. Primary specification, dependence fixed at zero, both priors.
    primary, saved = {}, {}
    for prior_name in ("moderate", "weak"):
        res, idata, secs = fit_stage1(table, prior_name, 0.0)
        cp = class_probabilities(idata, kabco_idx, narr_idx)
        res["class_probability_cells"] = cell_table(cp["cells"], table)
        res["reassignment"] = reassignment_summary(cp["mean"], frame["sev3"].to_numpy())
        res["model_implied_class_share"] = cp["implied_class_share"]
        primary[prior_name] = res
        saved[prior_name] = cp
        print(f"{prior_name}: {secs:.0f}s, r-hat {res['diagnostics']['max_r_hat']}, "
              f"div {res['diagnostics']['divergences']}, "
              f"max Pearson {res['fit_against_raw_table']['max_abs_pearson_residual']}")

    # Reproduction against what numbers.json and the preliminary file carry.
    mod = primary["moderate"]
    new_mod = {"prevalence": mod["latent_prevalence"],
               "narrative_measurement": mod["narrative_measurement"],
               "class_alignment": mod["class_alignment"],
               "diagnostics": mod["diagnostics"],
               "fit_against_raw_table": {
                   k: mod["fit_against_raw_table"][k]
                   for k in ("max_abs_pearson_residual", "reproduces_raw_table")}}
    weak = primary["weak"]
    old_w = old_stage1["weak"]
    keys = ("latent_prevalence", "narrative_measurement", "class_alignment",
            "diagnostics", "fit_against_raw_table")
    out["reproduction"] = {
        "moderate_vs_numbers_json_audit_stage1": compare(new_mod, numbers["audit"]["stage1"]),
        "weak_vs_audit_stage1": compare(
            {k: weak[k] for k in keys}, {k: old_w[k] for k in keys}),
        "moderate_vs_audit_stage1": compare(
            {k: mod[k] for k in keys}, {k: old_stage1["moderate"][k] for k in keys}),
    }
    out["primary"] = primary

    # 2. Conditional dependence, estimated, both priors; and a fixed grid (fit check).
    dep = {}
    dep_saved = {}
    for prior_name in ("moderate", "weak"):
        short, _, _ = fit_stage1(table, prior_name, None)
        res, idata, secs = fit_stage1(table, prior_name, None, LONG_DRAWS, LONG_TUNE)
        cp = class_probabilities(idata, kabco_idx, narr_idx)
        res["class_probability_cells"] = cell_table(cp["cells"], table)
        res["reassignment"] = reassignment_summary(cp["mean"], frame["sev3"].to_numpy())
        res["model_implied_class_share"] = cp["implied_class_share"]
        res["latent_prevalence_note"] = (
            "With lambda estimated, the prevalence parameter is not the marginal class "
            "share; read model_implied_class_share instead.")
        res["pipeline_settings_run"] = {k: short[k] for k in (
            "dependence_posterior", "diagnostics", "latent_prevalence",
            "run_seconds", "draws_per_chain", "tune")}
        res["pipeline_settings_run"]["fit_against_raw_table"] = {
            k: short["fit_against_raw_table"][k]
            for k in ("max_abs_pearson_residual", "total_abs_count_error",
                      "max_abs_count_error", "reproduces_raw_table")}
        dep[prior_name] = res
        dep_saved[prior_name] = cp
        dp = res["dependence_posterior"]
        print(f"dependence {prior_name}: {dp['mean']} {dp['ci95']} r-hat {dp['r_hat']} "
              f"ESS {dp['ess_bulk']}, div {res['diagnostics']['divergences']}, "
              f"max Pearson {res['fit_against_raw_table']['max_abs_pearson_residual']}")
    grid = {}
    for val in FIXED_DEPENDENCE_GRID:
        res, _, secs = fit_stage1(table, "moderate", val)
        grid[str(val)] = {
            "diagnostics": res["diagnostics"],
            "fit_against_raw_table": {
                k: res["fit_against_raw_table"][k]
                for k in ("max_abs_pearson_residual", "total_abs_count_error",
                          "max_abs_count_error", "reproduces_raw_table")},
            "latent_prevalence": res["latent_prevalence"],
            "class_alignment_aligned": res["class_alignment"]["aligned"],
            "run_seconds": res["run_seconds"],
        }
        print(f"dependence fixed at {val}: max Pearson "
              f"{grid[str(val)]['fit_against_raw_table']['max_abs_pearson_residual']}")
    out["dependence_sensitivity"] = {
        "estimated": dep,
        "fixed_grid_moderate": grid,
        "parameterization": ("log-scale boost lambda on the cells where the KABCO code "
                             "and the narrative level imply the same severity position "
                             "(O; C or B with BC-consistent; A or K with KA-consistent); "
                             "the 'no cue' column never receives it. The factor is common "
                             "to all latent classes, so it cancels from the per-rider "
                             "class probabilities except through the other parameters."),
        "earlier_console_value_not_reproducible_from_files": (
            "The manuscript value +0.70 [0.23, 1.22] came from an unsaved run on "
            "2026-09-14 on pre-final labels; the values above replace it."),
    }

    # 3. Narrative-length sensitivity.
    length = {}
    length_probs = {}
    for tag, cs, ps in (("class_specific_with_prevalence_slopes", True, True),
                        ("class_specific_no_prevalence_slopes", True, False),
                        ("class_common_with_prevalence_slopes", False, True)):
        short, _ = fit_length_model(frame, table, kabco_idx, narr_idx, cs, ps)
        res, pm_ = fit_length_model(frame, table, kabco_idx, narr_idx, cs, ps,
                                    draws=LONG_DRAWS, tune=LONG_TUNE)
        res["pipeline_settings_run"] = short
        length[tag] = res
        if pm_ is not None:
            length_probs[tag] = pm_
        if res.get("estimated"):
            print(f"length {tag}: {res['run_seconds']}s r-hat "
                  f"{res['diagnostics']['max_r_hat']} ESS {res['diagnostics']['min_ess_bulk']} "
                  f"div {res['diagnostics']['divergences']} max Pearson "
                  f"{res['fit_against_raw_table']['max_abs_pearson_residual']} "
                  f"slopes excl 0: {res['n_narrative_slopes_excluding_zero']}")
        else:
            print(f"length {tag}: failed {res.get('error')}")
    out["narrative_length_sensitivity"] = {
        "association_length_with_coded_kabco": {
            k: round(float(frame.loc[frame["sev_kabco"] == k, "narrative_chars_z"].mean()), 3)
            for k in KABCO},
        "association_length_with_narrative_level": {
            n: round(float(frame.loc[frame["narrative_indicator"] == n,
                                     "narrative_chars_z"].mean()), 3)
            for n in N_LEVELS},
        "length_z_max": round(float(frame["narrative_chars_z"].max()), 2),
        "models": length,
        "note": ("Primary length sensitivity is class-specific narrative slopes with the "
                 "latent prevalence also logit-linear in length. Without the prevalence "
                 "slopes, the association between length and coded severity can only be "
                 "absorbed by the narrative slopes."),
    }

    # Saved posterior class probabilities, keyed by rider.
    npz_path = DATA / "audit_posterior_classprobs.npz"
    arrays = {
        "rider_row_id": frame["rider_row_id"].astype(str).to_numpy().astype("U"),
        "crash_id": frame["Crash_ID"].to_numpy(),
        "sev_kabco": frame["sev_kabco"].astype(str).to_numpy().astype("U"),
        "sev3": frame["sev3"].astype(str).to_numpy().astype("U"),
        "narrative_indicator": frame["narrative_indicator"].astype(str).to_numpy().astype("U"),
        "classes": np.array(SEV3, dtype="U"),
        "moderate_mean": saved["moderate"]["mean"],
        "moderate_draws": saved["moderate"]["draws"],
        "moderate_draw_indices": saved["moderate"]["draw_indices"],
        "weak_mean": saved["weak"]["mean"],
        "weak_draws": saved["weak"]["draws"],
        "weak_draw_indices": saved["weak"]["draw_indices"],
        "dependence_moderate_mean": dep_saved["moderate"]["mean"],
        "dependence_moderate_draws": dep_saved["moderate"]["draws"],
    }
    for tag, pm_ in length_probs.items():
        arrays[f"length_{tag}_mean"] = pm_
    np.savez_compressed(npz_path, **arrays)
    out["saved_class_probabilities"] = {
        "file": "data/audit_posterior_classprobs.npz",
        "key": "rider_row_id (row order of model_frame.csv); crash_id alongside",
        "class_order": list(SEV3),
        "primary_arrays": "moderate_mean (522, 3) and moderate_draws (100, 522, 3)",
        "draw_indices_moderate": saved["moderate"]["draw_indices"].tolist(),
        "n_total_posterior_draws": saved["moderate"]["n_total_draws"],
        "arrays": sorted(arrays),
    }
    out["total_run_seconds"] = round(time.perf_counter() - started, 1)

    path = DATA / "audit_final.json"
    path.write_text(json.dumps(out, indent=2, default=float), encoding="utf-8")
    print(f"written to {path} and {npz_path} ({out['total_run_seconds']} s)")
    return out


if __name__ == "__main__":
    main()
