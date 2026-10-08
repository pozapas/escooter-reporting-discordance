"""Supporting results that no other output contains.

One top-level key per item, written to `data/supporting_analyses.json`:

1. `annual` - riders and crashes per year, severity composition, and the yearly
   distribution of every covariate in the primary specification.
2. `cv_fold_level` - the 10 x 5 repeated stratified cross-validated log loss per fold for
   M0, M0c and M1, and the paired per-fold differences.
3. `transferability_detail` - per subgroup: n, severity counts, estimable parameters,
   convergence and separation, for every temporal breakpoint and the spatial split, and
   the exact membership of the "large city" group.
4. `benchmark_detail` - grid, resampling scheme, per-class precision, recall and F1, and
   what the code does about calibration.
5. `extraction_intervals_calibration` - per-cue bootstrap intervals for every route, the
   macro-F1 intervals, and a cross-fitted out-of-fold calibration of the Route B vote
   shares with reliability bins.
6. `data_sensitivities` - age bands and splines, complete-case, missingness mechanism,
   speed-limit treatment, ACS vintage and bike-facility buffers, four- and five-class
   orderings.
7. `facts` - extraction date, ACS product, riders against crashes, cue counts by severity,
   and the large-city definition.

Nothing here re-implements a fit. Every model goes through `severity.py`; the subgroup
screen goes through `transferability.estimable_in_both`; the four-class model goes
through `severity_appendix.four_class_model`. The bootstraps of the transferability tests
are not rerun; their stored p-values are carried over and the likelihood-ratio statistic
is recomputed only to confirm that the subgroup fits reported here are the ones tested.

The file is rewritten after every item so that a late failure does not lose finished
items.
"""

from __future__ import annotations

import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))

import severity as sv                      # noqa: E402
import severity_appendix as sa             # noqa: E402
import transferability as tr               # noqa: E402
import validation_metrics as vm            # noqa: E402
from labels import ALL_LABELS, INJURY_CUES, MECHANISMS   # noqa: E402

ROOT = sv.ROOT
DATA = sv.DATA
OUT = DATA / "supporting_analyses.json"
SCRIPT = "src/supporting_analyses.py"

FRAME_PATH = DATA / "model_frame.csv"
BASE_PATH = DATA / "base_frame.csv"

# The spatial split in `transferability.spatial_test` is a literal set inside the
# function and is not exported. It is restated here and checked against the stored
# group size, so a change to one without the other fails loudly.
LARGE_CITY_SET = ("Austin", "Houston", "San Antonio", "Dallas", "Fort Worth", "El Paso")

CAL_SEED = 20260919
CAL_FOLDS = 5
CAL_MIN_POSITIVES = 5
ECE_BINS = 10

FIVE_CLASS_ORDER = ("O", "C", "B", "A", "K")
# Fixed before the five-class fit: a model is called estimable only if the optimizer
# converged, the separation screen is clean, and the smallest class carries at least one
# event per parameter.
FIVE_CLASS_EPP_FLOOR = 1.0


# --------------------------------------------------------------------------
# Output handling
# --------------------------------------------------------------------------

def _clean(obj):
    """JSON-safe copy: numpy scalars to Python, NaN and infinities to None."""
    if isinstance(obj, dict):
        return {str(k): _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        v = float(obj)
        return None if not math.isfinite(v) else v
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return _clean(obj.tolist())
    return obj


RESULT: dict = {}


def save() -> None:
    OUT.write_text(json.dumps(_clean(RESULT), indent=2), encoding="utf-8")


def r(x, k: int = 4):
    return None if x is None or (isinstance(x, float) and not math.isfinite(x)) \
        else round(float(x), k)


def load_frame() -> pd.DataFrame:
    frame = pd.read_csv(FRAME_PATH)
    base = pd.read_csv(BASE_PATH, usecols=["rider_row_id", "rider_age",
                                           "Prsn_Ethnicity_ID", "Crash_Speed_Limit"])
    frame = frame.merge(base, on="rider_row_id", how="left", validate="one_to_one")
    if len(frame) != 522 or frame["rider_age"].notna().sum() != 488:
        raise SystemExit("rider-level merge with base_frame did not reproduce 522 rows "
                         "and 488 recorded ages")
    eth_raw = frame["Prsn_Ethnicity_ID"]
    frame["eth_recorded_unknown"] = (eth_raw.isna()
                                     | (eth_raw.astype(str).str.strip() == "Unknown")
                                     ).astype(int)
    return frame


def cells_by_sev(frame: pd.DataFrame, mask, outcome: str = "sev3",
                 order=sv.SEV_ORDER) -> dict:
    sub = frame.loc[np.asarray(mask, dtype=bool), outcome]
    return {k: int(v) for k, v in sub.value_counts().reindex(list(order))
            .fillna(0).items()}


# --------------------------------------------------------------------------
# 1. Annual table
# --------------------------------------------------------------------------

AGE_LEVEL_LABELS = {"under25": "under 25", "25to54": "25 to 44 (reference)",
                    "45plus": "45 and over", "unknown": "unrecorded"}
LEVEL_COLUMNS = ("speed_level", "light_level", "eth_level", "gender_level", "age_level",
                 "veh_level", "coll_level", "intx_level", "ctrl_level", "road_level")


def item_annual(frame: pd.DataFrame) -> dict:
    base = tr.annual_table(frame)
    cols = sv.ladder(frame)["M1"]
    years = sorted(int(y) for y in frame["crash_year"].unique())

    riders = frame.groupby("crash_year").size()
    crashes = frame.groupby("crash_year")["Crash_ID"].nunique()

    binaries, continuous = {}, {}
    for c in cols:
        vals = frame[c]
        if set(np.unique(vals.dropna())) <= {0.0, 1.0}:
            g = frame.groupby("crash_year")[c]
            binaries[c] = {str(y): {"count": int(g.sum()[y]),
                                    "share": r(g.mean()[y], 3)} for y in years}
            binaries[c]["all_years"] = {"count": int(vals.sum()),
                                        "share": r(vals.mean(), 3)}
        else:
            per = {}
            for y in years + ["all_years"]:
                v = vals if y == "all_years" else vals[frame["crash_year"] == y]
                q1, med, q3 = np.percentile(v, [25, 50, 75])
                per[str(y)] = {"median": r(med, 3), "q1": r(q1, 3), "q3": r(q3, 3),
                               "mean": r(v.mean(), 3)}
            continuous[c] = per

    levels = {}
    for lc in LEVEL_COLUMNS:
        tab = pd.crosstab(frame[lc], frame["crash_year"])
        entry = {}
        for lev in tab.index:
            name = AGE_LEVEL_LABELS.get(lev, lev) if lc == "age_level" else lev
            entry[name] = {str(y): int(tab.loc[lev, y]) for y in years}
        levels[lc] = entry

    nj = json.loads((DATA / "numbers.json").read_text(encoding="utf-8"))
    nj_annual = nj["transferability"]["annual"]
    matches = all(nj_annual[str(y)][k] == base["counts"][y][k]
                  for y in years for k in sv.SEV_ORDER)

    return {
        "inputs": ["data/model_frame.csv", "data/numbers.json",
                   "data/screening.json"],
        "unit_note": ("Counts are rider rows (one row per injured e-scooter rider), not "
                      "crashes. Two crashes carry two riders, so 522 rider rows come from "
                      "520 crashes. Both counts are given per year; severity is a rider "
                      "attribute and is counted on rider rows."),
        "years": years,
        "rider_rows_per_year": {str(y): int(riders[y]) for y in years},
        "distinct_crashes_per_year": {str(y): int(crashes[y]) for y in years},
        "severity_counts": {str(y): base["counts"][y] for y in years},
        "severity_share_percent": {str(y): base["severity_share_percent"][y]
                                   for y in years},
        "severity_counts_match_numbers_json_transferability_annual": bool(matches),
        "binary_covariates_by_year": binaries,
        "continuous_covariates_by_year": continuous,
        "categorical_levels_by_year": levels,
        "age_reference_note": ("The model frame names the age reference level '25to54', "
                               "but the stage-two merge set the older band at 45 and over, "
                               "so the reference level covers ages 25 to 44. It is "
                               "labelled 25 to 44 here."),
        "covariates_reported": cols,
        "note_2021": base["note"],
    }


# --------------------------------------------------------------------------
# 2. Fold-level cross-validation
# --------------------------------------------------------------------------

def _diff_summary(a: np.ndarray, b: np.ndarray, reps: int, folds: int) -> dict:
    d = a - b
    by_rep = d.reshape(reps, folds).mean(axis=1)
    return {
        "per_fold": [r(v, 5) for v in d],
        "mean": r(d.mean(), 5),
        "sd": r(d.std(ddof=1), 5),
        "min": r(d.min(), 5),
        "max": r(d.max(), 5),
        "share_of_folds_first_better": r((d < 0).mean(), 3),
        "n_folds_first_better": int((d < 0).sum()),
        "per_repeat_mean": [r(v, 5) for v in by_rep],
        "n_repeats_first_better": int((by_rep < 0).sum()),
    }


def item_cv(frame: pd.DataFrame) -> dict:
    spec = sv.ladder(frame)
    stored = json.loads((DATA / "severity.json").read_text(encoding="utf-8"))
    out: dict = {"inputs": ["data/model_frame.csv", "data/screening.json",
                            "data/severity.json"],
                 "function": "severity.cross_validated_log_loss (ordered logit, free=None)",
                 "repeats": sv.CV_REPEATS, "folds": sv.CV_FOLDS, "seed": sv.CV_SEED,
                 "rungs": {}, "skipped": {"M2": "not part of this task; not estimated"}}

    ys = {}
    for rung in ("M0", "M0c", "M1"):
        cols = spec[rung]
        _, y = sv._design(frame, cols)
        ys[rung] = y
        cv = sv.cross_validated_log_loss(frame, cols)
        st = stored["rungs"][rung]["cross_validation"]
        out["rungs"][rung] = {
            "n_covariates": len(cols),
            "mean_log_loss": cv["mean_log_loss"],
            "sd_across_folds": cv["sd_across_folds"],
            "min_log_loss": cv["min_log_loss"],
            "max_log_loss": cv["max_log_loss"],
            "n_folds_failed": cv["n_folds_failed"],
            "fold_level": cv["fold_level"],
            "reproduces_stored_mean": bool(cv["mean_log_loss"] == st["mean_log_loss"]),
            "stored_mean": st["mean_log_loss"],
        }
        if cv["n_folds_failed"]:
            out["rungs"][rung]["warning"] = "some folds failed; differences use ok folds only"

    same_y = all(np.array_equal(ys["M0"], ys[k]) for k in ys)
    sizes = {k: [f["n_test"] for f in out["rungs"][k]["fold_level"]] for k in ys}
    same_sizes = all(sizes["M0"] == sizes[k] for k in sizes)
    out["folds_identical_across_rungs"] = bool(same_y and same_sizes)
    out["pairing_basis"] = ("RepeatedStratifiedKFold splits depend only on the outcome "
                            "vector and n, which are identical across rungs, and the seed "
                            "is shared, so fold k of one rung holds the same test rows as "
                            "fold k of every other rung.")
    if not out["folds_identical_across_rungs"]:
        out["paired_differences"] = None
        out["paired_note"] = "folds differ across rungs; paired differences not computed"
        return out

    ll = {k: np.array([f["log_loss"] if f["ok"] else np.nan
                       for f in out["rungs"][k]["fold_level"]]) for k in ys}
    out["paired_differences"] = {
        "M1_minus_M0c": _diff_summary(ll["M1"], ll["M0c"], sv.CV_REPEATS, sv.CV_FOLDS),
        "M1_minus_M0": _diff_summary(ll["M1"], ll["M0"], sv.CV_REPEATS, sv.CV_FOLDS),
        "M0c_minus_M0": _diff_summary(ll["M0c"], ll["M0"], sv.CV_REPEATS, sv.CV_FOLDS),
    }
    out["reading"] = ("Lower log loss is better, so a negative difference means the first "
                      "model named is better on that fold. The fifty folds overlap across "
                      "repeats, so the SD of the differences is descriptive and is not a "
                      "standard error of the mean difference.")
    return out


# --------------------------------------------------------------------------
# 3. Transferability subgroup detail
# --------------------------------------------------------------------------

def _subgroup_fit(sub: pd.DataFrame, cols: list[str], pooled: sv.OrdinalFit,
                  name: str) -> dict:
    counts = {k: int(v) for k, v in sub["sev3"].value_counts()
              .reindex(list(sv.SEV_ORDER)).fillna(0).items()}
    entry: dict = {"n": int(len(sub)), "severity_counts": counts}
    try:
        fit = sv.fit_ordered_logit(sub, cols, name)
    except Exception as exc:                                       # noqa: BLE001
        entry["fit_error"] = type(exc).__name__
        return entry
    sep = sv.separation_diagnostics(fit)
    se_sub = np.sqrt(np.clip(np.diag(fit.cov)[:len(fit.beta)], 0, None))
    se_pool = np.sqrt(np.clip(np.diag(pooled.cov)[:len(pooled.beta)], 0, None))
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = se_sub / se_pool
    imax = int(np.nanargmax(ratio))
    entry.update({
        "estimable_parameters": int(fit.n_params),
        "events_per_parameter_smallest_class": r(min(counts.values()) / fit.n_params, 3),
        "converged": bool(fit.converged),
        "loglik": r(fit.loglik, 3),
        "separation": sep,
        "se_ratio_subgroup_over_pooled": {
            "max": r(np.nanmax(ratio), 3),
            "median": r(np.nanmedian(ratio), 3),
            "covariate_at_max": cols[imax],
        },
        "_loglik_raw": float(fit.loglik),
    })
    return entry


def item_transferability(frame: pd.DataFrame) -> dict:
    stored = json.loads((DATA / "transferability.json")
                        .read_text(encoding="utf-8"))
    cols = sv.ladder(frame)["M1"]
    out: dict = {
        "inputs": ["data/model_frame.csv", "data/transferability.json",
                   "src/transferability.py"],
        "covariate_pool": "M1 (34 covariates), screened per split by "
                          "transferability.estimable_in_both with min cell "
                          f"{tr.MIN_CELL_SUBSAMPLE}",
        "se_ratio_definition": ("standard error of a slope in the subgroup fit divided by "
                                "the standard error of the same slope in the pooled fit on "
                                "the same covariate set; the maximum and the median over "
                                "covariates are reported"),
        "bootstrap": {"replications": tr.BOOTSTRAP_REPS, "seed": tr.BOOTSTRAP_SEED,
                      "note": "not rerun; p-values carried from the stored file"},
        "splits": {},
    }

    yr = frame["crash_year"].to_numpy()
    m2022 = yr < 2022
    out["splits"]["temporal_2022"] = {
        "groups": {"before": {"years": "2021", "n": int(m2022.sum()),
                              "severity_counts": cells_by_sev(frame, m2022)},
                   "after": {"years": "2022 to 2025", "n": int((~m2022).sum()),
                             "severity_counts": cells_by_sev(frame, ~m2022)}},
        "testable": False,
        "reason": stored["untestable_breakpoints"]["2022"],
    }

    defs = {f"temporal_{bp}": (yr < bp, ("before", "after"),
                               stored["temporal"][str(bp)]) for bp in tr.BREAKPOINTS}
    big = frame["City_ID"].isin(LARGE_CITY_SET).to_numpy()
    defs["spatial"] = (big, ("large_city", "rest"), stored["spatial"])

    for key, (mask, names, st) in defs.items():
        keep, dropped = tr.estimable_in_both(frame, cols, mask)
        pooled = sv.fit_ordered_logit(frame, keep, f"pooled_{key}")
        groups = {names[0]: _subgroup_fit(frame[mask], keep, pooled, names[0]),
                  names[1]: _subgroup_fit(frame[~mask], keep, pooled, names[1])}
        ok = all("_loglik_raw" in g for g in groups.values())
        lr = (2 * (sum(g["_loglik_raw"] for g in groups.values()) - pooled.loglik)
              if ok else None)
        for g in groups.values():
            g.pop("_loglik_raw", None)
        stored_lr = st.get("lr_statistic")
        entry = {
            "groups": groups,
            "testable": True,
            "covariates_tested": len(keep),
            "covariates_tested_list": keep,
            "covariates_dropped": dropped,
            "matches_stored_covariates_tested": bool(len(keep) == st["covariates_tested"]),
            "pooled_converged": bool(pooled.converged),
            "pooled_separation": sv.separation_diagnostics(pooled),
            "lr_statistic_recomputed": r(lr, 3),
            "lr_statistic_stored": stored_lr,
            "lr_matches_stored": bool(lr is not None and stored_lr is not None
                                      and abs(round(lr, 3) - stored_lr) < 0.01),
            "df": st["df"],
            "p_asymptotic_stored": st["p_asymptotic"],
            "p_bootstrap_stored": st["p_bootstrap"],
            "rejects_at_5pct_bootstrap_stored": st["rejects_at_5pct_bootstrap"],
        }
        if key.startswith("temporal"):
            entry["bootstrap_replications_usable_stored"] = st.get(
                "bootstrap_replications_usable")
            entry["both_periods_converged_stored"] = st.get("both_periods_converged")
        else:
            entry["both_converged_stored"] = st.get("both_converged")
            entry["matches_stored_group_sizes"] = bool(
                int(mask.sum()) == st["n_large_city"]
                and int((~mask).sum()) == st["n_rest"])
        out["splits"][key] = entry

    counts = frame["City_ID"].value_counts()
    rank = {c: i + 1 for i, c in enumerate(counts.index)}
    out["large_city_definition"] = {
        "cities_in_code": list(LARGE_CITY_SET),
        "n_cities_in_code": len(LARGE_CITY_SET),
        "rider_rows_per_city": {c: int(counts.get(c, 0)) for c in LARGE_CITY_SET},
        "rank_by_rider_rows": {c: rank.get(c) for c in LARGE_CITY_SET},
        "n_large_city": int(big.sum()),
        "n_rest": int((~big).sum()),
        "rule_as_written": ("A literal set of six city names inside "
                            "transferability.spatial_test. No selection rule is stated in "
                            "the code or prespec.md. The result label in "
                            "the same function reads 'five largest reporting cities against "
                            "the rest', which does not match the six names."),
        "not_largest_by_reports": {
            "note": ("The set is not the largest reporting jurisdictions: records with "
                     "City_ID 'Not Listed' and several cities outside the set carry more "
                     "rider rows than Fort Worth and El Paso."),
            "outside_set_with_more_rows_than_smallest_member": {
                c: int(n) for c, n in counts.items()
                if c not in LARGE_CITY_SET
                and n > min(int(counts.get(x, 0)) for x in LARGE_CITY_SET)},
        },
    }
    return out


# --------------------------------------------------------------------------
# 4. Benchmark detail
# --------------------------------------------------------------------------

def item_benchmark(frame: pd.DataFrame) -> dict:
    import benchmark as bm

    stored = json.loads((DATA / "benchmark.json").read_text(encoding="utf-8"))
    res = stored["results"]
    per_class = {}
    for name, m in res.items():
        if name.startswith("_"):
            continue
        per_class[name] = {
            lvl: {k: m[f"{k}_{lvl}"] for k in ("precision", "recall", "f1")}
            for lvl in sv.SEV_ORDER}
        per_class[name]["n_folds"] = m["n_folds"]
    never_ka = [n for n, m in res.items() if not n.startswith("_")
                and m["recall_KA"]["mean"] == 0 and m["precision_KA"]["mean"] == 0]

    cols = sv.ladder(frame)["M1"]
    cue_cols = [c for c in cols if c in sv.CUES]
    cue_binary = all(set(np.unique(frame[c])) <= {0, 1} for c in cue_cols)

    return {
        "inputs": ["data/benchmark.json", "src/benchmark.py",
                   "data/model_frame.csv"],
        "grid": stored["grid"],
        "configurations_reported": [k for k in res if not k.startswith("_")],
        "selection": ("No hyperparameter is selected. Each grid point is fitted as its own "
                      "model on every fold and all four configurations are reported; there "
                      "is no inner tuning loop and nothing is chosen on the evaluation "
                      "folds."),
        "resampling": {"scheme": "RepeatedStratifiedKFold, stratified on the three-class "
                                 "outcome",
                       "repeats": stored["repeats"], "folds": stored["folds"],
                       "seed": stored["seed"],
                       "model_random_state": "seed + fold index (0 to 49)",
                       "same_folds_for_ordinal_logit": True,
                       "class_weighting": "none",
                       "seed_differs_from_severity_cv_seed": bool(stored["seed"]
                                                                  != sv.CV_SEED)},
        "inputs_to_models": {"n": stored["n"], "n_covariates": stored["n_covariates"],
                             "covariate_set": "M1",
                             "cue_columns": cue_cols,
                             "cue_columns_are_binary_labels": bool(cue_binary)},
        "per_class_metrics_mean_sd_over_50_folds": per_class,
        "models_that_never_predict_KA": never_ka,
        "baseline": res["_baseline_majority_class"],
        "calibration_inside_training_folds": {
            "claimed_in": "benchmark.py module docstring",
            "what_the_code_does": ("run() passes the binary final cue labels from "
                                   "model_frame.csv straight to every model. No "
                                   "probability is calibrated anywhere in the benchmark, "
                                   "inside or outside the folds, so the claim describes a "
                                   "step that does not exist. No fold leakage arises from "
                                   "calibration because there is none."),
            "confirmed": False,
        },
        "shap_settings": {"splitter": "StratifiedKFold, 5 folds, one repeat",
                          "seed": bm.SEED,
                          "model": "RandomForestClassifier(n_estimators=300, "
                                   "min_samples_leaf=5), not one of the grid points"},
    }


# --------------------------------------------------------------------------
# 5. Extraction intervals and out-of-sample calibration
# --------------------------------------------------------------------------

def _macro_f1_boot(truth: dict, pred: dict, w: np.ndarray, labels: list[str]) -> tuple:
    rng = np.random.default_rng(vm.BOOT_SEED)
    n = len(w)

    def macro(idx):
        f1s = []
        for lab in labels:
            c = vm._counts(truth[lab][idx], pred[lab][idx], w[idx])
            if c["f1"] == c["f1"]:
                f1s.append(c["f1"])
        return float(np.mean(f1s)) if f1s else float("nan")

    point = macro(np.arange(n))
    boots = np.array([macro(rng.integers(0, n, n)) for _ in range(vm.N_BOOT)])
    boots = boots[~np.isnan(boots)]
    return point, [r(np.percentile(boots, 2.5)), r(np.percentile(boots, 97.5))]


def _ece(p: np.ndarray, y: np.ndarray, w: np.ndarray, bins: int = ECE_BINS) -> tuple:
    edges = np.linspace(0, 1, bins + 1)
    # Bins are [b/bins, (b+1)/bins), the last closed at 1. The small offset keeps a vote
    # share such as 0.6 in the bin it names, which floating-point edges would not.
    idx = np.clip(np.floor(p * bins + 1e-9).astype(int), 0, bins - 1)
    total = w.sum()
    ece, rows = 0.0, []
    for b in range(bins):
        m = idx == b
        if not m.any():
            continue
        wb = w[m].sum()
        mp = float((w[m] * p[m]).sum() / wb)
        ob = float((w[m] * y[m]).sum() / wb)
        ece += wb / total * abs(mp - ob)
        rows.append({"bin": [r(edges[b], 1), r(edges[b + 1], 1)], "n": int(m.sum()),
                     "mean_predicted": r(mp), "observed_rate": r(ob)})
    return ece, rows


def _reliability_by_level(p: np.ndarray, y: np.ndarray, w: np.ndarray) -> list:
    out = []
    for lev in np.unique(np.round(p, 6)):
        m = np.isclose(p, lev)
        wb = w[m].sum()
        out.append({"vote_share": r(lev, 2), "n": int(m.sum()),
                    "observed_rate": r(float((w[m] * y[m]).sum() / wb)),
                    "observed_rate_unweighted": r(float(y[m].mean()))})
    return out


def item_extraction(frame: pd.DataFrame) -> dict:
    from sklearn.isotonic import IsotonicRegression
    from sklearn.model_selection import StratifiedKFold

    vmj = json.loads((DATA / "validation_metrics.json").read_text(encoding="utf-8"))
    routes = vmj["routes"]

    intervals: dict = {}
    for route, per in routes.items():
        intervals[route] = {}
        for lab, v in per.items():
            if lab.startswith("macro"):
                continue
            intervals[route][lab] = {
                kind: {"precision": r(v[kind]["precision"]),
                       "recall": r(v[kind]["recall"]),
                       "f1": r(v[kind]["f1"]),
                       "ci95": v[kind]["ci95"],
                       "tp": v[kind]["tp"], "fp": v[kind]["fp"],
                       "fn": v[kind]["fn"], "tn": v[kind]["tn"]}
                for kind in ("unweighted", "ip_weighted")}
            intervals[route][lab]["gold_positives"] = v["gold_positives"]
            intervals[route][lab]["predicted_positives"] = v["predicted_positives"]

    # Macro F1 intervals, resampling narratives jointly across cues.
    gold = pd.read_csv(DATA / "gold_labels.csv")
    ids = pd.read_csv(DATA / "validation_ids.csv")
    gold_ok = (len(gold) == 150 and set(gold["Crash_ID"]) == set(ids["Crash_ID"])
               and all(int(gold[l].sum()) == vmj["gold_label_detail"][l]["gold_positives"]
                       for l in ALL_LABELS))
    weights = dict(zip(ids["Crash_ID"].astype(int), ids["sampling_weight"]))
    route_files = {"route_a": DATA / "route_a_labels.csv",
                   "route_b_reference": DATA / "reference_labels.csv",
                   "fusion_automated": DATA / "fusion.csv"}
    vs = pd.read_csv(DATA / "vote_shares.csv")
    half = vs.copy()
    for m in MECHANISMS:
        half[m] = (vs[m] >= 0.5).astype(int)

    macro: dict = {}
    for route in routes:
        pred_frame = half if route == "route_b_vote_half" else pd.read_csv(route_files[route])
        merged = gold.merge(pred_frame, on="Crash_ID", how="inner",
                            suffixes=("_gold", "_pred"))
        labs = [l for l in ALL_LABELS if f"{l}_pred" in merged.columns]
        truth = {l: merged[f"{l}_gold"].to_numpy().astype(int) for l in labs}
        pred = {l: merged[f"{l}_pred"].to_numpy().astype(int) for l in labs}
        entry = {"labels": labs, "n": int(len(merged))}
        for kind in ("unweighted", "ip_weighted"):
            w = (np.ones(len(merged)) if kind == "unweighted"
                 else merged["Crash_ID"].map(weights).fillna(1.0).to_numpy())
            point, ci = _macro_f1_boot(truth, pred, w, labs)
            stored = routes[route][f"macro_{kind}"]["macro_f1"]
            entry[kind] = {"macro_f1": r(point), "ci95": ci,
                           "reproduces_stored": bool(round(point, 4) == stored)}
            # The stored macro averages whatever labels a route scores: fifteen for the
            # Route B reference run, ten for every other route. The ten-mechanism macro
            # is the one figure all routes share.
            mech = [l for l in labs if l in MECHANISMS]
            p_m, ci_m = _macro_f1_boot(truth, pred, w, mech)
            entry[kind]["macro_f1_mechanisms_only"] = r(p_m)
            entry[kind]["macro_f1_mechanisms_only_ci95"] = ci_m
        entry["n_labels_in_stored_macro"] = len(labs)
        macro[route] = entry

    # Cross-fitted isotonic calibration of the Route B vote shares.
    cal_frame = gold.merge(vs, on="Crash_ID", how="inner", suffixes=("_gold", "_vote"))
    w_ip = cal_frame["Crash_ID"].map(weights).fillna(1.0).to_numpy()
    per_cue: dict = {}
    pooled_p, pooled_c, pooled_y, pooled_w = [], [], [], []
    for cue in MECHANISMS:
        y = cal_frame[f"{cue}_gold"].to_numpy().astype(int)
        p = cal_frame[f"{cue}_vote"].to_numpy().astype(float)
        npos = int(y.sum())
        if npos < CAL_MIN_POSITIVES:
            per_cue[cue] = {"gold_positives": npos, "estimable": False,
                            "reason": f"fewer than {CAL_MIN_POSITIVES} gold positives"}
            continue
        k = min(CAL_FOLDS, npos)
        oof = np.empty(len(y))
        for tr_idx, te_idx in StratifiedKFold(k, shuffle=True, random_state=CAL_SEED
                                              ).split(p, y):
            iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
            iso.fit(p[tr_idx], y[tr_idx])
            oof[te_idx] = iso.predict(p[te_idx])
        entry = {"gold_positives": npos, "estimable": True, "folds": k}
        for kind, w in (("unweighted", np.ones(len(y))), ("ip_weighted", w_ip)):
            e_raw, bins_raw = _ece(p, y, w)
            e_cal, bins_cal = _ece(oof, y, w)
            entry[kind] = {
                "ece_raw": r(e_raw), "ece_calibrated_out_of_fold": r(e_cal),
                "brier_raw": r(float((w * (p - y) ** 2).sum() / w.sum())),
                "brier_calibrated_out_of_fold": r(float((w * (oof - y) ** 2).sum()
                                                        / w.sum())),
                "reliability_bins_calibrated": bins_cal,
            }
        entry["reliability_by_vote_share_ip_weighted"] = _reliability_by_level(p, y, w_ip)
        per_cue[cue] = entry
        pooled_p.append(p); pooled_c.append(oof); pooled_y.append(y); pooled_w.append(w_ip)

    P, C, Y, W = (np.concatenate(a) for a in (pooled_p, pooled_c, pooled_y, pooled_w))
    pooled: dict = {"cues_included": [c for c in MECHANISMS if per_cue[c]["estimable"]],
                    "n_pairs": int(len(Y))}
    for kind, w in (("unweighted", np.ones(len(Y))), ("ip_weighted", W)):
        e_raw, bins_raw = _ece(P, Y, w)
        e_cal, bins_cal = _ece(C, Y, w)
        pooled[kind] = {
            "ece_raw": r(e_raw), "ece_calibrated_out_of_fold": r(e_cal),
            "brier_raw": r(float((w * (P - Y) ** 2).sum() / w.sum())),
            "brier_calibrated_out_of_fold": r(float((w * (C - Y) ** 2).sum() / w.sum())),
            "reliability_bins_raw": bins_raw,
            "reliability_bins_calibrated": bins_cal,
        }

    return {
        "inputs": ["data/validation_metrics.json", "data/gold_labels.csv",
                   "data/validation_ids.csv", "data/route_a_labels.csv",
                   "data/reference_labels.csv", "data/vote_shares.csv",
                   "data/fusion.csv"],
        "validation_sample": "batch 1 probability sample, 150 narratives, three coders, "
                             "majority-vote gold",
        "gold_labels_file_matches_batch1": bool(gold_ok),
        "per_cue_intervals": {
            "source": "validation_metrics.json (already computed there; carried over)",
            "bootstrap": {"resamples": vm.N_BOOT, "seed": vm.BOOT_SEED,
                          "interval": "percentile 95 percent"},
            "routes": intervals,
        },
        "macro_f1_intervals": {
            "bootstrap": {"resamples": vm.N_BOOT, "seed": vm.BOOT_SEED,
                          "resampling_unit": "narrative, jointly across cues"},
            "routes": macro,
        },
        "calibration": {
            "status": ("New. No calibration code exists elsewhere in src: no isotonic "
                       "or other calibration map is fitted by any script before this one, "
                       "and no calibrated probability enters any model."),
            "scored_quantity": ("Route B vote share over the repeated runs "
                                "(runs per narrative: "
                                f"{sorted(int(v) for v in vs['n_runs'].unique())}) "
                                "against the batch-1 gold label"),
            "method": ("isotonic regression fitted on the training folds and applied to "
                       "the held-out fold, stratified K-fold per cue with K = min(5, gold "
                       "positives), so no narrative contributes to the map applied to "
                       "itself"),
            "seed": CAL_SEED,
            "ece_bins": f"{ECE_BINS} equal-width bins on [0, 1]",
            "weights": ("unweighted, and inverse-probability weighted by the validation "
                        "sampling weights; the sample oversamples predicted positives, so "
                        "the weighted figures are the ones that describe the population"),
            "injury_cues": "not scored; vote shares exist only for the ten mechanism cues",
            "per_cue": per_cue,
            "pooled_over_cues": pooled,
        },
    }


# --------------------------------------------------------------------------
# 6. Data-section sensitivities
# --------------------------------------------------------------------------

def _kr_sample(fit: sv.OrdinalFit, draws: int = sv.KR_DRAWS, seed: int = sv.KR_SEED):
    k = len(fit.beta)
    n_raw = k + len(fit.cut)
    cov = np.asarray(fit.cov)[:n_raw, :n_raw]
    cov = (cov + cov.T) / 2
    rng = np.random.default_rng(seed)
    return k, rng.multivariate_normal(np.asarray(fit.raw)[:n_raw], cov, size=draws,
                                      method="svd")


def _fit_summary(fit: sv.OrdinalFit, frame: pd.DataFrame, cols: list[str],
                 null_ll: float, outcome: str = "sev3", order=sv.SEV_ORDER) -> dict:
    return {"fit": sv.fit_metrics(fit, null_ll),
            "events_per_parameter": sv.events_per_parameter(frame, cols, outcome, order),
            "separation": sv.separation_diagnostics(fit)}


def _rcs_basis(x: np.ndarray, knots: np.ndarray) -> np.ndarray:
    """Restricted cubic spline basis (Harrell), linear term first, scaled by the knot
    span squared so the nonlinear columns sit on the scale of the linear one."""
    t = np.asarray(knots, dtype=float)
    k = len(t)
    cols = [x]
    span2 = (t[-1] - t[0]) ** 2

    def pos3(v):
        return np.clip(v, 0, None) ** 3

    for j in range(k - 2):
        term = (pos3(x - t[j])
                - pos3(x - t[k - 2]) * (t[k - 1] - t[j]) / (t[k - 1] - t[k - 2])
                + pos3(x - t[k - 1]) * (t[k - 2] - t[j]) / (t[k - 1] - t[k - 2]))
        cols.append(term / span2)
    return np.column_stack(cols)


def _age_profile(fit: sv.OrdinalFit, frame: pd.DataFrame, cols: list[str],
                 spl_cols: list[str], knots, ages, scale: float) -> dict:
    """Average predicted probabilities with every rider set to one recorded age, and the
    average derivative per ten years over riders with a recorded age, with Krinsky-Robb
    intervals from the same draws."""
    X, _ = sv._design(frame, cols)
    ispl = [cols.index(c) for c in spl_cols]
    iunk = cols.index("age_unknown")
    known = frame["age_unknown"].to_numpy() == 0
    k, sample = _kr_sample(fit)

    def profile(beta, cut):
        out = []
        for a in ages:
            Xa = X.copy()
            Xa[:, ispl] = _rcs_basis(np.full(len(X), a / scale), knots)
            Xa[:, iunk] = 0.0
            out.append(sv._probs(Xa, beta, cut).mean(axis=0))
        Xp, Xm = X[known].copy(), X[known].copy()
        age_k = frame.loc[known, "age_filled"].to_numpy()
        Xp[:, ispl] = _rcs_basis((age_k + 0.5) / scale, knots)
        Xm[:, ispl] = _rcs_basis((age_k - 0.5) / scale, knots)
        deriv = ((sv._probs(Xp, beta, cut) - sv._probs(Xm, beta, cut)) * 10).mean(axis=0)
        return np.array(out), deriv

    pt_prof, pt_der = profile(fit.beta, fit.cut)
    d_prof = np.empty((len(sample), len(ages), len(sv.SEV_ORDER)))
    d_der = np.empty((len(sample), len(sv.SEV_ORDER)))
    for d in range(len(sample)):
        d_prof[d], d_der[d] = profile(sample[d, :k], sv._cut_from_raw(sample[d, k:]))

    prof = {}
    for i, a in enumerate(ages):
        prof[str(a)] = {lvl: {"p": r(pt_prof[i, j], 4),
                              "ci95": [r(np.percentile(d_prof[:, i, j], 2.5), 4),
                                       r(np.percentile(d_prof[:, i, j], 97.5), 4)]}
                        for j, lvl in enumerate(sv.SEV_ORDER)}
    der = {lvl: {"ame_per_10_years": r(pt_der[j], 5),
                 "ci95": [r(np.percentile(d_der[:, j], 2.5), 5),
                          r(np.percentile(d_der[:, j], 97.5), 5)],
                 "excludes_zero": bool(np.percentile(d_der[:, j], 2.5) > 0
                                       or np.percentile(d_der[:, j], 97.5) < 0)}
           for j, lvl in enumerate(sv.SEV_ORDER)}
    return {"predicted_probability_at_age": prof,
            "average_derivative_recorded_age_riders": der}


def _ame_subset(ames: dict, keys: list[str]) -> dict:
    return {k: ames[k] for k in keys if k in ames}


def sens_age(frame: pd.DataFrame, cols: list[str], null_ll: float) -> dict:
    age = frame["rider_age"]
    known = age.notna()
    f = frame.copy()
    median_age = float(age[known].median())
    f["age_filled"] = age.fillna(median_age)

    band_cols = ["age_under25", "age_45plus"]
    base_cols = [c for c in cols if c not in band_cols]          # keeps age_unknown

    def add_bands(cuts: list[tuple[str, float, float]]) -> list[str]:
        names = []
        for name, lo, hi in cuts:
            f[name] = (known & (age >= lo) & (age < hi)).astype(int)
            names.append(name)
        return names

    specs: dict[str, dict] = {}
    specs["primary_lt25_25to44_ge45"] = {"cols": cols, "reference": "25 to 44",
                                         "bands": {"age_under25": "under 25",
                                                   "age_45plus": "45 and over"}}
    a_cols = add_bands([("age_under25", 0, 25), ("age_55plus", 55, 200)])
    specs["alt_lt25_25to54_ge55"] = {"cols": base_cols + a_cols, "reference": "25 to 54",
                                     "bands": {"age_under25": "under 25",
                                               "age_55plus": "55 and over"}}
    b_cols = add_bands([("age_lt21", 0, 21), ("age_21to29", 21, 30),
                        ("age_ge45", 45, 200)])
    specs["alt_lt21_21to29_30to44_ge45"] = {"cols": base_cols + b_cols,
                                            "reference": "30 to 44",
                                            "bands": {"age_lt21": "under 21",
                                                      "age_21to29": "21 to 29",
                                                      "age_ge45": "45 and over"}}

    base_fit = sv.fit_ordered_logit(f, base_cols, "age_none")
    out: dict = {
        "recorded_ages": int(known.sum()),
        "unrecorded_ages": int((~known).sum()),
        "unrecorded_age_handling": (f"age_unknown indicator kept in every specification; "
                                    f"for the spline and linear forms unrecorded ages are "
                                    f"filled with the median recorded age "
                                    f"({median_age:g}) so the indicator carries the shift"),
        "no_age_terms_model": {**_fit_summary(base_fit, f, base_cols, null_ll),
                               "note": "M1 without age bands, keeping age_unknown; the "
                                       "nested base for every likelihood-ratio test"},
        "band_specifications": {},
        "spline_specifications": {},
        "literature_for_cut_offs": "not compiled here; outside the scope of a data run",
    }

    for name, spec in specs.items():
        c = spec["cols"]
        fit = sv.fit_ordered_logit(f, c, name)
        ames = sv.average_marginal_effects(fit, f)
        band_names = list(spec["bands"])
        ref_mask = known & ~f[band_names].astype(bool).any(axis=1)
        out["band_specifications"][name] = {
            **_fit_summary(fit, f, c, null_ll),
            "reference_band": spec["reference"],
            "band_cells_by_severity": {spec["bands"][b]: cells_by_sev(f, f[b] == 1)
                                       for b in band_names},
            "reference_cells_by_severity": cells_by_sev(f, ref_mask),
            "lr_vs_no_age_terms": sv.nested_lr_test(base_fit, fit),
            "age_ames": _ame_subset(ames, band_names + ["age_unknown"]),
        }

    knots_used = {}
    lin_cols = base_cols + ["agespline_lin"]
    f["agespline_lin"] = f["age_filled"] / 10.0
    lin_fit = sv.fit_ordered_logit(f, lin_cols, "age_linear")
    lin_ames = sv.average_marginal_effects(lin_fit, f)
    out["spline_specifications"]["linear"] = {
        **_fit_summary(lin_fit, f, lin_cols, null_ll),
        "scale": "age in decades",
        "lr_vs_no_age_terms": sv.nested_lr_test(base_fit, lin_fit),
        "age_ames": {"age_per_10_years": lin_ames["agespline_lin"],
                     "age_unknown": lin_ames["age_unknown"]},
    }
    for nk, pct in ((3, (10, 50, 90)), (4, (5, 35, 65, 95))):
        knots = np.percentile(age[known], pct) / 10.0
        knots_used[nk] = knots
        basis = _rcs_basis(f["age_filled"].to_numpy() / 10.0, knots)
        spl = ["agespline_lin"] + [f"agespline_nl{j}" for j in range(1, nk - 1)]
        for j, cname in enumerate(spl):
            f[cname] = basis[:, j]
        c = base_cols + spl
        fit = sv.fit_ordered_logit(f, c, f"age_rcs{nk}")
        ames = sv.average_marginal_effects(fit, f)
        prof = _age_profile(fit, f, c, spl, knots, (16, 20, 25, 35, 45, 55, 65), 10.0)
        out["spline_specifications"][f"rcs_{nk}_knots"] = {
            **_fit_summary(fit, f, c, null_ll),
            "knot_percentiles": list(pct),
            "knots_years": [r(v * 10, 2) for v in knots],
            "lr_vs_no_age_terms": sv.nested_lr_test(base_fit, fit),
            "lr_vs_linear_age": sv.nested_lr_test(lin_fit, fit),
            "age_unknown_ame": ames["age_unknown"],
            **prof,
        }

    rows = {}
    for group in ("band_specifications", "spline_specifications"):
        for k, v in out[group].items():
            rows[k] = {"aic": v["fit"]["aic"], "bic": v["fit"]["bic"],
                       "loglik": v["fit"]["loglik"],
                       "n_parameters": v["fit"]["n_parameters"]}
    out["comparison"] = rows
    out["comparison_note"] = ("Band specifications are not nested in one another, so they "
                              "are compared by AIC and BIC; likelihood-ratio tests are "
                              "given only for nested pairs.")
    return out


def sens_complete_case(frame: pd.DataFrame, cols: list[str]) -> dict:
    mask = ((frame["coord_missing"] == 0) & (frame["age_unknown"] == 0)
            & (frame["gender_unknown"] == 0) & (frame["eth_recorded_unknown"] == 0))
    sub = frame[mask].copy()
    drop_ind = ["age_unknown", "gender_unknown", "coord_missing"]
    cc_cols = [c for c in cols if c not in drop_ind]
    constant = [c for c in cc_cols if sub[c].nunique() < 2]
    cc_cols = [c for c in cc_cols if c not in constant]
    null_ll = sv.null_loglikelihood(sub)
    fit = sv.fit_ordered_logit(sub, cc_cols, "complete_case")
    ames = sv.average_marginal_effects(fit, sub)

    primary = json.loads((DATA / "severity.json")
                         .read_text(encoding="utf-8"))["average_marginal_effects"]
    compare = {}
    for c in cc_cols:
        p, q = primary[c]["KA"], ames[c]["KA"]
        compare[c] = {"primary_KA": p, "complete_case_KA": q,
                      "same_sign": bool(np.sign(p["ame"]) == np.sign(q["ame"])),
                      "excludes_zero_primary": p["excludes_zero"],
                      "excludes_zero_complete_case": q["excludes_zero"]}
    sparse = {c: cells_by_sev(sub, sub[c] == 1) for c in cc_cols
              if set(np.unique(sub[c])) <= {0, 1}
              and min(cells_by_sev(sub, sub[c] == 1).values()) < sv_min_cell()}
    return {
        "definition": ("located crash (coordinates present) and age, gender and ethnicity "
                       "all recorded; ethnicity unknown read from the raw "
                       "Prsn_Ethnicity_ID (missing or 'Unknown'), because the model frame "
                       "pools it with 'other'"),
        "n": int(mask.sum()),
        "n_excluded": int((~mask).sum()),
        "severity_counts": cells_by_sev(sub, np.ones(len(sub), bool)),
        "covariates_dropped_as_constant_indicators": drop_ind,
        "covariates_dropped_for_no_variation": constant,
        "eth_other_or_unknown_meaning_here": "other recorded ethnicity only",
        **_fit_summary(fit, sub, cc_cols, null_ll),
        "covariates_with_a_severity_cell_below_10": sparse,
        "ames": ames,
        "ka_comparison_with_primary": compare,
        "ka_excluding_zero_primary": [c for c in cc_cols
                                      if compare[c]["excludes_zero_primary"]],
        "ka_excluding_zero_primary_all_34": [c for c in cols
                                             if primary[c]["KA"]["excludes_zero"]],
        "comparison_note": ("ka_excluding_zero_primary is restricted to the covariates "
                            "the complete-case model can carry; the indicators dropped as "
                            "constant cannot be compared"),
        "ka_excluding_zero_complete_case": [c for c in cc_cols
                                            if compare[c]["excludes_zero_complete_case"]],
        "multiple_imputation": ("prespec.md section 2.3 also names a chained-equations "
                                "multiple-imputation sensitivity (m = 20). No code in "
                                "src implements it and it is not computed here."),
    }


def sv_min_cell() -> int:
    return 10


def sens_missingness(frame: pd.DataFrame) -> dict:
    import statsmodels.api as sm

    f = frame.copy()
    f["sev_O"] = (f["sev3"] == "O").astype(int)
    f["sev_KA"] = (f["sev3"] == "KA").astype(int)
    f["year_c"] = f["crash_year"] - 2023
    f["large_city"] = f["City_ID"].isin(LARGE_CITY_SET).astype(int)
    indicators = {"age_unknown": "age_unknown", "gender_unknown": "gender_unknown",
                  "ethnicity_unknown": "eth_recorded_unknown",
                  "speed_limit_unknown": "speed_unknown",
                  "coordinates_missing": "coord_missing"}
    others = ["year_c", "nighttime", "weekend", "intx_intersection_or_driveway",
              "road_not_city_street", "large_city"]
    out: dict = {"predictors": ["sev_O", "sev_KA"] + others,
                 "severity_reference": "BC",
                 "year": "linear, centred at 2023",
                 "excluded_by_design": ("poverty_share and road_density_km_100m are left "
                                        "out of every model because they are imputed from "
                                        "the city median exactly for the coordinate-missing "
                                        "rows"),
                 "models": {}}
    for label, col in indicators.items():
        y = f[col].to_numpy()
        tab = pd.crosstab(f[col], f["sev3"]).reindex(columns=list(sv.SEV_ORDER)).fillna(0)
        chi2, p_chi, _, _ = stats.chi2_contingency(tab.to_numpy())
        rate = {k: r(float(f.loc[f["sev3"] == k, col].mean()), 3) for k in sv.SEV_ORDER}
        entry: dict = {"n_missing": int(y.sum()), "missing_rate_by_severity": rate,
                       "unadjusted_chi2": r(chi2, 3), "unadjusted_df": 2,
                       "unadjusted_p": r(p_chi, 4)}
        preds = ["sev_O", "sev_KA"] + others
        # A predictor that is constant among the missing or the recorded rows separates
        # perfectly; it is dropped and named rather than left to inflate the fit.
        sep_drop = [c for c in others
                    if f.loc[y == 1, c].nunique() < 2 or f.loc[y == 0, c].nunique() < 2]
        preds = [c for c in preds if c not in sep_drop]
        try:
            X = sm.add_constant(f[preds].astype(float))
            full = sm.Logit(y, X).fit(disp=False, maxiter=200)
            red = sm.Logit(y, sm.add_constant(f[[c for c in preds if not c.startswith("sev_")]]
                                              .astype(float))).fit(disp=False, maxiter=200)
            lr = 2 * (full.llf - red.llf)
            ci = np.asarray(full.conf_int())
            entry.update({
                "predictors_dropped_for_separation": sep_drop,
                "converged": bool(full.mle_retvals.get("converged", False)),
                "max_abs_coefficient": r(float(np.max(np.abs(full.params[1:]))), 3),
                "severity_odds_ratios": {
                    k: {"or": r(float(np.exp(full.params[k])), 3),
                        "ci95": [r(float(np.exp(ci[preds.index(k) + 1, 0])), 3),
                                 r(float(np.exp(ci[preds.index(k) + 1, 1])), 3)],
                        "p": r(float(full.pvalues[k]), 4)}
                    for k in ("sev_O", "sev_KA")},
                "other_odds_ratios": {
                    k: {"or": r(float(np.exp(full.params[k])), 3),
                        "p": r(float(full.pvalues[k]), 4)} for k in preds
                    if not k.startswith("sev_")},
                "severity_block_lr_chi2": r(lr, 3), "severity_block_df": 2,
                "severity_block_p": r(float(1 - stats.chi2.cdf(lr, 2)), 4),
            })
        except Exception as exc:                                   # noqa: BLE001
            entry["fit_error"] = f"{type(exc).__name__}: {exc}"
        out["models"][label] = entry
    out["reading"] = ("A severity block that is associated with missingness after "
                      "adjustment means the indicator is not missing completely at "
                      "random with respect to the outcome. None of these models can "
                      "distinguish missing at random from missing not at random.")
    return out


def sens_speed(frame: pd.DataFrame) -> dict:
    raw = frame["Crash_Speed_Limit"]
    return {
        "alternatives_in_pipeline": ("none. The only treatment coded anywhere in src "
                                     "is the pre-specified band coding in "
                                     "model_frame._speed_band; no other speed-limit form "
                                     "is fitted by any script"),
        "primary_treatment": "bands: 30 mph or below (reference), 35 to 40, 45 and over, "
                             "unknown (raw -1 or missing)",
        "band_counts": {k: int(v) for k, v in frame["speed_level"].value_counts().items()},
        "band_cells_by_severity": {lev: cells_by_sev(frame, frame["speed_level"] == lev)
                                   for lev in sorted(frame["speed_level"].unique())},
        "raw_value_counts": {str(int(k)): int(v) for k, v in
                             raw.value_counts().sort_index().items()},
        "raw_minus_one": int((raw == -1).sum()),
        "raw_missing": int(raw.isna().sum()),
    }


def sens_acs_bike(frame: pd.DataFrame, cols: list[str], null_ll: float) -> dict:
    acs_audit = json.loads((DATA / "acs_extraction_audit.json").read_text(encoding="utf-8"))
    primary_ames = json.loads((DATA / "severity.json")
                              .read_text(encoding="utf-8"))["average_marginal_effects"]
    prim_fit = sv.fit_ordered_logit(frame, cols, "M1")
    early = frame["crash_year"].isin([2021, 2022])

    variants = {}
    for name, series in (
        ("acs2017_2021_for_2021_2022_crashes",
         frame["poverty_share"].where(~early, frame["poverty_share_acs2021"])),
        ("acs2017_2021_for_all_crashes", frame["poverty_share_acs2021"]),
    ):
        f = frame.copy()
        f["poverty_share"] = series
        fit = sv.fit_ordered_logit(f, cols, name)
        ames = sv.average_marginal_effects(fit, f)
        coef = sv.coefficient_table(fit)["slopes"]["poverty_share"]
        variants[name] = {**_fit_summary(fit, f, cols, null_ll),
                          "rows_with_value_changed": int((series != frame["poverty_share"])
                                                         .sum()),
                          "poverty_share_coefficient": coef,
                          "poverty_share_ame": ames["poverty_share"],
                          "ka_excluding_zero": [c for c in cols
                                                if ames[c]["KA"]["excludes_zero"]]}
    located = frame["coord_missing"] == 0
    corr = float(np.corrcoef(frame.loc[located, "poverty_share"],
                             frame.loc[located, "poverty_share_acs2021"])[0, 1])
    bike = json.loads((DATA / "bike_facility_sensitivity.json").read_text(encoding="utf-8"))
    screening = json.loads((DATA / "screening.json").read_text(encoding="utf-8"))
    nearest = next((c for c in screening["candidates"]
                    if c["candidate"] == "bike_facility_nearest"), None)
    return {
        "acs_vintage": {
            "inputs": ["data/model_frame.csv", "data/acs_extraction_audit.json"],
            "primary_vintage": acs_audit["acs_primary_vintage"],
            "sensitivity_vintage": acs_audit["acs_sensitivity_vintage"],
            "table": {"numerator": acs_audit["poverty_numerator"],
                      "denominator": acs_audit["poverty_denominator"]},
            "missing_value_rule": "city median, then overall median, for both vintages "
                                  "(model_frame.build)",
            "correlation_between_vintages_located_rows": r(corr, 3),
            "primary": {"fit": sv.fit_metrics(prim_fit, null_ll),
                        "poverty_share_ame": primary_ames["poverty_share"]},
            "variants": variants,
            "status": ("acs_extraction_audit.json held only the column metadata; the model "
                       "refits are new here"),
        },
        "bike_facility_buffers": {
            "inputs": ["data/bike_facility_sensitivity.json", "data/screening.json"],
            "pre_specified_primary_form": "bike_facility_nearest",
            "nearest_segment_screening": ({"events_by_severity":
                                           nearest.get("events_by_severity"),
                                           "disposition": nearest.get("disposition")}
                                          if nearest else None),
            "buffers": {k: {"n_positive": v["n_positive"], "fit": v["fit"],
                            "ame": v["ame"]} for k, v in bike.items()},
            "any_ka_interval_excludes_zero": any(v["ame"]["KA"]["excludes_zero"]
                                                 for v in bike.values()),
            "note": "each buffer indicator is added to M1 one at a time",
        },
    }


def sens_classes(frame: pd.DataFrame, cols: list[str]) -> dict:
    stored = json.loads((DATA / "severity_appendix.json")
                        .read_text(encoding="utf-8"))["four_class"]
    four = sa.four_class_model(frame, cols)
    st_n = stored["events_per_parameter"]["n_covariates"]
    four_out = {
        "refit_on_current_M1": four,
        "stored_file_n_covariates": st_n,
        "stored_file_loglik": stored["fit"]["loglik"],
        "stored_file_events_per_parameter": stored["events_per_parameter"][
            "events_per_parameter_smallest_class"],
        "stored_matches_current_M1": bool(st_n == len(cols)
                                          and abs(stored["fit"]["loglik"]
                                                  - four["fit"]["loglik"]) < 0.01),
    }

    five: dict = {"order": list(FIVE_CLASS_ORDER),
                  "class_counts": cells_by_sev(frame, np.ones(len(frame), bool),
                                               "sev_kabco", FIVE_CLASS_ORDER),
                  "estimability_rule": ("fixed before fitting: optimizer converged, "
                                        "separation screen clean, and at least "
                                        f"{FIVE_CLASS_EPP_FLOOR:g} event per parameter in "
                                        "the smallest class"),
                  "prespec": ("prespec.md section 2 states the five-class ordering is not "
                              "estimated because of the fatality count; this is an attempt "
                              "made as an exploratory check and is reported as such")}
    five["events_per_parameter"] = sv.events_per_parameter(frame, cols, "sev_kabco",
                                                           FIVE_CLASS_ORDER)
    try:
        fit = sv.fit_ordered_logit(frame, cols, "M1_five_class", outcome="sev_kabco",
                                   order=FIVE_CLASS_ORDER)
        null5 = sv.null_loglikelihood(frame, "sev_kabco", FIVE_CLASS_ORDER)
        sep = sv.separation_diagnostics(fit)
        five["fit"] = sv.fit_metrics(fit, null5)
        five["separation"] = sep
        five["cutpoints"] = [r(c, 4) for c in fit.cut]
        epp = five["events_per_parameter"]["events_per_parameter_smallest_class"]
        crit = {"converged": bool(fit.converged),
                "no_separation_flag": not sep["suggests_separation"],
                "epp_at_or_above_floor": bool(epp >= FIVE_CLASS_EPP_FLOOR)}
        five["criteria"] = crit
        five["estimable"] = bool(all(crit.values()))
        if fit.converged and not sep["suggests_separation"]:
            ames = sv.average_marginal_effects(fit, frame, outcome="sev_kabco",
                                               order=FIVE_CLASS_ORDER)
            five["ames_K_and_A"] = {c: {"A": ames[c]["A"], "K": ames[c]["K"]}
                                    for c in cols}
            five["k_intervals_excluding_zero"] = [c for c in cols
                                                  if ames[c]["K"]["excludes_zero"]]
            five["ame_note"] = ("Reported for completeness. With the estimability rule "
                                "failed these are not interpreted.")
    except Exception as exc:                                       # noqa: BLE001
        five["fit_error"] = f"{type(exc).__name__}: {exc}"
        five["estimable"] = False
    return {"four_class": four_out, "five_class": five}


def item_sensitivities(frame: pd.DataFrame) -> dict:
    cols = sv.ladder(frame)["M1"]
    null_ll = sv.null_loglikelihood(frame)
    out: dict = {
        "inputs": ["data/model_frame.csv", "data/base_frame.csv",
                   "data/severity.json", "data/screening.json"],
        "primary_specification": "M1 ordered logit, 34 covariates",
        "krinsky_robb": {"draws": sv.KR_DRAWS, "seed": sv.KR_SEED},
    }
    for key, fn in (("age", lambda: sens_age(frame, cols, null_ll)),
                    ("complete_case", lambda: sens_complete_case(frame, cols)),
                    ("missingness_mechanism", lambda: sens_missingness(frame)),
                    ("speed_limit", lambda: sens_speed(frame)),
                    ("acs_and_bike_facility", lambda: sens_acs_bike(frame, cols, null_ll)),
                    ("severity_classes", lambda: sens_classes(frame, cols))):
        try:
            out[key] = fn()
        except Exception as exc:                                   # noqa: BLE001
            out[key] = {"error": f"{type(exc).__name__}: {exc}"}
        RESULT["data_sensitivities"] = out
        save()
        print(f"  6 {key} done")
    return out


# --------------------------------------------------------------------------
# 7. Small facts
# --------------------------------------------------------------------------

def item_facts(frame: pd.DataFrame) -> dict:
    osm = json.loads((DATA / "osm_extraction_audit.json").read_text(encoding="utf-8"))
    acs = json.loads((DATA / "acs_extraction_audit.json").read_text(encoding="utf-8"))

    cue_counts = {}
    for lab in ALL_LABELS:
        entry = {"total_rider_rows": int(frame[lab].sum())}
        entry["by_sev3"] = {k: int(frame.loc[frame["sev3"] == k, lab].sum())
                            for k in sv.SEV_ORDER}
        entry["share_within_sev3"] = {k: r(frame.loc[frame["sev3"] == k, lab].mean(), 3)
                                      for k in sv.SEV_ORDER}
        entry["by_kabco"] = {k: int(frame.loc[frame["sev_kabco"] == k, lab].sum())
                             for k in ("K", "A", "B", "C", "O")}
        entry["in_primary_model"] = lab in sv.ladder(frame)["M1"]
        entry["distinct_crashes_positive"] = int(frame.loc[frame[lab] == 1, "Crash_ID"]
                                                 .nunique())
        cue_counts[lab] = entry

    multi = frame.groupby("Crash_ID").size()
    return {
        "inputs": ["data/osm_extraction_audit.json", "data/acs_extraction_audit.json",
                   "data/model_frame.csv"],
        "osm": {"extraction_date": osm["extraction_date"],
                "located_crashes": osm["located_crashes"], "tiles": osm["tiles"],
                "failed_tiles": osm["failed_tiles"],
                "frame_column_osm_extraction_date": sorted(
                    frame["osm_extraction_date"].dropna().unique().tolist())},
        "acs": {"product": "American Community Survey 5-year estimates",
                "primary_vintage": acs["acs_primary_vintage"],
                "sensitivity_vintage": acs["acs_sensitivity_vintage"],
                "geography": "block group",
                "poverty_numerator": acs["poverty_numerator"],
                "poverty_denominator": acs["poverty_denominator"],
                "table": acs["poverty_numerator"].split("_")[0],
                "b17021_available_at_block_group": acs["b17021_available_at_block_group"],
                "b17021_block_group_check_date": acs["b17021_block_group_check_date"],
                "tiger_vintage": acs["tiger_vintage"],
                "tiger_url": acs["tiger_url"]},
        "riders_and_crashes": {
            "rider_rows": int(len(frame)),
            "distinct_crashes": int(frame["Crash_ID"].nunique()),
            "crashes_with_two_riders": int((multi == 2).sum()),
            "crashes_with_more_than_two_riders": int((multi > 2).sum()),
            "note": ("Cue labels are extracted per narrative, which is per crash, so the "
                     "two two-rider crashes carry the same cue values on both rider rows."),
        },
        "cue_counts_by_severity": {
            "unit": "rider rows (522); severity is the rider's coded severity",
            "label_source": {k: int(v) for k, v in
                             frame["label_source"].value_counts().items()},
            "labels": cue_counts,
        },
        "large_city": {
            "definition_in_code": list(LARGE_CITY_SET),
            "see": "transferability_detail.large_city_definition",
        },
    }


# --------------------------------------------------------------------------
# Recomputation of outputs that predate the current model frame
# --------------------------------------------------------------------------
# `transferability.json`, `benchmark.json`,
# `severity_appendix.json` and `bike_facility_sensitivity.json` were written
# on 2026-09-14, before `final_labels.csv` replaced the fused cue labels in
# `model_frame.csv` on 2026-09-16. Their cue columns therefore differ from the frame every
# other result uses. The tests below are rerun on the current frame with the same
# functions, seeds and replications, and the stored values are kept beside them.

STALE_NOTE = ("written 2026-09-14 from the frame with fused reference-run cue labels; "
              "model_frame.csv was rebuilt with the final adjudicated labels on "
              "2026-09-16")


def recompute_transferability(frame: pd.DataFrame) -> None:
    cols = sv.ladder(frame)["M1"]
    td = RESULT["transferability_detail"]
    for bp in tr.BREAKPOINTS:
        key = f"temporal_{bp}"
        entry = td["splits"][key]
        if "stored_values_pre_final_labels" not in entry:
            entry["stored_values_pre_final_labels"] = {
                k: entry.pop(k) for k in list(entry)
                if k.endswith("_stored") or k == "lr_matches_stored"
                or k == "matches_stored_covariates_tested"}
        res = tr.temporal_test(frame, cols, bp)
        entry.update({
            "lr_statistic": res["lr_statistic"],
            "lr_equals_subgroup_refit": bool(abs(res["lr_statistic"]
                                                 - entry["lr_statistic_recomputed"]) < 0.01),
            "df": res["df"],
            "p_asymptotic": res["p_asymptotic"],
            "p_bootstrap": res["p_bootstrap"],
            "bootstrap_replications_usable": res["bootstrap_replications_usable"],
            "bootstrap_null_median": res["bootstrap_null_median"],
            "both_periods_converged": res["both_periods_converged"],
            "rejects_at_5pct_bootstrap": res["rejects_at_5pct_bootstrap"],
        })
        save()
        print(f"  temporal {bp}: LR {res['lr_statistic']} p_boot {res['p_bootstrap']}")

    entry = td["splits"]["spatial"]
    if "stored_values_pre_final_labels" not in entry:
        entry["stored_values_pre_final_labels"] = {
            k: entry.pop(k) for k in list(entry)
            if k.endswith("_stored") or k in ("lr_matches_stored",
                                              "matches_stored_covariates_tested")}
    res = tr.spatial_test(frame, cols)
    entry.update({
        "lr_statistic": res["lr_statistic"],
        "lr_equals_subgroup_refit": bool(abs(res["lr_statistic"]
                                             - entry["lr_statistic_recomputed"]) < 0.01),
        "df": res["df"],
        "p_asymptotic": res["p_asymptotic"],
        "p_bootstrap": res["p_bootstrap"],
        "both_converged": res["both_converged"],
        "rejects_at_5pct_bootstrap": res["rejects_at_5pct_bootstrap"],
        "group_sizes_match_code": bool(res["n_large_city"] == entry["groups"]
                                       ["large_city"]["n"]),
    })
    td["bootstrap"] = {"replications": tr.BOOTSTRAP_REPS, "seed": tr.BOOTSTRAP_SEED,
                       "note": ("rerun here on the current frame with "
                                "transferability.temporal_test and spatial_test; the "
                                "stored file's values are kept under "
                                "stored_values_pre_final_labels in each split")}
    td["stored_file_status"] = "transferability.json " + STALE_NOTE
    td["inputs"] = ["data/model_frame.csv", "src/transferability.py",
                    "data/transferability.json (stored values only)"]
    save()
    print(f"  spatial: LR {res['lr_statistic']} p_boot {res['p_bootstrap']}")


def recompute_benchmark(frame: pd.DataFrame) -> None:
    import benchmark as bm

    cols = sv.ladder(frame)["M1"]
    run = bm.run(frame, cols)
    res = run["results"]
    per_class = {}
    for name, m in res.items():
        if name.startswith("_"):
            continue
        per_class[name] = {
            lvl: {k: m[f"{k}_{lvl}"] for k in ("precision", "recall", "f1")}
            for lvl in sv.SEV_ORDER}
        per_class[name]["n_folds"] = m["n_folds"]
    summary = {name: {k: m[k] for k in ("macro_f1", "balanced_accuracy", "pr_auc_KA",
                                        "brier_KA")}
               for name, m in res.items() if not name.startswith("_")}
    never_ka = [n for n, m in res.items() if not n.startswith("_")
                and m["recall_KA"]["mean"] == 0 and m["precision_KA"]["mean"] == 0]
    shap = bm.shap_salience(frame, cols)
    top10 = (dict(list(shap["mean_abs_shap"].items())[:10])
             if shap.get("available") else None)

    bd = RESULT["benchmark_detail"]
    if "stored_values_pre_final_labels" not in bd:
        bd["stored_values_pre_final_labels"] = {
            "per_class_metrics_mean_sd_over_50_folds":
                bd.pop("per_class_metrics_mean_sd_over_50_folds"),
            "models_that_never_predict_KA": bd.pop("models_that_never_predict_KA"),
        }
    bd.update({
        "stored_file_status": "benchmark.json " + STALE_NOTE,
        "recomputed_with": "benchmark.run and benchmark.shap_salience on the current frame",
        "summary_metrics_mean_sd_over_50_folds": summary,
        "per_class_metrics_mean_sd_over_50_folds": per_class,
        "models_that_never_predict_KA": never_ka,
        "baseline": res["_baseline_majority_class"],
        "shap_top10": top10,
        "shap_age_unknown_in_top10": bool(top10 and "age_unknown" in top10),
        "shap_coord_missing_in_top10": bool(top10 and "coord_missing" in top10),
        "shap_full": shap,
    })
    save()
    print("  benchmark rerun done")


def recompute_bike(frame: pd.DataFrame) -> None:
    cols = sv.ladder(frame)["M1"]
    null_ll = sv.null_loglikelihood(frame)
    stored = json.loads((DATA / "bike_facility_sensitivity.json").read_text(encoding="utf-8"))
    out = {}
    for b in ("bike_facility_nearest", "bike_facility_50m", "bike_facility_100m",
              "bike_facility_200m"):
        c = cols + [b]
        fit = sv.fit_ordered_logit(frame, c, f"M1_{b}")
        ames = sv.average_marginal_effects(fit, frame)
        out[b] = {"n_positive": int(frame[b].sum()),
                  "cells_by_severity": cells_by_sev(frame, frame[b] == 1),
                  "fit": sv.fit_metrics(fit, null_ll),
                  "separation": sv.separation_diagnostics(fit),
                  "ame": ames[b]}
    sens = RESULT["data_sensitivities"]["acs_and_bike_facility"]["bike_facility_buffers"]
    sens.update({
        "stored_file_status": "bike_facility_sensitivity.json " + STALE_NOTE,
        "stored_values_pre_final_labels": sens.pop("buffers", None),
        "buffers": out,
        "any_ka_interval_excludes_zero": any(v["ame"]["KA"]["excludes_zero"]
                                             for v in out.values()),
        "note": ("each indicator is added to M1 one at a time and refitted on the current "
                 "frame; the nearest-segment form is included for completeness although "
                 "it was excluded at screening"),
    })
    save()
    print("  bike buffers rerun done")


def stale_status() -> dict:
    """File times of the stored outputs against the model frame, and whether the stored
    values now agree with the recomputation. The stored files can be regenerated by
    other scripts after this one ran, so this is read at the time it is called."""
    def mtime(p: Path) -> str:
        return datetime.fromtimestamp(p.stat().st_mtime, timezone.utc).isoformat(
            timespec="seconds")

    frame_t = (DATA / "model_frame.csv").stat().st_mtime
    files = {}
    for name in ("transferability.json", "benchmark.json",
                 "severity_appendix.json", "bike_facility_sensitivity.json",
                 "numbers.json"):
        p = DATA / name
        files[name] = {"modified": mtime(p),
                       "older_than_model_frame": bool(p.stat().st_mtime < frame_t)}

    agree: dict = {}
    t = json.loads((DATA / "transferability.json").read_text(encoding="utf-8"))
    sp = RESULT["transferability_detail"]["splits"]
    agree["transferability.json"] = bool(
        t["spatial"]["lr_statistic"] == sp["spatial"].get("lr_statistic")
        and t["spatial"]["p_bootstrap"] == sp["spatial"].get("p_bootstrap")
        and all(t["temporal"][str(b)]["p_bootstrap"]
                == sp[f"temporal_{b}"].get("p_bootstrap") for b in tr.BREAKPOINTS))
    b = json.loads((DATA / "benchmark.json").read_text(encoding="utf-8"))
    bd = RESULT["benchmark_detail"]
    agree["benchmark.json"] = bool(all(
        b["results"][m]["balanced_accuracy"] == v["balanced_accuracy"]
        for m, v in bd["summary_metrics_mean_sd_over_50_folds"].items())
        and list(b["shap"]["mean_abs_shap"])[:10] == list(bd["shap_top10"]))
    a = json.loads((DATA / "severity_appendix.json").read_text(encoding="utf-8"))
    four = RESULT["data_sensitivities"]["severity_classes"]["four_class"][
        "refit_on_current_M1"]
    agree["severity_appendix.json (four_class only)"] = bool(
        a["four_class"]["fit"]["loglik"] == four["fit"]["loglik"])
    bike = json.loads((DATA / "bike_facility_sensitivity.json").read_text(encoding="utf-8"))
    cur = RESULT["data_sensitivities"]["acs_and_bike_facility"][
        "bike_facility_buffers"]["buffers"]
    agree["bike_facility_sensitivity.json"] = bool(all(
        bike[k]["ame"]["KA"] == cur[k]["ame"]["KA"] for k in bike))
    return {"checked": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "model_frame_modified": mtime(DATA / "model_frame.csv"),
            "files": files, "stored_values_agree_with_recomputation": agree}


def update_status() -> None:
    RESULT.update(json.loads(OUT.read_text(encoding="utf-8")))
    RESULT["recomputed_stale_outputs"]["status_at_check"] = stale_status()
    save()
    print(json.dumps(RESULT["recomputed_stale_outputs"]["status_at_check"], indent=2))


def recompute() -> None:
    frame = load_frame()
    RESULT.update(json.loads(OUT.read_text(encoding="utf-8")))
    RESULT["recomputed_stale_outputs"] = {
        "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "files_found_stale": ["transferability.json",
                              "benchmark.json",
                              "severity_appendix.json",
                              "bike_facility_sensitivity.json"],
        "reason": STALE_NOTE,
        "recomputed_here": ["transferability tests (all breakpoints and the spatial "
                            "split)", "benchmark and SHAP salience",
                            "bike-facility buffer refits",
                            "four-class model (data_sensitivities.severity_classes)"],
        "not_recomputed": ["multinomial logit and IIA tests in "
                           "severity_appendix.json (outside this task)"],
    }
    save()
    for name, fn in (("bike", recompute_bike), ("benchmark", recompute_benchmark),
                     ("transferability", recompute_transferability)):
        print(f"recompute {name} ...")
        try:
            fn(frame)
        except Exception as exc:                                   # noqa: BLE001
            RESULT["recomputed_stale_outputs"][f"{name}_error"] = (
                f"{type(exc).__name__}: {exc}")
            print(f"  FAILED: {type(exc).__name__}: {exc}")
        save()
    print(f"written to {OUT}")


# --------------------------------------------------------------------------

def main() -> None:
    frame = load_frame()
    RESULT.update({
        "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "script": SCRIPT,
        "inputs": ["data/model_frame.csv", "data/base_frame.csv",
                   "data/screening.json", "data/numbers.json",
                   "data/severity.json",
                   "data/severity_appendix.json",
                   "data/transferability.json",
                   "data/benchmark.json",
                   "data/validation_metrics.json", "data/gold_labels.csv",
                   "data/validation_ids.csv", "data/vote_shares.csv",
                   "data/route_a_labels.csv", "data/reference_labels.csv",
                   "data/fusion.csv", "data/acs_extraction_audit.json",
                   "data/osm_extraction_audit.json",
                   "data/bike_facility_sensitivity.json"],
        "seeds": {"cv": sv.CV_SEED, "krinsky_robb": sv.KR_SEED,
                  "validation_bootstrap": vm.BOOT_SEED, "calibration_cross_fit": CAL_SEED,
                  "transferability_bootstrap_stored": tr.BOOTSTRAP_SEED},
        "draws": {"krinsky_robb": sv.KR_DRAWS, "validation_bootstrap": vm.N_BOOT},
        "n_rider_rows": int(len(frame)),
    })
    steps = (("annual", item_annual), ("cv_fold_level", item_cv),
             ("transferability_detail", item_transferability),
             ("benchmark_detail", item_benchmark),
             ("extraction_intervals_calibration", item_extraction),
             ("data_sensitivities", item_sensitivities),
             ("facts", item_facts))
    for key, fn in steps:
        print(f"{key} ...")
        try:
            RESULT[key] = fn(frame)
        except Exception as exc:                                   # noqa: BLE001
            RESULT[key] = {"error": f"{type(exc).__name__}: {exc}"}
            print(f"  FAILED: {type(exc).__name__}: {exc}")
        save()
    print(f"written to {OUT}")


if __name__ == "__main__":
    # Full run, then the recomputation of the stale outputs. `--recompute` alone reruns
    # only the second stage on an existing result file.
    # `--status` only refreshes the file-time and agreement check.
    if "--status" in sys.argv:
        update_status()
    else:
        if "--recompute" not in sys.argv:
            main()
        recompute()
        update_status()
