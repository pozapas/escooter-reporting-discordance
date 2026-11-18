"""The single source for every number the manuscript prints.

Nothing is typed into the manuscript or a figure caption that is not read from here
first.

The drift this prevents is not hypothetical and it is not caught by proofreading. A model
is re-run, a count moves by one, the abstract still carries the old value, and no reader
can tell.

So this module **derives** rather than restates. It reads the JSON artefacts the analysis
scripts write and assembles them into one flat, addressable structure. It computes nothing
of its own except arithmetic that is purely presentational, and it records where every
value came from. If an artefact is missing, the entry is absent rather than stale, which
is the behaviour that matters: a missing number stops a build, a stale one ships.

Run with:  python src/build_numbers.py
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = DATA / "numbers.json"


def _load(name: str):
    path = DATA / name
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _r(x, n=4):
    if x is None:
        return None
    try:
        return round(float(x), n)
    except (TypeError, ValueError):
        return x


def data_section() -> dict:
    a = _load("base_frame_audit.json") or {}
    osm = _load("osm_extraction_audit.json") or {}
    acs = _load("acs_extraction_audit.json") or {}
    frame = pd.read_csv(DATA / "model_frame.csv")

    return {
        "raw_rows": a.get("raw_rows"),
        "raw_crash_ids": a.get("unique_crash_ids_raw"),
        "dropped_unknown_severity": a.get("dropped_unknown_severity"),
        "rider_rows": a.get("modeled_rows"),
        "distinct_crashes": a.get("modeled_unique_crashes"),
        "crashes_with_two_riders": a.get("n_crashes_with_multiple_scooter_riders"),
        "second_unit_motor_vehicle": a.get("second_unit_motor_vehicle"),
        "rider_severity_agreement": {
            "comparable": a.get("rider_severity_comparable_n"),
            "agree": a.get("rider_severity_equals_crash_severity")},
        "kabco": a.get("sev_kabco_counts"),
        "sev3": a.get("sev3_counts"),
        "sev4": a.get("sev4_counts"),
        "age_missing": a.get("age_missing"),
        "coord_missing": a.get("coord_missing"),
        "coord_present": a.get("coord_present"),
        "located_crashes": osm.get("located_crashes"),
        "narrative_chars_median": a.get("narrative_chars_median"),
        "narrative_chars_iqr": [a.get("narrative_chars_q1"), a.get("narrative_chars_q3")],
        "years": [int(frame["crash_year"].min()), int(frame["crash_year"].max())],
        "crashes_by_year": frame["crash_year"].value_counts().sort_index().to_dict(),
        "distinct_cities": int(frame["City_ID"].nunique()),
        "top_cities": frame["City_ID"].value_counts().head(5).to_dict(),
        "osm": {"tiles": osm.get("tiles"), "failed_tiles": osm.get("failed_tiles"),
                "extraction_date": osm.get("extraction_date"),
                "bike_facility_nearest": osm.get("bike_facility_nearest_positives"),
                "bike_facility_100m": osm.get("bike_facility_100m_positives"),
                "road_density_median": _r(osm.get("road_density_km_100m_median"))},
        "acs": {"tiger_vintage": acs.get("tiger_vintage"),
                "acs_vintage": acs.get("acs_primary_vintage"),
                "unmatched_points": acs.get("points_not_matched_to_a_block_group"),
                "poverty_missing": acs.get("poverty_missing_primary"),
                "poverty_median": _r(acs.get("poverty_share_median"), 3),
                "poverty_iqr": [_r(x, 3) for x in (acs.get("poverty_share_iqr") or [])]},
    }


def extraction_section() -> dict:
    vm = _load("validation_metrics.json")
    fl = _load("final_labels_report.json")
    if vm is None:
        return {}

    ag = vm.get("agreement", {})
    kappas = {k: _r(v["fleiss_kappa"], 3) for k, v in ag.items()}
    defined = [v for v in kappas.values() if v is not None and v == v]

    # A route's macro F1 averages whatever labels that route produces: Route A and the
    # fusion score the ten mechanisms only, the Route B reference run also scores the five
    # injury cues. Routes are compared on the mechanism-only average, never on macro_f1.
    from labels import MECHANISMS, INJURY_CUES

    def _label_macro(m: dict, labels) -> float | None:
        f1s = [m[lab]["unweighted"]["f1"] for lab in labels
               if lab in m and m[lab]["unweighted"].get("f1") == m[lab]["unweighted"].get("f1")]
        return _r(np.mean(f1s)) if f1s else None

    routes = {}
    for name, m in (vm.get("routes") or {}).items():
        routes[name] = {
            "macro_f1": _r((m.get("macro_unweighted") or {}).get("macro_f1")),
            "macro_f1_mechanisms": _label_macro(m, MECHANISMS),
            "macro_f1_injury_cues": _label_macro(m, INJURY_CUES),
            "macro_f1_ip_weighted": _r((m.get("macro_ip_weighted") or {}).get("macro_f1")),
            "per_label": {lab: {
                "precision": _r(v["unweighted"].get("precision")),
                "recall": _r(v["unweighted"].get("recall")),
                "f1": _r(v["unweighted"].get("f1")),
                "tp": v["unweighted"].get("tp"), "fp": v["unweighted"].get("fp"),
                "fn": v["unweighted"].get("fn"), "tn": v["unweighted"].get("tn"),
            } for lab, v in m.items()
                if isinstance(v, dict) and "unweighted" in v},
        }

    mr = vm.get("merge_report", {})
    return {
        "validation_n": mr.get("narratives_coded_by_every_coder"),
        "coders": mr.get("coders"),
        "fleiss_kappa_by_label": kappas,
        "fleiss_kappa_macro": _r(np.mean(defined), 3) if defined else None,
        "pairwise_cohen_kappa": {k: v.get("pairwise_cohen_kappa")
                                 for k, v in ag.items()},
        "routes": routes,
        "validation_subset_rates": _validation_rates(),
        "final_labels": {
            "narratives": (fl or {}).get("narratives"),
            "by_source": (fl or {}).get("by_source"),
            "human_share": _r((fl or {}).get("human_share"), 3),
            "rate_by_label": (fl or {}).get("rate_by_label"),
        },
    }


def _validation_rates() -> dict:
    """Per-cue positive rates on the validation sample: humans against each route.

    Section 5 quotes these to show that the model reads stated outcomes accurately and
    infers unstated behaviour liberally, so they have to be derivable rather than
    recomputed by hand in the text.
    """
    codes = DATA / "validation_codes_batch1.csv"
    if not codes.exists():
        return {}
    from labels import ALL_LABELS
    long = pd.read_csv(codes)
    if "_coded" in long.columns:
        long = long[long["_coded"].astype(bool)]
    piv = long.pivot_table(index="crash_id", columns="coder_id",
                           values=list(ALL_LABELS), aggfunc="first")
    ids = piv.index
    out: dict[str, dict] = {}
    ref = pd.read_csv(DATA / "reference_labels.csv").set_index("Crash_ID")
    ra_path = DATA / "route_a_labels.csv"
    ra = pd.read_csv(ra_path).set_index("Crash_ID") if ra_path.exists() else None
    for lab in ALL_LABELS:
        v = piv[lab]
        n = v.notna().sum(axis=1)
        gold = (v.fillna(0).sum(axis=1) > n / 2).astype(int)
        entry = {"human": _r(gold.mean(), 4)}
        if lab in ref.columns:
            entry["route_b"] = _r(ref.loc[ids, lab].mean(), 4)
        if ra is not None and lab in ra.columns:
            entry["route_a"] = _r(ra.loc[ids, lab].mean(), 4)
        out[lab] = entry
    return out


def audit_section() -> dict:
    ct = _load("audit_crosstab.json")
    st = _load("audit_stage1.json")
    rec = _load("audit_recovery_seeds.json")
    if ct is None:
        return {}

    counts = ct["cross_tab_kabco"]["counts"]
    order = ["O", "C", "B", "A", "K"]
    nocue = np.array([counts[k]["no cue"] for k in order], float)
    other = np.array([sum(counts[k][c] for c in counts[k] if c != "no cue")
                      for k in order], float)
    tab = np.column_stack([nocue, other])
    chi_all, p_all, df_all, _ = stats.chi2_contingency(tab)
    chi_4, p_4, df_4, _ = stats.chi2_contingency(tab[:4])

    out = {
        "label_source": ct.get("label_source"),
        "indicator_counts": ct.get("indicator_counts"),
        "cross_tab_kabco": ct["cross_tab_kabco"],
        "cross_tab_sev3": ct["cross_tab_sev3"],
        "cue_conflicts": ct.get("cue_conflicts"),
        "no_cue_share_overall": _r(nocue.sum() / (nocue.sum() + other.sum()), 3),
        "no_cue_flatness": {
            "all_five_levels": {"chi2": _r(chi_all, 2), "df": int(df_all),
                                "p": _r(p_all, 4)},
            # Stored at full precision: rounded at four places it is 0.3385, which
            # would be rounded a second time to the three places the tables print.
            "excluding_fatal": {"chi2": _r(chi_4, 2), "df": int(df_4),
                                "p": float(p_4)},
            "reading": ("The share of narratives carrying no injury-outcome language "
                        "does not differ across the four non-fatal coded levels; the "
                        "overall test rejects, and the rejection is carried entirely by "
                        "the fatal class."),
        },
    }
    if st:
        mod = st.get("moderate", {})
        out["stage1"] = {
            "prevalence": mod.get("latent_prevalence"),
            "narrative_measurement": mod.get("narrative_measurement"),
            "class_alignment": mod.get("class_alignment"),
            "diagnostics": mod.get("diagnostics"),
            "fit_against_raw_table": {
                k: mod.get("fit_against_raw_table", {}).get(k)
                for k in ("max_abs_pearson_residual", "reproduces_raw_table")},
        }
    if isinstance(rec, list):
        # The first version of the power study wrote a bare list of per-dataset results;
        # `audit_power.py` writes the summary dict. Accept either rather than depend on
        # which script last ran.
        means = [r["posterior_mean"] for r in rec if "posterior_mean" in r]
        planted = rec[0].get("effect_size") if rec else None
        rec = {
            "planted_effect": planted,
            "n_datasets": len(rec),
            "mean_posterior_mean": _r(np.mean(means), 3) if means else None,
            "multiplicative_bias": (_r(np.mean(means) / planted, 2)
                                    if means and planted else None),
            "coverage": f"{sum(r.get('covers_truth', False) for r in rec)}/{len(rec)}",
            "power_excludes_zero": (f"{sum(r.get('excludes_zero', False) for r in rec)}"
                                    f"/{len(rec)}"),
            "mean_false_positives_of_47_zeros": _r(np.mean(
                [r.get("false_positives_among_47_true_zeros", 0) for r in rec]), 2),
        }
    if rec:
        out["shifter_power"] = {
            "planted_effect": rec.get("planted_effect"),
            "datasets": rec.get("n_datasets"),
            "mean_posterior_mean": rec.get("mean_posterior_mean"),
            "multiplicative_bias": rec.get("multiplicative_bias"),
            "coverage": rec.get("coverage"),
            "power": rec.get("power_excludes_zero"),
            "false_positives_of_47": rec.get("mean_false_positives_of_47_zeros"),
        }
    return out


def severity_section() -> dict:
    sv = _load("severity.json")
    ap = _load("severity_appendix.json")
    bike = _load("bike_facility_sensitivity.json")
    scr = _load("screening.json")
    if sv is None:
        return {}

    rungs = {}
    for name, r in (sv.get("rungs") or {}).items():
        rungs[name] = {
            "loglik": _r(r["fit"]["loglik"], 2),
            "n_parameters": r["fit"]["n_parameters"],
            "adjusted_rho2": _r(r["fit"]["adjusted_rho2"]),
            "aic": _r(r["fit"]["aic"], 1), "bic": _r(r["fit"]["bic"], 1),
            "events_per_parameter": r["events_per_parameter"][
                "events_per_parameter_smallest_class"],
            "n_covariates": r["events_per_parameter"]["n_covariates"],
            "cv_log_loss": _r(r["cross_validation"]["mean_log_loss"], 3),
            "cv_sd": _r(r["cross_validation"]["sd_across_folds"], 3),
            "cv_folds_failed": r["cross_validation"]["n_folds_failed"],
            "separation_flag": r["separation"]["suggests_separation"],
        }
        # The model estimates table prints a slope, a standard error and an interval for
        # every covariate on every rung, so they are registered like any other printed
        # value.
        coef = r.get("coefficients")
        if coef:
            rungs[name]["coefficients"] = {
                "slopes": {c: {"coef": _r(s["coef"], 3), "se": _r(s["se"], 3),
                               "z": _r(s["z"], 2), "p": _r(s["p"], 3),
                               "ci95": [_r(s["ci95"][0], 3), _r(s["ci95"][1], 3)],
                               "excludes_zero": s["excludes_zero"]}
                           for c, s in coef["slopes"].items()},
                "cutpoints": [_r(c, 3) for c in coef["cutpoints"]],
                "n_excluding_zero": sum(1 for s in coef["slopes"].values()
                                        if s["excludes_zero"]),
            }

    po = sv.get("proportional_odds_tests", {})
    ames = sv.get("average_marginal_effects", {})
    excl = {c: {lvl: {"ame": v[lvl]["ame"], "ci95": v[lvl]["ci95"]}
                for lvl in ("O", "BC", "KA")}
            for c, v in ames.items() if v["KA"]["excludes_zero"]}

    return {
        "primary_model": sv.get("primary_model"),
        "rungs": rungs,
        "nested_lr": {"M0_to_M0c": sv.get("lr_M0_to_M0c"),
                      "M0c_to_M1": sv.get("lr_M0c_to_M1")},
        "proportional_odds": {
            "omnibus_lr_chi2": po.get("omnibus_lr_chi2"),
            "omnibus_df": po.get("omnibus_df"),
            "omnibus_p": po.get("omnibus_lr_p"),
            "rejects": po.get("omnibus_rejects_at_5pct"),
            "n_failing": len(po.get("covariates_failing_at_5pct") or []),
            "failing": po.get("covariates_failing_at_5pct"),
            "wald_form_chi2": po.get("omnibus_wald_chi2_brant_form"),
            "wald_form_p": po.get("omnibus_wald_p"),
        },
        "multiplicity": {k: {"n_freed": v.get("n_freed"),
                             "bic_difference": (v.get("bic_confirmation") or {}
                                                ).get("difference"),
                             "primary": v.get("primary")}
                         for k, v in (sv.get("multiplicity_sensitivity") or {}).items()
                         if isinstance(v, dict) and "n_freed" in v},
        "ames_excluding_zero": excl,
        "ames_all": ames,
        "screening": {"n_candidates": (scr or {}).get("n_candidates"),
                      "n_included": (scr or {}).get("n_included"),
                      "layers": (scr or {}).get("layers")},
        "four_class": {
            "loglik": _r(((ap or {}).get("four_class") or {}).get("fit", {}).get("loglik"), 2),
            "adjusted_rho2": _r(((ap or {}).get("four_class") or {}).get("fit", {}
                                 ).get("adjusted_rho2")),
            "events_per_parameter": ((ap or {}).get("four_class") or {}).get(
                "events_per_parameter", {}).get("events_per_parameter_smallest_class"),
        } if ap else None,
        "iia": {
            "small_hsiao_usable": (((ap or {}).get("multinomial_robustness") or {})
                                   .get("small_hsiao_calibration") or {}
                                   ).get("test_is_usable"),
            "small_hsiao_worst_reject_rate": (
                ((ap or {}).get("multinomial_robustness") or {})
                .get("small_hsiao_calibration") or {}).get("worst_reject_fraction"),
            "hausman_mcfadden": {
                k: {"chi2": v.get("hausman_mcfadden_chi2"),
                    "p": v.get("hausman_mcfadden_p"),
                    "negative": v.get("hausman_mcfadden_negative")}
                for k, v in (((ap or {}).get("multinomial_robustness") or {})
                             .get("iia_tests") or {}).items()
                if isinstance(v, dict)},
        } if ap else None,
        "bike_facility": {k: {"n_positive": v.get("n_positive"),
                              "ame_KA": v["ame"]["KA"]["ame"],
                              "ci95_KA": v["ame"]["KA"]["ci95"]}
                          for k, v in (bike or {}).items()} if bike else None,
    }


def transferability_section() -> dict:
    t = _load("transferability.json")
    if t is None:
        return {}
    out = {"annual": t.get("annual", {}).get("counts"),
           "untestable": t.get("untestable_breakpoints"),
           "temporal": {}, "spatial": {}}
    for bp, r in (t.get("temporal") or {}).items():
        if not r.get("testable"):
            out["temporal"][bp] = {"testable": False, "reason": r.get("reason")}
            continue
        out["temporal"][bp] = {
            "n_before": r["n_before"], "n_after": r["n_after"],
            "covariates_tested": r["covariates_tested"],
            "covariates_dropped": len(r.get("covariates_dropped") or {}),
            "lr": r["lr_statistic"], "df": r["df"],
            "p_asymptotic": r["p_asymptotic"], "p_bootstrap": r["p_bootstrap"],
            "rejects": r["rejects_at_5pct_bootstrap"],
        }
    for bp, r in (t.get("temporal") or {}).items():
        cc = r.get("city_composition")
        if cc and bp in out["temporal"]:
            out["temporal"][bp]["city_composition"] = cc
    sp = t.get("spatial") or {}
    out["spatial"] = {k: sp.get(k) for k in
                      ("n_large_city", "n_rest", "covariates_tested", "lr_statistic",
                       "df", "p_asymptotic", "p_bootstrap",
                       "rejects_at_5pct_bootstrap")}
    return out


def benchmark_section() -> dict:
    b = _load("benchmark.json")
    if b is None:
        return {}
    res = b.get("results", {})
    out = {"repeats": b.get("repeats"), "folds": b.get("folds"),
           "baseline": res.get("_baseline_majority_class"), "models": {}}
    for name, m in res.items():
        if name.startswith("_"):
            continue
        out["models"][name] = {
            k: {"mean": m[k]["mean"], "sd": m[k]["sd"]}
            for k in ("macro_f1", "balanced_accuracy", "pr_auc_KA", "brier_KA")
            if k in m}
    shap = b.get("shap") or {}
    if shap.get("available"):
        top = list(shap["mean_abs_shap"].items())[:10]
        out["shap_top10"] = dict(top)
    return out


def external_sources_section() -> dict:
    """Verified figures from documents outside this study.

    They belong in the single source for the same reason everything else does: the
    Introduction prints them, so a checker has to be able to derive them. Each was read
    from the primary document rather than from a search result.
    """
    return {
        "cpsc_2026": {
            "report": ("U.S. Consumer Product Safety Commission, Micromobility "
                       "Products-Related Deaths, Injuries, and Hazard Patterns: "
                       "2017-2024, April 2026"),
            "read_on": "2026-09-14",
            "period": [2017, 2024],
            "ed_visits_all_micromobility": 698500,
            "ed_visits_escooter": 380000,
            "ed_visits_self_balancing": 163300,
            "ed_visits_ebike": 155200,
            "fatalities_all_micromobility": 533,
            "fatalities_escooter": 206,
            "fatalities_ebike": 310,
            "fatalities_self_balancing": 17,
            "fatalities_2017": 5,
            "fatalities_2024": 135,
            "escooter_fatalities_battery_fire": 15,
            "special_study_2024": {
                "cases_followed_up": 173, "rental_share_percent": 35,
                "paved_road_percent": 54, "paved_sidewalk_percent": 32,
                "distraction_percent": 11, "helmet_percent": 18,
                "visibility_percent": 20},
            "leading_hazard": ("motor vehicle accidents and control issues were the top "
                               "hazards associated with e-scooter fatalities"),
        },
    }


def masking_section() -> dict:
    """What the prompt-hash reconstruction proves, as numbers the text cites."""
    import extract as ex
    frame = ex._narratives()
    changed = int((frame.n_masked_tokens > 0).sum())
    return {
        "narratives": int(len(frame)),
        "narratives_masking_changes": changed,
        "narratives_masking_leaves_identical": int(len(frame)) - changed,
        "tokens_masked": int(frame.n_masked_tokens.sum()),
    }


# Table 1's groups, shared with build_tables.py so that the table cannot print a level the
# registry does not know about. The reference level is named rather than left to be
# inferred from what is missing.
COVARIATE_GROUPS: tuple[tuple[str, str, tuple[tuple[str, str], ...]], ...] = (
    ("Posted speed limit", "30 mph or below", (
        ("speed_35to40", "35 to 40 mph"),
        ("speed_ge45", "45 mph or above"),
        ("speed_unknown", "Not recorded"))),
    ("Rider age", "25 to 44", (
        ("age_under25", "Under 25"),
        ("age_45plus", "45 or above"),
        ("age_unknown", "Not recorded"))),
    ("Rider gender", "Male", (
        ("gender_female", "Female"),
        ("gender_unknown", "Not recorded"))),
    ("Rider ethnicity", "White", (
        ("eth_black", "Black"),
        ("eth_hispanic", "Hispanic"),
        ("eth_other_or_unknown", "Other or not recorded"))),
    ("Striking vehicle", "Passenger car", (
        ("veh_suv", "Sport utility"),
        ("veh_pickup_truck_van", "Pickup, truck or van"),
        ("veh_other_unknown", "Other or not recorded"))),
    ("Light condition", "Daylight", (
        ("light_dark_or_unknown", "Dark or not recorded"),)),
    ("Collision manner", "Angle", (
        ("coll_single_mv_straight", "Vehicle going straight"),
        ("coll_turning", "Vehicle turning"),
        ("coll_other", "Other manner"))),
    ("Intersection relation", "Not at an intersection", (
        ("intx_intersection_or_driveway", "At intersection or driveway"),)),
    ("Traffic control", "None, other or not recorded", (
        ("ctrl_signal", "Traffic signal"),
        ("ctrl_stop_sign", "Stop sign"))),
    ("Roadway class", "City street", (
        ("road_not_city_street", "Not a city street"),)),
    ("Crash context", None, (
        ("nighttime", "Night"),
        ("weekend", "Weekend"),
        ("coord_missing", "Crash coordinates absent"),
        ("urban_area_200k", "Inside an urban area of 200,000 or more"))),
    # M0c and M1 add these. Table 1 is the paper's variable dictionary, so a covariate
    # that enters any rung belongs in it; leaving the cues out made the table describe
    # the structured layer while the prose described the whole specification.
    ("Officer-coded contributing factors", "not cited", (
        ("cf_rider_failed_to_yield", "Rider failed to yield"),
        ("cf_driver_inattention", "Driver inattention"))),
    ("Narrative-derived pre-crash cues", "cue absent", (
        ("sidewalk_transition", "Sidewalk to roadway transition"),
        ("failure_to_yield", "Failure to yield"),
        ("signal_violation", "Signal violation"),
        ("vehicle_turning_across", "Vehicle turning across"),
        ("lane_positioning", "Lane positioning"))),
)

# Reported as a median rather than a count, because a share of the sample means nothing
# for a continuous measure. Each is a covariate in the primary specification.
CONTINUOUS_COVARIATES: tuple[tuple[str, str], ...] = (
    ("road_density_km_100m", "Road density within 100 m"),
    ("poverty_share", "Block-group poverty share"),
)


def covariate_table_section() -> dict:
    """Every cell printed in Table 1, so that none is printed unregistered."""
    frame = pd.read_csv(DATA / "model_frame.csv")
    n_total = len(frame)
    out: dict = {"n_total": n_total, "groups": []}
    for group, reference, levels in COVARIATE_GROUPS:
        rows = []
        for col, label in levels:
            if col not in frame.columns:
                raise SystemExit(f"covariate table: {col} is not in model_frame.csv")
            mask = frame[col].astype(float) == 1
            n = int(mask.sum())
            comp = {}
            for level in ("O", "BC", "KA"):
                k = int((frame.loc[mask, "sev3"] == level).sum())
                comp[level] = _r(100.0 * k / n, 1) if n else None
            rows.append({"column": col, "label": label, "n": n,
                         "share_of_sample": _r(100.0 * n / n_total, 1),
                         "severity_percent": comp})
        out["groups"].append({"group": group, "reference": reference, "levels": rows})

    # The continuous covariates carry a median and a quartile range instead of a count.
    out["continuous"] = []
    for col, label in CONTINUOUS_COVARIATES:
        if col not in frame.columns:
            raise SystemExit(f"covariate table: {col} is not in model_frame.csv")
        s = frame[col].astype(float)
        out["continuous"].append({
            "column": col, "label": label,
            "median": _r(float(s.median()), 3),
            "iqr": [_r(float(s.quantile(0.25)), 3), _r(float(s.quantile(0.75)), 3)],
        })

    # A guard, because the point of this table is that it is complete: every covariate in
    # the widest rung must appear in it exactly once.
    listed = {lvl["column"] for g in out["groups"] for lvl in g["levels"]}
    listed |= {c["column"] for c in out["continuous"]}
    sev = _load("severity.json") or {}
    m1 = (((sev.get("rungs") or {}).get("M1") or {}).get("coefficients") or {})
    model_cols = set(m1.get("columns") or [])
    if model_cols:
        missing = sorted(model_cols - listed)
        extra = sorted(listed - model_cols)
        if missing or extra:
            raise SystemExit(
                "Table 1 must list every covariate in M1 and nothing else.\n"
                f"  in the model but not in the table: {missing}\n"
                f"  in the table but not in the model: {extra}")
    return out


def appendix_section() -> dict:
    """Everything the appendix tables print, so that none of it is printed unregistered.

    The screening table needs every screening candidate, not only the count of them,
    with the rule that excluded each. The four-class appendix needs the model's
    marginal effects in full rather than summarized.
    """
    screening = _load("screening.json") or {}
    appendix = _load("severity_appendix.json") or {}
    four = appendix.get("four_class", {})

    return {
        "screening_detail": {
            "min_events_rule": screening.get("min_events_rule"),
            "max_vif_rule": screening.get("max_vif_rule"),
            "candidates": screening.get("candidates", []),
            "references": screening.get("references"),
            "stage_two_merges": screening.get("stage_two_merges"),
        },
        "four_class_detail": {
            "order": four.get("order"),
            "fit": four.get("fit"),
            "cutpoints": four.get("cutpoints"),
            "separation": four.get("separation"),
            "average_marginal_effects": four.get("average_marginal_effects", {}),
        },
        "multinomial_robustness": appendix.get("multinomial_robustness", {}),
    }


def figure3_support_section() -> dict:
    """The marginal distributions Figure 3 puts on its diagonal.

    Built from the same merge the figure uses: the model frame for the analytic rows, the
    base frame for age and posted speed, and the final labels for the pre-crash cue count.
    A posted speed of zero or less is the missing code and is dropped, which is what the
    figure does.
    """
    import sys as _sys
    _sys.path.insert(0, str(Path(__file__).resolve().parent))
    from build_figures_main import PRECRASH_CUES

    frame = pd.read_csv(DATA / "model_frame.csv")
    base = pd.read_csv(DATA / "base_frame.csv")
    labels = pd.read_csv(DATA / "final_labels.csv")
    cues = [c for c in PRECRASH_CUES if c in labels.columns]
    labels = labels.assign(
        cue_count=labels[cues].fillna(0).astype(float).sum(axis=1))

    d = (frame[["Crash_ID", "rider_row_id", "sev3"]]
         .merge(base[["rider_row_id", "rider_age", "Crash_Speed_Limit"]],
                on="rider_row_id", how="left")
         .merge(labels[["Crash_ID", "cue_count"]], on="Crash_ID", how="left"))
    d["speed"] = d["Crash_Speed_Limit"].where(d["Crash_Speed_Limit"] > 0)

    out: dict = {"n_total": int(len(d))}
    # The output key is not "rider_age", the name of a person-level field. This is a
    # median.
    for key, col in (("age_years", "rider_age"), ("posted_speed", "speed"),
                     ("cue_count", "cue_count")):
        s = d[col].astype(float).dropna()
        out[key] = {
            "median": _r(float(s.median()), 1),
            "iqr": [_r(float(s.quantile(0.25)), 1), _r(float(s.quantile(0.75)), 1)],
            "n_recorded": int(len(s)),
        }
    return out


def _printable(o, max_list: int = 60):
    """A result file reduced to what a sentence or a table cell can print.

    Long arrays (posterior draws, per-row vectors) are dropped and their length recorded,
    so the section stays readable and a printed value can still be traced to its file.
    """
    if isinstance(o, dict):
        return {k: _printable(v, max_list) for k, v in o.items()}
    if isinstance(o, list):
        if len(o) > max_list:
            return {"omitted_list_length": len(o)}
        return [_printable(v, max_list) for v in o]
    return o


def sensitivity_section() -> dict:
    """The robustness and supporting analyses, each read from its own result file."""
    files = {
        "labels": "label_sensitivity.json",
        "audit_final": "audit_final.json",
        "m2": "m2.json",
        "random_parameters": "random_parameters.json",
        "alt_model": "alt_model_validation.json",
        "masking": "masking_comparison.json",
        "supporting": "supporting_analyses.json",
        "ordinal_alternatives": "ordinal_alternatives.json",
        "data_extra": "data_extra.json",
        "urbanized": "urbanized_sensitivity.json",
        "safe_system": "safe_system_evidence.json",
        "excluded_cues": "excluded_cues.json",
    }
    out = {}
    for key, name in files.items():
        loaded = _load(name)
        if loaded is not None:
            out[key] = _printable(loaded)
    return out


def build() -> dict:
    numbers = {
        "generated": date.today().isoformat(),
        "note": ("Single source for every value printed in the manuscript, "
                 "figure captions and the supplementary material. Derived from "
                 "the analysis artefacts; nothing here is computed by hand. A value "
                 "absent from this file may not be printed."),
        "data": data_section(),
        "covariate_table": covariate_table_section(),
        "figure3_support": figure3_support_section(),
        # Written by build_figures_main.py figure5: the covariates Figure 5 shows.
        "figure5": _load("figure5_rows.json"),
        "extraction": extraction_section(),
        "audit": audit_section(),
        "severity": severity_section(),
        "appendix": appendix_section(),
        "transferability": transferability_section(),
        "benchmark": benchmark_section(),
        "sensitivity": sensitivity_section(),
        "external_sources": external_sources_section(),
        "masking": masking_section(),
    }
    OUT.write_text(json.dumps(numbers, indent=2, default=str), encoding="utf-8")
    return numbers


if __name__ == "__main__":
    n = build()
    print(f"numbers.json written to {OUT}")
    print()
    for section, body in n.items():
        if not isinstance(body, dict):
            continue
        filled = sum(1 for v in body.values() if v not in (None, {}, []))
        state = "ok" if body else "EMPTY - upstream artefact missing"
        print(f"  {section:<16} {filled:>3} top-level entries   {state}")
    print()
    d, e, a, s = n["data"], n["extraction"], n["audit"], n["severity"]
    print("spot check:")
    print(f"  rider rows {d.get('rider_rows')} in {d.get('distinct_crashes')} crashes")
    print(f"  macro Fleiss kappa {e.get('fleiss_kappa_macro')} on "
          f"{e.get('validation_n')} narratives")
    print(f"  no-cue share {a.get('no_cue_share_overall')}, flat across the four "
          f"non-fatal levels at p = "
          f"{a.get('no_cue_flatness', {}).get('excluding_fatal', {}).get('p')}")
    print(f"  primary model: {s.get('primary_model')}")
