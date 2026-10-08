"""The pre-specified urbanized-area indicator in the primary model, and M1 without it.

prespec section 2.1 lists "Urbanized" (1 if the crash point falls inside a 2020 Census
Urban Area of population 200,000 or more, city lookup where coordinates are absent) among
the roadway and environment candidates. `model_frame.py` builds it as `urban_area_200k`
with the construction in `data_extra._urbanized_area`, and the section 3 inclusion rules
admit it (`screening.json`), so it is a covariate of the primary ordered logit M1. This
script reports the reverse sensitivity for transparency: M1 refitted without the
indicator, which is the specification reported before the indicator was screened.

What is written to `data/urbanized_sensitivity.json`:

* ``indicator``   - the construction, with the frame column checked against a fresh
  call of `data_extra._urbanized_area` and against the cells in `data_extra.json`;
* ``screening``   - the screening entry of the indicator from `screening.json` (event
  rule on both sides, VIF in the pooled screening design, disposition) and, labelled as
  auxiliary, the VIF in the M1 design and the association with the large-city indicator
  and the spatial covariates that take city-level values where coordinates are absent;
* ``primary_with_indicator`` - the indicator's coefficient, odds ratio and AMEs in M1;
* ``without_indicator`` - M1 minus the indicator: fit, LR test of the indicator, BIC and
  AIC against M1, and which interval conclusions change when the indicator is removed;
* ``kr_seed_check`` - the KA intervals of the requested covariates under the four
  Krinsky-Robb seeds of `ordinal_alternatives.json`, for M1 and for the model without the
  indicator, because the absent-coordinates interval sits on zero and a seed change alone
  moves it.

Run with:  python src/urbanized_sensitivity.py
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import data_extra as dx           # noqa: E402
import model_frame as mf          # noqa: E402
import severity as sv             # noqa: E402
from random_parameters import LARGE_CITIES  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = DATA / "urbanized_sensitivity.json"
PREVIOUS_PRIMARY = DATA / "backup_pre_urbanized" / "severity.json"

COL = "urban_area_200k"

KA_REQUESTED = ("poverty_share", "coord_missing", "age_unknown", "sidewalk_transition",
                "failure_to_yield", "signal_violation", "vehicle_turning_across",
                "lane_positioning")
KR_SEEDS = (20260917, 20260918, 20260919, 20260920)   # ordinal_alternatives seed check

INPUTS = {
    "model_frame": "data/model_frame.csv",
    "base_frame": "data/base_frame.csv",
    "screening": "data/screening.json",
    "severity_primary": "data/severity.json",
    "data_extra": "data/data_extra.json",
    "previous_primary_without_indicator":
        "data/backup_pre_urbanized/severity.json",
    "urban_areas": ["data/shapefiles/tl_2020_us_uac20.shp",
                    "data/shapefiles/ua_list_all.txt"],
    "prespec": "docs/prespec.md",
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _vif_table(X: pd.DataFrame) -> dict[str, float]:
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    X = X.astype(float).dropna()
    X = X.loc[:, X.std() > 0]
    Xc = np.column_stack([np.ones(len(X)), X.to_numpy()])
    return {name: round(float(variance_inflation_factor(Xc, i)), 3)
            for i, name in enumerate(X.columns, start=1)}


def check_indicator(frame: pd.DataFrame, base: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """The frame column against a fresh construction and against data_extra.json."""
    if COL not in frame.columns:
        raise SystemExit(f"{COL} is not in model_frame.csv; run model_frame.py first")
    fr = frame.copy()
    fr["_rural"] = (base["Rural_Fl"].astype(str).str.upper() == "Y").astype(int).to_numpy()
    ua, meta = dx._urbanized_area(fr)
    same = bool((ua.to_numpy().astype(int) == fr[COL].to_numpy().astype(int)).all())
    if not same:
        raise SystemExit(f"{COL} in model_frame.csv differs from data_extra._urbanized_area")
    counts = {"inside_ua_200k": dx._cells(fr, fr[COL] == 1),
              "outside": dx._cells(fr, fr[COL] == 0)}
    stored = json.loads((DATA / "data_extra.json").read_text(encoding="utf-8"))
    stored = stored["spatial_indicator"]["urbanized_area_indicator"]["counts_by_severity"]
    meta["counts_by_severity"] = counts
    meta["frame_column_matches_construction"] = same
    meta["reproduces_data_extra"] = bool(counts == stored)
    if not meta["reproduces_data_extra"]:
        raise SystemExit(f"indicator does not reproduce data_extra.json: {counts} vs {stored}")
    meta["construction"] = ("model_frame.py with data_extra._urbanized_area: "
                            "point-in-polygon against the 2020 Census Urban Areas of "
                            "population 200,000 or more; a record without coordinates "
                            "takes the majority value of the located records of its "
                            "City_ID, and a record whose city cannot be placed takes the "
                            "CRIS rural flag")
    meta["column"] = COL
    return fr, meta


def screening(fr: pd.DataFrame, m1_cols: list[str]) -> dict:
    scr = json.loads((DATA / "screening.json").read_text(encoding="utf-8"))
    entry = next(r for r in scr["candidates"] if r["candidate"] == COL)
    outside = dx._cells(fr, fr[COL] == 0)
    event = {
        "indicator_equal_1_by_severity": entry["events_by_severity"],
        "indicator_equal_0_by_severity": outside,
        "smallest_cell_indicator_1": entry["min_cell"],
        "smallest_cell_indicator_0": min(outside.values()),
        "min_events_rule": scr["min_events_rule"],
        "meets_event_rule": entry["meets_event_rule"],
        "reference_side_meets_event_rule": bool(min(outside.values()) >= mf.MIN_EVENTS),
        "rule_as_applied": ("model_frame.screen counts the rows where the indicator is 1 "
                            "in each severity class; the reference side is reported too"),
    }
    pooled = {
        "design": ("every screening candidate meeting the event rule (model_frame.screen), "
                   "complete cases"),
        "n_columns": sum(1 for r in scr["candidates"] if r["meets_event_rule"]),
        "vif_indicator": entry["vif"],
        "max_vif_rule": scr["max_vif_rule"],
        "meets_vif_rule": bool(entry["vif"] < scr["max_vif_rule"]),
    }

    m1_without = [c for c in m1_cols if c != COL]
    m1_vif = _vif_table(fr[m1_cols])
    large = fr["City_ID"].isin(LARGE_CITIES).astype(int)
    unloc = fr["coord_missing"] == 1

    def corr(a, b, mask=None):
        a = pd.Series(np.asarray(a, dtype=float))
        b = pd.Series(np.asarray(b, dtype=float))
        if mask is not None:
            m = np.asarray(mask, dtype=bool)
            a, b = a[m], b[m]
        if a.std() == 0 or b.std() == 0:
            return None
        return round(float(np.corrcoef(a, b)[0, 1]), 3)

    xt = pd.crosstab(fr[COL], large)
    aux = {
        "note": ("auxiliary diagnostics, not the prespec rule: the large-city indicator is "
                 f"not an M1 covariate, and the M1-design VIF is computed on the "
                 f"{len(m1_cols)} M1 covariates rather than the pooled screening design"),
        "vif_in_m1_design": {
            c: m1_vif[c] for c in (COL, "poverty_share", "road_density_km_100m",
                                   "coord_missing", "road_not_city_street")
            if c in m1_vif},
        "vif_in_m1_design_without_indicator": {
            c: v for c, v in _vif_table(fr[m1_without]).items()
            if c in ("poverty_share", "road_density_km_100m", "coord_missing",
                     "road_not_city_street")},
        "large_city_definition": ("City_ID in " + ", ".join(LARGE_CITIES)
                                  + " (random_parameters.LARGE_CITIES, the spatial split "
                                    "of transferability.spatial_test)"),
        "phi_with_large_city": corr(fr[COL], large),
        "crosstab_indicator_by_large_city": {
            f"{COL}={int(i)}": {f"large_city={int(j)}": int(xt.loc[i, j])
                                for j in xt.columns} for i in xt.index},
        "large_city_rows_outside_ua200k": int(((large == 1) & (fr[COL] == 0)).sum()),
        "correlation_all_rows": {
            c: corr(fr[COL], fr[c]) for c in ("poverty_share", "road_density_km_100m",
                                              "coord_missing", "road_not_city_street",
                                              "_rural")},
        "correlation_located_rows": {
            c: corr(fr[COL], fr[c], ~unloc) for c in ("poverty_share",
                                                      "road_density_km_100m")},
        "unlocated_rows": int(unloc.sum()),
        "unlocated_rows_city_level_fill": (
            "For the records without coordinates, poverty share and road density take the "
            "median of the city (model_frame.py) and the indicator takes the majority value "
            "of the city's located records or the CRIS rural flag, so on those records all "
            "three are functions of City_ID and move together by construction"),
        "correlation_unlocated_rows": {
            c: corr(fr[COL], fr[c], unloc) for c in ("poverty_share",
                                                     "road_density_km_100m")},
    }
    return {
        "screening_entry": entry,
        "event_rule": event,
        "vif_rule_pooled_screening_design": pooled,
        "disposition": entry["disposition"],
        "in_primary_specification": COL in m1_cols,
        "conceptual_role": ("Urbanized area is the conventional urban and rural context "
                            "covariate of police-record severity models and stands for "
                            "the speed environment, trauma care access and reporting "
                            "practice that differ between large urban areas and the rest; "
                            "its role is the one prespec section 2.1 assigns it"),
        "auxiliary_collinearity": aux,
    }


def _ka_by_seed(fit: sv.OrdinalFit, fr: pd.DataFrame, groups: dict,
                cols: tuple[str, ...]) -> dict:
    out: dict = {c: {} for c in cols}
    for seed in KR_SEEDS:
        ames = sv.average_marginal_effects(fit, fr, seed=seed, groups=groups)
        for c in cols:
            out[c][str(seed)] = {"ci95": ames[c]["KA"]["ci95"],
                                 "excludes_zero": ames[c]["KA"]["excludes_zero"]}
    return out


def reverse(fr: pd.DataFrame, primary: sv.OrdinalFit, cols: list[str],
            primary_ames: dict) -> tuple[dict, dict, dict]:
    without = [c for c in cols if c != COL]
    groups_p = sv.categorical_groups(fr, cols)
    groups_w = sv.categorical_groups(fr, without)
    fit = sv.fit_ordered_logit(fr, without, "M1_without_urban_area")
    null_ll = sv.null_loglikelihood(fr)
    ames = sv.average_marginal_effects(fit, fr, groups=groups_w)
    coef = sv.coefficient_table(primary)["slopes"][COL]

    prim_sig = dx.significant_sets({c: primary_ames[c] for c in without})
    new_sig = dx.significant_sets({c: ames[c] for c in without})
    # Changes are stated relative to M1: "lost" means an interval that excludes zero in
    # M1 and includes it once the indicator is removed, and "gained" the reverse.
    changes = {lvl: {"gained": sorted(set(new_sig[lvl]) - set(prim_sig[lvl])),
                     "lost": sorted(set(prim_sig[lvl]) - set(new_sig[lvl]))}
               for lvl in sv.SEV_ORDER}
    any_change = any(v["gained"] or v["lost"] for v in changes.values())
    shift = {c: abs(ames[c]["KA"]["ame"] - primary_ames[c]["KA"]["ame"]) for c in without}
    shift_col = max(shift, key=shift.get)
    sign_flips = sorted(c for c in without
                        if np.sign(ames[c]["KA"]["ame"]) != np.sign(primary_ames[c]["KA"]["ame"])
                        and (ames[c]["KA"]["excludes_zero"]
                             or primary_ames[c]["KA"]["excludes_zero"]))
    fit_m = sv.fit_metrics(fit, null_ll)
    prim_m = sv.fit_metrics(primary, null_ll)

    previous = None
    if PREVIOUS_PRIMARY.exists():
        prev = json.loads(PREVIOUS_PRIMARY.read_text(encoding="utf-8"))["rungs"]["M1"]
        prev_cols = prev["coefficients"]["columns"]
        previous = {
            "loglik_previous_primary": prev["fit"]["loglik"],
            "columns_match": bool(sorted(prev_cols) == sorted(without)),
            "loglik_matches": bool(abs(prev["fit"]["loglik"] - fit.loglik) < 1e-3),
        }

    with_ind = {
        "columns": list(cols),
        "n_covariates": len(cols),
        "fit": prim_m,
        "coefficient": coef,
        "odds_ratio": round(float(np.exp(coef["coef"])), 4),
        "ame": {k: primary_ames[COL][k] for k in sv.SEV_ORDER},
        "excludes_zero": dx.significant_sets({c: primary_ames[c] for c in cols}),
    }
    wo = {
        "columns": without,
        "n_covariates": len(without),
        "fit": fit_m,
        "converged": fit.converged,
        "separation": sv.separation_diagnostics(fit),
        "reproduces_previous_primary": previous,
        "bic_minus_primary": round(fit_m["bic"] - prim_m["bic"], 3),
        "aic_minus_primary": round(fit_m["aic"] - prim_m["aic"], 3),
        "lr_indicator": sv.nested_lr_test(fit, primary),
        "excludes_zero": dx.significant_sets(ames),
        "conclusion_changes_when_removed": changes,
        "any_other_conclusion_changes": bool(any_change),
        "max_abs_ka_ame_shift_other_covariates": round(float(shift[shift_col]), 5),
        "max_shift_covariate": shift_col,
        "sign_flips_among_significant": sign_flips,
        "ka_ame_requested": {c: {"without_indicator": ames[c]["KA"],
                                 "primary": primary_ames[c]["KA"]}
                             for c in KA_REQUESTED},
    }

    print("  Krinsky-Robb seed check, M1 and M1 without the indicator")
    seed_prim = _ka_by_seed(primary, fr, groups_p, KA_REQUESTED + (COL,))
    seed_wo = _ka_by_seed(fit, fr, groups_w, KA_REQUESTED)
    seed_cmp = {}
    for c in KA_REQUESTED:
        p = [seed_prim[c][str(s)]["excludes_zero"] for s in KR_SEEDS]
        q = [seed_wo[c][str(s)]["excludes_zero"] for s in KR_SEEDS]
        seed_cmp[c] = {"primary_excludes_zero_seeds": int(sum(p)),
                       "without_indicator_excludes_zero_seeds": int(sum(q)),
                       "stable_across_seeds_primary": bool(len(set(p)) == 1),
                       "stable_across_seeds_without_indicator": bool(len(set(q)) == 1)}
    seed_cmp[COL] = {"primary_excludes_zero_seeds": int(sum(
        seed_prim[COL][str(s)]["excludes_zero"] for s in KR_SEEDS))}
    unstable = sorted(c for c, v in seed_cmp.items() if c != COL
                      and not (v["stable_across_seeds_primary"]
                               and v["stable_across_seeds_without_indicator"]))
    # For a KA conclusion that changes, compare the bound nearest zero across the four
    # seeds in each model. Overlapping seed ranges put the change within the Monte Carlo
    # variation of the interval; separated ranges mean the indicator moves it.
    changed_ka = sorted(set(changes["KA"]["gained"] + changes["KA"]["lost"]))
    near_zero = {}
    for c in changed_ka:
        if c not in KA_REQUESTED:
            continue
        side = 1 if primary_ames[c]["KA"]["ame"] < 0 else 0   # upper bound if negative
        pr = [seed_prim[c][str(s)]["ci95"][side] for s in KR_SEEDS]
        rr = [seed_wo[c][str(s)]["ci95"][side] for s in KR_SEEDS]
        overlap = bool(max(min(pr), min(rr)) <= min(max(pr), max(rr)))
        near_zero[c] = {
            "bound": "upper" if side == 1 else "lower",
            "primary_range": [round(min(pr), 5), round(max(pr), 5)],
            "without_indicator_range": [round(min(rr), 5), round(max(rr), 5)],
            "seed_ranges_overlap": overlap,
            "seed_fragile": c in unstable,
        }
    kr = {
        "seeds": list(KR_SEEDS),
        "draws": sv.KR_DRAWS,
        "primary": seed_prim,
        "without_indicator": seed_wo,
        "comparison": seed_cmp,
        "covariates_unstable_across_seeds": unstable,
        "changed_ka_bound_nearest_zero_by_seed": near_zero,
        "ka_changes_within_seed_variation": sorted(
            c for c, v in near_zero.items() if v["seed_ranges_overlap"]),
        "ka_changes_beyond_seed_variation": sorted(
            c for c, v in near_zero.items() if not v["seed_ranges_overlap"]),
        "reading": {
            c: (f"KA interval excludes zero on {seed_cmp[c]['primary_excludes_zero_seeds']} "
                f"of {len(KR_SEEDS)} seeds in M1 and on "
                f"{seed_cmp[c]['without_indicator_excludes_zero_seeds']} of "
                f"{len(KR_SEEDS)} without the indicator. The {v['bound']} bound ranges "
                f"over {v['primary_range']} in M1 and {v['without_indicator_range']} "
                "without the indicator; "
                + ("the ranges overlap, so the change is within the Monte Carlo "
                   "variation of the interval" if v["seed_ranges_overlap"] else
                   "the ranges do not overlap, so the indicator moves the interval by "
                   "more than the seeds do"))
            for c, v in near_zero.items()},
        "note": ("The two models differ by one parameter, so the same seed gives different "
                 "Krinsky-Robb draws. A KA conclusion that changes between them is "
                 "compared on the bound nearest zero across the four seeds in each model; "
                 "overlapping seed ranges mean the change is Monte Carlo variation, "
                 "separated ranges mean the indicator moved it."),
    }
    return with_ind, wo, kr


def main() -> None:
    frame, base = dx.load_frames()
    primary, cols, check = dx.primary_fit(frame)
    print(f"primary M1: loglik {check['loglik_refit']} (matches {check['matches']}), "
          f"{len(cols)} covariates")
    if not check["matches"]:
        raise SystemExit("primary refit does not reproduce severity.json")
    if COL not in cols:
        raise SystemExit(f"{COL} is not in M1; rerun model_frame.py and severity.py")
    primary_ames = sv.average_marginal_effects(primary, frame)

    print("checking the urbanized-area indicator")
    fr, meta = check_indicator(frame, base)
    print(f"  inside {meta['counts_by_severity']['inside_ua_200k']}, "
          f"outside {meta['counts_by_severity']['outside']}")
    scr = screening(fr, cols)
    print(f"  disposition: {scr['disposition']}, VIF "
          f"{scr['vif_rule_pooled_screening_design']['vif_indicator']}")
    print("refitting M1 without the indicator")
    with_ind, wo, kr = reverse(fr, primary, cols, primary_ames)
    print(f"  coef in M1 {with_ind['coefficient']['coef']} "
          f"(se {with_ind['coefficient']['se']}), LR {wo['lr_indicator']}, "
          f"BIC without minus M1 {wo['bic_minus_primary']}")
    print(f"  reproduces the previous primary: {wo['reproduces_previous_primary']}")
    print(f"  conclusion changes when removed {wo['conclusion_changes_when_removed']}")

    out = {
        "generated": _now(),
        "script": "src/urbanized_sensitivity.py",
        "inputs": INPUTS,
        "settings": {"kr_draws": sv.KR_DRAWS, "kr_seed": sv.KR_SEED,
                     "kr_seed_check": list(KR_SEEDS),
                     "ua_population_threshold": dx.UA_POP_THRESHOLD,
                     "min_events_rule": mf.MIN_EVENTS, "max_vif_rule": mf.MAX_VIF},
        "direction": ("reverse sensitivity: the indicator is a screened covariate of the "
                      "primary M1, and the comparison model is M1 without it"),
        "primary_check": check,
        "prespec": ("section 2.1 lists Urbanized as a roadway and environment candidate; "
                    "section 3 gives the inclusion rules, applied in model_frame.screen"),
        "indicator": meta,
        "screening": scr,
        "primary_with_indicator": with_ind,
        "without_indicator": wo,
        "kr_seed_check": kr,
    }
    OUT.write_text(json.dumps(out, indent=2, default=float), encoding="utf-8")
    print(f"written to {OUT}")


if __name__ == "__main__":
    main()
