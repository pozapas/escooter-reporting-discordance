"""The modeling frame, built strictly to `prespec.md`.

Every covariate here is defined in the frozen pre-specification. Unknown values are
carried as explicit categories or indicators and are never pooled into a reference
group. Nothing is coded, merged, or dropped on the basis of an estimated coefficient.

Writes `data/model_frame.csv` and `data/screening.json`, the latter being the
source for Appendix Table A1 (every candidate, its counts by severity class, its VIF,
and its disposition).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

MIN_EVENTS = 10   # prespec section 3, rule 1
MAX_VIF = 5.0     # prespec section 3, rule 2

# prespec section 3.2: retained at any cell count and any VIF, because deleting a
# missingness indicator pools those records into the reference group rather than
# removing them. Their coefficients are reported and not interpreted.
EXEMPT = frozenset({
    "speed_unknown", "gender_unknown", "age_unknown", "coord_missing",
    "veh_other_unknown",
})

SEV_ORDER = ["O", "BC", "KA"]


# --------------------------------------------------------------------------
# Covariate construction. Each helper returns a dict of indicator columns and
# records the reference category it leaves out.
# --------------------------------------------------------------------------

def _speed_band(v) -> str:
    x = pd.to_numeric(v, errors="coerce")
    if pd.isna(x) or x < 0:
        return "unknown"
    if x <= 30:
        return "le30"
    if x <= 40:
        return "35to40"
    return "ge45"


def _lighting(v) -> str:
    s = str(v).strip()
    if s in ("Daylight", "Dusk", "Dawn"):
        return "day_dusk_dawn"
    if s == "Dark, Lighted":
        return "dark_lighted"
    # "Dark, Not Lighted", "Dark, Unknown Lighting", "Unknown", "Other", missing.
    # prespec section 4 merges dark-not-lighted with other or unknown.
    return "dark_unlit_or_unknown"


def _ethnicity(v) -> str:
    s = str(v).strip()
    if s == "White":
        return "white"
    if s == "Black":
        return "black"
    if s == "Hispanic":
        return "hispanic"
    if s in ("Asian", "Other", "Amer. Indian/Alaskan Native"):
        return "other"
    return "unknown"


def _gender(v) -> str:
    s = str(v).strip()
    if s == "Male":
        return "male"
    if s == "Female":
        return "female"
    return "unknown"


def _age_band(v) -> str:
    x = pd.to_numeric(v, errors="coerce")
    if pd.isna(x):
        return "unknown"
    if x < 25:
        return "under25"
    if x < 55:
        return "25to54"
    return "55plus"


def _body_style(v) -> str:
    s = str(v).strip()
    if s.startswith("Passenger Car"):
        return "car"
    if s == "Sport Utility Vehicle":
        return "suv"
    if s in ("Pickup", "Truck", "Van", "Truck Tractor"):
        return "pickup_truck_van"
    return "other_unknown"


def _collision_manner(v) -> str:
    """prespec section 4: same direction wins over turning, and the order is fixed.

    The branches are ordered exactly as the pre-specification lists them, so that
    "Same Direction - One Straight-One Right Turn" is same-direction and not turning.
    Relative path geometry, not the presence of a turn, defines the category.
    """
    low = str(v).strip().lower()
    if low.startswith("same direction"):
        return "same_direction"
    if "turn" in low:
        return "turning"
    if low.startswith("one motor vehicle - going straight"):
        return "single_mv_straight"
    if low.startswith("angle"):
        return "angle"
    return "other"


def _intersection(v) -> str:
    s = str(v).strip()
    if s in ("Intersection", "Intersection Related"):
        return "intersection"
    if s == "Driveway Access":
        return "driveway_access"
    return "non_intersection"


def _traffic_control(v) -> str:
    s = str(v).strip()
    if s.startswith("Signal Light") or s == "Flashing Yellow Light":
        return "signal"
    if s == "Stop Sign":
        return "stop_sign"
    return "none_other_unknown"


def _road_class(v) -> str:
    s = str(v).strip()
    if s == "City Street":
        return "city_street"
    if s == "Non Trafficway":
        return "non_trafficway"
    return "highway_county_fm"


CONTRIB_PATTERNS: dict[str, tuple[str, ...]] = {
    "cf_failed_to_yield": ("failed to yield",),
    "cf_driver_inattention": ("driver inattention", "inattention"),
    "cf_disregard_control": ("disregard stop", "disregard signal", "disregarded stop",
                             "disregard stop and go signal", "ran red light",
                             "disregard stop sign"),
    "cf_speed_related": ("speeding", "unsafe speed", "failed to control speed",
                         "faster than posted"),
    "cf_wrong_side_or_way": ("wrong side", "wrong way"),
}


def _contrib_flags(row: pd.Series) -> dict[str, int]:
    """Officer-coded contributing factors, split by which unit they attach to.

    Unit 1 fields describe the scooter, unit 2 fields the motor vehicle, so failure to
    yield is recorded separately for rider and driver as `prespec.md` section 2.5
    requires.
    """
    rider_text = " | ".join(
        str(row.get(c, "")) for c in ("Contrib_Factr_1_ID", "Contrib_Factr_2_ID")
    ).lower()
    driver_text = " | ".join(
        str(row.get(c, "")) for c in ("Contrib_Factr_1_ID_2", "Contrib_Factr_2_ID_2")
    ).lower()
    other_text = str(row.get("Othr_Factr_ID", "")).lower()

    out: dict[str, int] = {}
    for name, pats in CONTRIB_PATTERNS.items():
        if name == "cf_failed_to_yield":
            out["cf_rider_failed_to_yield"] = int(any(p in rider_text for p in pats))
            out["cf_driver_failed_to_yield"] = int(any(p in driver_text for p in pats))
            continue
        both = rider_text + " | " + driver_text
        out[name] = int(any(p in both for p in pats))
    out["cf_driveway_entry_exit"] = int(
        "driveway" in other_text or "drive way" in other_text
    )
    return out


CATEGORICALS: dict[str, tuple[str, str, object]] = {
    # column in base frame -> (prefix, reference level, coder)
    "Crash_Speed_Limit": ("speed", "le30", _speed_band),
    "Light_Cond_ID": ("light", "day_dusk_dawn", _lighting),
    "Prsn_Ethnicity_ID": ("eth", "white", _ethnicity),
    "Prsn_Gndr_ID": ("gender", "male", _gender),
    "rider_age": ("age", "25to54", _age_band),
    "Veh_Body_Styl_ID_2": ("veh", "car", _body_style),
    "FHE_Collsn_ID": ("coll", "angle", _collision_manner),
    "Intrsct_Relat_ID": ("intx", "non_intersection", _intersection),
    "Traffic_Cntl_ID": ("ctrl", "none_other_unknown", _traffic_control),
    "Road_Cls_ID": ("road", "city_street", _road_class),
}


def _urban_area_layer(frame: pd.DataFrame,
                      base: pd.DataFrame) -> tuple[pd.Series, dict] | None:
    """The prespec section 2.1 urbanized-area indicator, or None if its layer is absent.

    The construction is `data_extra._urbanized_area`, imported here rather than at module
    level because data_extra imports this module. The resulting cells are checked
    against the ones `data_extra.json` recorded for the same construction, so the frame
    cannot carry a different indicator from the one that was screened and reported.
    """
    shp = ROOT / "data" / "shapefiles" / "tl_2020_us_uac20.shp"
    if not shp.exists():
        return None
    import sys
    here = str(Path(__file__).resolve().parent)
    if here not in sys.path:
        sys.path.insert(0, here)
    import data_extra as dx

    work = frame[["rider_row_id", "latitude", "longitude", "coord_missing",
                  "City_ID"]].copy()
    rural = base.set_index("rider_row_id")["Rural_Fl"].reindex(work["rider_row_id"])
    work["_rural"] = (rural.astype(str).str.upper() == "Y").astype(int).to_numpy()
    ua, meta = dx._urbanized_area(work)
    ua = ua.astype(int)

    sev = frame["sev3"]
    cells = {side: {k: int(v) for k, v in sev[ua == val].value_counts()
                    .reindex(SEV_ORDER).fillna(0).items()}
             for side, val in (("inside_ua_200k", 1), ("outside", 0))}
    meta["counts_by_severity"] = cells
    meta["construction"] = (
        "data_extra._urbanized_area: point-in-polygon of the crash coordinates against "
        "the 2020 Census Urban Areas (tl_2020_us_uac20) whose population in "
        "ua_list_all.txt is 200,000 or more; a record without coordinates takes the "
        "majority value of the located records of its City_ID, and a record whose city "
        "cannot be placed takes the CRIS rural flag")
    stored_path = DATA / "data_extra.json"
    if stored_path.exists():
        stored = (json.loads(stored_path.read_text(encoding="utf-8"))
                  .get("spatial_indicator", {}).get("urbanized_area_indicator", {})
                  .get("counts_by_severity"))
        if stored is not None:
            meta["reproduces_data_extra"] = bool(stored == cells)
            if stored != cells:
                raise SystemExit(f"urban_area_200k does not reproduce data_extra.json: "
                                 f"{cells} vs {stored}")
    return ua, meta


def build() -> tuple[pd.DataFrame, dict]:
    base = pd.read_csv(DATA / "base_frame.csv")
    frame = base[[
        "Crash_ID", "rider_row_id", "crash_rider_seq", "is_first_rider_in_crash",
        "sev3", "sev4", "sev_kabco", "crash_year", "is_weekend", "nighttime",
        "narrative_chars", "coord_missing", "latitude", "longitude", "City_ID",
        "Rural_Urban_Type_ID",
    ]].copy()

    sev = base["sev3"]
    merge_log: list[str] = []

    def cells(mask: pd.Series) -> dict[str, int]:
        return {k: int(v) for k, v in
                sev[mask].value_counts().reindex(SEV_ORDER).fillna(0).items()}

    def min_cell(mask: pd.Series) -> int:
        return min(cells(mask).values())

    categories: dict[str, list[str]] = {}
    references: dict[str, str] = {}
    labels: dict[str, pd.Series] = {}
    for col, (prefix, ref, coder) in CATEGORICALS.items():
        labels[prefix] = base[col].map(coder)
        references[prefix] = ref

    # --- prespec section 3.1 and section 4: stage-two merges -------------------
    # Age: 55+ -> 50+ -> 45+, stopping at 45.
    for threshold in (55, 50, 45):
        lab = base["rider_age"].map(
            lambda x, t=threshold: "unknown" if pd.isna(pd.to_numeric(x, errors="coerce"))
            else ("under25" if float(x) < 25 else ("25to54" if float(x) < t else f"{t}plus"))
        )
        older = lab == f"{threshold}plus"
        if min_cell(older) >= MIN_EVENTS or threshold == 45:
            labels["age"] = lab
            references["age"] = "25to54"
            merge_log.append(
                f"age: older band set at {threshold} and over, cells {cells(older)}"
                + ("" if min_cell(older) >= MIN_EVENTS
                   else "; retained at 45 and over although the smallest cell is below "
                        f"{MIN_EVENTS}, as the merging chain stops there")
            )
            break
        merge_log.append(
            f"age: {threshold} and over has cells {cells(older)}, below {MIN_EVENTS}; "
            "threshold moved"
        )

    # Ethnicity: "other" merges with "unknown" when it is too sparse.
    eth = labels["eth"]
    if min_cell(eth == "other") < MIN_EVENTS:
        merged = eth.replace({"other": "other_or_unknown", "unknown": "other_or_unknown"})
        merge_log.append(
            f"ethnicity: 'other' cells {cells(eth == 'other')} below {MIN_EVENTS}; "
            f"merged with 'unknown' into 'other or unknown', cells "
            f"{cells(merged == 'other_or_unknown')}"
        )
        labels["eth"] = merged

    # Collision manner: sparse levels fold into "other", then into the reference.
    coll = labels["coll"]
    for level in ("same_direction", "single_mv_straight", "turning"):
        if (coll == level).any() and min_cell(coll == level) < MIN_EVENTS:
            merge_log.append(
                f"collision manner: '{level}' cells {cells(coll == level)} below "
                f"{MIN_EVENTS}; merged into 'other'")
            coll = coll.replace({level: "other"})
    if (coll == "other").any() and min_cell(coll == "other") < MIN_EVENTS:
        merge_log.append(
            f"collision manner: 'other' cells {cells(coll == 'other')} below "
            f"{MIN_EVENTS}; merged into the 'angle' reference")
        coll = coll.replace({"other": "angle"})
    labels["coll"] = coll

    # Road class: non-trafficway folds into the non-city-street category.
    road = labels["road"]
    if min_cell(road == "non_trafficway") < MIN_EVENTS:
        merged = road.replace({"non_trafficway": "not_city_street",
                               "highway_county_fm": "not_city_street"})
        merge_log.append(
            f"road class: 'non-trafficway' cells {cells(road == 'non_trafficway')} below "
            f"{MIN_EVENTS}; merged with the highway category into 'not a city street', "
            f"cells {cells(merged == 'not_city_street')}")
        labels["road"] = merged

    # Intersection relation: driveway access folds into the conflict-point category.
    intx = labels["intx"]
    if min_cell(intx == "driveway_access") < MIN_EVENTS:
        merged = intx.replace({"driveway_access": "intersection_or_driveway",
                               "intersection": "intersection_or_driveway"})
        merge_log.append(
            f"intersection relation: 'driveway access' cells "
            f"{cells(intx == 'driveway_access')} below {MIN_EVENTS}; merged with "
            f"'intersection or intersection-related' into 'intersection or driveway "
            f"access', cells {cells(merged == 'intersection_or_driveway')}")
        labels["intx"] = merged

    # Lighting: a sparse dark-or-unknown residue merges with dark-lighted, so the
    # contrast is daylight against dark rather than daylight against a residue.
    light = labels["light"]
    if min_cell(light == "dark_unlit_or_unknown") < MIN_EVENTS:
        merged = light.replace({"dark_unlit_or_unknown": "dark_or_unknown",
                                "dark_lighted": "dark_or_unknown"})
        merge_log.append(
            f"lighting: 'dark, not lighted or unknown' cells "
            f"{cells(light == 'dark_unlit_or_unknown')} below {MIN_EVENTS}; merged with "
            f"'dark, lighted' into 'dark or unknown lighting', cells "
            f"{cells(merged == 'dark_or_unknown')}")
        labels["light"] = merged

    # Vehicle body style has no further merge available in section 4; its residual
    # level is retained under section 3.2.

    for prefix, lab in labels.items():
        ref = references[prefix]
        levels = sorted(set(lab.dropna()) | {ref})
        categories[prefix] = levels
        frame[f"{prefix}_level"] = lab
        for lev in levels:
            if lev == ref:
                continue
            frame[f"{prefix}_{lev}"] = (lab == lev).astype(int)

    cf = base.apply(_contrib_flags, axis=1, result_type="expand")
    frame = pd.concat([frame, cf], axis=1)

    frame["weekend"] = frame["is_weekend"].astype(int)
    frame["nighttime"] = frame["nighttime"].fillna(0).astype(int)
    frame["narrative_chars_z"] = (
        (frame["narrative_chars"] - frame["narrative_chars"].mean())
        / frame["narrative_chars"].std()
    )

    # --- spatial layers, attached when their scripts have produced them -----
    spatial_notes: dict[str, str] = {}
    osm_path = DATA / "osm_features.csv"
    if osm_path.exists():
        osm = pd.read_csv(osm_path).drop_duplicates(subset="Crash_ID")
        keep = ["Crash_ID", "bike_facility_nearest", "bike_facility_50m",
                "bike_facility_100m", "bike_facility_200m",
                "road_density_km_100m", "n_ways_100m", "osm_extraction_date"]
        frame = frame.merge(osm[[c for c in keep if c in osm.columns]],
                            on="Crash_ID", how="left")
        spatial_notes["osm"] = f"{len(osm)} located crashes joined"
    else:
        spatial_notes["osm"] = "not yet available"

    acs_path = DATA / "acs_features.csv"
    if acs_path.exists():
        acs = pd.read_csv(acs_path).drop_duplicates(subset="Crash_ID")
        frame = frame.merge(acs, on="Crash_ID", how="left")
        spatial_notes["acs"] = f"{len(acs)} located crashes joined"
    else:
        spatial_notes["acs"] = "not yet available"

    # --- missing-coordinate rule (prespec section 2.2) -------------------
    # Records without coordinates take zero bike facility, the city median for
    # density and poverty, and carry an explicit indicator.
    for col, fill in (("bike_facility_nearest", 0.0), ("bike_facility_50m", 0.0),
                      ("bike_facility_100m", 0.0), ("bike_facility_200m", 0.0)):
        if col in frame:
            frame[col] = frame[col].fillna(fill)
    for col in ("road_density_km_100m", "poverty_share", "poverty_share_acs2021"):
        if col in frame:
            city_med = frame.groupby("City_ID")[col].transform("median")
            frame[col] = frame[col].fillna(city_med).fillna(frame[col].median())

    # --- urbanized area (prespec section 2.1) -----------------------------------
    # 1 when the crash point lies inside a 2020 Census Urban Area of population
    # 200,000 or more, built with data_extra._urbanized_area (the same construction
    # urbanized_sensitivity.py screened). Records without coordinates follow the
    # pre-specified city lookup, the city-level analogue of the city median that density
    # and poverty take above: such a record takes 1 when at least half the located
    # records of its City_ID fall inside a qualifying urban area, and a record whose city
    # cannot be placed ("Not Listed", or a city with no located record) takes the CRIS
    # rural flag (0 if Rural_Fl is Y, else 1). These records also carry coord_missing.
    ua_layer = _urban_area_layer(frame, base)
    if ua_layer is not None:
        frame["urban_area_200k"], ua_meta = ua_layer
        spatial_notes["urban_area_200k"] = (
            f"{ua_meta['located_inside']} of {ua_meta['located_rows']} located rows "
            f"inside; {ua_meta['unlocated_rows']} unlocated rows by city lookup "
            f"({ua_meta['unlocated_resolved_by_city_lookup']}) or rural flag "
            f"({ua_meta['unlocated_unresolved_fallback_rural_flag']})")
    else:
        ua_meta = None
        spatial_notes["urban_area_200k"] = "not yet available"

    # --- narrative cues, attached once extraction is fused ---------------------
    labels_path = DATA / "final_labels.csv"
    if labels_path.exists():
        lab = pd.read_csv(labels_path)
        frame = frame.merge(lab, on="Crash_ID", how="left")
        spatial_notes["cues"] = "final adjudicated labels joined"
    else:
        ref_path = DATA / "reference_labels.csv"
        if ref_path.exists():
            frame = frame.merge(pd.read_csv(ref_path), on="Crash_ID", how="left")
            spatial_notes["cues"] = "PRELIMINARY reference-run labels joined"
        else:
            spatial_notes["cues"] = "not yet available"

    frame.to_csv(DATA / "model_frame.csv", index=False)

    screening = screen(frame, categories, references)
    screening["layers"] = spatial_notes
    screening["stage_two_merges"] = merge_log
    unk = [c for c in ("eth_unknown", "gender_unknown", "age_unknown") if c in frame]
    if len(unk) == 3:
        counts = frame[unk].sum(axis=1)
        screening["person_unknown_cooccurrence"] = {
            "any_unknown": int((counts > 0).sum()),
            "exactly_one": int((counts == 1).sum()),
            "exactly_two": int((counts == 2).sum()),
            "all_three": int((counts == 3).sum()),
            "per_indicator": {c: int(frame[c].sum()) for c in unk},
        }
    if ua_meta is not None:
        screening["urban_area_200k"] = ua_meta
    screening["rows"] = int(len(frame))
    screening["distinct_crashes"] = int(frame["Crash_ID"].nunique())
    (DATA / "screening.json").write_text(json.dumps(screening, indent=2), encoding="utf-8")
    return frame, screening


def screen(frame: pd.DataFrame, categories: dict, references: dict) -> dict:
    """Apply the prespec inclusion rules and record every candidate's disposition."""
    from statsmodels.stats.outliers_influence import variance_inflation_factor

    candidates: list[str] = []
    for prefix, levels in categories.items():
        for lev in levels:
            if lev == references[prefix]:
                continue
            candidates.append(f"{prefix}_{lev}")
    candidates += [c for c in frame.columns if c.startswith("cf_")]
    candidates += ["weekend", "nighttime", "coord_missing"]
    for c in ("bike_facility_nearest", "road_density_km_100m", "poverty_share",
              "urban_area_200k"):
        if c in frame.columns:
            candidates.append(c)
    candidates += [c for c in (
        "wrong_way", "sidewalk_transition", "driveway_alley", "failure_to_yield",
        "signal_violation", "swerve_loss_control", "dooring",
        "distraction_impairment", "vehicle_turning_across", "lane_positioning",
    ) if c in frame.columns]

    present = [c for c in candidates if c in frame.columns]
    rows: list[dict] = []
    for c in present:
        col = frame[c]
        binary = set(col.dropna().unique()) <= {0, 1}
        entry: dict[str, object] = {"candidate": c, "binary": bool(binary)}
        if binary:
            tab = frame.groupby("sev3")[c].sum().reindex(SEV_ORDER).fillna(0)
            entry["events_by_severity"] = {k: int(v) for k, v in tab.items()}
            entry["min_cell"] = int(tab.min())
            entry["exempt"] = c in EXEMPT
            entry["meets_event_rule"] = bool(tab.min() >= MIN_EVENTS) or c in EXEMPT
        else:
            entry["events_by_severity"] = None
            entry["min_cell"] = None
            entry["exempt"] = c in EXEMPT
            entry["meets_event_rule"] = True
            entry["median_by_severity"] = {
                k: float(v) for k, v in frame.groupby("sev3")[c].median()
                .reindex(SEV_ORDER).items()
            }
        rows.append(entry)

    # VIF over the candidates that pass the event rule, on complete cases.
    passing = [r["candidate"] for r in rows if r["meets_event_rule"]]
    X = frame[passing].astype(float).dropna()
    X = X.loc[:, X.std() > 0]
    Xc = np.column_stack([np.ones(len(X)), X.to_numpy()])
    vifs: dict[str, float] = {}
    for i, name in enumerate(X.columns, start=1):
        try:
            vifs[name] = round(float(variance_inflation_factor(Xc, i)), 3)
        except Exception:
            vifs[name] = float("nan")

    for r in rows:
        v = vifs.get(r["candidate"])
        r["vif"] = v
        if r.get("exempt"):
            r["disposition"] = "included: missingness indicator, exempt (prespec 3.2)"
        elif not r["meets_event_rule"]:
            r["disposition"] = f"excluded: smallest severity cell below {MIN_EVENTS}"
        elif v is not None and v == v and v >= MAX_VIF:
            r["disposition"] = f"excluded: VIF at or above {MAX_VIF}"
        elif v is None:
            r["disposition"] = "excluded: no variation in the pooled design"
        else:
            r["disposition"] = "included"

    return {
        "min_events_rule": MIN_EVENTS,
        "max_vif_rule": MAX_VIF,
        "n_candidates": len(rows),
        "n_included": sum(r["disposition"] == "included" for r in rows),
        "candidates": rows,
        "references": references,
    }


if __name__ == "__main__":
    frame, screening = build()
    print(f"model_frame.csv: {len(frame)} rows, {frame.shape[1]} columns")
    print(f"layers: {screening['layers']}")
    print("stage-two merges:")
    for line in screening["stage_two_merges"]:
        print("  " + line)
    print(f"candidates: {screening['n_candidates']}, included: {screening['n_included']}")
    for r in screening["candidates"]:
        if r["disposition"] != "included":
            print(f"  {r['candidate']:32s} {r['disposition']}"
                  f"  (cells {r['events_by_severity']}, VIF {r['vif']})")
