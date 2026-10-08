"""Build the base frame from the raw CRIS extract.

Reads `data/E_Scooter_IDYes2017_2025a.xlsx` and writes `data/base_frame.csv`:
one row per modeled crash, carrying the identifiers, the severity outcome, the
narrative, and every raw field the later steps need.

Every count this script prints is written to `data/base_frame_audit.json` and is
the source for the corresponding numbers in the manuscript.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "E_Scooter_IDYes2017_2025a.xlsx"
OUT_DIR = ROOT / "data"

SEVERITY_5 = {
    "Not Injured": "O",
    "Possible Injury": "C",
    "Non-Incapacitating": "B",
    "Incapacitating Injury": "A",
    "Killed": "K",
}
SEVERITY_3 = {"O": "O", "C": "BC", "B": "BC", "A": "KA", "K": "KA"}
SEVERITY_3_ORDER = ["O", "BC", "KA"]
SEVERITY_4_ORDER = ["O", "C", "B", "KA"]


def _parse_hour(val) -> float:
    if pd.isna(val):
        return np.nan
    if hasattr(val, "hour"):
        return float(val.hour)
    s = str(val).strip()
    for fmt in ("%I:%M %p", "%H:%M:%S", "%H:%M"):
        try:
            return float(pd.to_datetime(s, format=fmt).hour)
        except ValueError:
            continue
    try:
        return float(pd.to_datetime(s).hour)
    except Exception:
        return np.nan


def build() -> pd.DataFrame:
    audit: dict[str, object] = {}

    df = pd.read_excel(RAW)
    audit["raw_rows"] = int(len(df))
    audit["raw_cols"] = int(df.shape[1])

    # --- identifier integrity -------------------------------------------------
    # The extract is one row per scooter rider, not one row per crash. Two crashes
    # involve two scooter units each, so 527 rows correspond to 525 crashes. The
    # unit of analysis is the rider; the duplicated crashes are recorded here and a
    # one-rider-per-crash sensitivity is reported.
    audit["unique_crash_ids_raw"] = int(df["Crash_ID"].nunique())
    dup_ids = sorted(df.loc[df["Crash_ID"].duplicated(keep=False), "Crash_ID"].unique())
    audit["crashes_with_multiple_scooter_riders"] = [int(x) for x in dup_ids]
    audit["n_crashes_with_multiple_scooter_riders"] = int(len(dup_ids))
    df["rider_row_id"] = df["Unit_ID"].astype(str)
    df["crash_rider_seq"] = df.groupby("Crash_ID").cumcount() + 1
    df["is_first_rider_in_crash"] = (df["crash_rider_seq"] == 1).astype(int)

    # --- outcome --------------------------------------------------------------
    audit["crash_sev_raw_counts"] = (
        df["Crash_Sev_ID"].value_counts(dropna=False).to_dict()
    )
    unknown_mask = ~df["Crash_Sev_ID"].isin(SEVERITY_5)
    audit["dropped_unknown_severity"] = int(unknown_mask.sum())
    audit["dropped_unknown_severity_values"] = sorted(
        str(v) for v in df.loc[unknown_mask, "Crash_Sev_ID"].unique()
    )
    df = df.loc[~unknown_mask].copy()
    audit["modeled_rows"] = int(len(df))
    audit["modeled_unique_crashes"] = int(df["Crash_ID"].nunique())

    df["sev_kabco"] = df["Crash_Sev_ID"].map(SEVERITY_5)
    df["sev3"] = df["sev_kabco"].map(SEVERITY_3)
    df["sev4"] = np.where(df["sev_kabco"].isin(["A", "K"]), "KA", df["sev_kabco"])
    audit["sev_kabco_counts"] = df["sev_kabco"].value_counts().to_dict()
    audit["sev3_counts"] = df["sev3"].value_counts().reindex(SEVERITY_3_ORDER).to_dict()
    audit["sev4_counts"] = df["sev4"].value_counts().reindex(SEVERITY_4_ORDER).to_dict()

    # --- scope: second unit is a motor vehicle --------------------------------
    audit["unit_desc_2_counts"] = (
        df["Unit_Desc_ID_2"].value_counts(dropna=False).to_dict()
    )
    audit["second_unit_motor_vehicle"] = int(
        (df["Unit_Desc_ID_2"].astype(str).str.strip() == "Motor Vehicle").sum()
    )
    audit["unit_desc_1_counts"] = (
        df["Unit_Desc_ID"].value_counts(dropna=False).to_dict()
    )

    # Agreement between the crash-level KABCO code and the rider's own coded
    # injury severity, reported in the manuscript as a definitional check.
    rider_sev = df["Prsn_Injry_Sev_ID"].astype(str).str.strip()
    crash_sev = df["Crash_Sev_ID"].astype(str).str.strip()
    align = {"Non-Incapacitating": "Non-Incapacitating Injury"}
    crash_sev_aligned = crash_sev.replace(align)
    comparable = rider_sev.isin(SEVERITY_5.keys() | {"Non-Incapacitating Injury"})
    audit["rider_severity_comparable_n"] = int(comparable.sum())
    audit["rider_severity_equals_crash_severity"] = int(
        (crash_sev_aligned[comparable] == rider_sev[comparable]).sum()
    )
    audit["rider_severity_missing_or_unknown"] = int((~comparable).sum())

    # --- narrative ------------------------------------------------------------
    narr = df["Investigator_Narrative"].astype("string")
    df["narrative"] = narr.fillna("").str.strip()
    df["narrative_chars"] = df["narrative"].str.len()
    audit["narrative_missing_or_empty"] = int((df["narrative_chars"] == 0).sum())
    audit["narrative_chars_median"] = float(
        df.loc[df["narrative_chars"] > 0, "narrative_chars"].median()
    )
    audit["narrative_chars_q1"] = float(
        df.loc[df["narrative_chars"] > 0, "narrative_chars"].quantile(0.25)
    )
    audit["narrative_chars_q3"] = float(
        df.loc[df["narrative_chars"] > 0, "narrative_chars"].quantile(0.75)
    )

    # --- temporal -------------------------------------------------------------
    dt = pd.to_datetime(df["Crash_Date"], errors="coerce")
    df["crash_year"] = dt.dt.year
    df["crash_month"] = dt.dt.month
    df["crash_hour"] = df["Crash_Time"].apply(_parse_hour)
    df["is_weekend"] = (
        df["Day_of_Week"].astype(str).str.upper().str.strip().isin(["SAT", "SUN"]).astype(int)
    )
    # Nighttime for the audit's reporting-context vector: 22:00 to 05:59.
    df["nighttime"] = np.where(
        df["crash_hour"].isna(),
        np.nan,
        ((df["crash_hour"] >= 22) | (df["crash_hour"] < 6)).astype(float),
    )
    audit["crash_year_counts"] = df["crash_year"].value_counts().sort_index().to_dict()
    audit["crash_hour_missing"] = int(df["crash_hour"].isna().sum())

    # --- coordinates ----------------------------------------------------------
    lat = pd.to_numeric(df["Latitude"], errors="coerce")
    lon = pd.to_numeric(df["Longitude"], errors="coerce")
    valid = lat.between(25.0, 37.0) & lon.between(-107.0, -93.0)
    df["latitude"] = lat.where(valid)
    df["longitude"] = lon.where(valid)
    df["coord_missing"] = (~valid).astype(int)
    audit["coord_missing"] = int(df["coord_missing"].sum())
    audit["coord_present"] = int((~df["coord_missing"].astype(bool)).sum())

    # --- rider person fields --------------------------------------------------
    age = pd.to_numeric(df["Prsn_Age"], errors="coerce")
    df["rider_age"] = age.where(age.between(3, 100))
    audit["age_missing"] = int(df["rider_age"].isna().sum())
    audit["gender_counts"] = df["Prsn_Gndr_ID"].value_counts(dropna=False).to_dict()
    audit["ethnicity_counts"] = df["Prsn_Ethnicity_ID"].value_counts(dropna=False).to_dict()

    # --- fields carried forward unchanged for the model frame ------------------------------
    carry = [
        "Crash_ID", "Crash_Date", "Crash_Time", "Day_of_Week", "Year",
        "Crash_Sev_ID", "Prsn_Injry_Sev_ID", "Tot_Injry_Cnt", "Death_Cnt",
        "Sus_Serious_Injry_Cnt",
        "Crash_Speed_Limit", "Wthr_Cond_ID", "Light_Cond_ID", "Surf_Cond_ID",
        "Traffic_Cntl_ID", "Intrsct_Relat_ID", "FHE_Collsn_ID", "Obj_Struck_ID",
        "Othr_Factr_ID", "Road_Type_ID", "Road_Cls_ID", "Road_Relat_ID",
        "Road_Part_Adj_ID", "Func_Sys_ID", "Onsys_Fl", "Rural_Fl",
        "Rural_Urban_Type_ID", "Pop_Group_ID", "Cnty_ID", "City_ID",
        "Adt_Curnt_Amt", "Adt_Curnt_Year",
        "Unit_Desc_ID", "Veh_Body_Styl_ID", "Contrib_Factr_1_ID", "Contrib_Factr_2_ID",
        "Prsn_Age", "Prsn_Ethnicity_ID", "Prsn_Gndr_ID", "Prsn_Helmet_ID",
        "Prsn_Alc_Rslt_ID", "Prsn_Drg_Rslt_ID", "Prsn_Type_ID",
        "Unit_Desc_ID_2", "Veh_Body_Styl_ID_2", "Veh_Mod_Year_2",
        "Contrib_Factr_1_ID_2", "Contrib_Factr_2_ID_2",
        "Prsn_Age_2", "Prsn_Gndr_ID_2", "Prsn_Injry_Sev_ID_2",
        "Unit_ID",
    ]
    derived = [
        "sev_kabco", "sev3", "sev4", "narrative", "narrative_chars",
        "crash_year", "crash_month", "crash_hour", "is_weekend", "nighttime",
        "latitude", "longitude", "coord_missing", "rider_age",
        "rider_row_id", "crash_rider_seq", "is_first_rider_in_crash",
    ]
    missing_cols = [c for c in carry if c not in df.columns]
    if missing_cols:
        raise KeyError(f"Expected columns absent from the extract: {missing_cols}")

    out = df[carry + derived].copy()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_DIR / "base_frame.csv", index=False)

    audit["output_rows"] = int(len(out))
    audit["output_path"] = str(OUT_DIR / "base_frame.csv")
    with open(OUT_DIR / "base_frame_audit.json", "w", encoding="utf-8") as fh:
        json.dump(audit, fh, indent=2, default=str)

    return out, audit


if __name__ == "__main__":
    frame, audit = build()
    for k in (
        "raw_rows", "dropped_unknown_severity", "dropped_unknown_severity_values",
        "modeled_rows", "modeled_unique_crashes",
        "n_crashes_with_multiple_scooter_riders",
        "crashes_with_multiple_scooter_riders",
        "rider_severity_comparable_n", "rider_severity_equals_crash_severity",
        "sev_kabco_counts", "sev3_counts", "sev4_counts",
        "second_unit_motor_vehicle", "unit_desc_2_counts",
        "narrative_missing_or_empty", "narrative_chars_median",
        "coord_missing", "crash_year_counts",
    ):
        print(f"{k}: {audit[k]}")
