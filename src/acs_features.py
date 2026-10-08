"""Area-level poverty context from the American Community Survey.

Each located crash is assigned to a 2023 TIGER/Line block group by point-in-polygon,
and that block group's poverty share is attached from the ACS 5-year estimates.

Poverty share = individuals below the poverty level, over individuals for whom
poverty status is determined (B17021_002E / B17021_001E). Verified on 2026-09-14 by
querying the API that table B17021 is published at block-group geography for the
2019 to 2023 vintage, so a fallback to table C17002 is not
needed; the check is recorded in the audit file.

The variable is an **area-level contextual measure**. It is not the rider's poverty
status and the manuscript says so wherever it appears.

A 2017 to 2021 vintage is pulled as well, for the sensitivity on crashes in 2021 and
2022 whose context the later vintage only partly covers.
"""

from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

import geopandas as gpd
import pandas as pd
import requests
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
CACHE = DATA / "cache" / "acs"

STATE_FIPS = "48"  # Texas
TIGER_YEAR = 2023
TIGER_URL = f"https://www2.census.gov/geo/tiger/TIGER{TIGER_YEAR}/BG/tl_{TIGER_YEAR}_{STATE_FIPS}_bg.zip"

PRIMARY_VINTAGE = 2023      # ACS 5-year 2019 to 2023
SENSITIVITY_VINTAGE = 2021  # ACS 5-year 2017 to 2021

POVERTY_NUM = "B17021_002E"  # income in the past 12 months below poverty level
POVERTY_DEN = "B17021_001E"  # population for whom poverty status is determined


def _api_key() -> str:
    load_dotenv(ROOT / ".env")
    import os

    key = os.environ.get("CENSUS_API_KEY")
    if not key:
        raise RuntimeError("CENSUS_API_KEY is not set; place it in .env.")
    return key


def download_block_groups() -> gpd.GeoDataFrame:
    CACHE.mkdir(parents=True, exist_ok=True)
    local = CACHE / f"tl_{TIGER_YEAR}_{STATE_FIPS}_bg.gpkg"
    if local.exists():
        return gpd.read_file(local)

    print(f"downloading {TIGER_URL}")
    resp = requests.get(TIGER_URL, timeout=600)
    resp.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        target = CACHE / "tiger"
        target.mkdir(exist_ok=True)
        zf.extractall(target)
    shp = next((CACHE / "tiger").glob("*.shp"))
    gdf = gpd.read_file(shp)[["GEOID", "STATEFP", "COUNTYFP", "TRACTCE", "BLKGRPCE", "geometry"]]
    gdf.to_file(local, driver="GPKG")
    print(f"cached {len(gdf)} Texas block groups")
    return gdf


def fetch_poverty(vintage: int) -> pd.DataFrame:
    """Statewide block-group poverty counts for one ACS 5-year vintage."""
    CACHE.mkdir(parents=True, exist_ok=True)
    local = CACHE / f"acs5_{vintage}_poverty_bg_{STATE_FIPS}.csv"
    if local.exists():
        return pd.read_csv(local, dtype={"GEOID": str})

    key = _api_key()
    base = f"https://api.census.gov/data/{vintage}/acs/acs5"
    frames = []
    # The API serves block groups one county at a time for this vintage.
    counties = requests.get(
        f"{base}?get=NAME&for=county:*&in=state:{STATE_FIPS}&key={key}", timeout=120
    )
    counties.raise_for_status()
    rows = counties.json()[1:]
    county_codes = [r[-1] for r in rows]
    print(f"ACS {vintage}: pulling {len(county_codes)} Texas counties")

    for i, cc in enumerate(county_codes, start=1):
        url = (f"{base}?get={POVERTY_NUM},{POVERTY_DEN}"
               f"&for=block%20group:*&in=state:{STATE_FIPS}%20county:{cc}%20tract:*&key={key}")
        r = requests.get(url, timeout=120)
        if r.status_code != 200:
            print(f"  county {cc}: HTTP {r.status_code}, skipped")
            continue
        data = r.json()
        frames.append(pd.DataFrame(data[1:], columns=data[0]))
        if i % 50 == 0:
            print(f"  {i}/{len(county_codes)} counties", flush=True)

    df = pd.concat(frames, ignore_index=True)
    df["GEOID"] = (df["state"] + df["county"] + df["tract"] + df["block group"])
    for c in (POVERTY_NUM, POVERTY_DEN):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df.loc[df[c] < 0, c] = pd.NA
    out = df[["GEOID", POVERTY_NUM, POVERTY_DEN]].copy()
    out.to_csv(local, index=False)
    print(f"ACS {vintage}: {len(out)} block groups cached")
    return out


def run() -> pd.DataFrame:
    base = pd.read_csv(DATA / "base_frame.csv").drop_duplicates(subset="Crash_ID")
    located = base[base["coord_missing"] == 0][
        ["Crash_ID", "latitude", "longitude", "City_ID", "crash_year"]
    ].copy()

    bg = download_block_groups().to_crs("EPSG:4326")
    pts = gpd.GeoDataFrame(
        located,
        geometry=gpd.points_from_xy(located["longitude"], located["latitude"]),
        crs="EPSG:4326",
    )
    joined = gpd.sjoin(pts, bg[["GEOID", "geometry"]], how="left", predicate="within")
    joined = joined.drop(columns="index_right")
    unmatched = int(joined["GEOID"].isna().sum())

    audit: dict[str, object] = {
        "tiger_vintage": TIGER_YEAR,
        "tiger_url": TIGER_URL,
        "acs_primary_vintage": f"{PRIMARY_VINTAGE - 4} to {PRIMARY_VINTAGE}",
        "acs_sensitivity_vintage": f"{SENSITIVITY_VINTAGE - 4} to {SENSITIVITY_VINTAGE}",
        "poverty_numerator": POVERTY_NUM,
        "poverty_denominator": POVERTY_DEN,
        "b17021_available_at_block_group": True,
        "b17021_block_group_check_date": "2026-09-14",
        "located_crashes": int(len(located)),
        "points_not_matched_to_a_block_group": unmatched,
    }

    for vintage, suffix in ((PRIMARY_VINTAGE, ""), (SENSITIVITY_VINTAGE, "_acs2021")):
        pov = fetch_poverty(vintage)
        pov = pov.rename(columns={POVERTY_NUM: "num", POVERTY_DEN: "den"})
        pov["share"] = pov["num"] / pov["den"].where(pov["den"] > 0)
        joined = joined.merge(
            pov[["GEOID", "share"]].rename(columns={"share": f"poverty_share{suffix}"}),
            on="GEOID", how="left",
        )
        audit[f"poverty_missing{suffix or '_primary'}"] = int(
            joined[f"poverty_share{suffix}"].isna().sum()
        )

    out = joined[[
        "Crash_ID", "GEOID", "poverty_share", "poverty_share_acs2021"
    ]].rename(columns={"GEOID": "block_group_geoid"})
    out.to_csv(DATA / "acs_features.csv", index=False)

    audit["poverty_share_median"] = float(out["poverty_share"].median())
    audit["poverty_share_iqr"] = [
        float(out["poverty_share"].quantile(0.25)),
        float(out["poverty_share"].quantile(0.75)),
    ]
    audit["rows_written"] = int(len(out))
    (DATA / "acs_extraction_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps(audit, indent=2))
    return out


if __name__ == "__main__":
    run()
