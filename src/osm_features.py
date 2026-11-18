"""Real OpenStreetMap features for every located crash.

For each crash with usable coordinates this derives, from OpenStreetMap:

* ``bike_facility_nearest`` - the drivable roadway segment nearest the crash point
  carries a cycle facility (the primary variable);
* ``bike_facility_50m`` / ``_100m`` / ``_200m`` - any cycle facility within the buffer
  (the buffer sensitivity);
* ``road_density_km_100m`` - drivable centerline length within 100 m, in km, with each
  segment clipped to the buffer before its length is counted;
* ``n_ways_100m`` - count of distinct drivable ways within 100 m.

Crash points are grouped into a coarse grid and one Overpass request is issued per
occupied tile, rather than one per crash. Per-point requests against the public
Overpass instance took upwards of forty seconds each. Tiles are cached on disk as GeoPackage, so an interrupted run resumes and
a rerun costs nothing.

The extraction date and the Overpass endpoint are recorded on every row, so the
manuscript states the snapshot date and the direction of its bias.
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
CACHE = DATA / "cache" / "osm"
TILES = CACHE / "tiles"

TILE_DEG = 0.2          # about 20 km; 444 located crashes fall in ~105 tiles
TILE_PAD_DEG = 0.004    # about 400 m, so a 200 m buffer never leaves the tile
BUFFERS = (50, 100, 200)
SLEEP_S = 1.0

DRIVABLE = {
    "motorway", "motorway_link", "trunk", "trunk_link", "primary", "primary_link",
    "secondary", "secondary_link", "tertiary", "tertiary_link",
    "unclassified", "residential", "living_street", "service",
}
CYCLE_KEYS = ("cycleway", "cycleway:left", "cycleway:right", "cycleway:both")
NON_FACILITY = {"no", "none", "separate", "0", "false", "nan"}

OSM_TAGS = {
    "highway": True,
    "cycleway": True,
    "cycleway:left": True,
    "cycleway:right": True,
    "cycleway:both": True,
    "bicycle": True,
}


def _truthy_cycle(val) -> bool:
    return isinstance(val, str) and val.strip().lower() not in NON_FACILITY


def _is_cycle_facility(row) -> bool:
    hw = row.get("highway")
    if isinstance(hw, str) and hw == "cycleway":
        return True
    if isinstance(hw, list) and "cycleway" in hw:
        return True
    if any(_truthy_cycle(row.get(k)) for k in CYCLE_KEYS):
        return True
    bike = row.get("bicycle")
    return isinstance(bike, str) and bike.strip().lower() == "designated"


def _is_drivable(row) -> bool:
    hw = row.get("highway")
    if isinstance(hw, str):
        return hw in DRIVABLE
    if isinstance(hw, list):
        return any(h in DRIVABLE for h in hw)
    return False


def tile_key(lat: float, lon: float) -> tuple[int, int]:
    return (int(round(lat / TILE_DEG)), int(round(lon / TILE_DEG)))


def tile_bbox(key: tuple[int, int]) -> tuple[float, float, float, float]:
    ky, kx = key
    south = ky * TILE_DEG - TILE_DEG / 2 - TILE_PAD_DEG
    north = ky * TILE_DEG + TILE_DEG / 2 + TILE_PAD_DEG
    west = kx * TILE_DEG - TILE_DEG / 2 - TILE_PAD_DEG
    east = kx * TILE_DEG + TILE_DEG / 2 + TILE_PAD_DEG
    return west, south, east, north


# The main Overpass instance refuses connections under load. The mirrors below are
# tried in order; whichever answers is recorded in the audit file, because the
# manuscript states where the snapshot came from as well as when.
# Ordered by measured responsiveness on 2026-09-14. The mirrors reject requests that
# do not carry a meaningful User-Agent, so `USER_AGENT` below is set on osmnx before
# any query is issued; without it every mirror returns HTTP 429.
OVERPASS_ENDPOINTS = (
    "https://overpass.osm.ch/api",
    "https://overpass.kumi.systems/api",
    "https://overpass.private.coffee/api",
    "https://overpass-api.de/api",
)

SUBDIVIDED: set[tuple[int, int]] = set()

USER_AGENT = (
    "escooter-severity-research/1.0 (academic research; contact via "
    "contact via repository README)"
)


def configure_osmnx() -> None:
    """Settings every entry point must apply before issuing an Overpass query."""
    import osmnx as ox

    ox.settings.use_cache = True
    ox.settings.cache_folder = str(CACHE / "osmnx")
    ox.settings.log_console = False
    ox.settings.requests_timeout = 90
    ox.settings.http_user_agent = USER_AGENT
    # The downloader spreads concurrent requests across four mirrors, so osmnx's
    # own per-endpoint limiter would serialize them without adding politeness.
    ox.settings.overpass_rate_limit = False
ENDPOINTS_USED: dict[str, int] = {}


def _fetch_bbox(bbox: tuple[float, float, float, float]):
    """Fetch one bounding box, rotating over the Overpass mirrors until one answers."""
    import osmnx as ox

    last: Exception | None = None
    for url in OVERPASS_ENDPOINTS:
        ox.settings.overpass_url = url
        try:
            gdf = ox.features_from_bbox(bbox, tags=OSM_TAGS)
            ENDPOINTS_USED[url] = ENDPOINTS_USED.get(url, 0) + 1
            return gdf
        except Exception as exc:  # noqa: BLE001 - try the next mirror
            last = exc
            time.sleep(2)
    raise last if last else RuntimeError("no Overpass endpoint answered")


def _quarters(bbox: tuple[float, float, float, float]):
    """Split a bounding box into four, overlapping slightly so nothing falls in a seam."""
    west, south, east, north = bbox
    mid_lon = (west + east) / 2
    mid_lat = (south + north) / 2
    pad = TILE_PAD_DEG
    return [
        (west, south, mid_lon + pad, mid_lat + pad),
        (mid_lon - pad, south, east, mid_lat + pad),
        (west, mid_lat - pad, mid_lon + pad, north),
        (mid_lon - pad, mid_lat - pad, east, north),
    ]


def _download_tile(key: tuple[int, int]):
    """Download one tile whole, over the mirrors.

    The subdivision fallback for dense tiles lives in `osm_download._fetch`, which is
    what the bulk downloader actually calls and which owns the endpoint rotation and its
    lock. It used to be duplicated here as well. Two copies of the same geometry rule is
    one copy too many: a change to the tile pad or the quadrant overlap would have to be
    made in both places, and the copy nothing calls would drift silently. `_quarters` is
    still defined here and imported from there, so the split itself has one definition.
    """
    return _fetch_bbox(tile_bbox(key))


def load_tile(key: tuple[int, int]) -> gpd.GeoDataFrame | None:
    """Return the tile's line features, downloading and caching it if needed."""
    TILES.mkdir(parents=True, exist_ok=True)
    path = TILES / f"tile_{key[0]}_{key[1]}.gpkg"
    if path.exists():
        try:
            return gpd.read_file(path)
        except Exception:
            path.unlink(missing_ok=True)

    gdf = _download_tile(key)
    if gdf is None or gdf.empty:
        gpd.GeoDataFrame(geometry=[], crs="EPSG:4326").to_file(path, driver="GPKG")
        return None

    lines = gdf[gdf.geometry.geom_type.isin(["LineString", "MultiLineString"])].copy()
    if lines.empty:
        gpd.GeoDataFrame(geometry=[], crs="EPSG:4326").to_file(path, driver="GPKG")
        return None

    lines["is_cycle"] = lines.apply(_is_cycle_facility, axis=1)
    lines["is_drive"] = lines.apply(_is_drivable, axis=1)
    lines["highway_str"] = lines["highway"].astype(str)
    keep = lines[lines["is_cycle"] | lines["is_drive"]][
        ["geometry", "is_cycle", "is_drive", "highway_str"]
    ].reset_index(drop=True)
    keep.to_file(path, driver="GPKG")
    return keep


def features_from_tile(tile: gpd.GeoDataFrame, lat: float, lon: float) -> dict:
    out: dict[str, object] = {
        "bike_facility_nearest": 0,
        "nearest_segment_dist_m": None,
        "nearest_segment_highway": None,
        "road_density_km_100m": 0.0,
        "n_ways_100m": 0,
        "osm_n_features": 0 if tile is None else int(len(tile)),
        **{f"bike_facility_{b}m": 0 for b in BUFFERS},
    }
    if tile is None or tile.empty:
        return out

    utm = tile.estimate_utm_crs()
    lines = tile.to_crs(utm)
    pt = gpd.GeoSeries([Point(lon, lat)], crs="EPSG:4326").to_crs(utm).iloc[0]

    lines = lines.copy()
    lines["dist_m"] = lines.geometry.distance(pt)
    near = lines[lines["dist_m"] <= max(BUFFERS)]
    if near.empty:
        return out

    drivable = near[near["is_drive"]]
    if not drivable.empty:
        nearest = drivable.loc[drivable["dist_m"].idxmin()]
        out["bike_facility_nearest"] = int(bool(nearest["is_cycle"]))
        out["nearest_segment_dist_m"] = round(float(nearest["dist_m"]), 2)
        out["nearest_segment_highway"] = str(nearest["highway_str"])

    for b in BUFFERS:
        out[f"bike_facility_{b}m"] = int(
            bool(((near["dist_m"] <= b) & near["is_cycle"]).any())
        )

    buffer100 = pt.buffer(100)
    within100 = drivable[drivable["dist_m"] <= 100]
    if not within100.empty:
        clipped_m = float(within100.geometry.intersection(buffer100).length.sum())
        out["road_density_km_100m"] = round(clipped_m / 1000.0, 4)
        out["n_ways_100m"] = int(len(within100))
    out["osm_n_features"] = int(len(near))
    return out


def run() -> pd.DataFrame:
    import osmnx as ox

    CACHE.mkdir(parents=True, exist_ok=True)
    base = pd.read_csv(DATA / "base_frame.csv").drop_duplicates(subset="Crash_ID")
    located = base[base["coord_missing"] == 0][["Crash_ID", "latitude", "longitude"]].copy()
    located["tile"] = [
        tile_key(float(r.latitude), float(r.longitude)) for r in located.itertuples()
    ]
    tiles = sorted(set(located["tile"]))
    today = datetime.now(timezone.utc).date().isoformat()
    print(f"{len(located)} located crashes in {len(tiles)} tiles of {TILE_DEG} degrees")
    print(f"overpass endpoints (in order): {OVERPASS_ENDPOINTS}")

    rows: list[dict] = []
    failed_tiles = 0
    for i, key in enumerate(tiles, start=1):
        started = time.time()
        try:
            tile = load_tile(key)
            err = ""
        except Exception as exc:  # noqa: BLE001 - recorded, run continues
            tile, err = None, f"{type(exc).__name__}: {exc}"
            failed_tiles += 1
        elapsed = round(time.time() - started, 1)

        subset = located[located["tile"] == key]
        for r in subset.itertuples():
            feat = features_from_tile(tile, float(r.latitude), float(r.longitude))
            feat["Crash_ID"] = int(r.Crash_ID)
            feat["osm_extraction_date"] = today
            feat["osm_tile"] = f"{key[0]}_{key[1]}"
            feat["osm_error"] = err
            rows.append(feat)

        print(f"  tile {i}/{len(tiles)} {key} -> {len(subset)} crashes, {elapsed}s"
              f"{' ERROR ' + err if err else ''}", flush=True)
        time.sleep(SLEEP_S)

    out = pd.DataFrame(rows).sort_values("Crash_ID").reset_index(drop=True)
    out.to_csv(DATA / "osm_features.csv", index=False)

    summary = {
        "located_crashes": int(len(located)),
        "tiles": len(tiles),
        "failed_tiles": failed_tiles,
        "extraction_date": today,
        "overpass_endpoints_used": dict(ENDPOINTS_USED),
        "tile_degrees": TILE_DEG,
        "rows_written": int(len(out)),
        "bike_facility_nearest_positives": int(out["bike_facility_nearest"].sum()),
        "bike_facility_100m_positives": int(out["bike_facility_100m"].sum()),
        "road_density_km_100m_median": float(out["road_density_km_100m"].median()),
        "rows_with_error": int((out["osm_error"] != "").sum()),
    }
    (DATA / "osm_extraction_audit.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return out


if __name__ == "__main__":
    import osmnx as ox

    configure_osmnx()
    run()
