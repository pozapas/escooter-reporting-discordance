"""Download the OpenStreetMap tiles that `osm_features.py` reads.

Separated from feature computation so the slow, network-bound part can run on its
own, in parallel, and be restarted freely: every tile is cached to GeoPackage and an
existing cache file is never re-fetched.

The public Overpass instance answers slowly and intermittently refuses connections,
so tiles are distributed round-robin over four mirrors and fetched by a small pool of
workers. Each worker starts at its assigned mirror and falls through the rest.
"""

from __future__ import annotations

import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import geopandas as gpd
import pandas as pd

from osm_features import (
    SUBDIVIDED,
    _quarters,
    CACHE, OVERPASS_ENDPOINTS, OSM_TAGS, TILES, DATA, configure_osmnx,
    _is_cycle_facility, _is_drivable, tile_bbox, tile_key,
)

# Two, not four. The tiles left are the dense urban ones, and each is now fetched as
# four subdivided quadrant queries rather than one, so four workers would put sixteen
# heavy queries on the mirrors at once and earn a rate limit instead of data.
WORKERS = 2
PER_ENDPOINT_TIMEOUT = 90

_lock = threading.Lock()
_stats: dict[str, int] = {}


def _fetch(key: tuple[int, int], start: int) -> tuple[tuple[int, int], str, float]:
    """Fetch and cache one tile. Returns the tile key, an error string, and seconds."""
    import osmnx as ox

    path = TILES / f"tile_{key[0]}_{key[1]}.gpkg"
    if path.exists():
        return key, "", 0.0

    started = time.time()
    order = [OVERPASS_ENDPOINTS[(start + i) % len(OVERPASS_ENDPOINTS)]
             for i in range(len(OVERPASS_ENDPOINTS))]

    def over_mirrors(bbox):
        """Try one bounding box on each mirror in turn; raise if none answers."""
        west, south, east, north = bbox
        err: Exception | None = None
        for url in order:
            try:
                with _lock:
                    ox.settings.overpass_url = url
                    ox.settings.requests_timeout = PER_ENDPOINT_TIMEOUT
                got = ox.features_from_bbox((west, south, east, north), tags=OSM_TAGS)
                with _lock:
                    _stats[url] = _stats.get(url, 0) + 1
                return got
            except Exception as exc:  # noqa: BLE001 - try the next mirror
                err = exc
                time.sleep(1)
        raise err if err else RuntimeError("no mirror answered")

    def fetch_whole_or_subdivided():
        """Fetch the tile, falling back to quadrants when the servers will not send it.

        The mirrors answer a count query for these tiles in under a second, so they are
        reachable; what they will not do inside their timeout is stream every highway
        geometry in a 20 km square of Houston or Austin. Those dense tiles are also
        where most crashes are, so abandoning them would drop 73 of 446 located rider
        rows. Quadrants overlap by the tile pad so nothing is lost on a seam, and
        duplicates are dropped by index after concatenation.
        """
        try:
            return over_mirrors(tile_bbox(key)), False
        except Exception:  # noqa: BLE001
            pass

        parts = []
        for sub in _quarters(tile_bbox(key)):
            try:
                parts.append(over_mirrors(sub))
            except Exception:  # noqa: BLE001
                for sub2 in _quarters(sub):
                    try:
                        parts.append(over_mirrors(sub2))
                    except Exception:  # noqa: BLE001
                        continue
        parts = [g for g in parts if g is not None and not g.empty]
        if not parts:
            raise RuntimeError("failed whole and in every subdivision")
        merged = pd.concat(parts)
        return merged[~merged.index.duplicated(keep="first")], True

    last = ""
    for _attempt in (0,):
        try:
            gdf, was_subdivided = fetch_whole_or_subdivided()
            if was_subdivided:
                SUBDIVIDED.add(key)

            if gdf is None or gdf.empty:
                gpd.GeoDataFrame(geometry=[], crs="EPSG:4326").to_file(path, driver="GPKG")
                return key, "", time.time() - started

            lines = gdf[gdf.geometry.geom_type.isin(["LineString", "MultiLineString"])].copy()
            if lines.empty:
                gpd.GeoDataFrame(geometry=[], crs="EPSG:4326").to_file(path, driver="GPKG")
                return key, "", time.time() - started

            lines["is_cycle"] = lines.apply(_is_cycle_facility, axis=1)
            lines["is_drive"] = lines.apply(_is_drivable, axis=1)
            lines["highway_str"] = lines["highway"].astype(str)
            keep = lines[lines["is_cycle"] | lines["is_drive"]][
                ["geometry", "is_cycle", "is_drive", "highway_str"]
            ].reset_index(drop=True)
            keep.to_file(path, driver="GPKG")
            return key, "", time.time() - started
        except Exception as exc:  # noqa: BLE001
            last = f"{type(exc).__name__}"
    return key, last or "all mirrors failed", time.time() - started


def main() -> None:
    base = pd.read_csv(DATA / "base_frame.csv").drop_duplicates(subset="Crash_ID")
    located = base[base["coord_missing"] == 0]
    keys = sorted({tile_key(float(r.latitude), float(r.longitude))
                   for r in located.itertuples()})
    TILES.mkdir(parents=True, exist_ok=True)
    todo = [k for k in keys if not (TILES / f"tile_{k[0]}_{k[1]}.gpkg").exists()]
    print(f"{len(keys)} tiles total, {len(todo)} to fetch, {WORKERS} workers, "
          f"{len(OVERPASS_ENDPOINTS)} mirrors", flush=True)

    done = failed = 0
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futures = {pool.submit(_fetch, k, i % len(OVERPASS_ENDPOINTS)): k
                   for i, k in enumerate(todo)}
        for fut in as_completed(futures):
            key, err, secs = fut.result()
            done += 1
            if err:
                failed += 1
            print(f"  [{done}/{len(todo)}] tile {key} {secs:.0f}s"
                  f"{' FAILED ' + err if err else ''}", flush=True)

    cached = len(list(TILES.glob("*.gpkg")))
    print(json.dumps({
        "tiles_total": len(keys),
        "tiles_cached": cached,
        "fetched_this_run": done - failed,
        "failed_this_run": failed,
        "endpoint_use": _stats,
    }, indent=2))
    if failed:
        print("rerun this script to retry the failed tiles; cached tiles are skipped")


if __name__ == "__main__":
    import osmnx as ox

    configure_osmnx()
    sys.exit(main())
