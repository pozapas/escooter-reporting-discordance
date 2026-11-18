"""Assemble extraction logs into labels, and apply the fusion rule.

Reads the per-call JSONL logs written by `extract.py` and produces:

* ``reference_labels.csv``  - the temperature-0 reference run, one row per crash;
* ``vote_shares.csv``       - the share of the five temperature-0.7 runs labelling
                              each mechanism positive;
* ``route_a_labels.csv``    - the deterministic lexical classifier's labels;
* ``fusion.csv``            - the automated fusion label, the agreement and
                              disagreement flags, and the adjudication flag.

The fusion rule, applied per crash and per mechanism:

* the routes agree            -> that label stands;
* they disagree and the vote share is at or below 0.2 or at or above 0.8
                              -> the Route B label stands;
* they disagree and the vote share lies strictly between 0.2 and 0.8
                              -> the narrative is flagged for human adjudication.

If the flagged set exceeds the cap of 120 narratives, the band narrows to 0.3 to 0.7,
as the locked decision provides, and the narrowing is recorded in the audit file.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from labels import INJURY_CUES, MECHANISMS
from route_a import classify

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
LOG_DIR = DATA / "logs"

VOTE_RUNS = ("vote1", "vote2", "vote3", "vote4", "vote5")
BAND = (0.2, 0.8)
NARROW_BAND = (0.3, 0.7)
FLAG_CAP = 120


def read_log(run_id: str) -> pd.DataFrame:
    """Latest successfully parsed record per (crash, pass) for one run."""
    path = LOG_DIR / f"extract_{run_id}.jsonl"
    if not path.exists():
        return pd.DataFrame()
    rows: list[dict] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not rec.get("parsed"):
                continue
            flat = {
                "Crash_ID": int(rec["crash_id"]),
                "pass": rec["pass"],
                "run_id": rec["run_id"],
                "provider": rec.get("provider"),
                "timestamp_utc": rec.get("timestamp_utc"),
            }
            flat.update(rec["parsed"])
            rows.append(flat)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    return df.drop_duplicates(subset=["Crash_ID", "pass"], keep="last")


def build_reference() -> pd.DataFrame:
    df = read_log("reference")
    if df.empty:
        raise SystemExit("no parsed records in the reference log")

    mech = df[df["pass"] == "mechanism"].copy()
    cue = df[df["pass"] == "cue"].copy()

    mech_cols = ["Crash_ID", *MECHANISMS, "scooter_unit",
                 "failure_to_yield_actor", "signal_violation_actor"]
    cue_cols = ["Crash_ID", *INJURY_CUES]

    out = mech[[c for c in mech_cols if c in mech.columns]].merge(
        cue[[c for c in cue_cols if c in cue.columns]],
        on="Crash_ID", how="outer", validate="one_to_one",
    )
    out = out.rename(columns={"scooter_unit": "llm_scooter_unit"})
    out.to_csv(DATA / "reference_labels.csv", index=False)
    return out


def build_vote_shares() -> pd.DataFrame:
    frames = []
    for run in VOTE_RUNS:
        d = read_log(run)
        if d.empty:
            continue
        d = d[d["pass"] == "mechanism"]
        frames.append(d[["Crash_ID", *MECHANISMS]].assign(run=run))
    if not frames:
        return pd.DataFrame()
    stacked = pd.concat(frames, ignore_index=True)
    shares = stacked.groupby("Crash_ID")[list(MECHANISMS)].mean().reset_index()
    counts = stacked.groupby("Crash_ID").size().rename("n_runs").reset_index()
    shares = shares.merge(counts, on="Crash_ID")
    shares.to_csv(DATA / "vote_shares.csv", index=False)
    return shares


def build_route_a() -> pd.DataFrame:
    from masking import mask_narrative

    base = pd.read_csv(DATA / "base_frame.csv").drop_duplicates(subset="Crash_ID")
    rows = []
    for r in base.itertuples():
        masked, _ = mask_narrative(str(r.narrative))
        res = classify(masked)
        res.pop("evidence", None)
        res["Crash_ID"] = int(r.Crash_ID)
        rows.append(res)
    out = pd.DataFrame(rows)
    out = out.rename(columns={
        "scooter_unit": "routea_scooter_unit",
        "scooter_unit_confidence": "routea_scooter_confidence",
        "failure_to_yield_actor": "routea_failure_to_yield_actor",
        "signal_violation_actor": "routea_signal_violation_actor",
    })
    out.to_csv(DATA / "route_a_labels.csv", index=False)
    return out


def fuse() -> tuple[pd.DataFrame, dict]:
    ref = pd.read_csv(DATA / "reference_labels.csv")
    route_a = pd.read_csv(DATA / "route_a_labels.csv")
    shares_path = DATA / "vote_shares.csv"
    shares = pd.read_csv(shares_path) if shares_path.exists() else None

    a = route_a[["Crash_ID", *MECHANISMS]].rename(
        columns={m: f"a_{m}" for m in MECHANISMS})
    b = ref[["Crash_ID", *MECHANISMS]].rename(
        columns={m: f"b_{m}" for m in MECHANISMS})
    df = a.merge(b, on="Crash_ID", how="inner", validate="one_to_one")

    have_shares = shares is not None and not shares.empty
    if have_shares:
        s = shares[["Crash_ID", *MECHANISMS]].rename(
            columns={m: f"s_{m}" for m in MECHANISMS})
        df = df.merge(s, on="Crash_ID", how="left")

    audit: dict[str, object] = {
        "vote_shares_available": bool(have_shares),
        "band": list(BAND),
        "flag_cap": FLAG_CAP,
        "n_crashes": int(len(df)),
        "per_mechanism": {},
    }

    def apply_band(lo: float, hi: float) -> tuple[pd.DataFrame, int]:
        out = pd.DataFrame({"Crash_ID": df["Crash_ID"]})
        flags = pd.Series(False, index=df.index)
        for m in MECHANISMS:
            a_lab, b_lab = df[f"a_{m}"], df[f"b_{m}"]
            agree = a_lab == b_lab
            if have_shares:
                share = df[f"s_{m}"]
                decisive = (share <= lo) | (share >= hi)
                ambiguous = ~agree & ~decisive & share.notna()
                # A crash with no vote share yet falls back to the reference label.
                label = a_lab.where(agree, b_lab)
            else:
                # Before the vote runs exist, every disagreement is preliminary and
                # the reference label stands so downstream code can be proven.
                ambiguous = pd.Series(False, index=df.index)
                label = a_lab.where(agree, b_lab)
            out[m] = label.astype(int)
            out[f"{m}_agree"] = agree.astype(int)
            out[f"{m}_flagged"] = ambiguous.astype(int)
            flags |= ambiguous
        out["needs_adjudication"] = flags.astype(int)
        return out, int(flags.sum())

    fused, n_flagged = apply_band(*BAND)
    audit["band_used"] = list(BAND)
    if n_flagged > FLAG_CAP:
        fused, n_narrow = apply_band(*NARROW_BAND)
        audit["band_used"] = list(NARROW_BAND)
        audit["band_narrowed_because"] = (
            f"{n_flagged} narratives exceeded the cap of {FLAG_CAP}")
        n_flagged = n_narrow
    audit["n_flagged_narratives"] = n_flagged

    for m in MECHANISMS:
        audit["per_mechanism"][m] = {
            "route_a_positives": int(df[f"a_{m}"].sum()),
            "route_b_positives": int(df[f"b_{m}"].sum()),
            "disagreements": int((df[f"a_{m}"] != df[f"b_{m}"]).sum()),
            "flagged": int(fused[f"{m}_flagged"].sum()),
            "fused_positives": int(fused[m].sum()),
        }

    fused.to_csv(DATA / "fusion.csv", index=False)
    (DATA / "fusion_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")

    flagged_ids = fused.loc[fused["needs_adjudication"] == 1, ["Crash_ID"]]
    flagged_ids.to_csv(DATA / "flagged_ids.csv", index=False)
    return fused, audit


if __name__ == "__main__":
    import sys

    what = sys.argv[1:] or ["reference", "routea", "votes", "fuse"]
    if "reference" in what:
        r = build_reference()
        print(f"reference_labels.csv: {len(r)} crashes")
        print("  positives:", {m: int(r[m].sum()) for m in MECHANISMS if m in r})
        print("  cues:     ", {c: int(r[c].sum()) for c in INJURY_CUES if c in r})
    if "routea" in what:
        a = build_route_a()
        print(f"route_a_labels.csv: {len(a)} crashes")
        print("  positives:", {m: int(a[m].sum()) for m in MECHANISMS})
    if "votes" in what:
        v = build_vote_shares()
        print(f"vote_shares.csv: {len(v)} crashes" if len(v) else "vote runs not present yet")
    if "fuse" in what:
        f, audit = fuse()
        print(f"fusion.csv: {len(f)} crashes, {audit['n_flagged_narratives']} flagged "
              f"(band {audit['band_used']})")
        for m, d in audit["per_mechanism"].items():
            print(f"  {m:24s} A={d['route_a_positives']:3d} B={d['route_b_positives']:3d} "
                  f"disagree={d['disagreements']:3d} flagged={d['flagged']:3d} "
                  f"fused={d['fused_positives']:3d}")
