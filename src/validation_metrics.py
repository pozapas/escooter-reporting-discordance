"""Human validation metrics (Table 2 and Appendix C).

Merges the three coders' exported CSVs, computes agreement among them, forms the
majority-vote gold labels, and scores every extraction route against those labels.

Reported per label and macro-averaged:

* Fleiss' kappa across the three coders, and pairwise Cohen's kappa;
* counts of unanimous and two-to-one decisions;
* precision, recall, F1, and the four confusion counts for Route A, the Route B
  reference run, the Route B vote share thresholded at 0.5, and the automated fusion;
* bootstrap 95 percent intervals over 2,000 resamples;
* every metric a second time with inverse-probability weights, because the validation
  sample oversamples predicted positives and unweighted recall and F1 are biased
  upward by that design.

Run after the coder exports are in `data/coding/`.
"""

from __future__ import annotations

import json
import pathlib
from pathlib import Path

import numpy as np
import pandas as pd

from labels import ALL_LABELS, INJURY_CUES, MECHANISMS

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
CODING = DATA / "coding"

# A trailing run of untouched rows this long or longer is read as a coder
# having stopped, not as a run of genuine negative judgements.
TRAILING_BLANK_MIN = 5

N_BOOT = 2000
BOOT_SEED = 20260917


# --------------------------------------------------------------------------
# Coder exports
# --------------------------------------------------------------------------


def _coded_mask(d: pd.DataFrame, name: str) -> pd.Series:
    """Which rows in one export are actual judgements rather than untouched defaults.

    The app writes a per-row completeness flag precisely so this question has a direct
    answer. From batch 2 onward that is a plain `coded` 0/1 — it carries no time, no
    duration and no ordering, so it answers the integrity question without recording
    anything personal about when a coder worked. Older exports carry `timestamp_utc`
    instead. Either governs where present, and nothing below is inferred.

    Exports have arrived without it, and the distinction still has to be drawn, because
    an untouched narrative exports as fifteen zeros and is indistinguishable from a
    narrative a coder read and found nothing in. Averaging those in does not merely add
    noise: it manufactures disagreement with the coders who did read them, and it drags
    the reliability statistic down by an amount nobody can see afterwards. On batch 1 it
    was the difference between a macro Fleiss kappa of 0.859 and 0.919.

    Without the column, one pattern is still diagnosable. Narratives are presented in a
    per-coder randomised order and exported in that order, so a coder who stops partway
    leaves a *contiguous block of untouched rows running to the end of the file*. A
    genuine run of "nothing in this narrative" judgements would not be contiguous, would
    not end exactly at the last row, and would not coincide with narratives on which the
    other coders found more than usual. Only a trailing block is treated as uncoded, and
    only when it is long enough not to be chance; everything earlier is taken at face
    value.
    """
    if "coded" in d.columns:
        return d["coded"].fillna(0).astype(str).str.strip().isin(["1", "1.0", "True",
                                                                  "true"])
    if "timestamp_utc" in d.columns:
        return d["timestamp_utc"].notna() & (d["timestamp_utc"].astype(str).str.strip()
                                             != "")

    labels = [c for c in ALL_LABELS if c in d.columns]
    unsure = [c for c in d.columns if c.startswith("unsure_")]
    blank = (d[labels].sum(axis=1) == 0)
    if unsure:
        blank &= (d[unsure].sum(axis=1) == 0)
    if "note" in d.columns:
        blank &= ~(d["note"].notna() & (d["note"].astype(str).str.strip() != ""))

    coded = pd.Series(True, index=d.index)
    tail = 0
    for i in range(len(d) - 1, -1, -1):
        if blank.iloc[i]:
            tail += 1
        else:
            break
    if tail >= TRAILING_BLANK_MIN:
        coded.iloc[len(d) - tail:] = False
    return coded


def merge_coders(batch: int = 1) -> tuple[pd.DataFrame, dict]:
    """Read every coder CSV for one batch and return the long-form codings."""
    # Exports arrive in waves and get renamed on the way — a coder who finishes a
    # partial file sends a second, more complete one, and it may land in a subfolder.
    # Searching recursively and keeping the most complete file per coder is safer than
    # relying on anyone to delete the superseded copy, and silently concatenating both
    # would double every row.
    files = sorted(CODING.rglob(f"coder*_batch{batch}_*.csv"))
    if not files:
        raise SystemExit(f"no coder exports for batch {batch} in {CODING}")

    best: dict[int, tuple[int, pathlib.Path]] = {}
    superseded: list[str] = []
    for f in files:
        probe = pd.read_csv(f)
        coder = int(probe["coder_id"].iloc[0])
        n_coded = int(_coded_mask(probe, f.name).sum())
        if coder not in best or n_coded > best[coder][0]:
            if coder in best:
                superseded.append(f"{best[coder][1].name} ({best[coder][0]} coded) "
                                  f"superseded by {f.name} ({n_coded} coded)")
            best[coder] = (n_coded, f)
        else:
            superseded.append(f"{f.name} ({n_coded} coded) ignored; "
                              f"{best[coder][1].name} has {best[coder][0]}")
    files = [v[1] for _, v in sorted(best.items())]

    frames = []
    per_file: dict[str, dict] = {}
    for f in files:
        d = pd.read_csv(f)
        d["source_file"] = f.name
        coded = _coded_mask(d, f.name)
        d["_coded"] = coded
        per_file[f.name] = {
            "rows": int(len(d)),
            "coded": int(coded.sum()),
            "not_coded": int((~coded).sum()),
            "coder_id": int(d["coder_id"].iloc[0]),
            "has_timestamp_column": "timestamp_utc" in d.columns,
        }
        frames.append(d)

    long = pd.concat(frames, ignore_index=True)

    ids_file = DATA / ("validation_ids.csv" if batch == 1 else "flagged_ids.csv")
    expected = set(pd.read_csv(ids_file)["Crash_ID"].astype(int))
    excluded_overlap: list[int] = []
    if batch == 2:
        # The fusion rule flagged 47 narratives, ten of which are also in the batch-1
        # validation sample and already carry a three-coder panel. They were left out of
        # the batch-2 queue deliberately, so counting them as missing here would report a
        # gap that does not exist and would hide a real one behind it.
        prior = set(pd.read_csv(DATA / "validation_ids.csv")["Crash_ID"].astype(int))
        excluded_overlap = sorted(expected & prior)
        expected -= prior
    report: dict[str, object] = {
        "batch": batch,
        "files": per_file,
        "superseded_or_ignored": superseded,
        "excluded_because_batch1_already_has_a_panel": excluded_overlap,
        "expected_narratives": len(expected),
        "coders": sorted(long["coder_id"].unique().tolist()),
    }

    # Completeness: which coder is missing which narrative.
    missing: dict[str, list[int]] = {}
    for coder, grp in long.groupby("coder_id"):
        coded_ids = set(grp.loc[grp["_coded"], "crash_id"].astype(int))
        gap = sorted(expected - coded_ids)
        if gap:
            missing[f"coder{coder}"] = gap
    report["missing_by_coder"] = {k: len(v) for k, v in missing.items()}
    report["missing_ids_by_coder"] = missing

    dupes = long.duplicated(subset=["coder_id", "crash_id"]).sum()
    report["duplicate_rows"] = int(dupes)
    if dupes:
        long = long.drop_duplicates(subset=["coder_id", "crash_id"], keep="last")

    # An uncoded row exports as fifteen zeros. Blank the labels so that nothing
    # downstream can mistake a row the coder never reached for a row they read and found
    # nothing in: `gold_labels` and `agreement` both count raters with `notna`, so a
    # blanked row correctly reduces the rater count instead of casting a silent negative
    # vote.
    for col in ALL_LABELS:
        if col in long.columns:
            long.loc[~long["_coded"], col] = np.nan

    fully = (long.groupby("crash_id")["_coded"].sum() == len(per_file))
    report["narratives_coded_by_every_coder"] = int(fully.sum())
    report["narratives_short_of_a_full_panel"] = int((~fully).sum())
    report["primary_metrics_restricted_to_full_panel"] = True

    long.to_csv(DATA / f"validation_codes_batch{batch}.csv", index=False)
    return long, report


def gold_labels(long: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Majority vote of the three coders, with unanimity counts per label."""
    pivot = long.pivot_table(index="crash_id", columns="coder_id",
                            values=list(ALL_LABELS), aggfunc="first")
    gold = pd.DataFrame(index=pivot.index)
    detail: dict[str, dict] = {}
    for lab in ALL_LABELS:
        votes = pivot[lab]
        n_raters = votes.notna().sum(axis=1)
        positives = votes.fillna(0).sum(axis=1)
        gold[lab] = (positives > n_raters / 2).astype(int)
        detail[lab] = {
            "unanimous_positive": int(((positives == n_raters) & (n_raters > 0)).sum()),
            "unanimous_negative": int((positives == 0).sum()),
            "split_two_to_one": int(((positives > 0) & (positives < n_raters)).sum()),
            "gold_positives": int(gold[lab].sum()),
        }
    gold = gold.reset_index().rename(columns={"crash_id": "Crash_ID"})
    return gold, detail


# --------------------------------------------------------------------------
# Agreement
# --------------------------------------------------------------------------

def fleiss_kappa(votes: pd.DataFrame) -> float:
    """Fleiss' kappa for a binary label over subjects by raters."""
    v = votes.dropna(how="all")
    n_pos = v.fillna(0).sum(axis=1).to_numpy()
    n_rat = v.notna().sum(axis=1).to_numpy()
    keep = n_rat > 1
    n_pos, n_rat = n_pos[keep], n_rat[keep]
    if len(n_pos) == 0:
        return float("nan")
    n_neg = n_rat - n_pos
    p_i = (n_pos * (n_pos - 1) + n_neg * (n_neg - 1)) / (n_rat * (n_rat - 1))
    p_bar = p_i.mean()
    p_pos = (n_pos / n_rat).mean()
    p_e = p_pos ** 2 + (1 - p_pos) ** 2
    if np.isclose(p_e, 1.0):
        return float("nan")
    return float((p_bar - p_e) / (1 - p_e))


def cohen_kappa(a: np.ndarray, b: np.ndarray) -> float:
    mask = ~(pd.isna(a) | pd.isna(b))
    a, b = np.asarray(a)[mask].astype(int), np.asarray(b)[mask].astype(int)
    if len(a) == 0:
        return float("nan")
    po = (a == b).mean()
    pe = (a.mean() * b.mean()) + ((1 - a.mean()) * (1 - b.mean()))
    if np.isclose(pe, 1.0):
        return float("nan")
    return float((po - pe) / (1 - pe))


def agreement(long: pd.DataFrame) -> dict:
    out: dict[str, dict] = {}
    pivot = long.pivot_table(index="crash_id", columns="coder_id",
                            values=list(ALL_LABELS), aggfunc="first")
    coders = sorted(long["coder_id"].unique())
    for lab in ALL_LABELS:
        votes = pivot[lab]
        entry: dict[str, object] = {"fleiss_kappa": round(fleiss_kappa(votes), 4)}
        pairwise = {}
        for i in range(len(coders)):
            for j in range(i + 1, len(coders)):
                ci, cj = coders[i], coders[j]
                if ci in votes.columns and cj in votes.columns:
                    pairwise[f"{ci}v{cj}"] = round(
                        cohen_kappa(votes[ci].to_numpy(), votes[cj].to_numpy()), 4)
        entry["pairwise_cohen_kappa"] = pairwise
        entry["positives_by_coder"] = {
            str(c): int(votes[c].fillna(0).sum()) for c in coders if c in votes.columns
        }
        out[lab] = entry
    return out


# --------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------

def _counts(truth: np.ndarray, pred: np.ndarray, w: np.ndarray) -> dict:
    tp = float((w * ((truth == 1) & (pred == 1))).sum())
    fp = float((w * ((truth == 0) & (pred == 1))).sum())
    fn = float((w * ((truth == 1) & (pred == 0))).sum())
    tn = float((w * ((truth == 0) & (pred == 0))).sum())
    prec = tp / (tp + fp) if tp + fp else float("nan")
    rec = tp / (tp + fn) if tp + fn else float("nan")
    f1 = (2 * prec * rec / (prec + rec)
          if prec == prec and rec == rec and (prec + rec) > 0 else float("nan"))
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": prec, "recall": rec, "f1": f1}


def _bootstrap(truth: np.ndarray, pred: np.ndarray, w: np.ndarray) -> dict:
    rng = np.random.default_rng(BOOT_SEED)
    n = len(truth)
    stats = {"precision": [], "recall": [], "f1": []}
    for _ in range(N_BOOT):
        idx = rng.integers(0, n, n)
        c = _counts(truth[idx], pred[idx], w[idx])
        for k in stats:
            stats[k].append(c[k])
    out = {}
    for k, vals in stats.items():
        arr = np.asarray(vals, dtype=float)
        arr = arr[~np.isnan(arr)]
        out[k] = ([round(float(np.percentile(arr, 2.5)), 4),
                   round(float(np.percentile(arr, 97.5)), 4)]
                  if len(arr) else [float("nan"), float("nan")])
    return out


def score_routes(gold: pd.DataFrame) -> dict:
    """Score every route against the gold labels, weighted and unweighted."""
    ids = pd.read_csv(DATA / "validation_ids.csv")
    weights = dict(zip(ids["Crash_ID"].astype(int), ids["sampling_weight"]))

    routes: dict[str, pd.DataFrame] = {}
    if (DATA / "route_a_labels.csv").exists():
        routes["route_a"] = pd.read_csv(DATA / "route_a_labels.csv")
    if (DATA / "reference_labels.csv").exists():
        routes["route_b_reference"] = pd.read_csv(DATA / "reference_labels.csv")
    if (DATA / "vote_shares.csv").exists():
        vs = pd.read_csv(DATA / "vote_shares.csv")
        thr = vs.copy()
        for m in MECHANISMS:
            thr[m] = (vs[m] >= 0.5).astype(int)
        routes["route_b_vote_half"] = thr
    if (DATA / "fusion.csv").exists():
        routes["fusion_automated"] = pd.read_csv(DATA / "fusion.csv")

    results: dict[str, dict] = {}
    for name, frame in routes.items():
        merged = gold.merge(frame, on="Crash_ID", how="inner", suffixes=("_gold", "_pred"))
        labs = [l for l in ALL_LABELS if f"{l}_pred" in merged.columns
                or (l in frame.columns and f"{l}_gold" in merged.columns)]
        per_label: dict[str, dict] = {}
        for lab in labs:
            truth = merged[f"{lab}_gold"].to_numpy().astype(int)
            pred = merged[f"{lab}_pred"].to_numpy().astype(int)
            w1 = np.ones(len(truth))
            wv = merged["Crash_ID"].map(weights).fillna(1.0).to_numpy()
            per_label[lab] = {
                "unweighted": {**_counts(truth, pred, w1),
                               "ci95": _bootstrap(truth, pred, w1)},
                "ip_weighted": {**_counts(truth, pred, wv),
                                "ci95": _bootstrap(truth, pred, wv)},
                "gold_positives": int(truth.sum()),
                "predicted_positives": int(pred.sum()),
            }
        # Snapshot the per-label entries before adding any macro row: the second pass
        # would otherwise read the macro row written by the first and fail on it.
        label_rows = list(per_label.values())
        for kind in ("unweighted", "ip_weighted"):
            f1s = [v[kind]["f1"] for v in label_rows
                   if kind in v and v[kind]["f1"] == v[kind]["f1"]]
            per_label[f"macro_{kind}"] = {"macro_f1": round(float(np.mean(f1s)), 4)
                                          if f1s else float("nan")}
        results[name] = per_label
    return results


def run(batch: int = 1) -> dict:
    long, report = merge_coders(batch)

    # Agreement among three coders means three coders. Narratives that any coder has not
    # yet reached are held out of the primary statistics and reported as outstanding,
    # rather than being scored on whoever happened to finish. On batch 1 this is the
    # difference between a macro Fleiss kappa of 0.859 and 0.919, and the lower number is
    # an artefact of counting one coder's unreached rows as negative judgements.
    full_ids = long.groupby("crash_id")["_coded"].sum()
    full_ids = set(full_ids[full_ids == len(report["files"])].index)
    held_out = sorted(set(long["crash_id"]) - full_ids)
    long = long[long["crash_id"].isin(full_ids)].copy()
    report["held_out_pending_completion"] = len(held_out)
    report["held_out_ids"] = held_out
    gold, detail = gold_labels(long)
    gold.to_csv(DATA / "gold_labels.csv", index=False)

    out = {
        "merge_report": report,
        "gold_label_detail": detail,
        "agreement": agreement(long),
        "routes": score_routes(gold),
    }
    # One file per batch. A single shared filename meant running batch 2 silently
    # replaced batch 1's reliability statistics with a purposive sample of 37 hard cases,
    # and nothing downstream could tell. `validation_metrics.json` is kept as an alias
    # for batch 1, which is the probability sample and the reliability estimate.
    (DATA / f"validation_metrics_batch{batch}.json").write_text(
        json.dumps(out, indent=2, default=float), encoding="utf-8")
    if batch == 1:
        (DATA / "validation_metrics.json").write_text(
            json.dumps(out, indent=2, default=float), encoding="utf-8")
    return out


if __name__ == "__main__":
    import sys

    batch = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    res = run(batch)
    print(json.dumps(res["merge_report"], indent=2))
    print("\nFleiss kappa by label:")
    for lab, d in res["agreement"].items():
        print(f"  {lab:24s} {d['fleiss_kappa']}")
    for route, per in res["routes"].items():
        macro = per.get("macro_unweighted", {}).get("macro_f1")
        macro_w = per.get("macro_ip_weighted", {}).get("macro_f1")
        print(f"\n{route}: macro F1 unweighted {macro}, IP-weighted {macro_w}")
