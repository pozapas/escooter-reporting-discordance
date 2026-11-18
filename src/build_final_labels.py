"""Assemble the final cue labels the severity models use.

`prespec.md` section 8 says the primary models use the final adjudicated hard 0/1 cue
labels. Those come from three sources, and every narrative records which one it got, so
that no reader has to guess how any single label was produced:

* **human** — the 150 narratives of the validation sample and the 37 of the adjudication
  round each carry three independent codings; the label is the majority of three.
* **automated** — every remaining narrative takes the Route B reference run, which scored
  highest against the human labels.

The split is not incidental to the paper's argument, so it is counted and reported rather
than buried: 187 of 520 narratives carry a human label and 333 do not, and the severity
estimates rest on a mixture whose error properties differ between the two parts. That is
what the pre-specified sensitivities in section 8 exist to probe, and the provenance
column here is what makes them possible to run.

Two decisions are worth stating because a reader could reasonably expect the opposite.

**Route B alone, not the fusion.** The fusion rule was pre-specified and it is what
selected the adjudication set, which was its job. But measured against the human labels it
scores *below* Route B on its own — 0.779 against 0.811 macro F1 — so using it for the
narratives with no human label would knowingly choose the weaker measurement. The fusion's
role was to decide which narratives needed a human, and it did that; it is not used as a
label source.

**Ties go to zero.** Three coders cannot tie, so this only matters if a coding is ever
missing, and a two-to-two or one-to-one split resolves negative. A cue is recorded when
the narrative says so; silence is not evidence.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from labels import ALL_LABELS

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

ACTOR_FIELDS = ("failure_to_yield_actor", "signal_violation_actor")


def _panel(batch: int) -> pd.DataFrame | None:
    path = DATA / f"validation_codes_batch{batch}.csv"
    if not path.exists():
        return None
    long = pd.read_csv(path)
    if "_coded" in long.columns:
        long = long[long["_coded"].astype(bool)]
    return long


def majority(long: pd.DataFrame) -> pd.DataFrame:
    """Majority of the coders who actually coded each narrative."""
    out = pd.DataFrame(index=sorted(long["crash_id"].unique()))
    out.index.name = "Crash_ID"

    pivot = long.pivot_table(index="crash_id", columns="coder_id",
                             values=list(ALL_LABELS), aggfunc="first")
    for lab in ALL_LABELS:
        votes = pivot[lab]
        n = votes.notna().sum(axis=1)
        pos = votes.fillna(0).sum(axis=1)
        out[lab] = (pos > n / 2).astype(int).reindex(out.index).fillna(0).astype(int)
        out[f"{lab}__n_coders"] = n.reindex(out.index).fillna(0).astype(int)
        out[f"{lab}__unanimous"] = ((pos == 0) | (pos == n)).reindex(
            out.index).fillna(False).astype(int)

    # Actor attribution: the majority string among coders who gave one.
    for field in ACTOR_FIELDS:
        if field not in long.columns:
            continue
        got = long.dropna(subset=[field])
        if got.empty:
            out[field] = np.nan
            continue
        mode = (got.groupby("crash_id")[field]
                   .agg(lambda s: s.value_counts().idxmax()))
        out[field] = mode.reindex(out.index)
    return out


def build() -> dict:
    frames, report = [], {}
    for batch in (1, 2):
        long = _panel(batch)
        if long is None:
            report[f"batch{batch}"] = "no merged codings found"
            continue
        m = majority(long)
        m["label_source"] = "human"
        m["label_batch"] = batch
        frames.append(m)
        report[f"batch{batch}_human_labelled"] = int(len(m))

    if not frames:
        raise SystemExit("no human codings available; run validation_metrics.py first")

    human = pd.concat(frames)
    dupes = human.index.duplicated()
    if dupes.any():
        # A narrative in both rounds would otherwise be counted twice. The batch-2 queue
        # was built to exclude these, so this is a guard rather than an expectation.
        report["duplicate_across_batches"] = int(dupes.sum())
        human = human[~dupes]

    auto = pd.read_csv(DATA / "reference_labels.csv").set_index("Crash_ID")
    base = pd.read_csv(DATA / "base_frame.csv").drop_duplicates(subset="Crash_ID")
    all_ids = sorted(base["Crash_ID"].astype(int))

    rows = []
    for cid in all_ids:
        if cid in human.index:
            r = human.loc[cid]
            rec = {"Crash_ID": cid, "label_source": "human",
                   "label_batch": int(r["label_batch"])}
            for lab in ALL_LABELS:
                rec[lab] = int(r[lab])
            for f in ACTOR_FIELDS:
                rec[f] = r.get(f, np.nan)
        elif cid in auto.index:
            r = auto.loc[cid]
            rec = {"Crash_ID": cid, "label_source": "automated", "label_batch": np.nan}
            for lab in ALL_LABELS:
                rec[lab] = int(r[lab]) if lab in auto.columns else 0
            for f in ACTOR_FIELDS:
                rec[f] = r.get(f, np.nan)
        else:
            continue
        rows.append(rec)

    final = pd.DataFrame(rows)
    final.to_csv(DATA / "final_labels.csv", index=False)

    counts = final["label_source"].value_counts().to_dict()
    report.update({
        "narratives": int(len(final)),
        "by_source": counts,
        "human_share": round(float(counts.get("human", 0) / len(final)), 4),
        "automated_source": "route_b_reference",
        "why_not_fusion": ("the fusion rule scores 0.779 macro F1 against the human "
                           "labels against Route B's 0.811, so using it where no human "
                           "label exists would choose the weaker measurement; its role "
                           "was to select the adjudication set, which it did"),
        "rate_by_label": {lab: {
            "human_subset": round(float(
                final.loc[final.label_source == "human", lab].mean()), 4),
            "automated_subset": round(float(
                final.loc[final.label_source == "automated", lab].mean()), 4),
            "overall": round(float(final[lab].mean()), 4),
        } for lab in ALL_LABELS},
    })
    (DATA / "final_labels_report.json").write_text(
        json.dumps(report, indent=2, default=str), encoding="utf-8")
    return report


if __name__ == "__main__":
    rep = build()
    print("Final cue labels for the severity models.")
    print()
    print(f"{rep['narratives']} narratives: {rep['by_source']}")
    print(f"human-labelled share: {rep['human_share']:.1%}")
    print()
    print(f"{'cue':<26}{'human':>9}{'automated':>12}{'overall':>10}")
    for lab, v in rep["rate_by_label"].items():
        print(f"{lab:<26}{v['human_subset']:>8.1%}{v['automated_subset']:>12.1%}"
              f"{v['overall']:>10.1%}")
    print()
    print("The human and automated subsets are not random halves: the human set is the")
    print("validation sample plus the narratives the routes disagreed on, so a rate")
    print("difference between the columns is expected and is not by itself evidence of")
    print("bias in either.")
    print()
    print(f"written to {DATA / 'final_labels.csv'}")
