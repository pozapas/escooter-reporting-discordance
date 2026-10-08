"""The four-level narrative indicator, and the raw cross-tabulation.

The mapping is fixed in advance. The narrative indicator
N takes four observed levels:

* **KA-consistent** — a fatality, fracture, or unconsciousness cue is present;
* **O-consistent** — an explicit no-injury cue is present and no transport cue;
* **BC-consistent** — a transport cue only;
* **no cue** — otherwise.

Two things about this module are deliberate.

First, it produces the **raw cross-tabulation before any model runs**. It is the first
table of the audit results, and it is the one part of
the audit that involves no priors, no latent class, and no assumptions: it is two
observed columns of the same police record, counted. A reader who distrusts everything
downstream can still read it.

Second, the language. This compares two indicators drawn from the *same source* — the
officer's KABCO code and the officer's own narrative. Neither is a measurement of
injury. Disagreement between them is **discordance**, not error, and nothing here
licenses calling one of them right. That is
why the mapping is fixed in advance and reported as a table rather than tuned.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

N_LEVELS = ("no cue", "O-consistent", "BC-consistent", "KA-consistent")
KABCO_ORDER = ("O", "C", "B", "A", "K")
SEV3_ORDER = ("O", "BC", "KA")

# The mapping table reproduced in section 4. Order matters: the first row
# whose condition holds assigns the level, so a narrative carrying both a fracture cue
# and a no-injury cue is KA-consistent. That precedence is stated rather than left to
# the reader to infer from code.
MAPPING = (
    ("KA-consistent", "fatality, fracture, or unconsciousness cue present",
     "Severe or fatal injury is described explicitly."),
    ("O-consistent", "explicit no-injury cue present and no transport cue",
     "The narrative states that no one was injured and describes no transport."),
    ("BC-consistent", "transport cue present, no severe cue",
     "Transport to hospital is described without any severe-injury language."),
    ("no cue", "none of the above",
     "The narrative carries no injury-outcome language of any kind."),
)


def build_indicator(labels: pd.DataFrame) -> pd.Series:
    """Assign the four-level narrative indicator, in the precedence of `MAPPING`."""
    severe = (labels["fatality"].astype(bool)
              | labels["fracture"].astype(bool)
              | labels["unconsciousness"].astype(bool))
    transport = labels["transported_hospital"].astype(bool)
    no_injury = labels["explicit_no_injury"].astype(bool)

    n = pd.Series("no cue", index=labels.index, dtype=object)
    n[transport & ~severe] = "BC-consistent"
    n[no_injury & ~transport & ~severe] = "O-consistent"
    n[severe] = "KA-consistent"
    return pd.Categorical(n, categories=list(N_LEVELS), ordered=False)


def conflicts(labels: pd.DataFrame) -> dict:
    """Narratives whose cues point two ways at once.

    The precedence rule resolves these silently, so they are counted and reported. A
    narrative that records both a hospital transport and an explicit statement of no
    injury is not a coding mistake — a rider can decline treatment at the scene and be
    transported later, and reports do correct themselves mid-narrative — but the reader
    should know how often the mapping had to choose.
    """
    severe = (labels["fatality"].astype(bool)
              | labels["fracture"].astype(bool)
              | labels["unconsciousness"].astype(bool))
    transport = labels["transported_hospital"].astype(bool)
    no_injury = labels["explicit_no_injury"].astype(bool)
    return {
        "no_injury_and_transport": int((no_injury & transport).sum()),
        "no_injury_and_severe": int((no_injury & severe).sum()),
        "severe_without_transport": int((severe & ~transport).sum()),
        "note": ("Resolved by the precedence in MAPPING: a severe cue wins over "
                 "everything, and a transport cue wins over an explicit no-injury cue."),
    }


def cross_tabulation(frame: pd.DataFrame, coded: str = "sev_kabco",
                     order: tuple[str, ...] = KABCO_ORDER) -> dict:
    """Coded severity by narrative indicator, with row percentages."""
    ct = pd.crosstab(frame[coded], frame["narrative_indicator"])
    ct = ct.reindex(index=[c for c in order if c in ct.index],
                    columns=list(N_LEVELS)).fillna(0).astype(int)
    row_pct = ct.div(ct.sum(axis=1).replace(0, np.nan), axis=0) * 100

    return {
        "counts": ct.to_dict(orient="index"),
        "row_percent": row_pct.round(1).to_dict(orient="index"),
        "row_totals": ct.sum(axis=1).to_dict(),
        "column_totals": ct.sum(axis=0).to_dict(),
        "n": int(ct.to_numpy().sum()),
    }


def _render(ct: dict, order: tuple[str, ...], title: str) -> str:
    lines = [title, ""]
    head = f"{'coded':<8}" + "".join(f"{c:>16}" for c in N_LEVELS) + f"{'total':>8}"
    lines.append(head)
    lines.append("-" * len(head))
    for lvl in order:
        if lvl not in ct["counts"]:
            continue
        row = ct["counts"][lvl]
        pct = ct["row_percent"][lvl]
        cells = "".join(
            f"{row[c]:>9} {('(' + format(pct[c], '.1f') + ')'):>6}" for c in N_LEVELS)
        lines.append(f"{lvl:<8}" + cells + f"{ct['row_totals'][lvl]:>8}")
    lines.append("-" * len(head))
    lines.append(f"{'total':<8}" + "".join(
        f"{ct['column_totals'][c]:>16}" for c in N_LEVELS) + f"{ct['n']:>8}")
    return "\n".join(lines)


def build() -> dict:
    # The four-level indicator is built from the five injury cues, so it must use the
    # same labels the models use: the adjudicated human panel where one exists and the
    # Route B reference run elsewhere. Building it from the reference run alone would
    # mean the audit and the severity models disagreed about what a narrative says.
    final = DATA / "final_labels.csv"
    labels = pd.read_csv(final if final.exists() else DATA / "reference_labels.csv")
    if "Crash_ID" in labels.columns and "crash_id" not in labels.columns:
        labels = labels.rename(columns={"Crash_ID": "Crash_ID"})
    label_source = "final adjudicated" if final.exists() else "reference run only"
    frame = pd.read_csv(DATA / "model_frame.csv")

    labels = labels.copy()
    labels["narrative_indicator"] = build_indicator(labels)
    merged = frame.merge(
        labels[["Crash_ID", "narrative_indicator"]], on="Crash_ID", how="left")

    unmatched = int(merged["narrative_indicator"].isna().sum())
    if unmatched:
        raise SystemExit(
            f"{unmatched} rider rows have no narrative indicator; the label file and "
            f"the model frame disagree on crash identifiers")

    out = {
        "mapping": [{"level": a, "condition": b, "reading": c} for a, b, c in MAPPING],
        "levels": list(N_LEVELS),
        "n_rider_rows": int(len(merged)),
        "n_distinct_crashes": int(merged["Crash_ID"].nunique()),
        "indicator_counts": merged["narrative_indicator"].value_counts()
                            .reindex(N_LEVELS).fillna(0).astype(int).to_dict(),
        "cue_conflicts": conflicts(labels),
        "cross_tab_kabco": cross_tabulation(merged, "sev_kabco", KABCO_ORDER),
        "cross_tab_sev3": cross_tabulation(merged, "sev3", SEV3_ORDER),
        "label_source": label_source,
        "labels_are_preliminary": label_source != "final adjudicated",
        "note": ("Built from the cue labels the severity models use. The "
                 "indicator compares two fields of the same police record, the "
                 "officer's KABCO code and the officer's narrative, so disagreement "
                 "between them is discordance and not evidence that either is wrong."),
    }

    merged[["Crash_ID", "sev3", "sev_kabco", "narrative_indicator"]].to_csv(
        DATA / "narrative_indicator.csv", index=False)
    (DATA / "audit_crosstab.json").write_text(
        json.dumps(out, indent=2, default=str), encoding="utf-8")
    return out


if __name__ == "__main__":
    out = build()
    print("Narrative indicator. PRELIMINARY: built from the reference run, not the")
    print("coders' adjudicated labels.")
    print()
    print(f"{out['n_rider_rows']} rider rows, "
          f"{out['n_distinct_crashes']} distinct crashes")
    print()
    print("mapping (precedence order):")
    for row in out["mapping"]:
        print(f"  {row['level']:<15} {row['condition']}")
    print()
    print("indicator distribution:")
    for lvl, k in out["indicator_counts"].items():
        print(f"  {lvl:<15} {k:>4}  ({100 * k / out['n_rider_rows']:.1f}%)")
    print()
    print("cues pointing two ways, resolved by precedence:")
    for k, v in out["cue_conflicts"].items():
        if k != "note":
            print(f"  {k:<28} {v}")
    print()
    print(_render(out["cross_tab_kabco"], KABCO_ORDER,
                  "Coded KABCO by narrative indicator, count (row %)"))
    print()
    print(_render(out["cross_tab_sev3"], SEV3_ORDER,
                  "Collapsed three-class severity by narrative indicator, count (row %)"))
    print()
    print(f"written to {DATA / 'audit_crosstab.json'} and "
          f"{DATA / 'narrative_indicator.csv'}")
