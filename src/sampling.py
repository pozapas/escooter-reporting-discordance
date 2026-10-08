"""Frozen samples: the 50 guide-development narratives and the 150 validation narratives.

Both draws are seeded and written to `data/`. The guide-development set is drawn
first and excluded from the validation set, so no narrative that shaped the coding
guide or the lexical rules can also serve as a validation record.

The validation draw needs the reference-run labels and is therefore run after
extraction; the guide draw needs nothing but the base frame.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from labels import MECHANISMS

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

PILOT_SEED = 20260914
PILOT_N = 50

VALIDATION_SEED = 20260915
VALIDATION_N = 150
PER_MECHANISM_TARGET = 15


def draw_pilot() -> pd.DataFrame:
    """Fifty narratives drawn at random, frozen as the guide-development set."""
    base = pd.read_csv(DATA / "base_frame.csv")
    crashes = (
        base.drop_duplicates(subset="Crash_ID")
        .loc[:, ["Crash_ID", "sev3", "narrative", "narrative_chars"]]
        .sort_values("Crash_ID")
        .reset_index(drop=True)
    )
    rng = np.random.default_rng(PILOT_SEED)
    idx = rng.choice(len(crashes), size=PILOT_N, replace=False)
    pilot = crashes.iloc[np.sort(idx)].copy()
    pilot.to_csv(DATA / "pilot_ids.csv", index=False)
    return pilot


def draw_validation(labels_path: Path | None = None) -> pd.DataFrame:
    """One hundred and fifty narratives stratified on the reference-run labels.

    All predicted dooring positives enter. For every other mechanism, up to
    ``PER_MECHANISM_TARGET`` predicted positives are drawn at random. The remainder
    is filled by simple random sampling from records not already selected. The
    guide-development narratives are excluded throughout.

    The inclusion probability of every crash under this rule is recorded in
    `validation_ids.csv` as ``sampling_weight`` (the reciprocal of the selection
    probability), so that the validation metrics can be reported with
    inverse-probability weights as well as unweighted.
    """
    labels_path = labels_path or (DATA / "reference_labels.csv")
    base = pd.read_csv(DATA / "base_frame.csv").drop_duplicates(subset="Crash_ID")
    lab = pd.read_csv(labels_path)
    pilot_ids = set(pd.read_csv(DATA / "pilot_ids.csv")["Crash_ID"])

    frame = base[["Crash_ID", "sev3", "narrative_chars"]].merge(
        lab, on="Crash_ID", how="inner", validate="one_to_one"
    )
    eligible = frame[~frame["Crash_ID"].isin(pilot_ids)].sort_values("Crash_ID").reset_index(drop=True)

    rng = np.random.default_rng(VALIDATION_SEED)
    chosen: list[int] = []
    strata: dict[int, str] = {}

    # Stratum 1: every predicted dooring positive.
    dooring = eligible.loc[eligible["dooring"] == 1, "Crash_ID"].tolist()
    for cid in dooring:
        chosen.append(cid)
        strata[cid] = "dooring_all"

    # Stratum 2: up to PER_MECHANISM_TARGET predicted positives per other mechanism.
    stratum_sizes: dict[str, dict[str, int]] = {}
    for mech in MECHANISMS:
        if mech == "dooring":
            continue
        pool = eligible.loc[
            (eligible[mech] == 1) & (~eligible["Crash_ID"].isin(chosen)), "Crash_ID"
        ].to_numpy()
        take = min(PER_MECHANISM_TARGET, len(pool))
        stratum_sizes[mech] = {"pool": int(len(pool)), "drawn": int(take)}
        if take:
            picks = rng.choice(pool, size=take, replace=False)
            for cid in picks:
                chosen.append(int(cid))
                strata[int(cid)] = f"positive_{mech}"

    # Stratum 3: simple random fill from the remainder.
    remainder = eligible.loc[~eligible["Crash_ID"].isin(chosen), "Crash_ID"].to_numpy()
    fill = VALIDATION_N - len(chosen)
    if fill < 0:
        raise ValueError(
            f"Stratified strata already exceed {VALIDATION_N} records ({len(chosen)}); "
            "narrow PER_MECHANISM_TARGET and record the deviation."
        )
    fill_picks = rng.choice(remainder, size=fill, replace=False) if fill else np.array([])
    for cid in fill_picks:
        chosen.append(int(cid))
        strata[int(cid)] = "random_fill"

    # --- inclusion probabilities ---------------------------------------------
    # A record's selection probability is one minus the probability it is missed by
    # every stratum it belongs to. Dooring positives are certainties.
    n_remainder_pool = int(len(remainder)) + 0
    probs: dict[int, float] = {}
    for _, row in eligible.iterrows():
        cid = int(row["Crash_ID"])
        if row["dooring"] == 1:
            probs[cid] = 1.0
            continue
        p_miss = 1.0
        for mech in MECHANISMS:
            if mech == "dooring" or row[mech] != 1:
                continue
            info = stratum_sizes[mech]
            if info["pool"]:
                p_miss *= 1.0 - info["drawn"] / info["pool"]
        # Probability of being reached by the random fill, approximated by the
        # realized fill fraction over the pool that reached stratum 3.
        p_fill = (fill / n_remainder_pool) if n_remainder_pool else 0.0
        p_miss *= 1.0 - p_fill
        probs[cid] = max(1.0 - p_miss, 1e-6)

    out = eligible[eligible["Crash_ID"].isin(chosen)][
        ["Crash_ID", "sev3", "narrative_chars"]
    ].copy()
    out["stratum"] = out["Crash_ID"].map(strata)
    out["inclusion_prob"] = out["Crash_ID"].map(probs)
    out["sampling_weight"] = 1.0 / out["inclusion_prob"]
    out = out.sort_values("Crash_ID").reset_index(drop=True)

    if len(out) != VALIDATION_N:
        raise ValueError(f"Validation sample is {len(out)} records, expected {VALIDATION_N}.")

    out.to_csv(DATA / "validation_ids.csv", index=False)
    with open(DATA / "validation_sampling_audit.json", "w", encoding="utf-8") as fh:
        json.dump(
            {
                "seed": VALIDATION_SEED,
                "n": VALIDATION_N,
                "per_mechanism_target": PER_MECHANISM_TARGET,
                "eligible_n": int(len(eligible)),
                "pilot_excluded_n": int(len(pilot_ids)),
                "dooring_all_n": int(len(dooring)),
                "stratum_sizes": stratum_sizes,
                "random_fill_n": int(fill),
                "stratum_counts": out["stratum"].value_counts().to_dict(),
            },
            fh,
            indent=2,
        )
    return out


if __name__ == "__main__":
    pilot = draw_pilot()
    print(f"Guide-development sample frozen: {len(pilot)} narratives -> data/pilot_ids.csv")
    print(pilot["sev3"].value_counts().to_dict())
