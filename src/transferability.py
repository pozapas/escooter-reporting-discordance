"""Transferability of the severity specification across time and space.

The question is whether one pooled model is defensible
across five years and across Texas. The test is a likelihood-ratio comparison of a pooled
model against period-specific models, and the honest version of it has three parts the
usual version omits.

**The reference distribution is simulated, not assumed.** With 522 records split in two,
the asymptotic chi-squared reference for a structural-break statistic is not trustworthy —
the same problem that made the Small-Hsiao test uninterpretable in the appendix
specification. So the p-value comes from a parametric bootstrap: data are generated from
the pooled model, the split is refitted, and the observed statistic is placed in that
distribution. If the asymptotic and bootstrap p-values disagree, the bootstrap governs and
both are reported.

**Covariates are screened for estimability inside every subsample, not only in the pooled
data.** A covariate with ten events overall can have zero in one period, and a model
fitted with it is not comparable to one fitted without. The common estimable set is
computed per breakpoint and reported, so a reader can see what the test was actually
performed on.

**A breakpoint that cannot be tested is reported as untestable rather than forced.** The
2021/2022 split is the clearest case: 2021 contains thirty crashes and **not one of them
is coded O**. A period with an empty outcome category cannot support an ordered model, and
that emptiness is itself a finding — it belongs in the reporting-discordance discussion
rather than being patched over with a merge.

Nothing here is causal. A rejected pooled specification says the coefficients differ
across periods; it does not say why, and the composition of reporting cities changes over
the same window.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

import severity as sv

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

BREAKPOINTS = (2023, 2024, 2025)
UNTESTABLE_NOTE = (
    "2021 contains 30 crashes and none coded O, so a 2021/2022 split has an empty "
    "outcome category in the earlier period and no ordered model can be fitted to it. "
    "Reported as untestable rather than forced by merging categories."
)
BOOTSTRAP_REPS = 500
BOOTSTRAP_SEED = 20260914
MIN_CELL_SUBSAMPLE = 5


def annual_table(frame: pd.DataFrame) -> dict:
    """Crashes, severity composition and principal covariates by year."""
    by_year = pd.crosstab(frame["crash_year"], frame["sev3"])
    by_year = by_year.reindex(columns=list(sv.SEV_ORDER)).fillna(0).astype(int)
    total = by_year.sum(axis=1)
    share = (by_year.div(total, axis=0) * 100).round(1)

    principal = ["nighttime", "weekend", "speed_ge45", "intx_intersection_or_driveway",
                 "age_under25", "gender_female", "coord_missing", "poverty_share"]
    covars = {c: frame.groupby("crash_year")[c].mean().round(3).to_dict()
              for c in principal if c in frame.columns}

    return {
        "counts": by_year.to_dict(orient="index"),
        "totals": total.to_dict(),
        "severity_share_percent": share.to_dict(orient="index"),
        "principal_covariate_means": covars,
        "note": ("2021 carries no O-coded crash at all. That is reported rather than "
                 "smoothed, and it is the reason the 2021/2022 breakpoint is untestable."),
    }


def city_composition(frame: pd.DataFrame, breakpoint: int, top: int = 6) -> dict:
    """Which cities report, before and after a breakpoint."""
    before = frame.loc[frame["crash_year"] < breakpoint, "City_ID"]
    after = frame.loc[frame["crash_year"] >= breakpoint, "City_ID"]
    names = [c for c, _ in frame["City_ID"].value_counts().head(top).items()]
    return {
        "breakpoint": breakpoint,
        "n_before": int(len(before)), "n_after": int(len(after)),
        "share_before_percent": {c: round(100 * float((before == c).mean()), 1)
                                 for c in names},
        "share_after_percent": {c: round(100 * float((after == c).mean()), 1)
                                for c in names},
        "distinct_cities_before": int(before.nunique()),
        "distinct_cities_after": int(after.nunique()),
    }


def estimable_in_both(frame: pd.DataFrame, columns: list[str], mask: np.ndarray,
                      min_cell: int = MIN_CELL_SUBSAMPLE) -> tuple[list[str], dict]:
    """Covariates that carry enough events in both subsamples to be comparable."""
    keep, dropped = [], {}
    for col in columns:
        ok = True
        reason = []
        for label, sub in (("before", frame[mask]), ("after", frame[~mask])):
            vals = sub[col].dropna()
            if set(np.unique(vals)) <= {0.0, 1.0}:
                cells = pd.crosstab(sub[col], sub["sev3"])
                smallest = int(cells.to_numpy().min()) if cells.size else 0
                # An indicator that is 1 on every row of a subsample is a constant there
                # and is not identified beside the cutpoints (the urban-area indicator in
                # the six-city subsample), so both levels must be present.
                if 1 not in cells.index or 0 not in cells.index or smallest < min_cell:
                    ok = False
                    reason.append(f"{label}: smallest cell {smallest}")
            elif vals.std() == 0:
                ok = False
                reason.append(f"{label}: no variation")
        if ok:
            keep.append(col)
        else:
            dropped[col] = "; ".join(reason)
    return keep, dropped


def _split_loglik(frame: pd.DataFrame, columns: list[str],
                  mask: np.ndarray) -> tuple[float, bool]:
    total, converged = 0.0, True
    for sub in (frame[mask], frame[~mask]):
        fit = sv.fit_ordered_logit(sub, columns, "split")
        total += fit.loglik
        converged = converged and fit.converged
    return total, converged


def _simulate_from_pooled(frame: pd.DataFrame, columns: list[str],
                          fit: sv.OrdinalFit, rng) -> pd.DataFrame:
    """Draw a new outcome column from the pooled model — the bootstrap null."""
    X, _ = sv._design(frame, columns)
    probs = sv._probs(X, fit.beta, fit.cut)
    draws = np.array([rng.choice(len(sv.SEV_ORDER), p=p / p.sum()) for p in probs])
    out = frame.copy()
    out["sev3"] = [sv.SEV_ORDER[i] for i in draws]
    return out


def temporal_test(frame: pd.DataFrame, columns: list[str], breakpoint: int,
                  reps: int = BOOTSTRAP_REPS, seed: int = BOOTSTRAP_SEED) -> dict:
    mask = (frame["crash_year"] < breakpoint).to_numpy()
    keep, dropped = estimable_in_both(frame, columns, mask)

    counts = {
        "before": pd.crosstab(frame.loc[mask, "sev3"], columns="n")["n"]
                  .reindex(sv.SEV_ORDER).fillna(0).astype(int).to_dict(),
        "after": pd.crosstab(frame.loc[~mask, "sev3"], columns="n")["n"]
                 .reindex(sv.SEV_ORDER).fillna(0).astype(int).to_dict(),
    }
    if min(counts["before"].values()) == 0 or min(counts["after"].values()) == 0:
        return {"breakpoint": breakpoint, "testable": False,
                "severity_counts": counts,
                "reason": "an outcome category is empty in one period"}

    pooled = sv.fit_ordered_logit(frame, keep, f"pooled_{breakpoint}")
    split_ll, split_ok = _split_loglik(frame, keep, mask)
    stat = 2 * (split_ll - pooled.loglik)
    df = len(keep) + len(sv.SEV_ORDER) - 1
    p_asym = float(1 - stats.chi2.cdf(stat, df=df))

    rng = np.random.default_rng(seed)
    null = []
    for _ in range(reps):
        sim = _simulate_from_pooled(frame, keep, pooled, rng)
        try:
            p_fit = sv.fit_ordered_logit(sim, keep, "boot_pooled")
            s_ll, _ = _split_loglik(sim, keep, mask)
            null.append(2 * (s_ll - p_fit.loglik))
        except Exception:                                         # noqa: BLE001
            continue

    null = np.array(null)
    p_boot = float((null >= stat).sum() + 1) / (len(null) + 1) if len(null) else None

    return {
        "breakpoint": breakpoint,
        "testable": True,
        "n_before": int(mask.sum()), "n_after": int((~mask).sum()),
        "severity_counts": counts,
        "covariates_tested": len(keep),
        "covariates_dropped": dropped,
        "estimable_parameters_per_period": df,
        "both_periods_converged": split_ok,
        "lr_statistic": round(float(stat), 3),
        "df": df,
        "p_asymptotic": round(p_asym, 5),
        "bootstrap_replications_usable": int(len(null)),
        "p_bootstrap": round(p_boot, 5) if p_boot is not None else None,
        "bootstrap_null_median": round(float(np.median(null)), 2) if len(null) else None,
        "rejects_at_5pct_bootstrap": bool(p_boot is not None and p_boot < 0.05),
        "note": ("The bootstrap p-value governs. The asymptotic reference assumes a "
                 "sample size this design does not have, and where the two disagree "
                 "the asymptotic one is the optimistic one."),
    }


def spatial_test(frame: pd.DataFrame, columns: list[str],
                 reps: int = BOOTSTRAP_REPS, seed: int = BOOTSTRAP_SEED) -> dict:
    """Urbanized against the rest. Exploratory, and labelled so."""
    big = {"Austin", "Houston", "San Antonio", "Dallas", "Fort Worth", "El Paso"}
    mask = frame["City_ID"].isin(big).to_numpy()
    keep, dropped = estimable_in_both(frame, columns, mask)

    pooled = sv.fit_ordered_logit(frame, keep, "pooled_spatial")
    split_ll, split_ok = _split_loglik(frame, keep, mask)
    stat = 2 * (split_ll - pooled.loglik)
    df = len(keep) + len(sv.SEV_ORDER) - 1

    rng = np.random.default_rng(seed)
    null = []
    for _ in range(reps):
        sim = _simulate_from_pooled(frame, keep, pooled, rng)
        try:
            p_fit = sv.fit_ordered_logit(sim, keep, "boot")
            s_ll, _ = _split_loglik(sim, keep, mask)
            null.append(2 * (s_ll - p_fit.loglik))
        except Exception:                                         # noqa: BLE001
            continue
    null = np.array(null)
    p_boot = float((null >= stat).sum() + 1) / (len(null) + 1) if len(null) else None

    return {
        "split": "six most populous cities against the rest",
        "exploratory": True,
        "n_large_city": int(mask.sum()), "n_rest": int((~mask).sum()),
        "covariates_tested": len(keep),
        "covariates_dropped": dropped,
        "both_converged": split_ok,
        "lr_statistic": round(float(stat), 3),
        "df": df,
        "p_asymptotic": round(float(1 - stats.chi2.cdf(stat, df=df)), 5),
        "p_bootstrap": round(p_boot, 5) if p_boot is not None else None,
        "rejects_at_5pct_bootstrap": bool(p_boot is not None and p_boot < 0.05),
    }


if __name__ == "__main__":
    frame = pd.read_csv(DATA / "model_frame.csv")
    cols = sv.ladder(frame)["M1"]

    print("Transferability. PRELIMINARY: cues are the fused reference-run labels.")
    print()

    out: dict = {"annual": annual_table(frame),
                 "untestable_breakpoints": {"2022": UNTESTABLE_NOTE}}

    ann = out["annual"]
    print("crashes by year and severity")
    for yr in sorted(ann["totals"]):
        c = ann["counts"][yr]
        print(f"  {yr}  O {c['O']:>3}  BC {c['BC']:>3}  KA {c['KA']:>3}   "
              f"total {ann['totals'][yr]:>3}")
    print(f"  note: {ann['note']}")
    print()

    out["temporal"] = {}
    for bp in BREAKPOINTS:
        res = temporal_test(frame, cols, bp)
        out["temporal"][str(bp)] = res
        if not res["testable"]:
            print(f"break at {bp}: NOT TESTABLE — {res['reason']}")
            continue
        print(f"break at {bp}: n {res['n_before']} vs {res['n_after']}, "
              f"{res['covariates_tested']} covariates tested "
              f"({len(res['covariates_dropped'])} dropped as not estimable in both)")
        print(f"   LR {res['lr_statistic']} on {res['df']} df | "
              f"asymptotic p {res['p_asymptotic']} | "
              f"bootstrap p {res['p_bootstrap']} "
              f"({res['bootstrap_replications_usable']} usable replications)")
        print(f"   rejects the pooled specification: "
              f"{res['rejects_at_5pct_bootstrap']}")
        out["temporal"][str(bp)]["city_composition"] = city_composition(frame, bp)

    print()
    sp = spatial_test(frame, cols)
    out["spatial"] = sp
    print(f"spatial (exploratory): {sp['n_large_city']} large-city vs {sp['n_rest']} rest, "
          f"{sp['covariates_tested']} covariates")
    print(f"   LR {sp['lr_statistic']} on {sp['df']} df | asymptotic p "
          f"{sp['p_asymptotic']} | bootstrap p {sp['p_bootstrap']} | "
          f"rejects {sp['rejects_at_5pct_bootstrap']}")

    path = DATA / "transferability.json"
    path.write_text(json.dumps(out, indent=2, default=float), encoding="utf-8")
    print()
    print(f"written to {path}")
