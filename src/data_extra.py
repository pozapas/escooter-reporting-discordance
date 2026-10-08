"""Data-side sensitivity analyses and one pre-specified item.

Five items, each written to its own top-level key of `data/data_extra.json`:

* ``actor_attribution``  - prespec section 8b. Who the narrative says failed to yield or
  violated the signal, by severity class; agreement of the automated attribution with the
  coders; and the pre-specified gate on whether a rider/driver/unclear split may enter M1.
* ``multiple_imputation`` - prespec section 2.3. Chained-equations imputation (m = 20) of
  rider age, gender, ethnicity and posted speed limit, the primary ordered logit on every
  completed data set, and Rubin's rules.
* ``unknown_ethnicity``  - M1 refitted with unknown ethnicity as its own indicator,
  separate from the recorded "other" level it is merged with in the primary.
* ``spatial_indicator``  - the rural / non-rural split and the prespec urbanized-area
  indicator, each under the transferability bootstrap likelihood-ratio test, plus a
  three-way description.
* ``text_counts``        - counts the text quotes (weather, surface, helmet, alcohol and
  drug results, rider age by class, the flagged set and its overlap with validation).

The primary specification is refitted here from `model_frame.csv` with the functions in
`severity.py`, and its log-likelihood is checked against `severity.json`, so
every "what changes" comparison is against the same fit the paper reports.

Rows of `base_frame.csv` are joined to the model frame on `rider_row_id`. Two crashes
carry two riders, so `Crash_ID` is not a row key.

Run all items, or a subset, e.g. ``python src/data_extra.py --items 1 3 5``. Each item
is merged into the result file as soon as it finishes.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import optimize, stats

sys.path.insert(0, str(Path(__file__).resolve().parent))

import fuse                      # noqa: E402
import model_frame as mf         # noqa: E402
import severity as sv            # noqa: E402
import transferability as tf     # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = DATA / "data_extra.json"

CUES_WITH_ACTOR = ("failure_to_yield", "signal_violation")
ACTOR_LEVELS = ("rider", "driver", "both", "unclear")
MIN_EVENTS = mf.MIN_EVENTS

MI_M = 20
MI_SEED = 20261005
MI_CYCLES = 10
MI_PMM_DONORS = 5
MI_RIDGE_LINEAR = 1e-5
MI_RIDGE_LOGIT = 0.1
MI_KR_DRAWS = sv.KR_DRAWS

SIX_CITIES = ("Austin", "Houston", "San Antonio", "Dallas", "Fort Worth", "El Paso")
UA_POP_THRESHOLD = 200_000

INPUTS = {
    "model_frame": "data/model_frame.csv",
    "base_frame": "data/base_frame.csv",
    "final_labels": "data/final_labels.csv",
    "reference_labels": "data/reference_labels.csv",
    "validation_codes": ["data/validation_codes_batch1.csv",
                         "data/validation_codes_batch2.csv"],
    "screening": "data/screening.json",
    "severity_primary": "data/severity.json",
    "transferability": "data/transferability.json",
    "fusion_audit": "data/fusion_audit.json",
    "flagged_ids": "data/flagged_ids.csv",
    "validation_ids": "data/validation_ids.csv",
    "urban_areas": ["data/shapefiles/tl_2020_us_uac20.shp",
                    "data/shapefiles/ua_list_all.txt"],
    "prespec": "docs/prespec.md",
}


# --------------------------------------------------------------------------
# Shared
# --------------------------------------------------------------------------

def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _cells(frame: pd.DataFrame, mask) -> dict[str, int]:
    mask = pd.Series(np.asarray(mask, dtype=bool), index=frame.index)
    return {k: int(v) for k, v in frame.loc[mask, "sev3"].value_counts()
            .reindex(sv.SEV_ORDER).fillna(0).items()}


def load_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = pd.read_csv(DATA / "model_frame.csv")
    base = pd.read_csv(DATA / "base_frame.csv")
    base = base.set_index("rider_row_id").loc[frame["rider_row_id"]].reset_index()
    if not (base["Crash_ID"].to_numpy() == frame["Crash_ID"].to_numpy()).all():
        raise SystemExit("base_frame and model_frame do not align on rider_row_id")
    return frame, base


def primary_fit(frame: pd.DataFrame) -> tuple[sv.OrdinalFit, list[str], dict]:
    cols = sv.ladder(frame)["M1"]
    fit = sv.fit_ordered_logit(frame, cols, "M1")
    stored = json.loads((DATA / "severity.json").read_text(encoding="utf-8"))
    stored_ll = stored["rungs"]["M1"]["fit"]["loglik"]
    check = {
        "loglik_refit": round(fit.loglik, 4),
        "loglik_stored": stored_ll,
        "matches": bool(abs(fit.loglik - stored_ll) < 1e-3),
        "stored_primary_model": stored["primary_model"],
        "n_covariates": len(cols),
        "converged": fit.converged,
    }
    return fit, cols, check


def ka_summary(ames: dict, cols: list[str]) -> dict:
    return {c: ames[c]["KA"] for c in cols if c in ames}


def significant_sets(ames: dict) -> dict[str, list[str]]:
    return {lvl: sorted(c for c, v in ames.items()
                        if isinstance(v, dict) and lvl in v and v[lvl]["excludes_zero"])
            for lvl in sv.SEV_ORDER}


def merge_result(key: str, value: dict) -> None:
    data = json.loads(OUT.read_text(encoding="utf-8")) if OUT.exists() else {}
    data[key] = value
    data["generated"] = _now()
    data["script"] = "src/data_extra.py"
    data["inputs"] = INPUTS
    OUT.write_text(json.dumps(data, indent=2, default=float), encoding="utf-8")


# --------------------------------------------------------------------------
# 1. Actor attribution (prespec section 8b)
# --------------------------------------------------------------------------

def _coder_long() -> pd.DataFrame:
    frames = []
    for b in (1, 2):
        long = pd.read_csv(DATA / f"validation_codes_batch{b}.csv")
        if "_coded" in long.columns:
            long = long[long["_coded"].astype(bool)]
        long = long.assign(batch=b)
        frames.append(long)
    return pd.concat(frames, ignore_index=True)


def coder_majority_actor(long: pd.DataFrame, cue: str) -> pd.DataFrame:
    """Majority attribution among coders, with its strength recorded.

    The cue itself is the strict majority of the three coders, as in
    `build_final_labels.py`. The attribution is taken from the coders who coded the cue
    positive and gave an attribution. It is a strict majority of those coders, or a tie,
    which is recorded rather than broken arbitrarily.
    """
    field = f"{cue}_actor"
    rows = []
    for cid, g in long.groupby("crash_id"):
        n = int(g[cue].notna().sum())
        pos = int(g[cue].fillna(0).sum())
        cue_major = int(pos > n / 2)
        given = g[field].dropna().astype(str)
        rec = {"Crash_ID": int(cid), "batch": int(g["batch"].iloc[0]),
               "cue_majority": cue_major, "n_coders": n, "n_positive": pos,
               "n_attributions": int(len(given))}
        if len(given):
            vc = given.value_counts()
            top = vc.iloc[0]
            if len(vc) > 1 and vc.iloc[1] == top:
                rec["actor"] = "tie"
                rec["strength"] = "tie"
            else:
                rec["actor"] = vc.index[0]
                rec["strength"] = ("unanimous" if len(vc) == 1 and len(given) == n
                                   else "agreed by all who attributed" if len(vc) == 1
                                   else "strict majority")
        else:
            rec["actor"] = None
            rec["strength"] = None
        rows.append(rec)
    return pd.DataFrame(rows).set_index("Crash_ID")


def _kappa(a: list[str], b: list[str]) -> float | None:
    from sklearn.metrics import cohen_kappa_score
    if len(a) < 2 or len(set(a) | set(b)) < 2:
        return None
    return round(float(cohen_kappa_score(a, b)), 4)


def attribution_agreement(cue: str, majority: pd.DataFrame, ref: pd.DataFrame,
                          batches: tuple[int, ...]) -> dict:
    field = f"{cue}_actor"
    m = majority[majority["batch"].isin(batches)]
    r = ref.reindex(m.index)
    llm_attr = r[field].where(r[cue] == 1)
    coder_attr = m["actor"].where(m["cue_majority"] == 1)

    both_have = llm_attr.notna() & coder_attr.notna() & (coder_attr != "tie")
    a = coder_attr[both_have].astype(str).tolist()
    b = llm_attr[both_have].astype(str).tolist()
    levels = sorted(set(a) | set(b))
    confusion = pd.crosstab(pd.Series(a, name="coders"), pd.Series(b, name="llm"))
    agree = int(sum(x == y for x, y in zip(a, b)))
    return {
        "batches": list(batches),
        "narratives": int(len(m)),
        "n_both_attributed": int(both_have.sum()),
        "n_agree": agree,
        "percent_agreement": round(100 * agree / len(a), 1) if a else None,
        "cohen_kappa": _kappa(a, b),
        "levels_seen": levels,
        "confusion_rows_coders_cols_llm": {
            str(i): {str(j): int(confusion.loc[i, j]) for j in confusion.columns}
            for i in confusion.index},
        "coders_attributed_llm_cue_absent": int(
            (coder_attr.notna() & (coder_attr != "tie") & llm_attr.isna()).sum()),
        "llm_attributed_coders_cue_absent": int(
            (llm_attr.notna() & coder_attr.isna()).sum()),
        "coder_ties_excluded": int((coder_attr == "tie").sum()),
    }


def actor_attribution(frame: pd.DataFrame, primary: sv.OrdinalFit,
                      cols: list[str]) -> dict:
    ref = pd.read_csv(DATA / "reference_labels.csv").set_index("Crash_ID")
    long = _coder_long()
    out: dict = {
        "source_rule": ("Attribution follows the final labels' source: the coders' "
                        "majority for human-coded narratives (validation batch 1 and "
                        "adjudication batch 2), the reference-run attribution for "
                        "automated narratives. A record enters the cross-tabulation only "
                        "when its final cue is 1."),
        "split_rule": ("rider-attributed = rider or both; driver-attributed = driver or "
                       "both; unclear-attributed = unclear, or cue positive with no "
                       "attribution recorded (prespec section 8b)"),
        "event_rule_min_cell": MIN_EVENTS,
        "cues": {},
    }
    split_ok_all = True
    for cue in CUES_WITH_ACTOR:
        field = f"{cue}_actor"
        pos = frame[cue] == 1
        actor = frame[field].where(pos)
        actor_lab = actor.fillna("not recorded")

        tab = pd.crosstab(actor_lab[pos], frame.loc[pos, "sev3"])
        tab = tab.reindex(columns=list(sv.SEV_ORDER)).fillna(0).astype(int)
        crosstab = {str(k): {s: int(tab.loc[k, s]) for s in sv.SEV_ORDER}
                    for k in tab.index}
        by_source = pd.crosstab(actor_lab[pos], frame.loc[pos, "label_source"])
        by_source_d = {str(k): {str(s): int(by_source.loc[k, s])
                                for s in by_source.columns} for k in by_source.index}

        indicators = {
            "rider_attributed": pos & frame[field].isin(["rider", "both"]),
            "driver_attributed": pos & frame[field].isin(["driver", "both"]),
            "unclear_attributed": pos & (frame[field].isin(["unclear"])
                                         | frame[field].isna()),
        }
        ind_out = {}
        cue_ok = True
        for name, mask in indicators.items():
            c = _cells(frame, mask)
            ok = min(c.values()) >= MIN_EVENTS
            cue_ok = cue_ok and ok
            ind_out[name] = {"n": int(mask.sum()), "events_by_severity": c,
                             "smallest_cell": min(c.values()),
                             "meets_event_rule": bool(ok)}
        split_ok_all = split_ok_all and cue_ok

        stray = frame[field].notna() & ~pos
        stray_d = {
            "n": int(stray.sum()),
            "by_actor_and_severity": {
                str(k): v for k, v in
                pd.crosstab(frame.loc[stray, field], frame.loc[stray, "sev3"])
                .to_dict(orient="index").items()} if stray.any() else {},
            "by_source": frame.loc[stray, "label_source"].value_counts().to_dict(),
            "note": ("Human-coded records where at least one coder recorded an "
                     "attribution but the majority cue is 0. They are not cue-positive "
                     "and are excluded from the cross-tabulation and the indicators."),
        }

        majority = coder_majority_actor(long, cue)
        hum = frame.loc[frame["label_source"] == "human", ["Crash_ID", field, cue]]
        hum = hum.drop_duplicates("Crash_ID").set_index("Crash_ID")
        maj_pos = majority[majority["cue_majority"] == 1]
        strength = maj_pos["strength"].fillna("no attribution").value_counts().to_dict()
        joined = maj_pos.join(hum, how="left")
        differs = joined[(joined["actor"].notna()) & (joined["actor"] != "tie")
                         & (joined["actor"] != joined[field])]
        ties = joined[joined["actor"] == "tie"]

        out["cues"][cue] = {
            "cue_positive_rows": int(pos.sum()),
            "crosstab_actor_by_severity": crosstab,
            "crosstab_actor_by_label_source": by_source_d,
            "actor_percent_of_cue_positive": {
                str(k): round(100 * float(v) / int(pos.sum()), 1)
                for k, v in actor_lab[pos].value_counts().items()},
            "split_indicators": ind_out,
            "split_meets_event_rule": bool(cue_ok),
            "actor_recorded_but_cue_zero": stray_d,
            "coder_majority_strength_cue_positive": {str(k): int(v)
                                                    for k, v in strength.items()},
            "coder_ties": {
                "n": int(len(ties)),
                "crash_ids": [int(i) for i in ties.index],
                "final_labels_value": {str(int(i)): (None if pd.isna(v) else str(v))
                                       for i, v in ties[field].items()},
                "note": ("build_final_labels.py resolves a tied attribution by the first "
                         "most frequent value. The tie is listed so the reader can see "
                         "which records carry an arbitrary attribution."),
            },
            "final_labels_differ_from_strict_majority": int(len(differs)),
            "validation_agreement_llm_vs_coders": {
                "batch1_validation_sample": attribution_agreement(cue, majority, ref, (1,)),
                "batch2_flagged_adjudication": attribution_agreement(cue, majority, ref, (2,)),
                "pooled": attribution_agreement(cue, majority, ref, (1, 2)),
                "note": ("Agreement is computed on narratives where the coders' majority "
                         "codes the cue positive with a non-tied attribution and the "
                         "reference run codes the cue positive. Batch 1 is the "
                         "validation sample; batch 2 is the flagged adjudication set, "
                         "selected for disagreement and not a random sample. Agreement "
                         "is unweighted (validation_metrics.py also reports "
                         "inverse-probability-weighted label metrics; these are not). "
                         "Kappa is uninformative where either side uses one level only, "
                         "e.g. signal_violation in batch 2, where the coders used only "
                         "'rider'."),
            },
        }

    out["split_enters_m1"] = bool(split_ok_all)
    if split_ok_all:
        split_cols = []
        fr = frame.copy()
        for cue in CUES_WITH_ACTOR:
            field = f"{cue}_actor"
            pos = fr[cue] == 1
            fr[f"{cue}_rider"] = (pos & fr[field].isin(["rider", "both"])).astype(int)
            fr[f"{cue}_driver"] = (pos & fr[field].isin(["driver", "both"])).astype(int)
            fr[f"{cue}_unclear"] = (pos & (fr[field].isin(["unclear"])
                                           | fr[field].isna())).astype(int)
            split_cols += [f"{cue}_rider", f"{cue}_driver", f"{cue}_unclear"]
        new_cols = [c for c in cols if c not in CUES_WITH_ACTOR] + split_cols
        sv.CUES = tuple(sv.CUES) + tuple(split_cols)
        fit = sv.fit_ordered_logit(fr, new_cols, "M1_actor_split")
        out["split_model"] = {
            "fit": sv.fit_metrics(fit, sv.null_loglikelihood(fr)),
            "coefficients": sv.coefficient_table(fit),
            "separation": sv.separation_diagnostics(fit),
            "average_marginal_effects": sv.average_marginal_effects(fit, fr),
        }
        out["disposition"] = "split estimated (every indicator meets the 10-event rule)"
    else:
        failing = {cue: [k for k, v in out["cues"][cue]["split_indicators"].items()
                         if not v["meets_event_rule"]] for cue in CUES_WITH_ACTOR}
        out["failing_indicators"] = failing
        out["disposition"] = (
            "descriptive cross-tabulation only: at least one resulting indicator falls "
            "below the 10-event rule in a severity class, so under prespec section 8b "
            "the split does not enter M1 and no split model is estimated")
    return out


# --------------------------------------------------------------------------
# 2. Multiple imputation by chained equations (prespec section 2.3)
# --------------------------------------------------------------------------

IMPUTED_PREFIXES = ("speed", "eth", "gender", "age")
ETH_LEVELS = ("white", "black", "hispanic", "other")


def _mi_targets(base: pd.DataFrame) -> pd.DataFrame:
    age = pd.to_numeric(base["rider_age"], errors="coerce")
    speed = pd.to_numeric(base["Crash_Speed_Limit"], errors="coerce")
    speed = speed.where(speed >= 0)
    gender = base["Prsn_Gndr_ID"].map(mf._gender).map({"male": 0.0, "female": 1.0})
    eth = base["Prsn_Ethnicity_ID"].map(mf._ethnicity).replace({"unknown": np.nan})
    return pd.DataFrame({"age": age.to_numpy(), "speed": speed.to_numpy(),
                         "gender": gender.to_numpy(), "eth": eth.to_numpy()})


def _std(x: np.ndarray) -> np.ndarray:
    sd = x.std(axis=0)
    sd[sd == 0] = 1.0
    return (x - x.mean(axis=0)) / sd


def _pmm(y: np.ndarray, X: np.ndarray, miss: np.ndarray, rng) -> np.ndarray:
    """Predictive mean matching with a Bayesian linear-regression parameter draw."""
    Xo, yo = X[~miss], y[~miss]
    p = X.shape[1]
    xtx = Xo.T @ Xo + MI_RIDGE_LINEAR * np.eye(p)
    inv = np.linalg.inv(xtx)
    bhat = inv @ Xo.T @ yo
    resid = yo - Xo @ bhat
    df = max(len(yo) - p, 1)
    sigma = np.sqrt(resid @ resid / rng.chisquare(df))
    bstar = bhat + sigma * np.linalg.cholesky((inv + inv.T) / 2) @ rng.standard_normal(p)
    yhat_obs = Xo @ bhat
    yhat_mis = X[miss] @ bstar
    out = y.copy()
    fill = np.empty(int(miss.sum()))
    for i, v in enumerate(yhat_mis):
        idx = np.argpartition(np.abs(yhat_obs - v), MI_PMM_DONORS)[:MI_PMM_DONORS]
        fill[i] = yo[rng.choice(idx)]
    out[miss] = fill
    return out


def _mlogit_fit(X: np.ndarray, y: np.ndarray, K: int) -> tuple[np.ndarray, np.ndarray]:
    """Ridge-penalized multinomial logit, class 0 the base. Returns (theta, cov)."""
    n, p = X.shape
    pen = np.full(p, MI_RIDGE_LOGIT)
    pen[0] = 0.0                                  # no penalty on the intercept
    Y = np.eye(K)[y]

    def unpack(t):
        return np.column_stack([np.zeros(p), t.reshape(K - 1, p).T])

    def f(t):
        eta = X @ unpack(t)
        eta -= eta.max(axis=1, keepdims=True)
        lp = eta - np.log(np.exp(eta).sum(axis=1, keepdims=True))
        B = t.reshape(K - 1, p)
        nll = -(Y * lp).sum() + 0.5 * (pen * B ** 2).sum()
        P = np.exp(lp)
        grad = ((P - Y)[:, 1:].T @ X) + pen * B
        return nll, grad.ravel()

    res = optimize.minimize(f, np.zeros((K - 1) * p), jac=True, method="L-BFGS-B",
                            options={"maxiter": 2000})
    t = res.x
    eta = X @ unpack(t)
    eta -= eta.max(axis=1, keepdims=True)
    P = np.exp(eta) / np.exp(eta).sum(axis=1, keepdims=True)
    H = np.zeros(((K - 1) * p, (K - 1) * p))
    for a in range(1, K):
        for b in range(1, K):
            w = P[:, a] * ((a == b) - P[:, b])
            H[(a - 1) * p:a * p, (b - 1) * p:b * p] = (X * w[:, None]).T @ X
    H += np.diag(np.tile(pen, K - 1))
    cov = np.linalg.inv(H)
    return t, (cov + cov.T) / 2


def _logit_draw(y: np.ndarray, X: np.ndarray, miss: np.ndarray, K: int,
                rng) -> np.ndarray:
    t, cov = _mlogit_fit(X[~miss], y[~miss].astype(int), K)
    tstar = rng.multivariate_normal(t, cov, method="svd")
    p = X.shape[1]
    B = np.column_stack([np.zeros(p), tstar.reshape(K - 1, p).T])
    eta = X[miss] @ B
    eta -= eta.max(axis=1, keepdims=True)
    P = np.exp(eta) / np.exp(eta).sum(axis=1, keepdims=True)
    out = y.copy()
    out[miss] = [rng.choice(K, p=pr) for pr in P]
    return out


def _chain(targets: pd.DataFrame, Z: np.ndarray, rng) -> pd.DataFrame:
    age = targets["age"].to_numpy(float)
    speed = targets["speed"].to_numpy(float)
    gender = targets["gender"].to_numpy(float)
    eth = pd.Categorical(targets["eth"], categories=list(ETH_LEVELS)).codes.astype(float)
    eth[eth < 0] = np.nan
    miss = {"age": np.isnan(age), "speed": np.isnan(speed),
            "gender": np.isnan(gender), "eth": np.isnan(eth)}
    cur = {"age": age.copy(), "speed": speed.copy(), "gender": gender.copy(),
           "eth": eth.copy()}
    for k, v in cur.items():
        obs = v[~miss[k]]
        v[miss[k]] = rng.choice(obs, size=int(miss[k].sum()))

    def preds(exclude: str) -> np.ndarray:
        parts = [Z]
        if exclude != "age":
            parts.append(_std(cur["age"][:, None]))
        if exclude != "speed":
            parts.append(_std(cur["speed"][:, None]))
        if exclude != "gender":
            parts.append(cur["gender"][:, None])
        if exclude != "eth":
            parts.append(np.eye(len(ETH_LEVELS))[cur["eth"].astype(int)][:, 1:])
        return np.column_stack([np.ones(len(Z))] + parts)

    for _ in range(MI_CYCLES):
        cur["age"] = _pmm(cur["age"], preds("age"), miss["age"], rng)
        cur["speed"] = _pmm(cur["speed"], preds("speed"), miss["speed"], rng)
        cur["gender"] = _logit_draw(cur["gender"], preds("gender"), miss["gender"],
                                    2, rng).astype(float)
        cur["eth"] = _logit_draw(cur["eth"], preds("eth"), miss["eth"],
                                 len(ETH_LEVELS), rng).astype(float)
    return pd.DataFrame({"age": cur["age"], "speed": cur["speed"],
                         "gender": cur["gender"],
                         "eth": [ETH_LEVELS[int(i)] for i in cur["eth"]]})


def _completed_frame(frame: pd.DataFrame, comp: pd.DataFrame,
                     cols: list[str]) -> tuple[pd.DataFrame, list[str]]:
    """Bands from imputed values, coded once and identically for every data set."""
    fr = frame.copy()
    age, speed = comp["age"].to_numpy(), comp["speed"].to_numpy()
    fr["age_under25"] = (age < 25).astype(int)
    fr["age_45plus"] = (age >= 45).astype(int)
    fr["speed_35to40"] = ((speed > 30) & (speed <= 40)).astype(int)
    fr["speed_ge45"] = (speed > 40).astype(int)
    fr["gender_female"] = (comp["gender"].to_numpy() == 1).astype(int)
    for lev in ("black", "hispanic", "other"):
        fr[f"eth_{lev}"] = (comp["eth"].to_numpy() == lev).astype(int)

    new_cols = []
    for c in cols:
        if c in ("speed_unknown", "gender_unknown", "age_unknown"):
            continue
        if c == "eth_other_or_unknown":
            new_cols.append("eth_other")
            continue
        new_cols.append(c)
    return fr, new_cols


def _ame_point_and_var(fit: sv.OrdinalFit, frame: pd.DataFrame, draws: int,
                       seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Per-imputation AME and its Krinsky-Robb variance, with the same mechanics as
    `severity.average_marginal_effects`."""
    cols = fit.columns
    X, _ = sv._design(frame, cols)
    groups = sv.categorical_groups(frame, cols)
    sibling: dict[str, list[int]] = {}
    for _, members in groups.items():
        for c in members:
            sibling[c] = [cols.index(m) for m in members if m != c and m in cols]
    is_binary = {c: set(np.unique(X[:, i])) <= {0.0, 1.0} for i, c in enumerate(cols)}
    k = len(fit.beta)
    n_raw = k + len(fit.cut)
    cov = np.asarray(fit.cov)[:n_raw, :n_raw]
    rng = np.random.default_rng(seed)
    sample = rng.multivariate_normal(np.asarray(fit.raw)[:n_raw], (cov + cov.T) / 2,
                                     size=draws, method="svd")
    point = sv._ame_once(X, fit.beta, fit.cut, cols, sibling, is_binary)
    drawn = np.empty((draws, len(cols), len(sv.SEV_ORDER)))
    for d in range(draws):
        drawn[d] = sv._ame_once(X, sample[d, :k], sv._cut_from_raw(sample[d, k:]),
                                cols, sibling, is_binary)
    return point, drawn.var(axis=0, ddof=1)


def rubin(est: np.ndarray, var: np.ndarray, nu_com: float) -> dict:
    """Rubin's rules for one scalar, with the Barnard-Rubin degrees of freedom."""
    m = len(est)
    qbar = float(np.mean(est))
    W = float(np.mean(var))
    B = float(np.var(est, ddof=1))
    T = W + (1 + 1 / m) * B
    lam = (1 + 1 / m) * B / T if T > 0 else 0.0
    r = (1 + 1 / m) * B / W if W > 0 else np.inf
    nu_old = (m - 1) / lam ** 2 if lam > 0 else np.inf
    nu_obs = (nu_com + 1) / (nu_com + 3) * nu_com * (1 - lam)
    nu = 1 / (1 / nu_old + 1 / nu_obs) if np.isfinite(nu_old) else nu_obs
    fmi = (r + 2 / (nu + 3)) / (r + 1) if np.isfinite(r) else 1.0
    se = np.sqrt(T)
    tcrit = stats.t.ppf(0.975, nu)
    lo, hi = qbar - tcrit * se, qbar + tcrit * se
    return {"estimate": qbar, "se": float(se), "ci95": [float(lo), float(hi)],
            "excludes_zero": bool(lo > 0 or hi < 0),
            "within_var": W, "between_var": B, "df": float(nu),
            "lambda": float(lam), "fmi": float(fmi)}


def _round_rubin(d: dict, digits: int = 5) -> dict:
    return {
        "estimate": round(d["estimate"], digits), "se": round(d["se"], digits),
        "ci95": [round(d["ci95"][0], digits), round(d["ci95"][1], digits)],
        "excludes_zero": d["excludes_zero"],
        "df": round(d["df"], 1), "lambda": round(d["lambda"], 4),
        "fmi": round(d["fmi"], 4),
    }


def multiple_imputation(frame: pd.DataFrame, base: pd.DataFrame,
                        primary: sv.OrdinalFit, cols: list[str],
                        primary_ames: dict) -> dict:
    targets = _mi_targets(base)
    miss_counts = {k: int(targets[k].isna().sum()) for k in targets.columns}
    any_miss = targets.isna().any(axis=1)
    miss_by_class = {k: _cells(frame, targets[k].isna()) for k in targets.columns}

    fixed = [c for c in cols if c.split("_")[0] not in IMPUTED_PREFIXES]
    sev_d = pd.get_dummies(frame["sev3"]).reindex(columns=["BC", "KA"]).astype(float)
    Z = np.column_stack([_std(frame[fixed].astype(float).to_numpy()), sev_d.to_numpy()])

    ss = np.random.SeedSequence(MI_SEED)
    child = ss.spawn(MI_M)
    fits, ames_pt, ames_var, completed_cols = [], [], [], None
    band_cells: dict[str, list[dict]] = {}
    imputed_dist: dict[str, list] = {"eth": [], "gender_female_share_imputed": [],
                                     "age_median_imputed": [], "speed_median_imputed": []}
    sep_flags, conv = [], []
    t0 = time.time()
    for i in range(MI_M):
        rng = np.random.default_rng(child[i])
        comp = _chain(targets, Z, rng)
        # Observed values must pass through untouched.
        for k in ("age", "speed", "gender"):
            obs = targets[k].notna().to_numpy()
            assert np.allclose(comp[k].to_numpy()[obs], targets[k].to_numpy()[obs])
        assert (comp["eth"][targets["eth"].notna()].to_numpy()
                == targets["eth"][targets["eth"].notna()].to_numpy()).all()

        fr, new_cols = _completed_frame(frame, comp, cols)
        completed_cols = new_cols
        fit = sv.fit_ordered_logit(fr, new_cols, f"M1_mi{i + 1}")
        fits.append(fit)
        conv.append(fit.converged)
        sep_flags.append(sv.separation_diagnostics(fit)["suggests_separation"])
        pt, var = _ame_point_and_var(fit, fr, MI_KR_DRAWS, sv.KR_SEED + i)
        ames_pt.append(pt)
        ames_var.append(var)
        for c in ("age_under25", "age_45plus", "speed_35to40", "speed_ge45",
                  "gender_female", "eth_black", "eth_hispanic", "eth_other"):
            band_cells.setdefault(c, []).append(_cells(fr, fr[c] == 1))
        mi = targets["eth"].isna().to_numpy()
        imputed_dist["eth"].append(comp["eth"][mi].value_counts().to_dict())
        gm = targets["gender"].isna().to_numpy()
        imputed_dist["gender_female_share_imputed"].append(
            float(comp["gender"].to_numpy()[gm].mean()))
        imputed_dist["age_median_imputed"].append(
            float(np.median(comp["age"].to_numpy()[targets["age"].isna().to_numpy()])))
        imputed_dist["speed_median_imputed"].append(
            float(np.median(comp["speed"].to_numpy()[targets["speed"].isna().to_numpy()])))
        print(f"  imputation {i + 1}/{MI_M}: loglik {fit.loglik:.3f}, "
              f"converged {fit.converged}, {time.time() - t0:.0f}s")

    k = len(completed_cols)
    nu_com = float(len(frame) - (k + len(sv.SEV_ORDER) - 1))
    betas = np.array([f.beta for f in fits])
    bvars = np.array([np.diag(np.asarray(f.cov))[:k] for f in fits])
    coef = {c: _round_rubin(rubin(betas[:, j], bvars[:, j], nu_com), 4)
            for j, c in enumerate(completed_cols)}

    pts = np.array(ames_pt)
    vrs = np.array(ames_var)
    ames: dict = {}
    for j, c in enumerate(completed_cols):
        ames[c] = {lvl: _round_rubin(rubin(pts[:, j, li], vrs[:, j, li], nu_com))
                   for li, lvl in enumerate(sv.SEV_ORDER)}

    report = ["poverty_share", "coord_missing", "age_under25", "age_45plus",
              "gender_female", "eth_black", "eth_hispanic", "eth_other",
              "speed_35to40", "speed_ge45", "sidewalk_transition", "failure_to_yield",
              "signal_violation", "vehicle_turning_across", "lane_positioning"]
    primary_map = {"eth_other": "eth_other_or_unknown"}
    comparison = {}
    for c in report:
        pc = primary_map.get(c, c)
        prim = primary_ames.get(pc, {}).get("KA")
        comparison[c] = {
            "mi_ka": {kk: ames[c]["KA"][kk] for kk in ("estimate", "ci95",
                                                      "excludes_zero", "fmi")},
            "primary_ka": prim,
            "primary_column": pc,
            "significance_changes": (None if prim is None else
                                     bool(prim["excludes_zero"]
                                          != ames[c]["KA"]["excludes_zero"])),
        }

    mi_sig = {lvl: sorted(c for c in completed_cols if ames[c][lvl]["excludes_zero"])
              for lvl in sv.SEV_ORDER}
    prim_sig = significant_sets({c: primary_ames[c] for c in cols})

    cells_summary = {}
    for c, lst in band_cells.items():
        arr = np.array([[d[s] for s in sv.SEV_ORDER] for d in lst])
        cells_summary[c] = {
            "mean_by_severity": {s: round(float(v), 1)
                                 for s, v in zip(sv.SEV_ORDER, arr.mean(axis=0))},
            "min_by_severity": {s: int(v) for s, v in zip(sv.SEV_ORDER, arr.min(axis=0))},
            "meets_event_rule_in_every_imputation": bool(arr.min() >= MIN_EVENTS),
        }
    eth_imp = pd.DataFrame(imputed_dist["eth"]).fillna(0)

    return {
        "settings": {
            "m": MI_M, "seed": MI_SEED, "seed_scheme": "numpy SeedSequence(seed).spawn(m)",
            "cycles_per_chain": MI_CYCLES, "pmm_donors": MI_PMM_DONORS,
            "ridge_linear": MI_RIDGE_LINEAR, "ridge_logit": MI_RIDGE_LOGIT,
            "kr_draws_per_imputation": MI_KR_DRAWS,
            "kr_seed": f"{sv.KR_SEED} + imputation index (0..{MI_M - 1})",
            "pooling": ("Rubin's rules; total variance W + (1 + 1/m) B; t intervals "
                        "with Barnard-Rubin degrees of freedom; complete-data df "
                        f"{nu_com:.0f} (n minus parameters)"),
            "fmi_definition": "(r + 2/(df + 3)) / (r + 1), r = (1 + 1/m) B / W",
        },
        "imputation_models": {
            "rider_age": ("predictive mean matching, 5 donors, on a Bayesian linear "
                          "regression draw (rider_age, years)"),
            "posted_speed_limit": ("predictive mean matching, 5 donors, on a Bayesian "
                                   "linear regression draw (Crash_Speed_Limit; -1 and "
                                   "missing treated as missing), so imputed values are "
                                   "posted limits that occur in the data"),
            "gender": ("binary logit (female vs male), ridge 0.1 on slopes, parameters "
                       "drawn from the normal approximation; Unknown and missing "
                       "treated as missing"),
            "ethnicity": ("multinomial logit over white, black, hispanic, other (Asian, "
                          "other, American Indian), ridge 0.1 on slopes, parameters "
                          "drawn from the normal approximation; Unknown and missing "
                          "treated as missing"),
            "predictors_every_model": fixed + ["sev3 = BC", "sev3 = KA"],
            "predictors_cross": ("each model also uses the current values of the other "
                                 "three imputed fields (age and speed standardized, "
                                 "female, ethnicity dummies)"),
            "visit_order": ["age", "speed", "gender", "ethnicity"],
            "initialization": "random draws from the observed values",
        },
        "missing_counts": miss_counts,
        "missing_by_severity": miss_by_class,
        "rows_with_any_imputed_field": int(any_miss.sum()),
        "rows": int(len(frame)),
        "band_coding": {
            "age": "under 25; 25 to 54 (ref); 45 and over (threshold of the primary)",
            "speed": "30 or less (ref); 35 to 40; 45 and over",
            "gender": "male (ref); female",
            "ethnicity": ("white (ref); black; hispanic; other. With unknown imputed, "
                          "the primary's other-or-unknown level becomes other alone. "
                          "It is kept as its own level although it fails the 10-event "
                          "rule, because section 4 offers no other merge and pooling it "
                          "into White is not permitted."),
            "dropped": ["speed_unknown", "gender_unknown", "age_unknown"],
            "kept_unimputed_indicators": ["coord_missing", "veh_other_unknown"],
        },
        "band_cells_across_imputations": cells_summary,
        "imputed_values_summary": {
            "ethnicity_of_imputed_mean_counts": {
                str(c): round(float(eth_imp[c].mean()), 1) for c in eth_imp.columns},
            "female_share_of_imputed_gender_mean": round(
                float(np.mean(imputed_dist["gender_female_share_imputed"])), 3),
            "median_imputed_age_mean_across_m": round(
                float(np.mean(imputed_dist["age_median_imputed"])), 1),
            "median_imputed_speed_mean_across_m": round(
                float(np.mean(imputed_dist["speed_median_imputed"])), 1),
        },
        "fits_converged": int(sum(conv)),
        "fits_flagging_separation": int(sum(sep_flags)),
        "columns": completed_cols,
        "coefficients_pooled": coef,
        "ame_pooled": ames,
        "requested_ka_comparison": comparison,
        "ka_excludes_zero": {"mi": mi_sig["KA"], "primary": prim_sig["KA"]},
        "excludes_zero_all_outcomes": {"mi": mi_sig, "primary": prim_sig},
        "fmi_range_coefficients": [round(min(v["fmi"] for v in coef.values()), 4),
                                   round(max(v["fmi"] for v in coef.values()), 4)],
        "fmi_range_ka_ame": [round(min(ames[c]["KA"]["fmi"] for c in completed_cols), 4),
                             round(max(ames[c]["KA"]["fmi"] for c in completed_cols), 4)],
        "mar_caveat": ("Imputation assumes the fields are missing at random given the "
                       "predictors, including severity. The three person-field unknowns "
                       "co-occur and concentrate in the O class, which is the pattern "
                       "expected if reporting completeness depends on the injury itself; "
                       "conditioning on severity carries that dependence into the "
                       "imputation, but missingness that depends on the unrecorded value "
                       "itself is not repaired."),
    }


# --------------------------------------------------------------------------
# 3. Unknown ethnicity as its own indicator
# --------------------------------------------------------------------------

def unknown_ethnicity(frame: pd.DataFrame, base: pd.DataFrame, primary: sv.OrdinalFit,
                      cols: list[str], primary_ames: dict) -> dict:
    from statsmodels.stats.outliers_influence import variance_inflation_factor

    eth = base["Prsn_Ethnicity_ID"].map(mf._ethnicity)
    fr = frame.copy()
    fr["eth_other"] = (eth == "other").astype(int).to_numpy()
    fr["eth_unknown"] = (eth == "unknown").astype(int).to_numpy()
    if not ((fr["eth_other"] + fr["eth_unknown"]) == fr["eth_other_or_unknown"]).all():
        raise SystemExit("other + unknown does not reproduce eth_other_or_unknown")

    new_cols = []
    for c in cols:
        if c == "eth_other_or_unknown":
            new_cols += ["eth_other", "eth_unknown"]
        else:
            new_cols.append(c)
    fit = sv.fit_ordered_logit(fr, new_cols, "M1_eth_unknown")
    null_ll = sv.null_loglikelihood(fr)
    coefs = sv.coefficient_table(fit)
    ames = sv.average_marginal_effects(fit, fr)

    X = fr[new_cols].astype(float)
    Xc = np.column_stack([np.ones(len(X)), X.to_numpy()])
    vif = {c: round(float(variance_inflation_factor(Xc, new_cols.index(c) + 1)), 3)
           for c in ("eth_other", "eth_unknown", "gender_unknown", "age_unknown")}

    unk_cooc = {
        "eth_unknown_and_gender_unknown": int((fr["eth_unknown"] & fr["gender_unknown"]).sum()),
        "eth_unknown_and_age_unknown": int((fr["eth_unknown"] & fr["age_unknown"]).sum()),
        "eth_unknown_all_three": int((fr["eth_unknown"] & fr["gender_unknown"]
                                      & fr["age_unknown"]).sum()),
        "correlation_with_gender_unknown": round(float(
            np.corrcoef(fr["eth_unknown"], fr["gender_unknown"])[0, 1]), 3),
        "correlation_with_age_unknown": round(float(
            np.corrcoef(fr["eth_unknown"], fr["age_unknown"])[0, 1]), 3),
    }

    prim_sig = significant_sets({c: primary_ames[c] for c in cols})
    new_sig = significant_sets({c: ames[c] for c in new_cols})
    shared = [c for c in cols if c in new_cols]
    changes = {}
    for lvl in sv.SEV_ORDER:
        gained = sorted(set(new_sig[lvl]) - set(prim_sig[lvl]) - {"eth_other", "eth_unknown"})
        lost = sorted(set(prim_sig[lvl]) - set(new_sig[lvl]) - {"eth_other_or_unknown"})
        changes[lvl] = {"gained": gained, "lost": lost}
    max_shift = max(abs(ames[c]["KA"]["ame"] - primary_ames[c]["KA"]["ame"])
                    for c in shared)
    shift_col = max(shared, key=lambda c: abs(ames[c]["KA"]["ame"]
                                              - primary_ames[c]["KA"]["ame"]))
    sign_flips = sorted(c for c in shared
                        if np.sign(ames[c]["KA"]["ame"]) != np.sign(primary_ames[c]["KA"]["ame"])
                        and (ames[c]["KA"]["excludes_zero"]
                             or primary_ames[c]["KA"]["excludes_zero"]))
    any_change = any(v["gained"] or v["lost"] for v in changes.values())

    return {
        "counts_by_severity": {
            "eth_other": _cells(fr, fr["eth_other"] == 1),
            "eth_unknown": _cells(fr, fr["eth_unknown"] == 1),
            "eth_other_or_unknown_primary": _cells(fr, fr["eth_other_or_unknown"] == 1),
        },
        "event_rule": {
            "eth_other_meets": bool(min(_cells(fr, fr["eth_other"] == 1).values())
                                    >= MIN_EVENTS),
            "eth_unknown_meets": bool(min(_cells(fr, fr["eth_unknown"] == 1).values())
                                      >= MIN_EVENTS),
            "note": "fitted regardless, as a sensitivity",
        },
        "unknown_recode": ("unknown = Prsn_Ethnicity_ID recorded Unknown or missing; "
                           "other = Asian, Other, American Indian"),
        "fit": sv.fit_metrics(fit, null_ll),
        "primary_fit": sv.fit_metrics(primary, null_ll),
        "lr_split_vs_primary": sv.nested_lr_test(primary, fit),
        "coefficients": {c: coefs["slopes"][c] for c in ("eth_other", "eth_unknown",
                                                          "eth_black", "eth_hispanic")},
        "primary_coefficient_eth_other_or_unknown":
            sv.coefficient_table(primary)["slopes"]["eth_other_or_unknown"],
        "ame": {c: ames[c] for c in ("eth_other", "eth_unknown", "eth_black",
                                     "eth_hispanic")},
        "separation": sv.separation_diagnostics(fit),
        "vif": vif,
        "unknown_cooccurrence": unk_cooc,
        "excludes_zero": {"refit": new_sig, "primary": prim_sig},
        "conclusion_changes": changes,
        "any_other_conclusion_changes": bool(any_change),
        "max_abs_ka_ame_shift_other_covariates": round(float(max_shift), 5),
        "max_shift_covariate": shift_col,
        "sign_flips_among_significant": sign_flips,
        "ka_ame_requested": {c: {"refit": ames[c]["KA"], "primary": primary_ames[c]["KA"]}
                             for c in ("poverty_share", "coord_missing", "gender_unknown",
                                       "age_unknown", "failure_to_yield",
                                       "signal_violation", "sidewalk_transition",
                                       "vehicle_turning_across", "lane_positioning")},
    }


# --------------------------------------------------------------------------
# 4. Spatial indicator
# --------------------------------------------------------------------------

def _split_via_temporal_test(frame: pd.DataFrame, cols: list[str], mask: np.ndarray,
                             label_true: str, label_false: str) -> dict:
    """The transferability bootstrap LR test with an arbitrary binary split.

    `transferability.temporal_test` splits on `crash_year < breakpoint`. Coding the
    split as 0 / 1 in that column and passing a breakpoint of 1 runs the identical
    function, with the same estimability screen, replications and seed, on any mask.
    """
    fr = frame.copy()
    fr["crash_year"] = np.where(mask, 0, 1)
    res = tf.temporal_test(fr, cols, breakpoint=1)
    res.pop("breakpoint", None)
    ren = {"before": label_true, "after": label_false}
    if "severity_counts" in res:
        res["severity_counts"] = {ren[k]: v for k, v in res["severity_counts"].items()}
    if "n_before" in res:
        res[f"n_{label_true}"] = res.pop("n_before")
        res[f"n_{label_false}"] = res.pop("n_after")
    if "covariates_dropped" in res:
        res["covariates_dropped"] = {
            c: v.replace("before:", f"{label_true}:").replace("after:", f"{label_false}:")
            for c, v in res["covariates_dropped"].items()}
    if "both_periods_converged" in res:
        res["both_sides_converged"] = res.pop("both_periods_converged")
    if "estimable_parameters_per_period" in res:
        res["estimable_parameters_per_side"] = res.pop("estimable_parameters_per_period")
    res["function"] = ("transferability.temporal_test on a 0/1 split column "
                       f"(replications {tf.BOOTSTRAP_REPS}, seed {tf.BOOTSTRAP_SEED}, "
                       f"subsample minimum cell {tf.MIN_CELL_SUBSAMPLE})")
    res["covariates_tested_names"] = [c for c in cols if c not in res.get(
        "covariates_dropped", {})] if res.get("testable") else []
    return res


def _urbanized_area(frame: pd.DataFrame) -> tuple[pd.Series, dict]:
    """prespec section 2.1: inside a 2020 Census Urban Area of 200,000 or more, with a
    city-lookup fallback where coordinates are absent."""
    import geopandas as gpd

    ua = gpd.read_file(ROOT / "data" / "shapefiles" / "tl_2020_us_uac20.shp")
    import re
    rows = []
    text = (ROOT / "data" / "shapefiles" / "ua_list_all.txt").read_text(encoding="latin-1")
    for line in text.splitlines()[1:]:
        m = re.match(r"^(\d{5})\s+(.*?)\s{2,}(\d+)\s", line)
        if m:
            rows.append({"UACE": m.group(1), "NAME": m.group(2), "POP": int(m.group(3))})
    pop = pd.DataFrame(rows)
    big = set(pop.loc[pop["POP"] >= UA_POP_THRESHOLD, "UACE"])
    ua = ua[ua["UACE20"].isin(big)].to_crs("EPSG:4326")

    located = frame["latitude"].notna() & frame["longitude"].notna() & (frame["coord_missing"] == 0)
    pts = gpd.GeoDataFrame(frame.loc[located, ["rider_row_id"]],
                           geometry=gpd.points_from_xy(frame.loc[located, "longitude"],
                                                       frame.loc[located, "latitude"]),
                           crs="EPSG:4326")
    joined = gpd.sjoin(pts, ua[["UACE20", "NAME20", "geometry"]], how="left",
                       predicate="within")
    joined = joined[~joined.index.duplicated()]
    inside = joined["UACE20"].notna()
    out = pd.Series(np.nan, index=frame.index)
    out.loc[inside.index] = inside.astype(float)

    city_share = (pd.DataFrame({"city": frame.loc[located, "City_ID"],
                                "ua": out[located]})
                  .groupby("city")["ua"].mean())
    fallback_rows = frame.index[~located]
    resolved, unresolved = 0, []
    for i in fallback_rows:
        city = frame.at[i, "City_ID"]
        if city != "Not Listed" and city in city_share.index:
            out.at[i] = float(city_share[city] >= 0.5)
            resolved += 1
        else:
            unresolved.append(i)
    # Rows the city lookup cannot place take the CRIS rural flag.
    for i in unresolved:
        out.at[i] = 0.0 if frame.at[i, "_rural"] == 1 else 1.0
    names = joined.loc[inside, "NAME20"].value_counts().to_dict()
    return out.astype(int), {
        "ua_population_threshold": UA_POP_THRESHOLD,
        "urban_areas_at_or_above_threshold_nationally": int(len(big)),
        "located_rows": int(located.sum()),
        "located_inside": int(inside.sum()),
        "located_outside": int((~inside).sum()),
        "unlocated_rows": int((~located).sum()),
        "unlocated_resolved_by_city_lookup": resolved,
        "unlocated_unresolved_fallback_rural_flag": len(unresolved),
        "city_lookup_rule": ("a record without coordinates takes 1 if at least half the "
                             "located records of its City_ID fall inside a qualifying "
                             "urban area; 'Not Listed' or a city with no located record "
                             "falls back to the CRIS rural flag"),
        "urban_areas_hit": {str(k): int(v) for k, v in names.items()},
    }


def spatial_indicator(frame: pd.DataFrame, base: pd.DataFrame, cols: list[str],
                      reproduce: bool = True) -> dict:
    fr = frame.copy()
    fr["_rural"] = (base["Rural_Fl"].astype(str).str.upper() == "Y").astype(int).to_numpy()
    rural = fr["_rural"].to_numpy().astype(bool)
    six = fr["City_ID"].isin(SIX_CITIES).to_numpy()

    out: dict = {
        "fields_available": {
            "Rural_Fl": base["Rural_Fl"].fillna("missing").value_counts().to_dict(),
            "Rural_Urban_Type_ID": base["Rural_Urban_Type_ID"].fillna("missing")
                                   .value_counts().to_dict(),
            "Pop_Group_ID": base["Pop_Group_ID"].fillna("missing").value_counts().to_dict(),
        },
        "prespec_indicator": ("prespec section 2.1 'Urbanized': 1 if the crash point "
                              "falls inside a 2020 Census Urban Area of population "
                              "200,000 or more, city lookup where coordinates are "
                              "absent. It is not in model_frame.csv and was not among "
                              "the screened candidates; it is built here from "
                              "tl_2020_us_uac20 and ua_list_all.txt."),
        "note_rural_fl": ("Rural_Fl is the CRIS rural flag, complete for all rows; "
                          "Rural_Urban_Type_ID is missing for 415 of 522 rows and "
                          "cannot carry a split."),
    }

    print("  rural vs non-rural (CRIS Rural_Fl)")
    out["rural_vs_nonrural_test"] = _split_via_temporal_test(
        fr, cols, rural, "rural", "non_rural")

    ua, ua_meta = _urbanized_area(fr)
    fr["_ua200k"] = ua.to_numpy()
    out["urbanized_area_indicator"] = ua_meta
    out["urbanized_area_indicator"]["counts_by_severity"] = {
        "inside_ua_200k": _cells(fr, fr["_ua200k"] == 1),
        "outside": _cells(fr, fr["_ua200k"] == 0)}
    xt = pd.crosstab(fr["_ua200k"], fr["_rural"])
    out["urbanized_area_indicator"]["crosstab_ua200k_by_rural_fl"] = {
        f"ua200k={int(i)}": {f"rural_fl={int(j)}": int(xt.loc[i, j]) for j in xt.columns}
        for i in xt.index}
    print("  inside vs outside a 2020 urban area of 200,000 or more")
    out["ua200k_test"] = _split_via_temporal_test(
        fr, cols, fr["_ua200k"].to_numpy().astype(bool), "inside_ua200k", "outside_ua200k")
    out["ua200k_test"]["mask_note"] = "first side (the split mask) is inside the urban areas"

    three = np.where(six, "six_major_cities", np.where(rural, "rural", "other_urban"))
    desc = {}
    for g in ("six_major_cities", "other_urban", "rural"):
        m = three == g
        c = _cells(fr, m)
        n = int(m.sum())
        desc[g] = {"n": n, "counts": c,
                   "share_percent": {k: round(100 * v / n, 1) for k, v in c.items()}}
    out["three_way"] = {
        "rule": ("six major cities = City_ID in Austin, Houston, San Antonio, Dallas, "
                 "Fort Worth, El Paso, taking precedence; rural = CRIS Rural_Fl Y among "
                 "the rest; other urban = the remainder"),
        "groups": desc,
        "six_city_rows_flagged_rural": int((six & rural).sum()),
    }

    if not reproduce and OUT.exists():
        prev = json.loads(OUT.read_text(encoding="utf-8")).get("spatial_indicator", {})
        if "reproduction_check_six_cities" in prev:
            out["reproduction_check_six_cities"] = prev["reproduction_check_six_cities"]
    if reproduce:
        print("  reproduction check: six cities vs rest through the same path")
        rep = _split_via_temporal_test(fr, cols, six, "six_cities", "rest")
        stored = json.loads((DATA / "transferability.json")
                            .read_text(encoding="utf-8"))["spatial"]
        out["reproduction_check_six_cities"] = {
            "lr_statistic": rep.get("lr_statistic"),
            "p_bootstrap": rep.get("p_bootstrap"),
            "stored_lr_statistic": stored.get("lr_statistic"),
            "stored_p_bootstrap": stored.get("p_bootstrap"),
            "matches": bool(rep.get("lr_statistic") == stored.get("lr_statistic")
                            and rep.get("p_bootstrap") == stored.get("p_bootstrap")),
        }
    return out


# --------------------------------------------------------------------------
# 5. Counts the text needs
# --------------------------------------------------------------------------

def text_counts(frame: pd.DataFrame, base: pd.DataFrame) -> dict:
    crashes = base.drop_duplicates("Crash_ID")

    def vc(s: pd.Series) -> dict:
        return {str(k): int(v) for k, v in s.fillna("missing").value_counts().items()}

    def dominant(s: pd.Series) -> dict:
        v = s.fillna("missing").value_counts()
        return {"category": str(v.index[0]), "count": int(v.iloc[0]),
                "percent": round(100 * float(v.iloc[0]) / len(s), 1), "of": int(len(s))}

    age_by_class = {}
    for s in sv.SEV_ORDER:
        a = pd.to_numeric(base.loc[base["sev3"] == s, "rider_age"], errors="coerce").dropna()
        q = np.percentile(a, [25, 50, 75])
        age_by_class[s] = {"n_known": int(len(a)), "q1": round(float(q[0]), 1),
                           "median": round(float(q[1]), 1), "q3": round(float(q[2]), 1)}
    a_all = pd.to_numeric(base["rider_age"], errors="coerce").dropna()
    q = np.percentile(a_all, [25, 50, 75])
    age_by_class["all"] = {"n_known": int(len(a_all)), "q1": round(float(q[0]), 1),
                           "median": round(float(q[1]), 1), "q3": round(float(q[2]), 1)}

    audit = json.loads((DATA / "fusion_audit.json").read_text(encoding="utf-8"))
    assert list(audit["band"]) == list(fuse.BAND)
    assert audit["flag_cap"] == fuse.FLAG_CAP
    flagged = set(pd.read_csv(DATA / "flagged_ids.csv")["Crash_ID"].astype(int))
    valid = set(pd.read_csv(DATA / "validation_ids.csv")["Crash_ID"].astype(int))
    report = json.loads((DATA / "final_labels_report.json").read_text(encoding="utf-8"))
    overlap = len(flagged & valid)
    batch2 = report.get("batch2_human_labelled")

    return {
        "denominators": {"rider_rows": int(len(base)), "crashes": int(len(crashes))},
        "weather": {"crash_level": vc(crashes["Wthr_Cond_ID"]),
                    "dominant_crash_level": dominant(crashes["Wthr_Cond_ID"]),
                    "rider_rows": vc(base["Wthr_Cond_ID"]),
                    "dominant_rider_rows": dominant(base["Wthr_Cond_ID"])},
        "surface": {"crash_level": vc(crashes["Surf_Cond_ID"]),
                    "dominant_crash_level": dominant(crashes["Surf_Cond_ID"]),
                    "rider_rows": vc(base["Surf_Cond_ID"]),
                    "dominant_rider_rows": dominant(base["Surf_Cond_ID"])},
        "helmet": {"rider_rows": vc(base["Prsn_Helmet_ID"]),
                   "not_applicable": int((base["Prsn_Helmet_ID"] == "Not Applicable").sum()),
                   "any_informative_value": int(base["Prsn_Helmet_ID"].notna().sum()
                                                - (base["Prsn_Helmet_ID"]
                                                   == "Not Applicable").sum())},
        "alcohol_test": {"rider_rows": vc(base["Prsn_Alc_Rslt_ID"]),
                         "result_present": int(base["Prsn_Alc_Rslt_ID"].notna().sum()),
                         "code_note": ("raw CRIS codes; no code lookup is held in the "
                                       "repository, so meanings are not attached here")},
        "drug_test": {"rider_rows": vc(base["Prsn_Drg_Rslt_ID"]),
                      "result_present_excluding_97": int(
                          (base["Prsn_Drg_Rslt_ID"].notna()
                           & (base["Prsn_Drg_Rslt_ID"] != 97)).sum()),
                      "code_note": ("raw CRIS codes; 97 is the most frequent code; no "
                                    "lookup is held in the repository")},
        "rider_age_by_class": age_by_class,
        "rider_age_quantile_method": "numpy.percentile, linear interpolation; unknown ages excluded",
        "fusion": {
            "n_flagged_narratives": int(audit["n_flagged_narratives"]),
            "flagged_ids_file_count": len(flagged),
            "band": list(audit["band"]),
            "band_used": list(audit["band_used"]),
            "narrow_band_if_cap_exceeded": list(fuse.NARROW_BAND),
            "flag_cap": int(audit["flag_cap"]),
            "cap_exceeded": bool(audit["n_flagged_narratives"] > audit["flag_cap"]),
        },
        "validation_flagged_overlap": overlap,
        "validation_sample_size": len(valid),
        "flagged_not_in_validation": len(flagged - valid),
        "batch2_human_labelled": batch2,
        "identity_check_flagged_minus_overlap_equals_batch2": bool(
            len(flagged) - overlap == batch2),
    }


# --------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", nargs="*", type=int, default=[1, 2, 3, 4, 5])
    ap.add_argument("--no-reproduce", action="store_true")
    args = ap.parse_args()

    frame, base = load_frames()
    primary, cols, check = primary_fit(frame)
    print(f"primary M1 refit: loglik {check['loglik_refit']} "
          f"(stored {check['loglik_stored']}, matches {check['matches']})")
    if not check["matches"]:
        raise SystemExit("primary refit does not reproduce severity.json")
    primary_ames = sv.average_marginal_effects(primary, frame)
    merge_result("primary_check", {**check, "kr_draws": sv.KR_DRAWS,
                                   "kr_seed": sv.KR_SEED, "columns": cols})

    if 1 in args.items:
        print("1. actor attribution")
        merge_result("actor_attribution", actor_attribution(frame, primary, cols))
    if 3 in args.items:
        print("3. unknown ethnicity")
        merge_result("unknown_ethnicity",
                     unknown_ethnicity(frame, base, primary, cols, primary_ames))
    if 5 in args.items:
        print("5. text counts")
        merge_result("text_counts", text_counts(frame, base))
    if 2 in args.items:
        print("2. multiple imputation")
        merge_result("multiple_imputation",
                     multiple_imputation(frame, base, primary, cols, primary_ames))
    if 4 in args.items:
        print("4. spatial indicator")
        merge_result("spatial_indicator",
                     spatial_indicator(frame, base, cols, reproduce=not args.no_reproduce))
    print(f"written to {OUT}")


if __name__ == "__main__":
    main()
