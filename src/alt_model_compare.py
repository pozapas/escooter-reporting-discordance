"""The reference extraction model against one alternative open-weight model.

The choice of Llama 3.3 70B Instruct is checked here against one reasonable
alternative. Qwen 2.5 72B Instruct was run through `extract.py` with
everything else held fixed: the v1.3 mechanism prompt, the same JSON schema, the same
injury-outcome masking, temperature 0, seed 42, one narrative per call, one pinned
provider with fallbacks disabled. Both runs are scored here against the three-coder
majority gold labels on the 150 validation narratives, using the metric functions of
`validation_metrics.py` so the numbers are on exactly the footing of Table 2.

Reported:

* per-cue precision, recall, F1, confusion counts and positives, with the 2,000-resample
  bootstrap intervals of `validation_metrics._bootstrap`, unweighted and with the
  inverse-probability sampling weights;
* macro F1 over the ten mechanism cues, with a bootstrap interval;
* a paired bootstrap of the macro-F1 difference and of each cue's F1 difference, both
  runs scored on the same resampled narratives;
* per-cue discordant-correctness counts with an exact McNemar test, and inter-model
  Cohen's kappa;
* the model id and provider the API returned for every call, token usage and cost.

Only the mechanism pass is compared. Its prompt (v1.3) is the one every Llama reference
mechanism label was produced with; the reference injury-cue labels span three prompt
versions, so no single-prompt comparison of that pass exists.

Run with:  python src/alt_model_compare.py
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from labels import MECHANISMS
from validation_metrics import BOOT_SEED, N_BOOT, _bootstrap, _counts, cohen_kappa

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
LOG_DIR = DATA / "logs"

REFERENCE_LOG = LOG_DIR / "extract_reference.jsonl"
ALT_LOG = LOG_DIR / "extract_alt_qwen.jsonl"
# Llama with the reference settings, rerun on the validation narratives in the same
# period as the Qwen run (written by the masking comparison, masking_compare.py).
LLAMA_RERUN_LOG = LOG_DIR / "extract_masked_rerun.jsonl"
GOLD = DATA / "gold_labels.csv"
VALIDATION_IDS = DATA / "validation_ids.csv"
VALIDATION_METRICS = DATA / "validation_metrics.json"
OUT = DATA / "alt_model_validation.json"

MECHANISM_PROMPT_VERSION = "v1.3-2026-09-14"
ALT_MODEL = "qwen/qwen-2.5-72b-instruct"
ALT_PROVIDER = "DeepInfra"


def _r(x, nd: int = 4):
    if x is None:
        return None
    x = float(x)
    return None if x != x else round(x, nd)


# --------------------------------------------------------------------------
# Logs
# --------------------------------------------------------------------------

def load_run(path: Path, pass_name: str = "mechanism",
             prompt_version: str | None = MECHANISM_PROMPT_VERSION,
             ids: set[int] | None = None) -> tuple[pd.DataFrame, dict]:
    """Latest parsed record per crash for one pass, and a summary of the call metadata.

    Only labels and metadata are read; the log holds no narrative text.
    """
    recs: list[dict] = []
    errors: list[dict] = []
    invalid: list[dict] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("pass") != pass_name:
                continue
            if prompt_version and rec.get("prompt_version") != prompt_version:
                continue
            if ids is not None and int(rec["crash_id"]) not in ids:
                continue
            if rec.get("parsed") and not all(m in rec["parsed"] for m in MECHANISMS):
                # The provider returned JSON that omits schema-required labels (a reply
                # cut off at the token limit). It is not a label; a later call for the
                # same crash, if any, supersedes it.
                invalid.append(rec)
                continue
            (recs if rec.get("parsed") else errors).append(rec)

    latest: dict[int, dict] = {}
    for rec in recs:
        latest[int(rec["crash_id"])] = rec
    rows = []
    for cid, rec in sorted(latest.items()):
        row = {"Crash_ID": cid, "prompt_sha256": rec.get("prompt_sha256")}
        for m in MECHANISMS:
            row[m] = int(rec["parsed"][m])
        rows.append(row)
    labels = pd.DataFrame(rows)

    used = list(latest.values())
    usage = [r.get("usage") or {} for r in used]
    cost = [u.get("cost") for u in usage if u.get("cost") is not None]
    meta = {
        "log": str(path.relative_to(ROOT)).replace("\\", "/"),
        "records_parsed_in_scope": len(recs),
        "records_failed_in_scope": len(errors),
        "records_failed_crash_ids": sorted({int(r["crash_id"]) for r in errors}),
        "records_schema_invalid_in_scope": len(invalid),
        "records_schema_invalid_crash_ids": sorted({int(r["crash_id"]) for r in invalid}),
        "calls_in_scope_total": len(recs) + len(errors) + len(invalid),
        "crashes_labelled": len(latest),
        "models_returned": sorted({str(r.get("model")) for r in used}),
        "providers_returned": sorted({str(r.get("provider")) for r in used}),
        "quantization_logged": sorted({str(r.get("quantization")) for r in used}),
        "response_format": sorted({str(r.get("response_format", "json_schema"))
                                   for r in used}),
        "temperature": sorted({r.get("temperature") for r in used}),
        "seed": sorted({r.get("seed") for r in used}),
        "prompt_version": sorted({str(r.get("prompt_version")) for r in used}),
        "first_call_utc": min((r["timestamp_utc"] for r in used), default=None),
        "last_call_utc": max((r["timestamp_utc"] for r in used), default=None),
        "calls_needing_retry": sum(1 for r in used if (r.get("attempt") or 1) > 1),
        "mean_latency_ms": _r(np.mean([r["latency_ms"] for r in used
                                       if r.get("latency_ms") is not None]), 0)
        if used else None,
        "prompt_tokens": int(sum(u.get("prompt_tokens", 0) for u in usage)),
        "completion_tokens": int(sum(u.get("completion_tokens", 0) for u in usage)),
        "cost_usd_reported": _r(sum(cost), 4) if cost else None,
        "calls_with_cost_reported": len(cost),
    }
    return labels, meta


# --------------------------------------------------------------------------
# Scoring, on the footing of validation_metrics.score_routes
# --------------------------------------------------------------------------

def _macro(f1s: list[float]) -> float:
    vals = [f for f in f1s if f == f]
    return float(np.mean(vals)) if vals else float("nan")


def score_labels(gold: pd.DataFrame, pred: pd.DataFrame, weights: dict,
                 labels=MECHANISMS) -> dict:
    """Per-label metrics and macro F1, as `validation_metrics.score_routes` reports them."""
    merged = gold.merge(pred, on="Crash_ID", how="inner", suffixes=("_gold", "_pred"))
    per_label: dict[str, dict] = {}
    for lab in labels:
        truth = merged[f"{lab}_gold"].to_numpy().astype(int)
        p = merged[f"{lab}_pred"].to_numpy().astype(int)
        w1 = np.ones(len(truth))
        wv = merged["Crash_ID"].map(weights).fillna(1.0).to_numpy()
        entry = {}
        for kind, w in (("unweighted", w1), ("ip_weighted", wv)):
            c = _counts(truth, p, w)
            entry[kind] = {k: (_r(v, 4) if k in ("precision", "recall", "f1")
                               else _r(v, 4 if kind == "ip_weighted" else 0))
                           for k, v in c.items()}
            entry[kind]["ci95"] = _bootstrap(truth, p, w)
        entry["gold_positives"] = int(truth.sum())
        entry["predicted_positives"] = int(p.sum())
        per_label[lab] = entry
    raw_f1 = {
        kind: [_counts(merged[f"{l}_gold"].to_numpy().astype(int),
                       merged[f"{l}_pred"].to_numpy().astype(int),
                       (np.ones(len(merged)) if kind == "unweighted"
                        else merged["Crash_ID"].map(weights).fillna(1.0).to_numpy())
                       )["f1"] for l in labels]
        for kind in ("unweighted", "ip_weighted")
    }
    return {
        "n": int(len(merged)),
        "per_label": per_label,
        "macro_f1": _r(_macro(raw_f1["unweighted"])),
        "macro_f1_ip_weighted": _r(_macro(raw_f1["ip_weighted"])),
        "labels_in_macro": [l for l, f in zip(labels, raw_f1["unweighted"]) if f == f],
    }


def paired_bootstrap(gold: pd.DataFrame, a: pd.DataFrame, b: pd.DataFrame,
                     name_a: str, name_b: str, labels=MECHANISMS,
                     n_boot: int = N_BOOT, seed: int = BOOT_SEED) -> dict:
    """Macro F1 of each run and their difference, on the same resampled narratives.

    Macro F1 follows the existing convention: the mean over labels whose F1 is defined
    in that sample. A label with no gold or no predicted positive in a resample drops out
    of that run's mean, exactly as it would from Table 2.
    """
    m = gold.merge(a, on="Crash_ID", suffixes=("", "_a")).merge(
        b, on="Crash_ID", suffixes=("", "_b"))
    truth = {l: m[l].to_numpy().astype(int) for l in labels}
    pa = {l: m[f"{l}_a"].to_numpy().astype(int) for l in labels}
    pb = {l: m[f"{l}_b"].to_numpy().astype(int) for l in labels}
    n = len(m)
    w = np.ones(n)

    def f1(t, p):
        return _counts(t, p, w)["f1"]

    point_a = _macro([f1(truth[l], pa[l]) for l in labels])
    point_b = _macro([f1(truth[l], pb[l]) for l in labels])

    rng = np.random.default_rng(seed)
    ma, mb, diff = [], [], []
    per = {l: [] for l in labels}
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        fa = [f1(truth[l][idx], pa[l][idx]) for l in labels]
        fb = [f1(truth[l][idx], pb[l][idx]) for l in labels]
        xa, xb = _macro(fa), _macro(fb)
        ma.append(xa)
        mb.append(xb)
        diff.append(xa - xb)
        for l, ya, yb in zip(labels, fa, fb):
            if ya == ya and yb == yb:
                per[l].append(ya - yb)

    def ci(v):
        v = np.asarray(v, dtype=float)
        v = v[~np.isnan(v)]
        return [_r(np.percentile(v, 2.5)), _r(np.percentile(v, 97.5))]

    d = np.asarray(diff, dtype=float)
    d = d[~np.isnan(d)]
    per_label = {}
    for l in labels:
        fa, fb = f1(truth[l], pa[l]), f1(truth[l], pb[l])
        v = np.asarray(per[l], dtype=float)
        # Exact McNemar on correctness: narratives one run gets right and the other wrong.
        a_only = int(((pa[l] == truth[l]) & (pb[l] != truth[l])).sum())
        b_only = int(((pb[l] == truth[l]) & (pa[l] != truth[l])).sum())
        p_mc = (float(stats.binomtest(a_only, a_only + b_only, 0.5).pvalue)
                if a_only + b_only else None)
        per_label[l] = {
            f"f1_{name_a}": _r(fa), f"f1_{name_b}": _r(fb),
            "f1_difference": _r(fa - fb) if fa == fa and fb == fb else None,
            "f1_difference_ci95": ci(v) if len(v) else [None, None],
            "resamples_with_both_defined": int(len(v)),
            f"correct_only_{name_a}": a_only,
            f"correct_only_{name_b}": b_only,
            "mcnemar_exact_p": float(f"{p_mc:.3g}") if p_mc is not None else None,
            "cohen_kappa_between_runs": _r(cohen_kappa(pa[l], pb[l])),
            "label_agreement_share": _r(float((pa[l] == pb[l]).mean())),
        }
    a_tot = sum(v[f"correct_only_{name_a}"] for v in per_label.values())
    b_tot = sum(v[f"correct_only_{name_b}"] for v in per_label.values())
    return {
        "n": int(n),
        "n_boot": n_boot,
        "seed": seed,
        "difference_definition": f"macro F1 {name_a} minus macro F1 {name_b}",
        f"macro_f1_{name_a}": _r(point_a),
        f"macro_f1_{name_a}_ci95": ci(ma),
        f"macro_f1_{name_b}": _r(point_b),
        f"macro_f1_{name_b}_ci95": ci(mb),
        "macro_f1_difference": _r(point_a - point_b),
        "macro_f1_difference_ci95": ci(d),
        "share_of_resamples_difference_above_zero": _r(float((d > 0).mean())),
        "difference_ci_excludes_zero": bool(np.percentile(d, 2.5) > 0
                                            or np.percentile(d, 97.5) < 0),
        "per_label": per_label,
        "pooled_decisions": {
            f"correct_only_{name_a}": int(a_tot),
            f"correct_only_{name_b}": int(b_tot),
            "mcnemar_exact_p": _r(float(stats.binomtest(a_tot, a_tot + b_tot, 0.5)
                                        .pvalue), 6) if a_tot + b_tot else None,
            "note": "all ten labels pooled over the narratives; decisions within a "
                    "narrative are not independent, so this p-value is descriptive",
        },
    }


def check_against_table2(score: dict) -> dict:
    """The Llama reference scored here must match validation_metrics.json exactly."""
    vm = json.loads(VALIDATION_METRICS.read_text(encoding="utf-8"))
    ref = vm["routes"]["route_b_reference"]
    gaps = {}
    for lab in MECHANISMS:
        mine = score["per_label"][lab]["unweighted"]
        theirs = ref[lab]["unweighted"]
        for k in ("tp", "fp", "fn", "tn"):
            if float(mine[k]) != float(theirs[k]):
                gaps[f"{lab}.{k}"] = [mine[k], theirs[k]]
    return {"matches_validation_metrics_json": not gaps, "mismatches": gaps,
            "note": ("Table 2's route_b_reference macro F1 averages fifteen labels "
                     "(ten mechanisms and five injury cues); the macro F1 here averages "
                     "the ten mechanism labels only, because the alternative model ran "
                     "the mechanism pass only.")}


def endpoint_listing(model: str) -> dict:
    """The provider endpoints OpenRouter lists for a model, read at run time."""
    import requests

    try:
        resp = requests.get(f"https://openrouter.ai/api/v1/models/{model}/endpoints",
                            timeout=30)
        resp.raise_for_status()
        data = resp.json()["data"]
        return {
            "accessed_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "endpoints": [{
                "provider": e.get("provider_name"),
                "quantization": e.get("quantization"),
                "context_length": e.get("context_length"),
                "supports_structured_outputs":
                    "structured_outputs" in (e.get("supported_parameters") or []),
                "supports_response_format":
                    "response_format" in (e.get("supported_parameters") or []),
                "supports_seed": "seed" in (e.get("supported_parameters") or []),
                "price_prompt_per_token": e.get("pricing", {}).get("prompt"),
                "price_completion_per_token": e.get("pricing", {}).get("completion"),
            } for e in data.get("endpoints", [])],
        }
    except Exception as exc:  # noqa: BLE001 - recorded, not fatal
        return {"error": f"{type(exc).__name__}: {exc}"}


def prompt_identity(a: pd.DataFrame, b: pd.DataFrame) -> dict:
    """Same prompt hash per narrative means the two models received byte-identical input."""
    m = a[["Crash_ID", "prompt_sha256"]].merge(b[["Crash_ID", "prompt_sha256"]],
                                               on="Crash_ID", suffixes=("_a", "_b"))
    same = int((m["prompt_sha256_a"] == m["prompt_sha256_b"]).sum())
    return {"narratives_compared": int(len(m)), "identical_prompt_sha256": same,
            "all_identical": bool(same == len(m))}


def main() -> dict:
    ids_frame = pd.read_csv(VALIDATION_IDS)
    ids = set(ids_frame["Crash_ID"].astype(int))
    weights = dict(zip(ids_frame["Crash_ID"].astype(int), ids_frame["sampling_weight"]))
    gold = pd.read_csv(GOLD)
    gold = gold[["Crash_ID", *MECHANISMS]]

    llama, llama_meta = load_run(REFERENCE_LOG, ids=ids)
    qwen, qwen_meta = load_run(ALT_LOG, ids=ids)

    missing = {"llama_reference": sorted(ids - set(llama["Crash_ID"])),
               "qwen": sorted(ids - set(qwen["Crash_ID"]))}
    if missing["qwen"] or missing["llama_reference"]:
        print(f"WARNING: narratives without a parsed label: "
              f"{ {k: len(v) for k, v in missing.items()} }")

    score_llama = score_labels(gold, llama[["Crash_ID", *MECHANISMS]], weights)
    score_qwen = score_labels(gold, qwen[["Crash_ID", *MECHANISMS]], weights)
    common = set(llama["Crash_ID"]) & set(qwen["Crash_ID"])
    paired = paired_bootstrap(
        gold[gold["Crash_ID"].isin(common)],
        llama.loc[llama["Crash_ID"].isin(common), ["Crash_ID", *MECHANISMS]],
        qwen.loc[qwen["Crash_ID"].isin(common), ["Crash_ID", *MECHANISMS]],
        "llama", "qwen")

    # The reference run dates from 2026-09-14 and the Qwen run from 2026-10-05. The same
    # comparison against the Llama rerun from the Qwen period removes any drift on the
    # provider side from the model contrast.
    same_period = None
    if LLAMA_RERUN_LOG.exists():
        rr, rr_meta = load_run(LLAMA_RERUN_LOG, ids=ids)
        both = set(rr["Crash_ID"]) & set(qwen["Crash_ID"])
        same_period = {
            "llama_rerun_provenance": rr_meta,
            "llama_rerun_score": score_labels(gold, rr[["Crash_ID", *MECHANISMS]],
                                              weights),
            "paired_llama_rerun_vs_qwen": paired_bootstrap(
                gold[gold["Crash_ID"].isin(both)],
                rr.loc[rr["Crash_ID"].isin(both), ["Crash_ID", *MECHANISMS]],
                qwen.loc[qwen["Crash_ID"].isin(both), ["Crash_ID", *MECHANISMS]],
                "llama_rerun", "qwen"),
        }

    out = {
        "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "script": "src/alt_model_compare.py",
        "inputs": [str(p.relative_to(ROOT)).replace("\\", "/") for p in
                   (REFERENCE_LOG, ALT_LOG, LLAMA_RERUN_LOG, GOLD, VALIDATION_IDS,
                    VALIDATION_METRICS)],
        "settings": {
            "pass": "mechanism (ten pre-crash cues)",
            "prompt_version": MECHANISM_PROMPT_VERSION,
            "masking": "injury-outcome masking on (masking.py), as in the reference run",
            "temperature": 0.0,
            "seed": 42,
            "one_narrative_per_call": True,
            "provider_fallbacks": "disabled (provider.order pinned, allow_fallbacks false)",
            "alternative_model_requested": ALT_MODEL,
            "alternative_provider_pinned": ALT_PROVIDER,
            "response_format": "json_schema, strict, same schema as the reference run",
            "bootstrap_resamples": N_BOOT,
            "bootstrap_seed": BOOT_SEED,
            "gold": "three-coder majority vote, batch 1 validation sample",
            "macro_f1_definition": ("mean F1 over the ten mechanism labels whose F1 is "
                                    "defined, as in validation_metrics.score_routes"),
            "extract_command": (
                "python extract.py custom --run-id alt_qwen --model "
                "qwen/qwen-2.5-72b-instruct --provider DeepInfra --quantization "
                "\"fp8 (provider endpoint listing)\" --passes mechanism --ids "
                "../data/validation_ids.csv --log "
                "../data/logs/extract_alt_qwen.jsonl"),
        },
        "provenance": {
            "llama_reference": llama_meta,
            "qwen": qwen_meta,
            "qwen_endpoint_listing": endpoint_listing(ALT_MODEL),
            "prompt_identity_llama_vs_qwen": prompt_identity(llama, qwen),
            "missing_labels": missing,
        },
        "validation": {
            "llama_reference": score_llama,
            "qwen": score_qwen,
        },
        "llama_reference_check": check_against_table2(score_llama),
        "paired_llama_vs_qwen": paired,
        "same_period_llama_rerun_vs_qwen": same_period,
    }
    OUT.write_text(json.dumps(out, indent=2, default=float), encoding="utf-8")

    print(f"Llama 3.3 70B macro F1 {paired['macro_f1_llama']} "
          f"{paired['macro_f1_llama_ci95']}")
    print(f"Qwen 2.5 72B  macro F1 {paired['macro_f1_qwen']} "
          f"{paired['macro_f1_qwen_ci95']}")
    print(f"difference {paired['macro_f1_difference']} "
          f"{paired['macro_f1_difference_ci95']}")
    for lab in MECHANISMS:
        p = paired["per_label"][lab]
        print(f"  {lab:24s} llama {p['f1_llama']}  qwen {p['f1_qwen']}  "
              f"diff {p['f1_difference']} {p['f1_difference_ci95']}  "
              f"McNemar p {p['mcnemar_exact_p']}")
    if same_period:
        q = same_period["paired_llama_rerun_vs_qwen"]
        print(f"same period: Llama rerun {q['macro_f1_llama_rerun']} "
              f"{q['macro_f1_llama_rerun_ci95']}, Qwen {q['macro_f1_qwen']} "
              f"{q['macro_f1_qwen_ci95']}, difference {q['macro_f1_difference']} "
              f"{q['macro_f1_difference_ci95']} (n {q['n']})")
    print(f"Table 2 reproduction: {out['llama_reference_check']['matches_validation_metrics_json']}")
    print(f"written to {OUT}")
    return out


if __name__ == "__main__":
    main()
