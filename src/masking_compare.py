"""Mechanism extraction on masked against unmasked narratives.

This checks whether masking injury-outcome language changes mechanism-extraction
accuracy, and how the severity model responds when the cues come from the complete
narratives instead. The Llama reference settings (temperature 0, seed 42, prompt v1.3,
pinned provider) were rerun on the unmasked text of all 520 narratives in the modeling
sample through `extract.py`; the system prompt is unchanged, so the only difference
between the two runs is the narrative text the model received.

Three comparisons:

1. Extraction accuracy on the 150 validation narratives, both runs against the same
   three-coder gold labels, with a paired bootstrap of the macro-F1 difference
   (functions shared with `alt_model_compare.py`).
2. Label agreement between the runs on all 520 narratives, split by whether masking
   changed the text at all. Where it did not, the two prompts are byte-identical and any
   disagreement is provider-side nondeterminism, which bounds how much of the remaining
   disagreement masking can be credited with.
3. The primary M1 specification (ordered logit, the screened covariate set of
   `severity.ladder`) fitted twice with LLM-only cue labels, once masked and once
   unmasked: the cue-block likelihood-ratio test against M0c, cue coefficients, KA
   average marginal effects with Krinsky-Robb intervals, and cue prevalence by severity
   class. Positives the unmasked run adds or removes are tabulated by severity class,
   which is where outcome leakage would show.

Because the reference run dates from 2026-09-14, the masked mechanism pass was also
rerun on the 150 validation narratives alongside the unmasked run. That gives a
same-period masked-versus-unmasked comparison and shows how closely the temperature-0
reference reproduces on the pinned provider.

The covariate set is held at the primary specification's, so the comparison isolates
the labels. The event-count screening rule is re-evaluated under each label set and
reported, but not allowed to change the specification.

Run with:  python src/masking_compare.py
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import extract as ex
from alt_model_compare import (MECHANISM_PROMPT_VERSION, load_run, paired_bootstrap,
                               score_labels, _r)
from labels import MECHANISMS
from severity import (CUES, KR_DRAWS, KR_SEED, SEV_ORDER, average_marginal_effects,
                      coefficient_table, fit_metrics, fit_ordered_logit, ladder,
                      nested_lr_test, null_loglikelihood, separation_diagnostics)
from validation_metrics import BOOT_SEED, N_BOOT, cohen_kappa

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
LOG_DIR = DATA / "logs"

REFERENCE_LOG = LOG_DIR / "extract_reference.jsonl"
UNMASKED_LOG = LOG_DIR / "extract_unmasked.jsonl"
MASKED_RERUN_LOG = LOG_DIR / "extract_masked_rerun.jsonl"
GOLD = DATA / "gold_labels.csv"
VALIDATION_IDS = DATA / "validation_ids.csv"
MODEL_FRAME = DATA / "model_frame.csv"
SCREENING = DATA / "screening.json"
SEVERITY = DATA / "severity.json"
OUT = DATA / "masking_comparison.json"

MIN_EVENTS = 10  # model_frame.screen event rule, re-evaluated per label set


def agreement_all(masked: pd.DataFrame, unmasked: pd.DataFrame) -> dict:
    """Label agreement on every narrative, split by whether masking changed the text."""
    frame = ex._narratives()[["Crash_ID", "n_masked_tokens"]]
    m = masked.merge(unmasked, on="Crash_ID", suffixes=("_m", "_u")).merge(
        frame, on="Crash_ID", how="left")
    m["text_changed"] = m["n_masked_tokens"] > 0
    out: dict = {"narratives": int(len(m)),
                 "masking_changed_text": int(m["text_changed"].sum()),
                 "masking_left_text_identical": int((~m["text_changed"]).sum())}

    same_prompt = m["prompt_sha256_m"] == m["prompt_sha256_u"]
    out["identical_prompt_hash"] = {
        "where_text_identical": int((same_prompt & ~m["text_changed"]).sum()),
        "where_text_changed": int((same_prompt & m["text_changed"]).sum()),
    }

    for subset_name, sel in (("all", np.ones(len(m), bool)),
                             ("text_changed", m["text_changed"].to_numpy()),
                             ("text_identical", ~m["text_changed"].to_numpy())):
        sub = m[sel]
        per = {}
        any_diff = np.zeros(len(sub), bool)
        for lab in MECHANISMS:
            a, b = sub[f"{lab}_m"].to_numpy(), sub[f"{lab}_u"].to_numpy()
            any_diff |= a != b
            per[lab] = {
                "masked_positive": int(a.sum()),
                "unmasked_positive": int(b.sum()),
                "gained_when_unmasked": int(((a == 0) & (b == 1)).sum()),
                "lost_when_unmasked": int(((a == 1) & (b == 0)).sum()),
                "agreement_share": _r(float((a == b).mean())) if len(a) else None,
                "cohen_kappa": _r(cohen_kappa(a, b)),
            }
        n_dec = len(sub) * len(MECHANISMS)
        n_diff = int(sum(v["gained_when_unmasked"] + v["lost_when_unmasked"]
                         for v in per.values()))
        out[subset_name] = {
            "narratives": int(len(sub)),
            "label_decisions": int(n_dec),
            "decisions_differing": n_diff,
            "share_of_decisions_differing": _r(n_diff / n_dec, 5) if n_dec else None,
            "narratives_with_any_difference": int(any_diff.sum()),
            "per_label": per,
        }
    return out


def discordance_by_text_change(gold: pd.DataFrame, masked: pd.DataFrame,
                               unmasked: pd.DataFrame) -> dict:
    """Validation decisions one run gets right and the other wrong, split by whether
    masking changed the narrative. Where it did not, the prompts are identical and the
    discordance is provider-side nondeterminism, not an effect of masking."""
    changed = ex._narratives()[["Crash_ID", "n_masked_tokens"]]
    m = masked.merge(unmasked, on="Crash_ID", suffixes=("_m", "_u")).merge(
        gold, on="Crash_ID").merge(changed, on="Crash_ID")
    out = {}
    for name, sel in (("text_changed", m["n_masked_tokens"] > 0),
                      ("text_identical", m["n_masked_tokens"] == 0)):
        sub = m[sel]
        um = mu = 0
        for lab in MECHANISMS:
            t, a, b = sub[lab], sub[f"{lab}_m"], sub[f"{lab}_u"]
            mu += int(((a == t) & (b != t)).sum())
            um += int(((b == t) & (a != t)).sum())
        out[name] = {"narratives": int(len(sub)), "correct_only_masked": mu,
                     "correct_only_unmasked": um}
    return out


def cue_frame(base: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    """The model frame with its ten cue columns replaced by one label set."""
    frame = base.drop(columns=[c for c in MECHANISMS if c in base.columns])
    frame = frame.merge(labels[["Crash_ID", *MECHANISMS]], on="Crash_ID", how="left")
    if frame[list(MECHANISMS)].isna().any().any():
        bad = int(frame[list(MECHANISMS)].isna().any(axis=1).sum())
        raise ValueError(f"{bad} model-frame rows have no label in this set")
    return frame


def prevalence(frame: pd.DataFrame) -> dict:
    counts = frame["sev3"].value_counts().reindex(SEV_ORDER)
    out = {"class_counts": {k: int(v) for k, v in counts.items()}}
    for lab in MECHANISMS:
        pos = frame.groupby("sev3")[lab].sum().reindex(SEV_ORDER)
        share = pos / counts
        out[lab] = {
            "positives_by_class": {k: int(v) for k, v in pos.items()},
            "share_by_class": {k: _r(v, 4) for k, v in share.items()},
            "share_KA_minus_O": _r(share["KA"] - share["O"], 4),
            "min_cell": int(pos.min()),
            "meets_event_rule": bool(pos.min() >= MIN_EVENTS),
        }
    return out


def label_shift_by_class(fm: pd.DataFrame, fu: pd.DataFrame) -> dict:
    """Positives the unmasked run gains or loses, by severity class, per cue."""
    out = {}
    for lab in MECHANISMS:
        a, b = fm[lab].to_numpy(), fu[lab].to_numpy()
        g = pd.Series((a == 0) & (b == 1)).groupby(fm["sev3"].to_numpy()).sum()
        l_ = pd.Series((a == 1) & (b == 0)).groupby(fm["sev3"].to_numpy()).sum()
        out[lab] = {
            "gained_by_class": {k: int(g.get(k, 0)) for k in SEV_ORDER},
            "lost_by_class": {k: int(l_.get(k, 0)) for k in SEV_ORDER},
        }
    return out


def severity_fit(frame: pd.DataFrame, spec: dict, null_ll: float, tag: str) -> dict:
    m0c = fit_ordered_logit(frame, spec["M0c"], f"M0c_{tag}")
    m1 = fit_ordered_logit(frame, spec["M1"], f"M1_{tag}")
    cues = [c for c in spec["M1"] if c in CUES]
    coef = coefficient_table(m1)["slopes"]
    ames = average_marginal_effects(m1, frame)
    return {
        "converged": {"M0c": m0c.converged, "M1": m1.converged},
        "fit_M0c": fit_metrics(m0c, null_ll),
        "fit_M1": fit_metrics(m1, null_ll),
        "separation_M1": separation_diagnostics(m1),
        "lr_M0c_to_M1_cue_block": nested_lr_test(m0c, m1),
        "cue_coefficients": {c: coef[c] for c in cues},
        "cue_ame": {c: {lvl: ames[c][lvl] for lvl in SEV_ORDER} for c in cues},
        "cue_ame_KA": {c: ames[c]["KA"] for c in cues},
    }, m1


def main() -> dict:
    ids_frame = pd.read_csv(VALIDATION_IDS)
    ids = set(ids_frame["Crash_ID"].astype(int))
    weights = dict(zip(ids_frame["Crash_ID"].astype(int), ids_frame["sampling_weight"]))
    gold = pd.read_csv(GOLD)[["Crash_ID", *MECHANISMS]]

    masked, masked_meta = load_run(REFERENCE_LOG)
    unmasked, unmasked_meta = load_run(UNMASKED_LOG)
    n_expected = int(ex._narratives()["Crash_ID"].nunique())
    missing_unmasked = sorted(set(masked["Crash_ID"]) - set(unmasked["Crash_ID"]))

    # 1. Validation, paired on the same 150 narratives.
    mv = masked[masked["Crash_ID"].isin(ids)]
    uv = unmasked[unmasked["Crash_ID"].isin(ids)]
    common = set(mv["Crash_ID"]) & set(uv["Crash_ID"])
    validation = {
        "masked": score_labels(gold, mv[["Crash_ID", *MECHANISMS]], weights),
        "unmasked": score_labels(gold, uv[["Crash_ID", *MECHANISMS]], weights),
        "paired_unmasked_vs_masked": paired_bootstrap(
            gold[gold["Crash_ID"].isin(common)],
            uv.loc[uv["Crash_ID"].isin(common), ["Crash_ID", *MECHANISMS]],
            mv.loc[mv["Crash_ID"].isin(common), ["Crash_ID", *MECHANISMS]],
            "unmasked", "masked"),
        "discordant_decisions_by_text_change": discordance_by_text_change(
            gold, mv[["Crash_ID", *MECHANISMS]], uv[["Crash_ID", *MECHANISMS]]),
    }

    # The same paired comparison on the narratives where masking changed the text. On
    # the rest the two prompts are byte-identical, so any difference there is run-to-run
    # variation and only dilutes the masking effect.
    changed_ids = set(ex._narratives().query("n_masked_tokens > 0")["Crash_ID"])
    cc = common & changed_ids
    validation["paired_unmasked_vs_masked_text_changed_only"] = paired_bootstrap(
        gold[gold["Crash_ID"].isin(cc)],
        uv.loc[uv["Crash_ID"].isin(cc), ["Crash_ID", *MECHANISMS]],
        mv.loc[mv["Crash_ID"].isin(cc), ["Crash_ID", *MECHANISMS]],
        "unmasked", "masked")

    # The reference run is from 2026-09-14 and the unmasked run from 2026-10-05, so a
    # masked rerun on the validation narratives, made alongside the unmasked run, gives
    # a same-period comparison and measures how far the reference itself reproduces.
    if MASKED_RERUN_LOG.exists():
        rr, rr_meta = load_run(MASKED_RERUN_LOG, ids=ids)
        rr = rr[["Crash_ID", "prompt_sha256", *MECHANISMS]]
        both = set(rr["Crash_ID"]) & set(uv["Crash_ID"])
        cc2 = both & changed_ids
        rerun_vs_ref = rr.merge(mv, on="Crash_ID", suffixes=("_r", "_m"))
        n_dec = len(rerun_vs_ref) * len(MECHANISMS)
        n_diff = int(sum((rerun_vs_ref[f"{l}_r"] != rerun_vs_ref[f"{l}_m"]).sum()
                         for l in MECHANISMS))
        validation["same_period_masked_rerun"] = {
            "provenance": rr_meta,
            "prompt_identical_to_reference": int(
                (rerun_vs_ref["prompt_sha256_r"] == rerun_vs_ref["prompt_sha256_m"]).sum()),
            "reference_reproduction": {
                "narratives": int(len(rerun_vs_ref)),
                "label_decisions": int(n_dec),
                "decisions_differing_from_reference": n_diff,
                "share_differing": _r(n_diff / n_dec, 5) if n_dec else None,
                "per_label_differing": {
                    l: int((rerun_vs_ref[f"{l}_r"] != rerun_vs_ref[f"{l}_m"]).sum())
                    for l in MECHANISMS},
            },
            "masked_rerun_score": score_labels(gold, rr[["Crash_ID", *MECHANISMS]],
                                               weights),
            "paired_unmasked_vs_masked_rerun": paired_bootstrap(
                gold[gold["Crash_ID"].isin(both)],
                uv.loc[uv["Crash_ID"].isin(both), ["Crash_ID", *MECHANISMS]],
                rr.loc[rr["Crash_ID"].isin(both), ["Crash_ID", *MECHANISMS]],
                "unmasked", "masked_rerun"),
            "paired_unmasked_vs_masked_rerun_text_changed_only": paired_bootstrap(
                gold[gold["Crash_ID"].isin(cc2)],
                uv.loc[uv["Crash_ID"].isin(cc2), ["Crash_ID", *MECHANISMS]],
                rr.loc[rr["Crash_ID"].isin(cc2), ["Crash_ID", *MECHANISMS]],
                "unmasked", "masked_rerun"),
            "paired_masked_rerun_vs_masked_reference": paired_bootstrap(
                gold[gold["Crash_ID"].isin(set(rr["Crash_ID"]))],
                rr[["Crash_ID", *MECHANISMS]],
                mv.loc[mv["Crash_ID"].isin(set(rr["Crash_ID"])), ["Crash_ID", *MECHANISMS]],
                "masked_rerun", "masked_reference"),
        }

    # The masked labels read from the log must be the reference labels the pipeline uses.
    ref_csv = pd.read_csv(DATA / "reference_labels.csv")[["Crash_ID", *MECHANISMS]]
    chk = masked[["Crash_ID", *MECHANISMS]].merge(ref_csv, on="Crash_ID",
                                                  suffixes=("_log", "_csv"))
    reference_csv_check = {
        "crashes_compared": int(len(chk)),
        "label_mismatches": int(sum((chk[f"{l}_log"] != chk[f"{l}_csv"]).sum()
                                    for l in MECHANISMS)),
    }
    reference_csv_check["matches"] = (reference_csv_check["label_mismatches"] == 0
                                      and len(chk) == len(masked) == len(ref_csv))

    # 2. Agreement on all narratives.
    agreement = agreement_all(masked, unmasked)

    # 3. Severity, primary M1 specification, LLM-only labels under each setting.
    base = pd.read_csv(MODEL_FRAME)
    spec = ladder(base)
    null_ll = null_loglikelihood(base)
    fm = cue_frame(base, masked)
    fu = cue_frame(base, unmasked)
    sev_masked, _ = severity_fit(fm, spec, null_ll, "masked")
    sev_unmasked, _ = severity_fit(fu, spec, null_ll, "unmasked")

    # The same code on the frame as built (adjudicated labels) must reproduce the
    # primary result, or nothing above is on the primary footing.
    sev_primary, _ = severity_fit(base, spec, null_ll, "primary")
    prim = json.loads(SEVERITY.read_text(encoding="utf-8"))
    reproduction = {
        "lr_here": sev_primary["lr_M0c_to_M1_cue_block"],
        "lr_severity": prim.get("lr_M0c_to_M1"),
        "matches": sev_primary["lr_M0c_to_M1_cue_block"] == prim.get("lr_M0c_to_M1"),
        "primary_model_in_severity": prim.get("primary_model"),
    }

    cues_in_m1 = [c for c in spec["M1"] if c in CUES]
    comparison = {}
    for c in cues_in_m1:
        km, ku = sev_masked["cue_ame_KA"][c], sev_unmasked["cue_ame_KA"][c]
        bm = sev_masked["cue_coefficients"][c]
        bu = sev_unmasked["cue_coefficients"][c]
        comparison[c] = {
            "coef_masked": bm["coef"], "coef_unmasked": bu["coef"],
            "coef_difference": _r(bu["coef"] - bm["coef"], 4),
            "ame_KA_masked": km["ame"], "ame_KA_unmasked": ku["ame"],
            "ame_KA_difference": _r(ku["ame"] - km["ame"], 5),
            "ka_interval_excludes_zero_masked": km["excludes_zero"],
            "ka_interval_excludes_zero_unmasked": ku["excludes_zero"],
            "significance_agrees": km["excludes_zero"] == ku["excludes_zero"],
            "sign_agrees": bool(np.sign(bm["coef"]) == np.sign(bu["coef"])),
        }

    prev_m, prev_u = prevalence(fm), prevalence(fu)
    out = {
        "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "script": "src/masking_compare.py",
        "inputs": [str(p.relative_to(ROOT)).replace("\\", "/") for p in
                   (REFERENCE_LOG, UNMASKED_LOG, MASKED_RERUN_LOG, GOLD, VALIDATION_IDS,
                    MODEL_FRAME, SCREENING, SEVERITY, DATA / "base_frame.csv",
                    DATA / "reference_labels.csv")],
        "settings": {
            "model": "meta-llama/llama-3.3-70b-instruct",
            "provider_pinned": "DeepInfra, fallbacks disabled",
            "pass": "mechanism (ten pre-crash cues)",
            "prompt_version": MECHANISM_PROMPT_VERSION,
            "system_prompt": ("identical in both runs, including its sentence about "
                              "[MASKED] tokens; only the narrative text differs"),
            "temperature": 0.0,
            "seed": 42,
            "masked_labels": "Route B reference run, mechanism pass (LLM only)",
            "unmasked_labels": "rerun on unmasked narratives, all 520 crashes (LLM only)",
            "bootstrap_resamples": N_BOOT,
            "bootstrap_seed": BOOT_SEED,
            "severity_model": ("ordered logit, primary M1 covariate set from "
                               "severity.ladder over the frozen screening; the cue set "
                               "is held fixed across label sets"),
            "krinsky_robb_draws": KR_DRAWS,
            "krinsky_robb_seed": KR_SEED,
            "event_rule_min_cell": MIN_EVENTS,
            "extract_command": (
                "python extract.py custom --run-id unmasked --mask off --passes "
                "mechanism --log ../data/logs/extract_unmasked.jsonl"),
            "masked_rerun_command": (
                "python extract.py custom --run-id masked_rerun --mask on --passes "
                "mechanism --ids ../data/validation_ids.csv --log "
                "../data/logs/extract_masked_rerun.jsonl"),
            "masked_rerun_note": ("rerun once to complete one HTTP 429 failure; one "
                                  "narrative returned a reply cut off at the token limit "
                                  "on both the run and one repeat with identical settings "
                                  "and has no masked-rerun label, so the same-period "
                                  "comparisons use the narratives listed under its "
                                  "provenance"),
            "reruns": ("the same command was rerun once to complete the calls that "
                       "failed with HTTP 429; one reply that omitted the schema-required "
                       "labels (cut off at the token limit) was repeated once with "
                       "extract._call and identical settings, and the complete reply "
                       "is the one used; both events are in the log and counted under "
                       "provenance.unmasked"),
        },
        "provenance": {
            "masked_reference": masked_meta,
            "unmasked": unmasked_meta,
            "narratives_in_modeling_sample": n_expected,
            "missing_unmasked_labels": missing_unmasked,
            "masked_log_matches_reference_labels_csv": reference_csv_check,
        },
        "validation": validation,
        "agreement_all_narratives": agreement,
        "severity": {
            "n_rows": int(len(base)),
            "n_crashes": int(base["Crash_ID"].nunique()),
            "cues_in_M1": cues_in_m1,
            "n_covariates_M0c": len(spec["M0c"]),
            "n_covariates_M1": len(spec["M1"]),
            "masked": sev_masked,
            "unmasked": sev_unmasked,
            "cue_comparison": comparison,
            "summary": {
                "lr_p_masked": sev_masked["lr_M0c_to_M1_cue_block"]["p"],
                "lr_p_unmasked": sev_unmasked["lr_M0c_to_M1_cue_block"]["p"],
                "cue_block_significant_at_5pct_masked":
                    sev_masked["lr_M0c_to_M1_cue_block"]["p"] < 0.05,
                "cue_block_significant_at_5pct_unmasked":
                    sev_unmasked["lr_M0c_to_M1_cue_block"]["p"] < 0.05,
                "all_coefficient_signs_agree":
                    all(v["sign_agrees"] for v in comparison.values()),
                "all_ka_significance_agrees":
                    all(v["significance_agrees"] for v in comparison.values()),
                "any_ka_interval_excludes_zero": any(
                    v["ka_interval_excludes_zero_masked"]
                    or v["ka_interval_excludes_zero_unmasked"]
                    for v in comparison.values()),
                "max_abs_ame_KA_difference": _r(max(
                    abs(v["ame_KA_difference"]) for v in comparison.values()), 5),
            },
            "prevalence_masked": prev_m,
            "prevalence_unmasked": prev_u,
            "label_shift_by_class": label_shift_by_class(fm, fu),
            "event_rule_would_admit": {
                "masked": [c for c in MECHANISMS if prev_m[c]["meets_event_rule"]],
                "unmasked": [c for c in MECHANISMS if prev_u[c]["meets_event_rule"]],
            },
            "primary_labels_reproduction": reproduction,
            "primary_labels_fit": sev_primary,
        },
    }
    OUT.write_text(json.dumps(out, indent=2, default=float), encoding="utf-8")

    p = validation["paired_unmasked_vs_masked"]
    print(f"validation n {p['n']}: unmasked macro F1 {p['macro_f1_unmasked']} "
          f"{p['macro_f1_unmasked_ci95']}, masked {p['macro_f1_masked']} "
          f"{p['macro_f1_masked_ci95']}, difference {p['macro_f1_difference']} "
          f"{p['macro_f1_difference_ci95']}")
    a = agreement
    for k in ("all", "text_changed", "text_identical"):
        print(f"  {k:15s} {a[k]['decisions_differing']} of {a[k]['label_decisions']} "
              f"decisions differ, {a[k]['narratives_with_any_difference']} narratives")
    for tag, s in (("masked", sev_masked), ("unmasked", sev_unmasked)):
        print(f"{tag}: cue-block LR {s['lr_M0c_to_M1_cue_block']}")
        for c in cues_in_m1:
            k = s["cue_ame_KA"][c]
            print(f"    {c:24s} b {s['cue_coefficients'][c]['coef']:+.4f}  "
                  f"KA AME {k['ame']:+.4f} {k['ci95']}")
    print(f"primary reproduction: {reproduction['matches']}")
    print(f"written to {OUT}")
    return out


if __name__ == "__main__":
    main()
