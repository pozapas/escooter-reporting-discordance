"""Evidence for the Safe System reading of the results (Table 7 and Supplementary S11).

Three pieces, each an analysis output rather than a restated count:

1. Posted speed at the statutory line. Texas Transportation Code 551.352(a) allows a
   motor-assisted scooter only on a street posted at 35 mph or less. The pre-specified
   speed bands (30 or below, 35 to 40, 45 and above) put 35 and 40 mph in one band, so
   the line the law draws is not tested by them. This exploratory, policy-motivated
   sensitivity replaces the two recorded bands with one indicator for a posted limit
   above 35 mph, keeps the unrecorded level, and fits the same partial proportional-odds
   form as the pre-specified Holm set (unrecorded age and the speed term freed). The
   pre-specified bands are refitted alongside so the two can be compared. As in the
   manuscript, a fit whose cumulative probabilities cross on some rows is refitted with
   the middle-class probability constrained to stay non-negative, and that refit is
   reported. It is recorded as a departure from the pre-specification.
2. The striking vehicle as a block: a likelihood-ratio test of the three body-type
   indicators in M1.
3. What the crash record holds about the rider's own device, read from the raw extract
   for the modeled crashes, beside the same fields for the striking vehicle.

Run with:  python src/safe_system_evidence.py
Writes:    data/safe_system_evidence.json
"""

from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ordinal_alternatives as oa  # noqa: E402
import severity as sv  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RAW = ROOT / "data" / "E_Scooter_IDYes2017_2025a.xlsx"
OUT = DATA / "safe_system_evidence.json"
STATUTORY_MPH = 35
J = len(sv.SEV_ORDER)


def _r(x, nd=4):
    return None if x is None else round(float(x), nd)


def _mix(d: pd.DataFrame) -> dict:
    n = len(d)
    out = {"n": int(n)}
    for lvl in sv.SEV_ORDER:
        k = int((d.sev3 == lvl).sum())
        out[lvl] = {"count": k, "percent": _r(100 * k / n, 1) if n else None}
    return out


def load() -> pd.DataFrame:
    mf = pd.read_csv(DATA / "model_frame.csv")
    bf = pd.read_csv(DATA / "base_frame.csv", low_memory=False)
    key = ["Crash_ID", "rider_row_id"]
    mf = mf.merge(bf[key + ["Crash_Speed_Limit"]], on=key, how="left", validate="one_to_one")
    recorded = mf["Crash_Speed_Limit"] > 0
    if int((~recorded).sum()) != int(mf["speed_unknown"].sum()):
        raise SystemExit("the unrecorded speed level does not match the raw speed field")
    mf["speed_above35"] = ((mf["Crash_Speed_Limit"] > STATUTORY_MPH) & recorded).astype(int)
    return mf


def ppo_variant(frame: pd.DataFrame, cols: list[str], free: list[str], focal: str,
                m1_loglik: float, m1_k: int) -> dict:
    """Partial proportional-odds fit with `free` threshold-specific, as in ordinal_alternatives."""
    X, y = sv._design(frame, cols)
    n = len(y)
    ol = sv.fit_ordered_logit(frame, cols, "ol")
    fit = sv.fit_ppo(frame, cols, free=list(free), name="ppo", compute_cov=False)
    prop = [c for c in cols if c not in free]
    ip, jf = [cols.index(c) for c in prop], [cols.index(c) for c in free]
    kp, kf = len(ip), len(jf)
    pfun = lambda t, Z, ip=ip, jf=jf: oa.cumulative_probs(t, Z, ip, jf)  # noqa: E731
    _, cross_hat = pfun(fit.raw, X)

    def summarize(theta, cov, loglik, label):
        se = np.sqrt(np.clip(np.diag(cov), 0, None))
        i = free.index(focal)
        g1, g2 = theta[kp + i], theta[kp + kf + i]
        a = np.zeros(len(theta))
        a[kp + i], a[kp + kf + i] = 1.0, -1.0
        vd = float(a @ cov @ a)
        wd = (g1 - g2) ** 2 / vd if vd > 0 else float("nan")
        ames = oa.ame_draws(pfun, theta, oa.kr_sample(theta, cov), X, frame, cols,
                            report=[focal])
        k = kp + kf * (J - 1) + (J - 1)
        bic = -2 * loglik + k * np.log(n)
        return {
            "fit": label,
            "loglik": _r(loglik, 3), "n_params": int(k),
            "bic_minus_ordered_logit": _r(bic - (-2 * ol.loglik + ol.n_params * np.log(n)), 1),
            "cut1_BC_or_KA_vs_O": oa.coef_rows(["g"], [g1], [se[kp + i]])["g"],
            "cut2_KA_vs_O_or_BC": oa.coef_rows(["g"], [g2], [se[kp + kf + i]])["g"],
            "difference_wald_chi2": _r(wd, 3),
            "difference_wald_p": _r(float(stats.chi2.sf(wd, 1)), 4) if vd > 0 else None,
            "ames": {lvl: ames[focal][lvl] for lvl in sv.SEV_ORDER},
            "ame_crossing": ames["_crossing"],
        }

    H = sv._numeric_hessian(lambda t: sv._ppo_negll(t, X[:, ip], X[:, jf], y, J), fit.raw)
    cov = np.linalg.inv(H) if oa.hessian_report(H)["hessian_positive_definite"] else np.linalg.pinv(H)
    out = {"covariates": list(cols), "freed": list(free), "focal": focal,
           "ordered_logit_coef": _r(ol.beta[cols.index(focal)], 3),
           "rows_crossing_unconstrained": int(cross_hat),
           "unconstrained": summarize(fit.raw, cov, fit.loglik, "unconstrained")}
    if cross_hat:
        nc = oa.fit_ppo_noncrossing(X, y, ip, jf, oa.ppo_start(ol, ip, jf))
        Hn = sv._numeric_hessian(nc["obj"], nc["theta"])
        covn = (np.linalg.inv(Hn) if oa.hessian_report(Hn)["hessian_positive_definite"]
                else np.linalg.pinv(Hn))
        out["noncrossing"] = summarize(nc["theta"], covn, nc["loglik"], "noncrossing")
        out["noncrossing"]["converged"] = bool(nc["success"])
        out["reported"] = "noncrossing"
    else:
        out["reported"] = "unconstrained"
    return out


def statutory_speed(frame: pd.DataFrame) -> dict:
    m1 = sv.ladder(frame)["M1"]
    ol_m1 = sv.fit_ordered_logit(frame, m1, "M1")
    rec = frame[frame["Crash_Speed_Limit"] > 0]
    above = rec[rec["speed_above35"] == 1]
    at_or_below = rec[rec["speed_above35"] == 0]
    raw_counts = above["Crash_Speed_Limit"].astype(int).value_counts().sort_index()
    descriptive = {
        "statutory_limit_mph": STATUTORY_MPH,
        "riders_with_recorded_limit": int(len(rec)),
        "riders_above_limit": int(len(above)),
        "riders_above_limit_percent_of_recorded": _r(100 * len(above) / len(rec), 1),
        "posted_limits_above": {str(k): int(v) for k, v in raw_counts.items()},
        "above_limit_at_intersection_or_driveway": int(above["intx_intersection_or_driveway"].sum()),
        "above_limit_at_intersection_or_driveway_percent": _r(
            100 * above["intx_intersection_or_driveway"].mean(), 1),
        "severity_mix_above": _mix(above),
        "severity_mix_at_or_below": _mix(at_or_below),
        "chi2_test": dict(zip(("chi2", "p", "df"), (
            _r(v, 4) if i < 2 else int(v) for i, v in enumerate(
                stats.chi2_contingency(pd.crosstab(rec["speed_above35"], rec["sev3"]))[:3])))),
    }
    holm_age = "age_unknown"
    bands = ppo_variant(frame, m1, [holm_age, "speed_ge45"], "speed_ge45",
                        ol_m1.loglik, ol_m1.n_params)
    stat_cols = [c for c in m1 if c not in ("speed_35to40", "speed_ge45")] + ["speed_above35"]
    statutory = ppo_variant(frame, stat_cols, [holm_age, "speed_above35"], "speed_above35",
                            ol_m1.loglik, ol_m1.n_params)
    return {"descriptive": descriptive,
            "prespecified_bands_holm_form": bands,
            "statutory_line_holm_form": statutory,
            "status": ("exploratory: the statutory split was not pre-specified; it is "
                       "recorded in prespec_deviations.md")}


def striking_vehicle_block(frame: pd.DataFrame) -> dict:
    m1 = sv.ladder(frame)["M1"]
    block = [c for c in ("veh_pickup_truck_van", "veh_suv", "veh_other_unknown") if c in m1]
    full = sv.fit_ordered_logit(frame, m1, "M1")
    small = sv.fit_ordered_logit(frame, [c for c in m1 if c not in block], "drop")
    return {"covariates": block, "lr_test": sv.nested_lr_test(small, full)}


def record_coverage(frame: pd.DataFrame) -> dict:
    """What the record holds about the scooter (unit 1) against the striking vehicle (unit 2)."""
    raw = pd.read_excel(RAW)
    # The extract has one row per person, so a crash with two riders appears twice; the
    # unit fields are the same on both rows, and each crash is counted once.
    raw = raw[raw["Crash_ID"].isin(set(frame["Crash_ID"]))].drop_duplicates("Crash_ID")
    n = int(len(raw))

    def filled(col):
        s = raw[col].astype("string").str.strip()
        return int((s.notna() & (s != "") & (s.str.lower() != "unknown")).sum())

    return {
        "crashes": n,
        "scooter_unit": {c: filled(c) for c in ("Veh_Make_ID", "Veh_Mod_ID", "Veh_Body_Styl_ID",
                                                "Veh_Mod_Year")},
        "striking_vehicle": {c: filled(c) for c in ("Veh_Make_ID_2", "Veh_Mod_ID_2",
                                                    "Veh_Body_Styl_ID_2", "Veh_Mod_Year_2")},
        "scooter_unit_description": {str(k): int(v) for k, v in
                                     raw["Unit_Desc_ID"].value_counts().items()},
        "note": ("a value counts as recorded when it is present and not 'Unknown'; unit 1 is "
                 "the scooter and unit 2 the motor vehicle in every modeled crash"),
    }


def cue_exposure(frame: pd.DataFrame) -> dict:
    out = {}
    for c in ("vehicle_turning_across", "sidewalk_transition", "driveway_alley",
              "distraction_impairment", "failure_to_yield", "signal_violation",
              "lane_positioning", "swerve_loss_control", "wrong_way", "dooring"):
        d = frame[frame[c] == 1]
        out[c] = {"riders": int(len(d)), "percent": _r(100 * len(d) / len(frame), 1),
                  "severity_mix": _mix(d)}
    return out


def main() -> int:
    frame = load()
    out = {
        "generated": date.today().isoformat(),
        "script": "src/safe_system_evidence.py",
        "inputs": ["data/model_frame.csv", "data/base_frame.csv",
                   "data/E_Scooter_IDYes2017_2025a.xlsx", "data/screening.json"],
        "krinsky_robb": {"draws": sv.KR_DRAWS, "seed": sv.KR_SEED},
        "statutory_speed": statutory_speed(frame),
        "striking_vehicle_block": striking_vehicle_block(frame),
        "record_coverage": record_coverage(frame),
        "cue_exposure": cue_exposure(frame),
    }
    OUT.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    s = out["statutory_speed"]
    for key in ("prespecified_bands_holm_form", "statutory_line_holm_form"):
        r = s[key][s[key]["reported"]]
        print(key, s[key]["reported"], "cut1", r["cut1_BC_or_KA_vs_O"], "cut2",
              r["cut2_KA_vs_O_or_BC"], "diff p", r["difference_wald_p"], "KA AME", r["ames"]["KA"])
    print("vehicle block", out["striking_vehicle_block"]["lr_test"])
    print("coverage", out["record_coverage"]["scooter_unit"], out["record_coverage"]["striking_vehicle"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
