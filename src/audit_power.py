"""Bias, coverage and power of the audit's reporting shifters, over independent datasets.

`audit_recovery.py` plants a known shifter in one simulated dataset. One dataset cannot
separate bias from Monte Carlo noise: the first run of that script suggested an upward
bias near 1.7x, and four datasets later suggested almost none. Six datasets give a stable
answer, and the answer is that both readings were partly wrong — there is a real bias of
about 1.4x, and the dominant limitation is power.

This module repeats the recovery across independent seeds and reports what the manuscript
needs in order to say what an interval covering zero means:

* how often the interval covers the truth (calibration);
* how often it excludes zero (power);
* how often a coefficient that is exactly zero is flagged (false positives);
* how far the average point estimate sits from the truth (bias).

Run with:  python src/audit_power.py
"""

from __future__ import annotations

import json

import numpy as np

import audit_recovery as ar

SEEDS = (20260914, 7, 101, 2024, 555, 88)
EFFECT = 0.5


def study(effect: float = EFFECT, seeds: tuple[int, ...] = SEEDS,
          draws: int = 400, tune: int = 800) -> dict:
    rows = []
    for seed in seeds:
        try:
            r = ar.recovery_check(effect, seed=seed, draws=draws, tune=tune)
            rows.append(r)
            print(f"  dataset seed {seed:>8}: {r['posterior_mean']:+.3f} "
                  f"[{r['ci95'][0]:+.3f}, {r['ci95'][1]:+.3f}] "
                  f"covers {r['covers_truth']} excludes zero {r['excludes_zero']} "
                  f"false positives {r['false_positives_among_47_true_zeros']}",
                  flush=True)
        except Exception as exc:                                  # noqa: BLE001
            print(f"  dataset seed {seed}: {type(exc).__name__}", flush=True)

    if not rows:
        raise SystemExit("no dataset produced a usable fit")

    means = np.array([r["posterior_mean"] for r in rows])
    fps = np.array([r["false_positives_among_47_true_zeros"] for r in rows])
    return {
        "planted_effect": effect,
        "n_datasets": len(rows),
        "mean_posterior_mean": round(float(means.mean()), 4),
        "sd_across_datasets": round(float(means.std(ddof=1)), 4),
        "standard_error_of_mean": round(float(means.std(ddof=1) / np.sqrt(len(means))), 4),
        "multiplicative_bias": round(float(means.mean() / effect), 3),
        "coverage": f"{sum(r['covers_truth'] for r in rows)}/{len(rows)}",
        "power_excludes_zero": f"{sum(r['excludes_zero'] for r in rows)}/{len(rows)}",
        "mean_false_positives_of_47_zeros": round(float(fps.mean()), 3),
        "false_positives_expected_at_5pct": round(47 * 0.05, 2),
        "per_dataset": rows,
    }


if __name__ == "__main__":
    print(f"Reporting-shifter recovery at a planted effect of {EFFECT}, across "
          f"{len(SEEDS)} independent datasets of n = 522.")
    print()
    out = study()
    print()
    print(f"mean posterior mean {out['mean_posterior_mean']:+.3f} "
          f"(sd across datasets {out['sd_across_datasets']}, "
          f"standard error {out['standard_error_of_mean']})")
    print(f"multiplicative bias {out['multiplicative_bias']}x")
    print(f"coverage of the truth {out['coverage']}")
    print(f"power, interval excludes zero {out['power_excludes_zero']}")
    print(f"false positives among the 47 true zeros: "
          f"{out['mean_false_positives_of_47_zeros']} on average, against "
          f"{out['false_positives_expected_at_5pct']} expected from a 5 percent rule")
    path = ar.DATA / "audit_recovery_seeds.json"
    path.write_text(json.dumps(out, indent=2, default=float), encoding="utf-8")
    print(f"written to {path}")
