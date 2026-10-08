"""Prove, from the logs alone, which text each extraction pass actually received.

Supplementary S2 claims that the mechanism pass saw narratives with injury-outcome
language masked, and that the injury-cue pass saw the unmasked text. The whole
reporting-discordance audit rests on the second half of that: if the cue pass had read
masked narratives, the four-level narrative indicator would be measuring the mask rather
than the report, and the audit would be void.

The claim is checkable rather than merely assertable. Every call logged a SHA-256 of the
exact prompt it sent, so the prompts can be rebuilt from the raw narratives and compared.
A match proves which text went out. Nothing here trusts the code path; it reconstructs
what was sent.

One subtlety makes the arithmetic look odd at first. Masking only changes a narrative
that contains injury-outcome language, and 248 of the 520 contain none. For those, the
masked and unmasked strings are identical and the hash matches both. The evidence
therefore lives in the 272 narratives where masking does change the text, and the test
is judged on those.

Run with:  python src/verify_masking.py
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import extract as ex

ROOT = Path(__file__).resolve().parents[1]
LOGS = ROOT / "data" / "logs"
NEWLINE = chr(10)

EXPECTED = {"mechanism": "masked", "cue": "raw"}


def prompt_sha(pass_name: str, text: str) -> str:
    """Rebuild the prompt exactly as `extract._call` did, and hash it."""
    system, _ = ex.PASS_SPEC[pass_name]
    user = "NARRATIVE:" + NEWLINE + str(text)
    return hashlib.sha256(
        (system + NEWLINE + NEWLINE + user).encode("utf-8")).hexdigest()


def verify(log_path: Path) -> dict:
    frame = ex._narratives()
    by_id = {int(r.Crash_ID): r for r in frame.itertuples()}
    differs = {int(r.Crash_ID) for r in frame.itertuples()
               if r.n_masked_tokens > 0}

    result: dict[str, dict] = {}
    with open(log_path, encoding="utf-8") as fh:
        for line in fh:
            rec = json.loads(line)
            cid, pass_name = rec["crash_id"], rec["pass"]
            row = by_id.get(cid)
            if row is None:
                continue

            want = rec["prompt_sha256"]
            hit_masked = prompt_sha(pass_name, row.masked) == want
            hit_raw = prompt_sha(pass_name, row.narrative) == want

            d = result.setdefault(pass_name, {
                "calls": 0, "informative": 0,
                "informative_masked": 0, "informative_raw": 0,
                "unreconstructed": 0, "ambiguous_identical_text": 0})
            d["calls"] += 1
            if not (hit_masked or hit_raw):
                d["unreconstructed"] += 1
            elif cid not in differs:
                d["ambiguous_identical_text"] += 1
            else:
                d["informative"] += 1
                d["informative_masked"] += int(hit_masked)
                d["informative_raw"] += int(hit_raw and not hit_masked)

    for pass_name, d in result.items():
        expected = EXPECTED.get(pass_name)
        got = d["informative_masked"], d["informative_raw"]
        d["expected_text"] = expected
        d["verdict"] = (
            "every informative call used the masked text"
            if expected == "masked" and got[1] == 0 and got[0] == d["informative"]
            else "every informative call used the unmasked text"
            if expected == "raw" and got[0] == 0 and got[1] == d["informative"]
            else "MIXED OR UNEXPECTED")
        d["passes"] = (d["unreconstructed"] == 0
                       and d["verdict"] != "MIXED OR UNEXPECTED")

    return {
        "log": log_path.name,
        "narratives": int(len(frame)),
        "narratives_masking_changes": len(differs),
        "tokens_masked": int(frame.n_masked_tokens.sum()),
        "per_pass": result,
        "all_passes_verified": all(d["passes"] for d in result.values()),
    }


if __name__ == "__main__":
    log = LOGS / (sys.argv[1] if len(sys.argv) > 1 else "extract_reference.jsonl")
    if not log.exists():
        raise SystemExit(f"{log} does not exist")

    out = verify(log)
    print(f"Reconstructing every prompt in {out['log']} and comparing to its logged "
          f"SHA-256.")
    print()
    print(f"{out['narratives']} narratives; masking changes "
          f"{out['narratives_masking_changes']} of them, replacing "
          f"{out['tokens_masked']} tokens. The other "
          f"{out['narratives'] - out['narratives_masking_changes']} are identical "
          f"masked or not, so they cannot distinguish the two and are set aside.")
    print()
    for pass_name, d in sorted(out["per_pass"].items()):
        print(f"{pass_name} pass — expected to receive the "
              f"{d['expected_text']} text")
        print(f"  {d['calls']} calls, all reconstructed "
              f"({d['unreconstructed']} could not be)")
        print(f"  {d['ambiguous_identical_text']} cannot distinguish "
              f"(masked and unmasked text identical)")
        print(f"  of the {d['informative']} that can: "
              f"{d['informative_masked']} used masked text, "
              f"{d['informative_raw']} used unmasked text")
        print(f"  verdict: {d['verdict']}")
        print()

    if not out["all_passes_verified"]:
        raise SystemExit("VERIFICATION FAILED: see above")
    print("Verified. The mechanism pass never saw injury-outcome language, and the "
          "injury-cue pass always saw the report as written.")
