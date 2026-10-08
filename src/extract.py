"""Route B: single-model narrative extraction with full per-call logging.

One model, one provider, one narrative per request, JSON-schema output. Every call is
logged to a JSONL file with the fields the manuscript's Supplementary S1 reports, so
the run is auditable end to end and resumable without losing provenance.

Two passes:

* ``mechanism`` receives the narrative with injury-outcome language masked and returns
  the ten pre-crash cues plus the actor attribution for the two shared cues.
* ``cue`` receives the unmasked narrative and returns the five injury cues.

Runs:

* the reference run, temperature 0 with seed 42, whose labels are the Route B reference;
* five vote-share runs, temperature 0.7 with seeds 1 to 5, whose mean label is the
  mechanism probability used for calibration and the fusion rule.

Usage
-----
    python extract.py reference            # both passes, temperature 0
    python extract.py votes                # five mechanism runs, temperature 0.7
    python extract.py unmasked-validation  # mechanism pass on unmasked text, validation IDs only

Options override the run settings; every default reproduces the runs above:

    --model MODEL            OpenRouter model id (default: the Llama reference model)
    --provider NAME          the single provider to pin, fallbacks disabled
    --quantization TEXT      quantization string written to the log
    --response-format KIND   json_schema (default) or json_object
    --mask on|off            mask injury-outcome language for the mechanism pass
    --ids FILE               CSV with a Crash_ID column restricting the narratives
    --log FILE               JSONL log path (default: logs/extract_<run_id>.jsonl)
    --run-id ID              run identifier written to every record
    --passes P [P ...]       mechanism and/or cue
    --temperature T, --seed S

e.g. the alternative-model run on the validation narratives:

    python extract.py custom --run-id alt_qwen --model qwen/qwen-2.5-72b-instruct \\
        --provider DeepInfra --passes mechanism --ids data/validation_ids.csv
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

from labels import INJURY_CUES, MECHANISMS
from masking import mask_narrative

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
LOG_DIR = DATA / "logs"

MODEL = "meta-llama/llama-3.3-70b-instruct"
PROVIDER = "DeepInfra"
QUANTIZATION = "fp8 (provider-reported)"
RESPONSE_FORMAT = "json_schema"
MAX_TOKENS = 600
ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"

REFERENCE_SEED = 42
VOTE_SEEDS = (1, 2, 3, 4, 5)
VOTE_TEMPERATURE = 0.7

MAX_WORKERS = 8
MAX_ATTEMPTS = 6

_lock = threading.Lock()

# ---------------------------------------------------------------------------
# Prompts. Definitions are the coding guide's, so the model and the human coders
# are given the same instructions. Version string is logged with every call.
# ---------------------------------------------------------------------------

PROMPT_VERSION = "v1.3-2026-09-14"

_UNIT_RULE = """Before assigning anything, decide which unit is the electric scooter.
Reports number the units inconsistently: the scooter is usually "Unit 2" but is
"Unit 1" in some reports, and a few reports renumber the units partway through.
Identify the scooter by description, never by number. Scooter descriptions include
"electric scooter", "e-scooter", "motorized scooter", "motor assisted scooter",
"stand up scooter", "motorized conveyance", and the brand names Bird, Lime, Gotrax,
Jetta, Spin and Veo. Some reports code the rider as a pedestrian; that unit is still
the scooter. Unit labels appear as "Unit 1", "Unit #2", "UNIT1", "U1", "Person 2"
and "Vehicle 1".

Every crash here involves one electric scooter and one motor vehicle. The rider is
the person on the scooter. The driver is the person in the motor vehicle. If the
report renumbers the units, use the corrected assignment the officer settled on.
Report the scooter's unit number in the field scooter_unit (1 or 2, or 0 if it
cannot be determined)."""

_GENERAL_RULE = """Record what the narrative states. If the narrative does not state
it, the answer is 0. Silence is 0. Do not infer what probably happened.

Ignore fault. Police narratives assign fault, cite statutes, and say who was placed
at fault. A narrative that says a unit was placed at fault for failure to yield but
never describes anyone failing to yield is 0 for that label. A narrative that
describes the action is 1 even if the officer blamed the other party.

Labels are not mutually exclusive. Assign every label the narrative supports."""

MECHANISM_DEFINITIONS = """1. wrong_way - The scooter was traveling against the legal direction of traffic for the space it was in: against traffic, the wrong way, the opposite direction of travel, or in a bike lane or shoulder on the wrong side for that direction. Riding on a sidewalk is not this label. Crossing a road from one side to the other is not contraflow. The motor vehicle traveling the wrong way is not this label.

2. sidewalk_transition - The scooter moved from a sidewalk, path, curb, or other off-roadway area into the roadway, or was crossing the roadway having come from such a space. Crossing a roadway from a sidewalk or curb counts whether or not a marked crosswalk is used. Riding along a sidewalk with no entry into the roadway is 0. Entering from a driveway, alley, or parking lot is label 3, not this one.
   The narrative must say where the rider came from. A narrative that says only that the rider was crossing, or was in a crosswalk, without stating a sidewalk, curb, path, or other off-roadway origin, is 0. Do not infer a sidewalk origin from the presence of a crosswalk.

3. driveway_alley - Either unit entered or left the roadway through a driveway, alley, private drive, or parking-lot exit. This includes the motor vehicle backing out of, turning into, or exiting a driveway or parking lot. A crash occurring wholly inside a parking lot, with neither unit entering or leaving the roadway, is 0. A street intersection is not a driveway.

4. failure_to_yield - THIS LABEL IS DECIDED BY WORDS, NOT BY EVENTS. Follow these two steps exactly and do not reason about the situation.
   STEP 1. Scan the narrative for the word "yield" in any form (yield, yields, yielded, yielding), or the phrase "right of way" (also written "right-of-way" or "ROW"). If NEITHER appears anywhere in the narrative, then failure_to_yield is 0. Stop. Do not consider what happened, who was at fault, or how the crash occurred. The answer is 0.
   STEP 2. Only if one of those words appeared: decide whether the narrative uses it to say that a party failed to yield or violated the right of way ("failed to yield", "did not yield", "without yielding", "failed to yield the right of way", "right of way violation"). If yes, 1. If the words appear only in some other sense, for example a statement that a party DID yield, or the name of a yield sign with no failure described, then 0.
   Running a red light, disregarding a stop sign, failing to stop at a signal, turning across a path, entering from a driveway, leaving a sidewalk, failing to control speed, and not seeing the other party are all 0 here. Those are labels 5, 9, 3, 2, 6 and 8. Assign them there. Many narratives describe a serious violation and still get 0 for failure_to_yield, and that is correct.

5. signal_violation - The narrative describes a party disregarding a traffic control device: running or disregarding a red light, a stop sign, a stop-and-go signal, or a "no walking" pedestrian signal; failing to stop at a stop sign; rolling through or past a stop line. Either party. Having a green light, or stopping properly, is 0. A location merely being signalized is 0.

6. swerve_loss_control - The scooter swerved, slid, skidded, braked hard, or could not stop or steer as intended: lost control, unable to stop, could not stop in time, failed to control speed, evasive action, or falling off before any impact. Being knocked off or falling as a result of being struck is 0. The motor vehicle losing control is 0.

7. dooring - A door of a parked or stopped vehicle was opened into the scooter's path, and the scooter struck it or swerved because of it. The scooter striking a door of a moving vehicle is 0. A door named only as the point of impact is 0. This is rare; most narratives mentioning a door are 0.

8. distraction_impairment - The narrative states something about attention, phone use, alcohol, or drugs, for either party: phone use, texting, navigating, looking away, inattention, distraction, intoxication, impairment, odor of alcohol, open container, field sobriety testing. Also 1 for "did not look", "did not observe", or "failed to see" where the narrative presents it as a failure to look. A driver saying the other party "came out of nowhere" is 0. A blocked view is 0. Speed alone is 0.

9. vehicle_turning_across - The motor vehicle was turning, and that turn brought it across the scooter's line of travel: turning left or right into, across, or in front of the scooter's path, turning into a driveway or side street across the scooter's direction, or a U-turn across the path. The vehicle traveling straight is 0. The scooter turning is 0.

10. lane_positioning - The narrative states where in the roadway the scooter was: in a bike lane, a travel lane, a numbered lane, a shoulder, a median, or between parked cars and the travel lane. On the sidewalk is 0. Crossing the road with no lane position given is 0. The motor vehicle's lane position is 0. This label is about the information being present, not about the position being wrong."""

ACTOR_RULE = """For failure_to_yield and signal_violation, also report which party the
narrative attributes the action to, in failure_to_yield_actor and
signal_violation_actor: "rider", "driver", "both", or "unclear". Use "both" when the
narrative describes each party doing it. Use "unclear" when the narrative uses the
phrase without saying who, or gives conflicting accounts. Leave the field as ""
when the label is 0."""

CUE_DEFINITIONS = """1. transported_hospital - The narrative states the rider was taken to a hospital, emergency room, or medical facility, by ambulance, EMS, helicopter, private vehicle, or on foot by a family member. Transport offered and refused is 0. Being checked or evaluated on scene only is 0.

2. unconsciousness - The narrative states the rider lost consciousness, was unresponsive, was knocked out, or was found unconscious. Awake, alert, dazed, or confused is 0.

3. fracture - The narrative names a broken bone, a fracture, or a laceration described as severe or deep, or an amputation or named internal injury. Scrapes, abrasions, road rash, bruising, swelling, a minor or light laceration, and a bloody lip are 0. A stated absence of fracture is 0.

4. fatality - The narrative states the rider died, was pronounced dead, was fatally injured, or is referred to as deceased. Injuries described as life-threatening or critical, without death, are 0.

5. explicit_no_injury - The narrative explicitly states that the rider was not injured, sustained no injuries, refused treatment or medical attention, said they were fine, or was cleared on scene. Silence about injury is 0; silence is not a statement of no injury. A narrative may be 1 for both transported_hospital and explicit_no_injury if it states both."""

MECHANISM_SYSTEM = f"""You are coding police crash narratives for a transportation safety study. Each narrative describes a crash between an electric scooter and a motor vehicle in Texas.

Your task is to assign ten binary labels describing the circumstances before the crash.

{_UNIT_RULE}

{_GENERAL_RULE}

The narrative you receive has had injury-outcome language replaced by the token [MASKED]. Ignore those tokens. They carry no information you need and you must not speculate about what they concealed.

LABEL DEFINITIONS

{MECHANISM_DEFINITIONS}

{ACTOR_RULE}

Give a one-sentence rationale in "rationale" naming the phrases you relied on.

Return JSON only, conforming to the schema. No prose outside the JSON."""

CUE_SYSTEM = f"""You are coding police crash narratives for a transportation safety study. Each narrative describes a crash between an electric scooter and a motor vehicle in Texas.

Your task is to assign five binary labels describing what the narrative states about the scooter rider's injury. Look only for explicit statements. A narrative that is silent about injury gets 0 on all five, which is common and correct.

{_UNIT_RULE}

The labels are about the RIDER, not the motor-vehicle driver. A statement that the driver was uninjured is 0 for explicit_no_injury unless the narrative also states it of the rider.

LABEL DEFINITIONS

{CUE_DEFINITIONS}

Give a one-sentence rationale in "rationale" naming the phrases you relied on.

Return JSON only, conforming to the schema. No prose outside the JSON."""


def _bin_prop(desc: str) -> dict:
    return {"type": "integer", "enum": [0, 1], "description": desc}


MECHANISM_SCHEMA = {
    "name": "mechanism_labels",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "scooter_unit": {"type": "integer", "enum": [0, 1, 2]},
            **{m: _bin_prop(m) for m in MECHANISMS},
            "failure_to_yield_actor": {
                "type": "string",
                "enum": ["rider", "driver", "both", "unclear", ""],
            },
            "signal_violation_actor": {
                "type": "string",
                "enum": ["rider", "driver", "both", "unclear", ""],
            },
            "rationale": {"type": "string"},
        },
        "required": ["scooter_unit", *MECHANISMS,
                     "failure_to_yield_actor", "signal_violation_actor", "rationale"],
        "additionalProperties": False,
    },
}

CUE_SCHEMA = {
    "name": "injury_cue_labels",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "scooter_unit": {"type": "integer", "enum": [0, 1, 2]},
            **{c: _bin_prop(c) for c in INJURY_CUES},
            "rationale": {"type": "string"},
        },
        "required": ["scooter_unit", *INJURY_CUES, "rationale"],
        "additionalProperties": False,
    },
}

PASS_SPEC = {
    "mechanism": (MECHANISM_SYSTEM, MECHANISM_SCHEMA),
    "cue": (CUE_SYSTEM, CUE_SCHEMA),
}


def _headers() -> dict:
    load_dotenv(ROOT / ".env")
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise RuntimeError("OPENROUTER_API_KEY is not set; place it in .env.")
    return {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://openrouter.ai",
        "X-Title": "E-scooter narrative extraction (DATA)",
    }


def _log_path(run_id: str, log_path: Path | None = None) -> Path:
    if log_path is not None:
        log_path = Path(log_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        return log_path
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    return LOG_DIR / f"extract_{run_id}.jsonl"


def _completed(run_id: str, log_path: Path | None = None) -> set[tuple[int, str]]:
    """Crash IDs already logged with a parsed response, for resumption."""
    path = _log_path(run_id, log_path)
    if not path.exists():
        return set()
    done: set[tuple[int, str]] = set()
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("parsed"):
                done.add((int(rec["crash_id"]), rec["pass"]))
    return done


def _append(run_id: str, record: dict, log_path: Path | None = None) -> None:
    with _lock:
        with open(_log_path(run_id, log_path), "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def _response_format(schema: dict, kind: str) -> dict:
    if kind == "json_schema":
        return {"type": "json_schema", "json_schema": schema}
    if kind == "json_object":
        return {"type": "json_object"}
    raise ValueError(f"unknown response format: {kind}")


def _call(crash_id: int, text: str, pass_name: str, temperature: float,
          seed: int, run_id: str, headers: dict, model: str = MODEL,
          provider: str = PROVIDER, quantization: str = QUANTIZATION,
          response_format: str = RESPONSE_FORMAT,
          log_path: Path | None = None) -> dict | None:
    system, schema = PASS_SPEC[pass_name]
    user = f"NARRATIVE:\n{text}"
    prompt_sha = hashlib.sha256((system + "\n\n" + user).encode("utf-8")).hexdigest()

    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "response_format": _response_format(schema, response_format),
        "temperature": temperature,
        "seed": seed,
        "max_tokens": MAX_TOKENS,
        "provider": {"order": [provider], "allow_fallbacks": False},
    }

    last_error = ""
    for attempt in range(MAX_ATTEMPTS):
        started = time.time()
        try:
            resp = requests.post(ENDPOINT, json=body, headers=headers, timeout=120)
            latency = int((time.time() - started) * 1000)
            if resp.status_code in (429, 500, 502, 503, 504):
                # The pinned provider is throttling or down. Wait; never switch.
                last_error = f"HTTP {resp.status_code}"
                time.sleep(min(2 ** attempt * 2, 60))
                continue
            resp.raise_for_status()
            data = resp.json()
            content = data["choices"][0]["message"]["content"]
            parsed = json.loads(content)
            record = {
                "crash_id": int(crash_id),
                "pass": pass_name,
                "run_id": run_id,
                "model": data.get("model", model),
                "provider": data.get("provider", provider),
                "quantization": quantization,
                "temperature": temperature,
                "seed": seed,
                "prompt_version": PROMPT_VERSION,
                "prompt_sha256": prompt_sha,
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "latency_ms": latency,
                "attempt": attempt + 1,
                "usage": data.get("usage", {}),
                "parsed": parsed,
            }
            if response_format != RESPONSE_FORMAT:
                record["response_format"] = response_format
            _append(run_id, record, log_path)
            return record
        except Exception as exc:  # noqa: BLE001 - logged and retried
            last_error = f"{type(exc).__name__}: {exc}"
            time.sleep(min(2 ** attempt * 2, 60))

    _append(run_id, {
        "crash_id": int(crash_id),
        "pass": pass_name,
        "run_id": run_id,
        "model": model,
        "provider": provider,
        "temperature": temperature,
        "seed": seed,
        "prompt_version": PROMPT_VERSION,
        "prompt_sha256": prompt_sha,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "error": last_error,
        "parsed": None,
    }, log_path)
    return None


def _narratives(ids: set[int] | None = None) -> pd.DataFrame:
    base = pd.read_csv(DATA / "base_frame.csv").drop_duplicates(subset="Crash_ID")
    frame = base[["Crash_ID", "narrative"]].copy()
    if ids is not None:
        frame = frame[frame["Crash_ID"].isin(ids)]
    masked = frame["narrative"].fillna("").map(mask_narrative)
    frame["masked"] = [m[0] for m in masked]
    frame["n_masked_tokens"] = [m[1] for m in masked]
    return frame.sort_values("Crash_ID").reset_index(drop=True)


def run(run_id: str, passes: tuple[str, ...], temperature: float, seed: int,
        masked_for_mechanism: bool = True, ids: set[int] | None = None,
        model: str = MODEL, provider: str = PROVIDER,
        quantization: str = QUANTIZATION, response_format: str = RESPONSE_FORMAT,
        log_path: Path | None = None) -> None:
    headers = _headers()
    frame = _narratives(ids)
    done = _completed(run_id, log_path)

    jobs: list[tuple[int, str, str]] = []
    for _, row in frame.iterrows():
        cid = int(row["Crash_ID"])
        for p in passes:
            if (cid, p) in done:
                continue
            if p == "mechanism":
                text = row["masked"] if masked_for_mechanism else row["narrative"]
            else:
                text = row["narrative"]
            jobs.append((cid, p, text))

    total = len(jobs)
    print(f"[{run_id}] {total} calls to make ({len(done)} already logged), "
          f"model={model} provider={provider} temp={temperature} seed={seed} "
          f"masked={masked_for_mechanism} log={_log_path(run_id, log_path)}")
    if not total:
        return

    completed = 0
    failures = 0
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {
            pool.submit(_call, cid, text, p, temperature, seed, run_id, headers,
                        model, provider, quantization, response_format,
                        log_path): (cid, p)
            for cid, p, text in jobs
        }
        for fut in as_completed(futures):
            completed += 1
            if fut.result() is None:
                failures += 1
            if completed % 25 == 0 or completed == total:
                print(f"[{run_id}] {completed}/{total} done, {failures} failed", flush=True)
    print(f"[{run_id}] finished: {completed - failures} succeeded, {failures} failed")


def _parse_args(argv: list[str]) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Route B narrative extraction.")
    ap.add_argument("mode", nargs="?", default="reference",
                    choices=["reference", "votes", "unmasked-validation", "custom"])
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--provider", default=PROVIDER)
    ap.add_argument("--quantization", default=QUANTIZATION)
    ap.add_argument("--response-format", default=RESPONSE_FORMAT,
                    choices=["json_schema", "json_object"])
    ap.add_argument("--mask", choices=["on", "off"], default=None,
                    help="mask the mechanism-pass narrative (mode default if omitted)")
    ap.add_argument("--ids", type=Path, default=None,
                    help="CSV with a Crash_ID column restricting the narratives")
    ap.add_argument("--log", type=Path, default=None, help="JSONL log path")
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--passes", nargs="+", choices=["mechanism", "cue"], default=None)
    ap.add_argument("--temperature", type=float, default=None)
    ap.add_argument("--seed", type=int, default=None)
    return ap.parse_args(argv)


def main() -> None:
    args = _parse_args(sys.argv[1:])
    common = {"model": args.model, "provider": args.provider,
              "quantization": args.quantization,
              "response_format": args.response_format, "log_path": args.log}
    ids = (set(pd.read_csv(args.ids)["Crash_ID"].astype(int))
           if args.ids is not None else None)
    mask = None if args.mask is None else args.mask == "on"

    def pick(value, default):
        return default if value is None else value

    if args.mode == "reference":
        run(pick(args.run_id, "reference"),
            tuple(pick(args.passes, ("mechanism", "cue"))),
            temperature=pick(args.temperature, 0.0),
            seed=pick(args.seed, REFERENCE_SEED),
            masked_for_mechanism=pick(mask, True), ids=ids, **common)
    elif args.mode == "votes":
        seeds = (args.seed,) if args.seed is not None else VOTE_SEEDS
        for s in seeds:
            run(pick(args.run_id, f"vote{s}"), tuple(pick(args.passes, ("mechanism",))),
                temperature=pick(args.temperature, VOTE_TEMPERATURE), seed=s,
                masked_for_mechanism=pick(mask, True), ids=ids, **common)
    elif args.mode == "unmasked-validation":
        if ids is None:
            ids = set(pd.read_csv(DATA / "validation_ids.csv")["Crash_ID"].astype(int))
        run(pick(args.run_id, "unmasked_validation"),
            tuple(pick(args.passes, ("mechanism",))),
            temperature=pick(args.temperature, 0.0),
            seed=pick(args.seed, REFERENCE_SEED),
            masked_for_mechanism=pick(mask, False), ids=ids, **common)
    else:
        if args.run_id is None:
            raise SystemExit("custom mode needs --run-id")
        run(args.run_id, tuple(pick(args.passes, ("mechanism",))),
            temperature=pick(args.temperature, 0.0),
            seed=pick(args.seed, REFERENCE_SEED),
            masked_for_mechanism=pick(mask, True), ids=ids, **common)


if __name__ == "__main__":
    main()
