"""Route A: deterministic lexical-anchor classifier (Supplementary S2, part 2).

Route A assigns the ten pre-crash cues from regular-expression dictionaries alone.
It has no parameters, is not fitted to any data, and therefore cannot leak the
validation labels. It runs on the same masked narrative the model's mechanism pass
receives, so the two routes see identical text.

Police narratives do not number the units consistently: the scooter is usually
"Unit 2" but is "Unit 1" in some reports, and a few reports renumber the units
partway through. The classifier therefore resolves which unit label denotes the
scooter before it attributes any action, exactly as the coding guide instructs the
human coders to do. The resolved unit is returned so that it can be compared with
the structured unit fields as a check on the whole procedure.

Frozen 2026-09-14.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from labels import MECHANISMS

# --------------------------------------------------------------------------
# Unit references
# --------------------------------------------------------------------------

# "Unit 1", "Unit #2", "UNIT1", "U1", "Person 2", "Vehicle 1", "Veh 2"
UNIT_REF = re.compile(
    r"\b(?:unit|units|u|person|persons|vehicle|veh|driver of unit|operator of unit)\s*#?\s*(?P<num>[12])\b",
    re.IGNORECASE,
)

SCOOTER_WORDS = re.compile(
    r"(?:"
    r"e-?\s?scooter|electric(?:al)?\s+(?:power\s+propelled\s+)?scooter|motor(?:ized|\s*assisted)?\s+scooter"
    r"|stand\s?-?up\s+scooter|sit\s?-?down\s+(?:motor\s+)?scooter|motorized\s+conveyance"
    r"|\bscooter\b|\bbird\b|\blime\b|\bgotrax\b|\bjetta\b|\bspin\b|\bveo\b"
    r")",
    re.IGNORECASE,
)

SENT_SPLIT = re.compile(r"(?<=[.!?])\s+|(?<=\.)(?=[A-Z]{2,})|\n+")


def _sentences(text: str) -> list[str]:
    parts = [s.strip() for s in SENT_SPLIT.split(text) if s and s.strip()]
    return parts or ([text.strip()] if text.strip() else [])


def resolve_scooter_unit(text: str) -> tuple[int | None, float]:
    """Return the unit number denoting the scooter and a confidence in [0, 1].

    Every scooter word votes for the unit reference nearest to it on its left within
    120 characters, or, failing that, the nearest on its right within 60. The unit
    with the most votes wins; confidence is its vote share.
    """
    if not text:
        return None, 0.0
    refs = [(m.start(), int(m.group("num"))) for m in UNIT_REF.finditer(text)]
    if not refs:
        return None, 0.0

    votes = {1: 0, 2: 0}
    for sw in SCOOTER_WORDS.finditer(text):
        pos = sw.start()
        left = [(pos - p, n) for p, n in refs if 0 <= pos - p <= 120]
        right = [(p - pos, n) for p, n in refs if 0 < p - pos <= 60]
        if left:
            votes[min(left)[1]] += 1
        elif right:
            votes[min(right)[1]] += 1

    total = votes[1] + votes[2]
    if total == 0:
        # No scooter word could be tied to a unit. Fall back to the study-wide
        # convention, flagged by zero confidence so it is auditable.
        return 2, 0.0
    unit = 1 if votes[1] > votes[2] else 2
    return unit, max(votes.values()) / total


def _subject_unit(sentence: str) -> int | None:
    """The unit a sentence is about: the first unit reference it contains."""
    m = UNIT_REF.search(sentence)
    return int(m.group("num")) if m else None


# --------------------------------------------------------------------------
# Lexical dictionaries
# --------------------------------------------------------------------------
# actor: "rider" = must be the scooter unit's sentence; "driver" = the motor
# vehicle's; "any" = either party; "none" = attribution not required.

@dataclass(frozen=True)
class Rule:
    include: tuple[str, ...]
    actor: str = "any"
    exclude: tuple[str, ...] = ()
    require_all: tuple[tuple[str, ...], ...] = ()
    label: str = field(default="")


RULES: dict[str, Rule] = {
    "wrong_way": Rule(
        include=(
            r"wrong\s*way",
            r"against\s+(?:the\s+)?(?:flow\s+of\s+)?traffic",
            r"opposite\s+direction\s+of\s+travel",
            r"contra\s*-?flow",
            r"on\s+the\s+(?:north|south|east|west)bound\s+side\s+of",
            r"facing\s+oncoming\s+traffic",
        ),
        actor="rider",
    ),
    "sidewalk_transition": Rule(
        include=(),
        actor="rider",
        require_all=(
            (
                r"\bsidewalk\b", r"\bside\s?walk\b", r"\bcurb\b", r"\bwalkway\b",
                r"\bfrom\s+the\s+(?:north|south|east|west)\s+curb\b", r"\btrail\b",
            ),
            (
                r"\bcross(?:ed|ing|es)?\b", r"\benter(?:ed|ing|s)?\b",
                r"\bproceed(?:ed|ing|s)?\s+to\s+cross\b", r"\bonto\s+the\s+(?:road|street)",
                r"\binto\s+the\s+(?:road|street|lane|path)",
                r"\battempt(?:ed|ing)?\s+to\s+cross\b", r"\brode\s+(?:out\s+)?into\b",
            ),
        ),
        exclude=(r"\bdriveway\b", r"\balley\b"),
    ),
    "driveway_alley": Rule(
        include=(
            r"\bdriveway\b", r"\bdrive\s?way\b", r"\balley(?:way)?\b",
            r"private\s+(?:drive|road|parking)", r"parking\s+lot\s+(?:entrance|exit)",
            r"(?:enter|exit|leav)(?:ing|ed)?\s+the\s+(?:parking\s+lot|private\s+drive)",
            r"backing\s+out\s+of\s+(?:a\s+)?driveway",
        ),
        actor="any",
    ),
    # Stated-only, matching the coding guide version 1.2. Nothing is inferred from
    # the geometry of the collision: in every crash in these data one unit was in
    # front of the other, so a geometric rule would mark nearly every narrative.
    # The described-situation cases the earlier versions carried are already held by
    # vehicle_turning_across, signal_violation, driveway_alley and sidewalk_transition.
    "failure_to_yield": Rule(
        include=(
            r"fail(?:ed|ing|s|ure)?\s+to\s+yield",
            r"did\s+not\s+yield",
            r"without\s+yielding",
            r"yield(?:ed|ing)?\s+(?:the\s+)?right[\s-]?of[\s-]?way",
            r"right[\s-]?of[\s-]?way\s+violation",
            r"violat\w*\s+(?:the\s+)?right[\s-]?of[\s-]?way",
            r"disregard\w*\s+(?:the\s+)?right[\s-]?of[\s-]?way",
            r"\brow\s+violation\b",
            r"fail(?:ed|ing|s|ure)?\s+to\s+yield\s+row",
        ),
        actor="any",
    ),
    "signal_violation": Rule(
        include=(
            r"ran\s+(?:the\s+)?(?:red\s+light|stop\s+sign|light)",
            r"disregard(?:ed|ing|s)?\s+(?:the\s+)?(?:red\s+light|stop(?:\s+and\s+go)?\s+(?:sign|signal)|signal|traffic\s+(?:light|signal|control)|no\s+walking\s+signal)",
            r"fail(?:ed|ing|s)?\s+to\s+stop\s+at\s+the\s+(?:designated\s+)?stop\s+sign",
            r"bypass(?:ed|ing)?\s+a\s+stop\s+sign",
            r"roll(?:ed|ing)?\s+(?:through|past)\s+the\s+stop",
            r"did\s+not\s+stop\s+at\s+the\s+stop\s+sign",
            r"run(?:ning)?\s+the\s+red\s+light",
            r"violat(?:ed|ion)\s+of\s+(?:a\s+)?(?:red\s+light|stop\s+sign)",
        ),
        actor="any",
    ),
    "swerve_loss_control": Rule(
        include=(
            r"lost\s+control", r"loss\s+of\s+control", r"los(?:e|ing)\s+control",
            r"swerv(?:e|ed|ing)", r"slid(?:e|ing)?\b", r"skid(?:ded|ding)?",
            r"unable\s+to\s+stop", r"could\s+not\s+stop\s+in\s+time",
            r"fail(?:ed|ing|s)?\s+to\s+control\s+speed",
            r"evasive\s+action",
            r"brake(?:d)?\s+aggressively", r"brak(?:e|ed|ing)\s+abruptly",
        ),
        actor="rider",
        exclude=(r"knock(?:ed|ing)\s+(?:the\s+)?\w*\s*off", r"impact\s+(?:from|knocked)"),
    ),
    "dooring": Rule(
        include=(),
        actor="any",
        require_all=(
            (r"\bdoor\b", r"\bdoors\b"),
            (r"\bopen(?:ed|ing|s)?\b", r"\bdoored\b", r"\bswung\b", r"\bajar\b"),
        ),
        exclude=(r"door\s+(?:panel|handle|frame)", r"passenger\s+side\s+door\s+of\s+unit"),
    ),
    "distraction_impairment": Rule(
        include=(
            r"cell\s*phone", r"\bphone\b", r"text(?:ing|ed)?\b",
            r"distract(?:ed|ion|ing)", r"inattenti(?:on|ve)",
            r"not\s+paying\s+attention", r"looking\s+(?:away|down)",
            r"navigat(?:e|ing|ion)",
            r"\bintoxicat(?:ed|ion)\b", r"\bimpair(?:ed|ment)\b",
            r"under\s+the\s+influence", r"\bdwi\b", r"\bdui\b",
            r"odor\s+of\s+(?:an\s+)?alcohol", r"smell(?:ed|ing)?\s+of\s+alcohol",
            r"open\s+container", r"field\s+sobriety", r"\bbreathalyzer\b",
            r"did\s+not\s+look", r"did\s+not\s+observe", r"fail(?:ed|ing)?\s+to\s+(?:see|observe|look)",
            r"never\s+saw",
        ),
        actor="any",
        exclude=(r"came\s+out\s+of\s+nowhere",),
    ),
    "vehicle_turning_across": Rule(
        include=(
            r"turn(?:ed|ing|s)?\s+(?:left|right|north|south|east|west)",
            r"attempt(?:ed|ing)?\s+to\s+(?:make\s+a\s+)?(?:left|right|u-?)\s*turn",
            r"mak(?:e|ing)\s+a\s+(?:left|right|northbound|southbound|eastbound|westbound|u-?)\s*turn",
            r"turn(?:ed|ing)?\s+(?:in)?to\s+(?:the\s+)?(?:driveway|parking|lot|street|road)",
            r"conduct(?:ed|ing)?\s+a\s+u-?turn",
            r"\bu-?turn\b",
            r"proceeded\s+(?:forward\s+)?to\s+turn",
        ),
        actor="driver",
    ),
    "lane_positioning": Rule(
        include=(
            r"\bbike\s+lane\b", r"\bbicycle\s+lane\b",
            r"\bin\s+(?:the\s+)?(?:right|left|inside|outside|number\s+one|middle|travel)\s+lane\b",
            r"\bin\s+lane\s*\d", r"\blane\s*\d\s*/\s*\d",
            r"\bimproved\s+shoulder\b", r"\bon\s+the\s+shoulder\b",
            r"\bcenter\s+lane\b", r"\bmedian\b",
            r"\bbetween\s+the\s+(?:allowed\s+)?parking\s+space",
            r"\bin\s+the\s+lane\s+for\s+traffic\b",
        ),
        actor="rider",
        exclude=(r"\bsidewalk\b",),
    ),
}

_COMPILED: dict[str, dict] = {}
for name, rule in RULES.items():
    _COMPILED[name] = {
        "include": [re.compile(p, re.IGNORECASE) for p in rule.include],
        "exclude": [re.compile(p, re.IGNORECASE) for p in rule.exclude],
        "require_all": [
            [re.compile(p, re.IGNORECASE) for p in group] for group in rule.require_all
        ],
        "actor": rule.actor,
    }

ACTOR_CUES = ("failure_to_yield", "signal_violation")


def _sentence_hits(sentence: str, spec: dict) -> bool:
    for ex in spec["exclude"]:
        if ex.search(sentence):
            return False
    if spec["require_all"]:
        return all(any(p.search(sentence) for p in group) for group in spec["require_all"])
    return any(p.search(sentence) for p in spec["include"])


def classify(text: str) -> dict:
    """Assign the ten cues to one narrative.

    Returns the ten 0/1 labels, the resolved scooter unit and its confidence, the
    actor attribution for the two shared cues, and the matched sentence for each
    positive label so that any assignment can be inspected.
    """
    out: dict[str, object] = {m: 0 for m in MECHANISMS}
    evidence: dict[str, str] = {}
    actors: dict[str, str] = {}

    scooter_unit, conf = resolve_scooter_unit(text or "")
    vehicle_unit = None if scooter_unit is None else (2 if scooter_unit == 1 else 1)
    out["scooter_unit"] = scooter_unit
    out["scooter_unit_confidence"] = round(conf, 3)

    sents = _sentences(text or "")
    # Multi-part cues (a stated position, then a movement) are often split across
    # two adjacent sentences, so those rules are evaluated over a two-sentence
    # window. Single-anchor rules stay sentence-local.
    windows: list[tuple[str, int | None]] = []
    for i, sent in enumerate(sents):
        windows.append((sent, _subject_unit(sent)))
    for i, sent in enumerate(sents):
        if i + 1 < len(sents):
            joined = sent + " " + sents[i + 1]
            windows.append((joined, _subject_unit(joined)))

    for sent, subj in windows:
        for mech, spec in _COMPILED.items():
            if out[mech] == 1 and mech not in ACTOR_CUES:
                continue
            if not _sentence_hits(sent, spec):
                continue
            actor_req = spec["actor"]
            if actor_req == "rider" and subj is not None and subj != scooter_unit:
                continue
            if actor_req == "driver" and subj is not None and subj != vehicle_unit:
                continue
            if out[mech] == 0:
                out[mech] = 1
                evidence[mech] = sent[:300]
            if mech in ACTOR_CUES:
                if subj == scooter_unit:
                    who = "rider"
                elif subj == vehicle_unit:
                    who = "driver"
                else:
                    who = "unclear"
                prev = actors.get(mech)
                if prev is None:
                    actors[mech] = who
                elif prev != who and "unclear" not in (prev, who):
                    actors[mech] = "both"

    for cue in ACTOR_CUES:
        out[f"{cue}_actor"] = actors.get(cue, "") if out[cue] == 1 else ""
    out["evidence"] = evidence
    return out


if __name__ == "__main__":
    import json
    import sys
    from pathlib import Path

    import pandas as pd

    DATA = Path(__file__).resolve().parents[1] / "data"
    pilot = pd.read_csv(DATA / "pilot_ids.csv")
    rows = []
    for _, r in pilot.iterrows():
        res = classify(str(r["narrative"]))
        res.pop("evidence")
        res["Crash_ID"] = r["Crash_ID"]
        rows.append(res)
    out = pd.DataFrame(rows)
    print("Route A on the 50 guide-development narratives (positives per cue):")
    for m in MECHANISMS:
        print(f"  {m:24s} {int(out[m].sum()):3d}")
    print("\nscooter unit resolved:", out["scooter_unit"].value_counts().to_dict())
    print("mean confidence:", round(float(out["scooter_unit_confidence"].mean()), 3))
    print("zero-confidence (fallback) narratives:", int((out["scooter_unit_confidence"] == 0).sum()))
