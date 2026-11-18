"""Injury-outcome masking for the mechanism extraction pass (Supplementary S2, part 1).

The mechanism pass must not see injury-outcome language, because the severity models
would otherwise regress a severity outcome on a covariate derived from that same
outcome. Every token matched by the patterns below is replaced by the sentinel
``[MASKED]`` before the narrative reaches the model. The count of masked tokens per
narrative is stored so the manuscript can report masking intensity, and so the
masked-versus-unmasked comparison on the validation subset is auditable.

The list is applied with word boundaries, case-insensitively. It is deliberately
broad: a false mask costs a little context, an unmasked outcome term costs the
identification of the whole mechanism layer.

Frozen 2026-09-14.
"""

from __future__ import annotations

import re

MASK_TOKEN = "[MASKED]"

# Each entry is a regular-expression fragment placed inside \b...\b.
# Grouped by what the term reveals, for the supplementary table.
MASK_PATTERNS: dict[str, tuple[str, ...]] = {
    "death": (
        r"fatal(?:ity|ities|ly)?",
        r"died",
        r"dies",
        r"dying",
        r"death",
        r"deaths",
        r"deceased",
        r"decedent",
        r"pronounced",
        r"doa",
        r"expired",
        r"coroner",
        r"medical examiner",
        r"autopsy",
        r"morgue",
    ),
    "consciousness": (
        r"unconscious(?:ness)?",
        r"unresponsive",
        r"non-?responsive",
        r"loss of consciousness",
        r"lost consciousness",
        r"knocked out",
        r"gcs",
        r"comatose",
        r"coma",
        r"semi-?conscious",
        r"altered mental status",
    ),
    "fracture_and_wound": (
        r"fractur(?:e|ed|es|ing)",
        r"broke",
        r"broken",
        r"breaks?",
        r"laceration(?:s)?",
        r"lacerated",
        r"abrasion(?:s)?",
        r"contusion(?:s)?",
        r"avuls(?:ion|ed)",
        r"amputat(?:ion|ed)",
        r"concussion",
        r"hemorrhag(?:e|ing)",
        r"bleeding",
        r"bled",
        r"blood(?:y)?",
        r"road rash",
        r"dislocat(?:ed|ion)",
        r"compound fracture",
        r"internal injur(?:y|ies)",
        r"skull",
        r"ribs?",
        r"femur",
        r"pelvis",
        r"collarbone",
        r"clavicle",
    ),
    "transport_and_care": (
        r"hospital(?:s|ized|ised)?",
        r"ems",
        r"e\.m\.s\.?",
        r"ambulance",
        r"medic(?:s|al)?",
        r"paramedic(?:s)?",
        r"transport(?:ed|ing|s|ation)?",
        r"transferred",
        r"airlift(?:ed)?",
        r"air ?lift",
        r"life ?flight",
        r"careflight",
        r"helicopter",
        r"emergency room",
        r"\ber\b",
        r"\bed\b",
        r"trauma center",
        r"icu",
        r"triage",
        r"stretcher",
        r"backboard",
        r"c-?collar",
        r"treated",
        r"treatment",
        r"first aid",
        r"cpr",
        r"resuscitat(?:e|ed|ion)",
        r"admitted",
    ),
    "injury_and_refusal": (
        r"injur(?:y|ies|ed|ious)",
        r"uninjured",
        r"non-?injur(?:y|ed)",
        r"\bhurt\b",
        r"\bpain(?:ful)?\b",
        r"complain(?:ed|ing|t|ts) of",
        r"refused",
        r"refusal",
        r"declined",
        r"incapacitat(?:ed|ing)",
        r"serious bodily injury",
        r"\bsbi\b",
        r"\bkabco\b",
        r"suspected serious",
    ),
}

_ALL_FRAGMENTS: tuple[str, ...] = tuple(
    frag for group in MASK_PATTERNS.values() for frag in group
)

# Longest-first so that multi-word terms mask as a unit.
_ORDERED = sorted(_ALL_FRAGMENTS, key=len, reverse=True)

MASK_REGEX = re.compile(
    r"\b(?:" + "|".join(_ORDERED) + r")\b",
    flags=re.IGNORECASE,
)


def mask_narrative(text: str) -> tuple[str, int]:
    """Return the masked narrative and the number of tokens masked.

    A run of adjacent masked tokens collapses to a single sentinel so that the
    model does not read a block of sentinels as emphasis, but the returned count
    is the number of matches before collapsing.
    """
    if not isinstance(text, str) or not text:
        return "", 0
    n = len(MASK_REGEX.findall(text))
    masked = MASK_REGEX.sub(MASK_TOKEN, text)
    masked = re.sub(
        r"(?:\[MASKED\])(?:[\s,;:./-]+\[MASKED\])+", MASK_TOKEN, masked
    )
    return masked, n


def mask_report() -> list[tuple[str, str]]:
    """Group name and comma-joined patterns, for Supplementary S2."""
    return [(group, ", ".join(pats)) for group, pats in MASK_PATTERNS.items()]


if __name__ == "__main__":
    demo = (
        "Unit 1, a scooter, entered the roadway from the sidewalk and was struck by "
        "Unit 2. Unit 1 sustained a fractured wrist and was transported to the "
        "hospital by EMS. Unit 2 driver was uninjured and refused treatment."
    )
    out, k = mask_narrative(demo)
    print(f"masked {k} tokens\n{out}")
