"""Canonical label inventory for the extraction, validation, and modeling code.

Nothing downstream defines its own list of mechanisms or injury cues. Every module
imports from here so that the coding app, the extraction prompt, the lexical
classifier, the validation metrics, and the model frame carry identical names in
identical order.

Frozen 2026-09-14 with `prespec.md`.
"""

from __future__ import annotations

# --- Ten narrative-derived pre-crash cues (mechanism pass, masked narrative) ---

MECHANISMS: tuple[str, ...] = (
    "wrong_way",
    "sidewalk_transition",
    "driveway_alley",
    "failure_to_yield",
    "signal_violation",
    "swerve_loss_control",
    "dooring",
    "distraction_impairment",
    "vehicle_turning_across",
    "lane_positioning",
)

MECHANISM_TITLES: dict[str, str] = {
    "wrong_way": "Wrong-way or contraflow travel",
    "sidewalk_transition": "Sidewalk to roadway transition",
    "driveway_alley": "Entry from a driveway or alley",
    "failure_to_yield": "Failure to yield the right of way",
    "signal_violation": "Signal or stop-sign violation",
    "swerve_loss_control": "Swerve or loss of control",
    "dooring": "Struck by an opening vehicle door",
    "distraction_impairment": "Distraction or impairment cue",
    "vehicle_turning_across": "Motor vehicle turning across the rider's path",
    "lane_positioning": "Rider lane positioning",
}

# --- Five narrative injury cues (cue pass, unmasked narrative) ---

INJURY_CUES: tuple[str, ...] = (
    "transported_hospital",
    "unconsciousness",
    "fracture",
    "fatality",
    "explicit_no_injury",
)

INJURY_CUE_TITLES: dict[str, str] = {
    "transported_hospital": "Transported for care",
    "unconsciousness": "Unconscious or unresponsive",
    "fracture": "Fracture or severe laceration",
    "fatality": "Death",
    "explicit_no_injury": "Explicit statement of no injury",
}

ALL_LABELS: tuple[str, ...] = MECHANISMS + INJURY_CUES

assert len(MECHANISMS) == 10
assert len(INJURY_CUES) == 5
assert len(ALL_LABELS) == 15
assert len(set(ALL_LABELS)) == 15
