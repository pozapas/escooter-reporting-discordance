"""Constructed worked examples for the published copy of the coding guide.

The guide the coders used illustrates each rule with short quotations from crash
narratives. The narrative text is not redistributed, so every published copy of the guide
(the Supplementary material and the replication files) replaces each quotation with a
constructed example that makes the same point and receives the same label. The examples
below were written for this purpose and describe no particular crash.

Each entry names the guide section it belongs to, so a change in the guide that moves an
example to another section stops the build instead of mislabeling it.
"""

from __future__ import annotations

import re

# (guide section, constructed example), in the order the quotations appear in the guide.
EXAMPLES: tuple[tuple[str, str], ...] = (
    ("5.1", "Unit 2, an e-scooter, rode north in the bike lane on the side of the street meant for southbound traffic."),
    ("5.1", "Unit 2, an e-scooter, rode in the right travel lane because the street had no sidewalk."),
    ("5.2", "Unit 2 rode an e-scooter east along the sidewalk, then went off the curb to cross the street."),
    ("5.2", "Unit 2 rode an e-scooter west along the sidewalk, could not slow down and hit Unit 1 as it backed out of a driveway."),
    ("5.2", "Unit 2 crossed the street in the crosswalk from the west sidewalk and was hit as it reached the east sidewalk."),
    ("5.2", "Unit 2 rode south through the crosswalk over the eastbound lanes."),
    ("5.3", "Unit 2 pulled out of a business parking lot and did not yield as it left the private drive."),
    ("5.3", "Unit 1 was reversing out of a residential driveway."),
    ("5.4", "Unit 2 did not give Unit 1 the right of way."),
    ("5.4", "Unit 1 said its light was green but did not yield to Unit 2 in the crosswalk."),
    ("5.4", "Unit 2 did not yield at the yield sign."),
    ("5.4", "Unit 1 turned left at the intersection and hit Unit 2, which was passing in front of it."),
    ("5.4", "Unit 1 did not stop at the stop sign and hit the scooter."),
    ("5.4", "Unit 1 was going too fast and hit Unit 2 as Unit 2 came out of a private driveway."),
    ("5.4", "Unit 1 possibly did not yield at the stop sign, and Unit 2 possibly did not yield to a vehicle."),
    ("5.5", "Unit 1 entered the intersection on a red signal and hit Unit 2."),
    ("5.5", "Unit 2 rode into the road without looking, against the don't walk signal."),
    ("5.5", "Unit 1 halted at the stop sign, checked both directions and thought it was clear before turning left."),
    ("5.6", "Unit 2 tried to swerve away, lost control and slid into the side of Unit 1."),
    ("5.6", "Unit 2 said the scooter could not be stopped in time to avoid Unit 1."),
    ("5.6", "The collision threw the young rider off the scooter onto the pavement."),
    ("5.7", "Unit 2 hit the front passenger door of Unit 1 while Unit 1 was moving."),
    ("5.7", "The parked driver swung the door open as the scooter passed, and the rider hit it."),
    ("5.8", "The driver of Unit 1 was looking at a phone for directions."),
    ("5.8", "Unit 2 never saw Unit 1 and ran into its side."),
    ("5.8", "The driver of Unit 1 said the scooter appeared out of nowhere."),
    ("5.9", "Unit 1 turned right into a driveway without yielding to Unit 2."),
    ("5.9", "Unit 1 turned left and hit Unit 2, which was riding across the road."),
    ("5.9", "Unit 1 was going straight west, and Unit 2 did not yield to Unit 1."),
    ("5.10", "Unit 2 rode an e-scooter north in the bike lane behind Unit 1."),
    ("5.10", "Unit 2 rode an e-scooter in the gap between the parked cars and the travel lane."),
    ("6.1", "An ambulance took Unit 2 to a hospital."),
    ("6.1", "Unit 2 was hurt but declined to be taken to a hospital."),
    ("6.1", "The rider's parents drove him to the hospital."),
    ("6.2", "Unit 2 was conscious and talking but in pain."),
    ("6.3", "Unit 2 has no broken bones so far, but both legs are swollen and scraped."),
    ("6.3", "Unit 2 has a small cut on one hand."),
    ("6.3", "The rider had a serious injury to the right leg."),
    ("6.5", "EMS checked both riders of Unit 2, and both declined treatment."),
    ("6.5", "Unit 1 stopped to ask whether Unit 2 was all right, and Unit 2 said yes."),
)

# The one quotation the guide makes inside a sentence rather than on its own line. It
# is found by its form, a quoted sentence followed by its label, so that this file does
# not reproduce the narrative it replaces.
INLINE: tuple[tuple[str, str], ...] = (
    (r'"[^"\n]+" is a \*\*1\*\*',
     '"Crept past the stop line to get a better view" is a **1**'),
)

NOTE = ("> *The worked examples in this copy are constructed. The guide the coders used "
        "quoted crash narratives here; each quotation is replaced by an example that makes "
        "the same point and keeps the same label, because the narrative text is not "
        "redistributed.*\n\n")

_QUOTE = re.compile(r'^> ".*?"(?=\s*(?:→|->|$))', re.M)
_SECTION = re.compile(r"^###\s+(\d+\.\d+)\s", re.M)


def constructed(text: str) -> str:
    """The guide with every quoted narrative replaced by its constructed example.

    The released guide, docs/coding_guide.md, is already in this form and carries the
    note, so it is returned unchanged.
    """
    if NOTE in text:
        return text
    quotes = list(_QUOTE.finditer(text))
    if len(quotes) != len(EXAMPLES):
        raise SystemExit(f"coding guide: {len(quotes)} quoted examples, "
                         f"{len(EXAMPLES)} constructed examples")
    sections = [(m.start(), m.group(1)) for m in _SECTION.finditer(text)]
    out, last = [], 0
    for m, (want, example) in zip(quotes, EXAMPLES):
        here = [s for pos, s in sections if pos < m.start()]
        if not here or here[-1] != want:
            raise SystemExit(f"coding guide: example for section {want} sits in "
                             f"section {here[-1] if here else 'none'}")
        out.append(text[last:m.start()])
        out.append(f'> "{example}"')
        last = m.end()
    out.append(text[last:])
    text = "".join(out)
    for pattern, new in INLINE:
        text, n = re.subn(pattern, lambda _m: new, text)
        if n != 1:
            raise SystemExit(f"coding guide: inline quotation found {n} time(s)")
    # The note goes after the title block, before the first section.
    first = text.find("\n## ")
    return text[:first + 1] + NOTE + text[first + 1:] if first >= 0 else NOTE + text
