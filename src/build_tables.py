"""Generate the main-text tables from the frozen numbers.

The manuscript has one table system and every table uses it: a gold header
row (`\\headerrow` with `\\hcell`), alternating cream rows (`\\altrow`), panel bands
(`\\panelrow` colour), `threeparttable` with keyed `tablenotes`, `tabularx` so a
description column wraps, `xltabular` where a table runs past a page, and `\\arraystretch`
for breathing room. A short caption states what the table is; the detail lives in the notes
underneath.

Numbering is by document order and is load-bearing, because the text refers to the
tables by number:

    Table 1  \\S3.3  structured covariates            Table 4  \\S5.4  model estimates
    Table 2  \\S5.2  extraction performance           Table 5  \\S5.5  specification ladder
    Table 3  \\S5.3  the audit, two panels            Table 6  \\S5.6  stability tests

Captions print in black except for the words a caller wraps in a highlight macro, and the
table number takes a highlight colour.

Notes follow the prose rules of the manuscript: each opens with "Note:" in italics, has no
colons or dashes as punctuation and no bold run-in labels, and every number in it is read
from `numbers.json`.

Every value comes from `numbers.json`, so every printed value agrees with it by
construction. The file read can be redirected with the environment variable
ESCOOTER_NUMBERS, which exists so a rebuilt numbers file can be checked before it replaces
the frozen one.

Run with:  python src/build_tables.py
"""

from __future__ import annotations

import json
import os
import sys
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "results" / "tables"

sys.path.insert(0, str(Path(__file__).resolve().parent))
from audit_indicator import MAPPING  # noqa: E402

SEV3 = ("O", "BC", "KA")
N_LEVELS = ("no cue", "O-consistent", "BC-consistent", "KA-consistent")
BS = chr(92)

# Row-percentage threshold above which a Table 3 cell is shaded. A display rule, printed
# in the note from this constant so the note and the shading cannot disagree.
SHADE_ABOVE = 55

ESCAPE = {"&": r"\&", "%": r"\%", "_": r"\_", "#": r"\#"}

# Readable names for the covariates. A table should not ask a reader to decode a column
# name: "cf" is the contributing-factor prefix, "ctrl" traffic control, "coll" collision
# manner, "intx" intersection relation, and a reader has no way to expand any of them.
NAMES: dict[str, str] = {
    "coord_missing": "Crash coordinates absent",
    "age_unknown": "Rider age not recorded",
    "speed_unknown": "Speed limit not recorded",
    "gender_unknown": "Rider gender not recorded",
    "eth_other_or_unknown": "Ethnicity other or not recorded",
    "veh_other_unknown": "Striking vehicle other or not recorded",
    "light_dark_or_unknown": "Dark or light not recorded",
    "speed_35to40": "Speed limit 35 to 40 mph",
    "speed_ge45": "Speed limit 45 mph or above",
    "age_under25": "Rider under 25",
    "age_45plus": "Rider 45 or above",
    "gender_female": "Rider female",
    "eth_black": "Rider Black",
    "eth_hispanic": "Rider Hispanic",
    "veh_suv": "Striking vehicle sport utility",
    "veh_pickup_truck_van": "Striking vehicle pickup, truck or van",
    "coll_other": "Collision manner other",
    "coll_single_mv_straight": "Collision with vehicle going straight",
    "coll_turning": "Collision while turning",
    "intx_intersection_or_driveway": "At intersection or driveway",
    "road_not_city_street": "Not a city street",
    "road_density_km_100m": "Road density within 100 m",
    "poverty_share": "Poverty share, block group",
    "urban_area_200k": "Inside an urban area of 200,000 or more",
    "bike_facility_nearest": "Distance to nearest cycle facility",
    "bike_facility_100m": "Cycle facility within 100 m",
    "nighttime": "Night",
    "large_city": "Major city",
    "weekend": "Weekend",
    "is_weekend": "Weekend",
    "ctrl_signal": "Traffic signal present",
    "ctrl_stop_sign": "Stop sign present",
    "cf_rider_failed_to_yield": "Officer cited rider failed to yield",
    "cf_driver_failed_to_yield": "Officer cited driver failed to yield",
    "cf_driver_inattention": "Officer cited driver inattention",
    "cf_rider_disregarded_signal": "Officer cited rider disregarded signal",
    "cf_disregard_control": "Officer cited disregard of control",
    "cf_speed_related": "Officer cited speed",
    "cf_wrong_side_or_way": "Officer cited wrong side or way",
    "cf_driveway_entry_exit": "Officer cited driveway entry or exit",
    "sidewalk_transition": "Cue, sidewalk to roadway transition",
    "failure_to_yield": "Cue, failure to yield",
    "signal_violation": "Cue, signal violation",
    "lane_positioning": "Cue, lane positioning",
    "vehicle_turning_across": "Cue, vehicle turning across",
    "driveway_alley_entry": "Cue, driveway or alley entry",
    "driveway_alley": "Cue, driveway or alley entry",
    "distraction_impairment": "Cue, distraction or impairment",
    "wrong_way_riding": "Cue, wrong-way riding",
    "wrong_way": "Cue, wrong-way riding",
    "dooring": "Cue, dooring",
    "swerve_loss_control": "Cue, swerve or loss of control",
    "pedestrian_conflict": "Cue, pedestrian conflict",
    "unconsciousness": "Cue, unconsciousness",
    "fracture": "Cue, fracture",
    "fatality": "Cue, fatality",
    "transported_hospital": "Cue, transported to hospital",
    "explicit_no_injury": "Cue, explicit no injury",
}

# The fifteen labels without the "Cue," prefix, for tables that list only cues.
CUE_NAMES: dict[str, str] = {
    "wrong_way": "Wrong-way riding",
    "sidewalk_transition": "Sidewalk to roadway transition",
    "driveway_alley": "Driveway or alley entry",
    "failure_to_yield": "Failure to yield",
    "signal_violation": "Signal violation",
    "swerve_loss_control": "Swerve or loss of control",
    "dooring": "Dooring",
    "distraction_impairment": "Distraction or impairment",
    "vehicle_turning_across": "Vehicle turning across",
    "lane_positioning": "Lane positioning",
    "transported_hospital": "Transported to hospital",
    "unconsciousness": "Unconsciousness",
    "fracture": "Fracture",
    "fatality": "Fatality",
    "explicit_no_injury": "Explicit no injury",
}


def pretty(name: str) -> str:
    return NAMES.get(name, name.replace("_", " "))


def cue_name(name: str) -> str:
    return CUE_NAMES.get(name, name.replace("_", " "))


def tex(s: str) -> str:
    return "".join(ESCAPE.get(c, c) for c in str(s))


# One rounding rule for every generated table and appendix: round half up at the printed
# precision, on the decimal value as stored. Python's own float formatting rounds the binary
# value, so 0.1705 can print as 0.170 or 0.171 depending on its representation, and two
# tables printing one stored value could disagree. A stored value that is itself rounded
# can still be rounded twice, so a cell that can be computed from counts is computed from
# them, and ties() lists the stored values whose next digit is exactly 5.
TIES: set[str] = set()


def _dec(x) -> Decimal:
    return Decimal(repr(float(x)))


def half_up(x, places: int) -> Decimal:
    """x rounded half up at `places` decimals, as a Decimal."""
    d = x if isinstance(x, Decimal) else _dec(x)
    norm = d.normalize()
    exp = norm.as_tuple().exponent
    if isinstance(exp, int) and -exp == places + 1 and norm.as_tuple().digits[-1] == 5:
        TIES.add(f"{norm} at {places}")
    return d.quantize(Decimal(1).scaleb(-places), rounding=ROUND_HALF_UP)


def rval(x, places: int) -> float:
    """x as printed at `places` decimals, as a number, for best-value comparisons."""
    return float(half_up(x, places))


def rnd(x, places: int, comma: bool = False) -> str:
    """x printed at `places` decimals, rounded half up, without a negative zero."""
    d = half_up(x, places)
    if d == 0:
        d = abs(d)
    return f"{d:,.{places}f}" if comma else f"{d:.{places}f}"


def pct(x: float, places: int = 1) -> str:
    """A share printed as a percentage, rounded half up on the decimal value."""
    d = half_up(_dec(x) * 100, places)
    if d == 0:
        d = abs(d)
    return f"{d:.{places}f}"


def prf(e: dict) -> tuple[float, float, float]:
    """Precision, recall and F1 from the confusion counts of a record, checked against the
    stored values, so a printed score is never a stored rounded score rounded again."""
    tp, fp, fn = (float(e[k]) for k in ("tp", "fp", "fn"))
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * tp / (2 * tp + fp + fn) if tp else 0.0
    for name, v in (("precision", p), ("recall", r), ("f1", f)):
        if name in e and abs(e[name] - v) > 6e-4:
            raise SystemExit(f"{name} {e[name]} does not follow from the counts ({v:.4f})")
    return p, r, f


def macro_f1(per: dict, labels, stored: float | None = None) -> float:
    """The mean F1 over `labels`, each from its confusion counts, checked against the
    stored mean."""
    labels = list(labels)
    value = sum(prf(per[k])[2] for k in labels) / len(labels)
    if stored is not None and abs(stored - value) > 6e-4:
        raise SystemExit(f"macro F1 {stored} does not follow from the counts ({value:.4f})")
    return value


def fmt(x: float, places: int = 3) -> str:
    """A number as the manuscript prints it, with a real minus sign.

    A value that rounds to zero without being zero gets the places it needs: printing
    $-0.00013$ as "$-$0.000" shows a highlighted cell that appears to be zero, which is
    the one cell a sceptical reader checks first.
    """
    s = rnd(x, places)
    while float(s) == 0 and x != 0 and places < 6:
        places += 1
        s = rnd(x, places)
    return s.replace("-", "$-$") if s.startswith("-") else s


def pfmt(p: float, places: int = 3) -> str:
    """A p-value as the manuscript prints it.

    One rule for every table: a value at or above 0.995 is shown as "> 0.99", a value
    below 0.001 as "< 0.001", and any other value at three decimals, so no p-value reads
    as exactly one or exactly zero.
    """
    if p is None:
        raise ValueError("p-value missing")
    if p >= 0.995:
        return r"$>0.99$"
    if p < 0.001:
        return r"$<0.001$"
    return rnd(p, places)


WORDS = ("zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
         "ten")


def words(n: int) -> str:
    """A small count spelled out, as running text prints it."""
    return WORDS[n] if 0 <= n < len(WORDS) else str(n)


def numbers_path() -> Path:
    override = os.environ.get("ESCOOTER_NUMBERS")
    return Path(override) if override else DATA / "numbers.json"


def load_numbers() -> dict:
    return json.loads(numbers_path().read_text(encoding="utf-8"))


def req(d: dict, *keys: str):
    """A nested value, or a stop naming the key that is missing.

    A missing value must stop the build. Printing a dash in its place would publish a
    table that looks complete and is not.
    """
    cur = d
    trail = []
    for k in keys:
        trail.append(str(k))
        if isinstance(cur, dict) and k in cur:
            cur = cur[k]
        elif isinstance(cur, list) and isinstance(k, int) and k < len(cur):
            cur = cur[k]
        else:
            raise SystemExit("numbers.json has no " + ".".join(trail))
    return cur


def header(name: str, what: str) -> str:
    return (f"% {name}: {what}\n"
            f"% GENERATED by src/build_tables.py from data/numbers.json.\n"
            f"% Do not edit by hand; rerun the generator.\n")


def headerline(cells: list[str]) -> str:
    """The gold header row."""
    return (BS + "headerrow\n"
            + " & ".join(BS + "hcell{" + c + "}" for c in cells) + r" \\")


def banded(rows: list[str]) -> str:
    """Alternating cream rows."""
    out = []
    for i, r in enumerate(rows):
        if i % 2 == 1:
            out.append(BS + "altrow")
        out.append(r)
    return "\n".join(out)


def panel(title: str, ncols: int, rule: bool = False) -> str:
    r"""A band naming a block of rows.

    Coloured with \rowcolor rather than \cellcolor. The band spans every column, so both
    of the default \tabcolsep overhangs fall outside the rules rather than into a gutter
    between columns. \rowcolor takes overhang arguments and \cellcolor does not, so this
    is the form that can be squared off; \panelrow in the preamble is the same idea.

    `rule` closes the band with a \midrule. It is wanted where a gold header row follows,
    because two coloured bands stacked directly on each other read as one block.
    """
    out = (BS + "rowcolor{PanelBg}\n"
           + r"\multicolumn{" + str(ncols) + r"}{l}{\textbf{"
           + title + r"}} \\")
    return out + "\n" + BS + "midrule" if rule else out


def float_caption(label_colour: str, text: str) -> str:
    r"""A float caption whose number takes a highlight colour.

    The number takes `label_colour`; the caption text is printed as given, in black
    except for the words the caller wraps in \hlone, \hlmulti and so on.
    """
    return BS + "hllabelcolour{" + label_colour + "}\n" + BS + "caption{" + text + "}"


# ---------------------------------------------------------------- Table 1

# How each variable is constructed. A reader should not have to infer what "Crash context"
# contains from the names of its levels, and the dependent variable should be described in
# the same table as the covariates rather than only in the prose.
DESCRIBE: dict[str, str] = {
    "Dependent variable": "Officer-coded KABCO injury severity, collapsed to three "
                          "ordered classes",
    "Posted speed limit": "Posted limit at the crash location, in bands",
    "Rider age": "Rider age in years, in bands",
    "Rider gender": "Rider gender as recorded on the crash report",
    "Rider ethnicity": "Rider ethnicity as recorded on the crash report",
    "Striking vehicle": "Body type of the vehicle that struck the rider",
    "Light condition": "Light at the time of the crash",
    "Collision manner": "Movement of the striking vehicle relative to the rider",
    "Intersection relation": "Whether the crash occurred at a junction or driveway",
    "Traffic control": "Control device present at the crash location",
    "Roadway class": "Roadway type carrying the crash",
    "Crash context": "Time and location circumstances of the crash",
    "Officer-coded contributing factors": "Recorded by the officer, enter at M0c",
    "Narrative-derived pre-crash cues": "Extracted from the narrative, enter at M1",
    "Spatial covariates": "Joined to the crash point by spatial join",
}

SEV3_LONG = {"O": "Not injured (O)",
             "BC": "Non-incapacitating or possible injury (BC)",
             "KA": "Killed or incapacitating injury (KA)"}


def table1(numbers: dict) -> str:
    r"""Variable construction, descriptive statistics, and modeling role.

    A data dictionary: what each variable is, how it was built, what role it plays, and
    how much of the sample sits in each level.

    Two kinds of level are marked, because they are different things and the earlier note
    conflated them. Superscript a marks a missingness indicator: the screening record flags
    it as exempt, and its count is the number of records missing that field. Superscript b
    marks a level that pools unrecorded values with recorded ones, so its count is a
    combined count and not a missing count. The exemption is read from the screening
    record, not inferred from the wording of the label.
    """
    section = numbers["covariate_table"]
    data = numbers["data"]
    total = section["n_total"]
    screening = req(numbers, "appendix", "screening_detail")
    exempt = {c["candidate"] for c in screening["candidates"] if c.get("exempt")}
    poverty_missing = req(data, "acs", "poverty_missing")

    # The age reference band printed in the table must be the one the model uses. The
    # model frame names the level '25to54', but the stage-two merge moved the older band to
    # 45, so the reference covers 25 to 44. The data-sensitivity record states the band the
    # fitted primary specification uses; the two are checked against each other here.
    age_group = next(g for g in section["groups"] if g["group"] == "Rider age")
    ds = (numbers.get("sensitivity", {}).get("supporting", {})
          .get("data_sensitivities", {}).get("age", {}))
    if ds:
        used = req(ds, "band_specifications", "primary_lt25_25to44_ge45", "reference_band")
        if used != age_group["reference"]:
            raise SystemExit(f"Table 1: age reference '{age_group['reference']}' differs "
                             f"from the band the model uses, '{used}'")

    rows: list[str] = []

    def band(name: str, ref: str | None = None) -> str:
        head = r"\textbf{" + tex(name) + "}"
        desc = DESCRIBE.get(name)
        if desc:
            head += " \\quad " + tex(desc)
        if ref:
            head += r" \textit{(reference " + tex(ref) + ")}"
        # A paragraph cell of fixed width rather than an `l` cell: a band whose natural
        # width can exceed the X column makes xltabular renegotiate the column widths on
        # every pass, and the build never settles.
        return (BS + "rowcolor{PanelBg}\n"
                + r"\multicolumn{5}{>{\raggedright\arraybackslash}p{\dimexpr\textwidth-2\tabcolsep}}{"
                + head + r"} \\")

    rows.append(band("Dependent variable"))
    for s in SEV3:
        n = data["sev3"][s]
        rows.append(f"{tex(SEV3_LONG[s])} & DV & {n} & {rnd(100 * n / total, 1)} & --"
                    + r" \\")

    for group in section["groups"]:
        rows.append(band(group["group"], group["reference"]))
        for level in group["levels"]:
            comp = " / ".join(
                f"{rnd(level['severity_percent'][s], 1)}"
                if level["severity_percent"][s] is not None else "--" for s in SEV3)
            marks = []
            if level.get("column") in exempt:
                marks.append("a")
            if "or not recorded" in level["label"].lower():
                marks.append("b")
            note = (r"\textsuperscript{" + ",".join(marks) + "}") if marks else ""
            rows.append(f"{tex(level['label'])}{note} & Cov & {level['n']} & "
                        f"{rnd(level['share_of_sample'], 1)} & {comp}" + r" \\")

    rows.append(band("Spatial covariates"))
    for c in section["continuous"]:
        lo, hi = c["iqr"]
        rows.append(f"{tex(c['label'])}\\textsuperscript{{c}} & Cov & -- & -- & "
                    f"{rnd(c['median'], 3)} [{rnd(lo, 3)}, {rnd(hi, 3)}]" + r" \\")

    head = headerline(["Variable and level", "Role", r"$n$", r"\% of sample",
                       r"O / BC / KA (\%)"])

    notes = (
        r"\textit{Note:} DV marks the dependent variable and Cov a covariate, and every "
        r"covariate of the widest specification appears once. The officer-coded "
        r"contributing factors enter at M0c, the narrative-derived cues enter at M1, and "
        r"all other covariates enter at M0. Posted speed limit and rider age enter as "
        r"bands with an explicit level for an unrecorded value, so no record is dropped "
        r"for an incomplete field. For a categorical level the last column gives the "
        r"severity composition within the level as row percentages, and for a continuous "
        r"covariate it gives the median with the quartile range in brackets. The analytic "
        f"sample holds $n = {total}$ rider records, one for each rider."
        r"\par\vskip2pt"
        r"\textsuperscript{a}~These levels record that a field was not completed. They "
        r"enter as their own indicators rather than in the reference level, they are "
        r"exempt from the screening rules, and their screening counts are in "
        r"Appendix~\ref{app:screening}."
        r"\par\vskip2pt"
        r"\textsuperscript{b}~These levels pool records with the field not completed and "
        r"records with a recorded value outside the other levels. Their counts are "
        r"combined counts rather than counts of missing values, which are given per field "
        r"in Appendix~\ref{app:missing}."
        r"\par\vskip2pt"
        r"\textsuperscript{c}~Road density is measured within 100~m of the crash "
        r"coordinate from \ac{OSM}, and poverty share is the \ac{ACS} block-group share. "
        r"Records without coordinates, and the "
        + f"{poverty_missing}"
        + r" located records without a poverty estimate, take the median of their city, "
        r"or the overall median where the city has no located record."
    )

    return (header("Table 1", "variable construction, descriptives and modeling role")
            + r"""\addvspace{12pt}
{\small
\renewcommand{\arraystretch}{1.15}
\setlength{\LTpre}{0pt}
\def\LTcaptype{}
\refstepcounter{table}\label{tab:covariates}
% The caption text is black; the table number carries the highlight colour.
\noindent\parbox{\textwidth}{\rightskip=0pt\small
\hltint{HlM}{\textbf{Table \thetable}}\par
Analytical variable construction, descriptive statistics, and modeling role.\par\vskip4pt}
\begin{xltabular}{\textwidth}{>{\raggedright\arraybackslash}X c r r >{\centering\arraybackslash}p{3.3cm}}
\toprule
""" + head + r"""
\midrule
\endfirsthead
\multicolumn{5}{l}{\cellcolor{HeaderGold}\textcolor{HeaderText}{\textbf{\tablename\ \thetable{} \textit{(continued)}}}} \\
\toprule
""" + head + r"""
\midrule
\endhead
\midrule
\multicolumn{5}{r}{\textit{Continued on next page}} \\
\endfoot
\bottomrule
\multicolumn{5}{p{\dimexpr\textwidth-4\tabcolsep}}{\footnotesize
""" + notes + r"""} \\
\endlastfoot
""" + banded(rows) + r"""
\end{xltabular}}
\addvspace{6pt}
""")


# ---------------------------------------------------------------- Table 2

def table2(numbers: dict) -> str:
    """Per-cue agreement and per-route performance.

    The cues sit in two panels because the two passes are different tasks: the ten
    pre-crash cues are read from the masked narrative by both routes, the five injury cues
    from the unmasked narrative by Route B only. Each panel carries its own macro mean,
    so the routes are compared on the same ten labels and no mean mixes the two passes.
    """
    ex = numbers["extraction"]
    kappa = ex["fleiss_kappa_by_label"]
    ra = ex["routes"]["route_a"]["per_label"]
    rb = ex["routes"]["route_b_reference"]["per_label"]
    n_val = req(ex, "validation_n")

    mechanisms = [k for k in kappa if k in ra]
    injury = [k for k in kappa if k not in ra]
    if len(mechanisms) + len(injury) != len(kappa) or not injury:
        raise SystemExit("Table 2: cannot split the labels into the two passes")
    # Scores from the confusion counts, so a stored score is not rounded twice.
    macro_a = macro_f1(ra, mechanisms, req(ex, "routes", "route_a", "macro_f1_mechanisms"))
    macro_b = macro_f1(rb, mechanisms,
                       req(ex, "routes", "route_b_reference", "macro_f1_mechanisms"))
    macro_b_cues = macro_f1(rb, injury,
                            req(ex, "routes", "route_b_reference", "macro_f1_injury_cues"))

    def f1cell(value: float, other: float | None) -> str:
        body = f"{rnd(value, 2)}"
        if other is not None and rval(value, 2) >= rval(other, 2):
            return r"\textbf{" + body + "}"
        return body

    def row(label: str) -> str:
        a = prf(ra[label]) if label in ra else None
        b = prf(rb[label])
        if a:
            acells = f"{rnd(a[0], 2)} & {rnd(a[1], 2)} & " + f1cell(a[2], b[2])
        else:
            acells = "-- & -- & --"
        bcells = (f"{rnd(b[0], 2)} & {rnd(b[1], 2)} & "
                  + f1cell(b[2], a[2] if a else None))
        return (f"{tex(cue_name(label))} & {rnd(kappa[label], 3)} & {acells} & {bcells}"
                + r" \\")

    def order(labels):
        return sorted(labels, key=lambda k: -kappa[k])

    body = [panel("Pre-crash cues, read from the masked narrative", 8)]
    body.append(banded([row(k) for k in order(mechanisms)]))
    body.append(r"\midrule" + "\n" + r"\textit{Macro mean, "
                + f"{len(mechanisms)}" + r" pre-crash cues} & & & & "
                + f"{rnd(macro_a, 2)}" + r" & & & " + f"{rnd(macro_b, 2)}" + r" \\")
    body.append(panel("Injury cues, read from the unmasked narrative", 8))
    body.append(banded([row(k) for k in order(injury)]))
    body.append(r"\midrule" + "\n" + r"\textit{Macro mean, "
                + f"{len(injury)}" + r" injury cues} & & & & -- & & & "
                + f"{rnd(macro_b_cues, 2)}" + r" \\")

    caption = r"E-scooter pre-crash \hlone{cue} taxonomy and extraction validation."

    return (header("Table 2", "per-cue agreement and per-route performance")
            + r"""\begin{table}[pos=htbp]
\tableserif
\centering
\begin{threeparttable}
""" + float_caption("HlM", caption) + r"""
\label{tab:validation}
\small
\renewcommand{\arraystretch}{1.12}
\begin{tabularx}{\textwidth}{>{\raggedright\arraybackslash}X c ccc ccc}
\toprule
\headerrow
\hcell{} & \hcell{Coders\textsuperscript{a}} & \multicolumn{3}{c}{\hcell{Route A (lexical)}} &
\multicolumn{3}{c}{\hcell{Route B (model)}} \\
\headerrow
\hcell{Label} & \hcell{Fleiss' $\kappa$} & \hcell{P} & \hcell{R} & \hcell{$F_1$} &
\hcell{P} & \hcell{R} & \hcell{$F_1$} \\
\midrule
""" + "\n".join(body) + r"""
\bottomrule
\end{tabularx}
\begin{tablenotes}[flushleft]\footnotesize
\item[] \textit{Note:} P is precision, R is recall, and $F_1$ is their harmonic mean,
each computed for an automated route against the majority vote of the three coders on
the """ + f"{n_val}" + r""" validation narratives. Route~A reads the masked narrative and
never receives injury-outcome text, so it has no score for the injury cues and those cells
hold a dash. Each macro row averages $F_1$ over the cues of its own panel, so the two routes
are compared on the same pre-crash cues. Bold marks the higher $F_1$ of the two routes for
a cue, with ties at the printed precision bolded in both. The mean Fleiss' $\kappa$ over all
""" + f"{len(kappa)}" + r""" labels is """ + f"{rnd(ex['fleiss_kappa_macro'], 3)}" + r""".
Confusion counts and 95\% bootstrap intervals for every $F_1$ are in
Appendix~\ref{app:validation}.
\item[a] Agreement among the three human coders involves no automated output, so it is a
different quantity from the route columns and is never averaged with them.
\end{tablenotes}
\end{threeparttable}
\end{table}
""")


# ---------------------------------------------------------------- Table 3

def table3(numbers: dict) -> str:
    ct = numbers["audit"]["cross_tab_kabco"]
    counts, rowpct, totals = ct["counts"], ct["row_percent"], ct["row_totals"]

    rows_a = []
    for k in ("O", "C", "B", "A", "K"):
        cells = []
        for level in N_LEVELS:
            shade = r"\cellcolor{DiagGreen}" if rowpct[k][level] > SHADE_ABOVE else ""
            cells.append(f"{shade}{counts[k][level]} ({rnd(rowpct[k][level], 1)})")
        rows_a.append(f"{k} & " + " & ".join(cells) + f" & {totals[k]}" + r" \\")
    col = ct["column_totals"]
    totals_row = (r"\midrule" + "\n" + r"\textbf{Total} & "
                  + " & ".join(r"\textbf{" + str(col[l]) + "}" for l in N_LEVELS)
                  + r" & \textbf{" + str(ct["n"]) + r"} \\")

    rows_b = [f"{i} & {tex(level)} & {tex(rule)}" + r" \\"
              for i, (level, rule, _reading) in enumerate(MAPPING, start=1)]

    n_conflict = sum(v for v in numbers["audit"]["cue_conflicts"].values()
                     if isinstance(v, int))

    return (header("Table 3", "raw cross-tabulation and the cue-to-level mapping")
            + r"""\begin{table}[pos=htbp]
\tableserif
\centering
\begin{threeparttable}
\hlcaption{HlM}{Coded severity against the narrative-implied injury indicator.}
\label{tab:audit}
\small
\renewcommand{\arraystretch}{1.15}
\begin{tabularx}{\textwidth}{l XXXX r}
\toprule
""" + panel("Panel A. Observed counts, with no model and no priors", 6, rule=True) + "\n"
            + headerline(["KABCO", "No cue", "O-consistent", "BC-consistent",
                          "KA-consistent", "Total"]) + r"""
\midrule
""" + banded(rows_a) + "\n" + totals_row + r"""
\bottomrule
\end{tabularx}

\vspace{6pt}
\begin{tabularx}{\textwidth}{c l X}
\toprule
""" + panel("Panel B. The mapping, applied in this order", 3, rule=True) + "\n"
            + headerline(["Order", "Level", "Condition"]) + r"""
\midrule
""" + banded(rows_b) + r"""
\bottomrule
\end{tabularx}
\begin{tablenotes}[flushleft]\footnotesize
\item[] \textit{Note:} Panel A gives counts with row percentages in parentheses, and
shaded cells hold more than """ + f"{SHADE_ABOVE}" + r"""\% of their row. Panel B is applied
in order, so a narrative with cues that point in two directions resolves the same way every
time. """ + f"{n_conflict}" + r""" narratives carried conflicting cues and were resolved by
this order.
\end{tablenotes}
\end{threeparttable}
\end{table}
""")


# ---------------------------------------------------------------- Table 4

def table4(numbers: dict) -> str:
    """Model estimates, Table 4.

    One column per rung, each cell a slope with its standard error underneath in
    parentheses, bold where the 95% interval excludes zero.
    """
    rungs = numbers["severity"]["rungs"]
    coefs = {k: rungs[k]["coefficients"] for k in ("M0", "M0c", "M1")}
    order = list(coefs["M1"]["slopes"])
    in_m0 = set(coefs["M0"]["slopes"])
    in_m0c = set(coefs["M0c"]["slopes"])

    def cell(rung: str, name: str) -> str:
        s = coefs[rung]["slopes"].get(name)
        if s is None:
            return "--"
        body = f"{fmt(s['coef'], 3)} ({rnd(s['se'], 3)})"
        return (r"\textbf{" + body + "}") if s["excludes_zero"] else body

    groups = [
        ("Structured covariates", [c for c in order if c in in_m0]),
        ("Officer-coded contributing factors",
         [c for c in order if c in in_m0c and c not in in_m0]),
        ("Narrative-derived pre-crash cues",
         [c for c in order if c not in in_m0c]),
    ]

    body: list[str] = []
    for title, names in groups:
        if not names:
            continue
        body.append(panel(title, 4))
        rows = [f"{tex(pretty(n))} & " + " & ".join(cell(r, n) for r in
                                                    ("M0", "M0c", "M1")) + r" \\"
                for n in names]
        body.append(banded(rows))

    body.append(panel("Thresholds", 4))
    cuts = [f"$\\tau_{{{i + 1}}}$ & "
            + " & ".join(fmt(coefs[r]["cutpoints"][i], 3)
                         for r in ("M0", "M0c", "M1")) + r" \\"
            for i in range(len(coefs["M1"]["cutpoints"]))]
    body.append(banded(cuts))

    n_excl = coefs["M1"]["n_excluding_zero"]

    # The caption, with three phrases highlighted.
    caption = (r"\hlmulti{Ordinal} severity model estimates, structured-only (M0), "
               r"\hlmulti{with officer-coded contributing factors (M0c)}, and "
               r"narrative-augmented \hlone{with pre-crash cues} (M1).")

    # A float in the type size and style of Tables 2, 3, 5 and 6, so it never splits across
    # pages or leaves a half-empty page before it.
    return (header("Table 4", "ordinal severity model estimates")
            + r"""\begin{table}[pos=htbp]
\tableserif
\centering
\begin{threeparttable}
\hllabelcolour{HlM}
\caption{""" + caption + r"""}
\label{tab:model_estimates}
\small
\renewcommand{\arraystretch}{1.10}
\begin{tabularx}{\textwidth}{>{\raggedright\arraybackslash}p{5.2cm} *{3}{>{\centering\arraybackslash}X}}
\toprule
""" + headerline(["Parameter", "M0", "M0c", "M1"]) + r"""
\midrule
""" + "\n".join(body) + r"""
\bottomrule
\end{tabularx}
\begin{tablenotes}[flushleft]\footnotesize
\item \textit{Note:} Cells give ordered logit slopes on the latent severity scale, with
standard errors in parentheses. A positive slope shifts probability toward the more severe
outcome. Bold marks a coefficient whose 95\% interval excludes zero, and a dash marks a
covariate that does not enter that rung. In M1, """ + f"{n_excl}" + r""" of """ + f"{len(order)}"
            + r""" coefficients have an interval that excludes zero. The thresholds
$\tau_1$ and $\tau_2$ separate O from BC and BC from KA. Fit statistics for the three rungs
are in Table~\ref{tab:ladder}, and the average marginal effects that the text reports are in
Figure~\ref{fig:evidence} and Appendix~\ref{app:ames}.
\end{tablenotes}
\end{threeparttable}
\end{table}
""")


# ---------------------------------------------------------------- Table 5

def table5(numbers: dict) -> str:
    """Explanatory gain along the ladder, metrics in rows and rungs in columns.

    No variance-share row is reported: such a row rests on a normalization that does not
    support comparison across specifications with different covariate sets. A table
    states the values; the text reads them. Bold marks the best value of each scale-robust metric, which is the
    one piece of reading a table can carry without arguing.
    """
    rungs = numbers["severity"]["rungs"]
    lr = numbers["severity"]["nested_lr"]
    cvf = req(numbers, "sensitivity", "supporting", "cv_fold_level")
    repeats, folds = req(cvf, "repeats"), req(cvf, "folds")
    names = ("M0", "M0c", "M1")
    r = {k: rungs[k] for k in names}

    for k in names:
        stored = req(cvf, "rungs", k, "mean_log_loss")
        if rval(stored, 3) != rval(r[k]["cv_log_loss"], 3):
            raise SystemExit(f"Table 5: fold-level mean for {k} ({stored}) differs from "
                             f"the rung's stored log loss ({r[k]['cv_log_loss']})")

    def best(values: dict, higher: bool, printed: dict) -> dict:
        target = max(printed.values()) if higher else min(printed.values())
        return {k: (r"\textbf{" + v + "}" if printed[k] == target else v)
                for k, v in values.items()}

    rho = {k: fmt(r[k]["adjusted_rho2"], 4) for k in names}
    rho = best(rho, True, {k: rval(r[k]["adjusted_rho2"], 4) for k in names})
    cvs = {k: f"{rnd(r[k]['cv_log_loss'], 3)} ({rnd(r[k]['cv_sd'], 3)})" for k in names}
    cvs = best(cvs, False, {k: rval(r[k]["cv_log_loss"], 3) for k in names})
    aic = best({k: f"{rnd(r[k]['aic'], 1)}" for k in names}, False,
               {k: rval(r[k]["aic"], 1) for k in names})
    bic = best({k: f"{rnd(r[k]['bic'], 1)}" for k in names}, False,
               {k: rval(r[k]["bic"], 1) for k in names})

    def row(metric, cells):
        return f"{metric} & " + " & ".join(cells[k] for k in names) + r" \\"

    rows = [
        row(r"Adjusted $\rho^2$ (McFadden)", rho),
        row("Cross-validated log loss (SD)", cvs),
        row("AIC", aic),
        row("BIC", bic),
        row("Events per parameter",
            {k: f"{rnd(r[k]['events_per_parameter'], 2)}" for k in names}),
        row(r"Log-likelihood $\log L$", {k: fmt(r[k]["loglik"], 2) for k in names}),
        row("Covariates / parameters",
            {k: f"{r[k]['n_covariates']} / {r[k]['n_parameters']}" for k in names}),
    ]

    trows = []
    for key, label in (("M0_to_M0c", r"M0 $\rightarrow$ M0c, coded contributing factors"),
                       ("M0c_to_M1", r"M0c $\rightarrow$ M1, narrative-derived cues")):
        e = lr[key]
        trows.append(f"{label} & {rnd(e['lr_chi2'], 2)} & {e['df']} & {pfmt(e['p'])}"
                     + r" \\")

    caption = (r"Scale-robust \hlmulti{explanatory gain from the structured to the} "
               r"narrative-augmented \hlmulti{specification}.")

    return (header("Table 5", "explanatory gain along the ladder")
            + r"""\begin{table}[pos=htbp]
\tableserif
\centering
\begin{threeparttable}
""" + float_caption("HlM", caption) + r"""
\label{tab:ladder}
\small
\renewcommand{\arraystretch}{1.2}
\begin{tabularx}{\textwidth}{>{\raggedright\arraybackslash}X ccc}
\toprule
""" + headerline(["Scale-robust metric", "M0", "M0c", "M1"]) + r"""
\midrule
""" + banded(rows) + r"""
\addlinespace
""" + panel("Likelihood-ratio tests between adjacent rungs", 4, rule=True) + r"""
""" + headerline(["Comparison", r"$\chi^2$", "d.f.", "$p$"]) + r"""
\midrule
""" + banded(trows) + r"""
\bottomrule
\end{tabularx}
\begin{tablenotes}[flushleft]\footnotesize
\item[] \textit{Note:} The ladder is the sequence of nested specifications, and each
specification on it is a rung. M0 holds the screened structured covariates, M0c adds the
officer-coded contributing factors, and M1 adds the narrative-derived pre-crash cues.
Adjusted $\rho^2$ penalizes the McFadden $\rho^2$ for the additional parameters of each
rung. Cross-validated log loss is the mean over """ + f"{repeats}" + r""" repeats of
""" + f"{folds}" + r"""-fold stratified cross-validation, with the standard deviation across
folds in parentheses. AIC and BIC add to $-2\log L$ a penalty of 2 and of $\ln n$ per
parameter, with $n$ the number of riders. Bold marks the best value in each of the first four rows, the highest adjusted
$\rho^2$ and the lowest log loss, AIC and BIC. Fold-level values and the paired differences
between rungs are in Table~\ref{tab:a_cvfolds}. A large $p$ in a likelihood-ratio test
means that no gain was detected at this sample size, not that no gain exists.
\end{tablenotes}
\end{threeparttable}
\end{table}
""")


# ---------------------------------------------------------------- Table 6

def untestable_sentences(tr: dict, det: dict) -> str:
    """One sentence per break that could not be tested, built from the split record.

    The stored reason string is a working note with a sentence fragment in it, so the
    sentence is composed here from the counts that make the break untestable.
    """
    out = []
    for year in sorted(tr.get("untestable", {})):
        split = req(det, "splits", f"temporal_{year}")
        if split.get("testable", True):
            raise SystemExit(f"Table 6: {year} is listed as untestable but its split "
                             f"record says it was tested")
        before = req(split, "groups", "before")
        empty = [s for s in SEV3 if req(before, "severity_counts", s) == 0]
        if not empty:
            raise SystemExit(f"Table 6: no empty class explains the untested {year} break")
        out.append(f"The {year} break is not tested, because the {before['n']} riders "
                   f"before it include none coded {' or '.join(empty)}, which leaves an "
                   f"outcome class empty on that side.")
    return " ".join(out)


def table6(numbers: dict) -> str:
    """Stability and transferability tests.

    The test statistics, degrees of freedom, p-values and covariate counts are read from
    the top-level transferability section, which is what the analysis reports. The
    supporting detail is read only for the definition of the large-city group and the
    bootstrap settings, and its group sizes are checked against the test record.
    """
    tr = numbers["transferability"]
    det = req(numbers, "sensitivity", "supporting", "transferability_detail")
    cities = req(det, "large_city_definition", "cities_in_code")
    reps = req(det, "bootstrap", "replications")
    n_primary = req(numbers, "severity", "rungs", "M1", "n_covariates")

    sp = tr["spatial"]
    det_sp = req(det, "splits", "spatial", "groups")
    if (req(det_sp, "large_city", "n"), req(det_sp, "rest", "n")) != (
            sp["n_large_city"], sp["n_rest"]):
        raise SystemExit("Table 6: spatial group sizes differ between the test record and "
                         "the supporting detail")
    for year, e in tr["temporal"].items():
        d = det["splits"].get(f"temporal_{year}", {}).get("groups")
        if d and (d["before"]["n"], d["after"]["n"]) != (e["n_before"], e["n_after"]):
            raise SystemExit(f"Table 6: {year} group sizes differ between the test record "
                             f"and the supporting detail")

    def assess(rejects: bool) -> str:
        return (BS + "cellcolor{" + ("RedLo" if rejects else "GreenHi") + "}"
                + (r"\textbf{Rejected}" if rejects else "Not rejected"))

    rows = []
    for year in sorted(tr["temporal"]):
        e = tr["temporal"][year]
        rows.append(
            f"Temporal, {year} break & {e['n_before']}/{e['n_after']} & "
            f"{e['covariates_tested']} & {fmt(e['lr'], 2)} & {e['df']} & "
            f"{pfmt(e['p_asymptotic'])} & {pfmt(e['p_bootstrap'])} & "
            + assess(bool(e["rejects"])) + r" \\")

    rows.append(
        f"Spatial, {words(len(cities))} major cities vs.\\ rest & "
        f"{sp['n_large_city']}/{sp['n_rest']} & {sp['covariates_tested']} & "
        f"{rnd(sp['lr_statistic'], 2)} & {sp['df']} & {pfmt(sp['p_asymptotic'])} & "
        f"{pfmt(sp['p_bootstrap'])} & "
        + assess(bool(sp["rejects_at_5pct_bootstrap"])) + r" \\")

    untestable_note = untestable_sentences(tr, det)
    city_list = ", ".join(cities[:-1]) + " and " + cities[-1]

    return (header("Table 6", "stability and transferability tests")
            + r"""\begin{table}[pos=htbp]
\tableserif
\centering
\begin{threeparttable}
""" + float_caption("HlM", "Likelihood-ratio stability and transferability tests.") + r"""
\label{tab:stability_tests}
\small
\renewcommand{\arraystretch}{1.2}
\begin{tabularx}{\textwidth}{>{\raggedright\arraybackslash}X c c c c c c c}
\toprule
""" + headerline(["Split", "$n$ split", "Cov.", r"$\chi^2$", "d.f.",
                  "$p$ asym.", "$p$ boot.", "Assessment"]) + r"""
\midrule
""" + banded(rows) + r"""
\bottomrule
\end{tabularx}
\begin{tablenotes}[flushleft]\footnotesize
\item[] \textit{Note:} The statistic is $\chi^2 = -2[\log L_{\text{pooled}} - (\log L_A +
\log L_B)]$, computed on the covariates estimable within both sides of each split. The
asymptotic reference distribution is not trusted at these subsample sizes, so a parametric
bootstrap reference distribution is built from """ + f"{reps}" + r""" replications under the
pooled model. Both $p$-values are shown, and the bootstrap value governs where they
disagree. The tests are exploratory and are not used to select the primary specification.
The covariate count is the number estimable within both sides of the split rather than the
""" + f"{n_primary}" + r""" of the primary specification, so a non-rejection describes that
reduced set only. The """ + words(len(cities)) + r""" major cities are """ + tex(city_list) + r""". """
            + untestable_note + r""" The detail of every split and of each subgroup fit is in
Appendix~\ref{app:transfer}.
\end{tablenotes}
\end{threeparttable}
\end{table}
""")


def main() -> int:
    numbers = load_numbers()
    print(f"  reading {numbers_path()}")
    OUT.mkdir(parents=True, exist_ok=True)
    built = {
        "table1_covariates.tex": table1(numbers),
        "table2_validation.tex": table2(numbers),
        "table3_audit.tex": table3(numbers),
        "table4_estimates.tex": table4(numbers),
        "table5_ladder.tex": table5(numbers),
        "table6_transferability.tex": table6(numbers),
    }
    for name, content in built.items():
        # Appendix tables printed in the supplement are named by their S number.
        from build_appendix import refs_to_supplement_static
        content = refs_to_supplement_static(content)
        (OUT / name).write_text(content, encoding="utf-8")
        print(f"  wrote tables/{name} ({len(content.splitlines())} lines)")
    print(f"\n{len(built)} table(s) generated.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
