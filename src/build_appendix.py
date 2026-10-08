"""Generate the appendix from the frozen numbers.

The appendices follow the Elsevier layout: they sit after the Conclusion and before the
references, they are lettered A, B, C, and every table inside one carries its letter and
restarts its count, as in Table A.1 and Table B.1. Each opens with a paragraph that says
what the material is and how it was produced, and each table is introduced in the text
before it appears.

    A  covariate screening                       app:screening
    B  missing data                              app:missing
    C  extraction validation, alternative model
       and masking                               app:validation
    D  audit diagnostics                         app:auditdiag
    E  the ordinal family                        app:ordinal
    F  the four-class ordering                   app:fourclass
    G  average marginal effects                  app:ames
    H  cue label variants                        app:labels
    I  the latent-class-weighted model M2        app:m2
    J  random parameters                         app:randpar
    K  transferability                           app:transfer
    L  out-of-sample evaluation                  app:benchmark

`LETTERS` below is the single place the order is recorded.

Seven tables that hold data only are generated here but printed in the Supplementary
material, numbered S1, S2, ...: the annual composition (`annual_block`), the
shifter-recovery simulation (`shifter_block`), the partial
proportional-odds coefficients (`ppo_coef_block`), the actor attribution (`actor_block`),
the single random coefficients (`randpar_candidates_block`), the rural and urban-area
spatial splits (`spatial_alt_block`) and the per-class benchmark metrics
(`perclass_block`). build_supplementary.py calls these functions and writes
`results/supp_labels.tex`, whose macros (\\suppAnnual and so on) carry the printed S
numbers; the pointer sentences here use those macros, so no S number is typed by hand.

Every printed value comes from numbers.json, read through ESCOOTER_NUMBERS when that is
set (see build_tables.py). A value the file does not hold stops the build rather than
printing as a dash.

Run with:  python src/build_appendix.py
"""

from __future__ import annotations

import re
import string
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "results"

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_tables import (cue_name, fmt, load_numbers, numbers_path, pct, pfmt,  # noqa: E402
                          macro_f1, pretty, prf, req, rnd, rval, tex, words)

SEV3 = ("O", "BC", "KA")
MECH5 = ("sidewalk_transition", "failure_to_yield", "signal_violation",
         "vehicle_turning_across", "lane_positioning")
MECH5_SHORT = {"sidewalk_transition": "SW", "failure_to_yield": "FTY",
               "signal_violation": "SV", "vehicle_turning_across": "VTA",
               "lane_positioning": "LP"}

# The appendix order, as (label, title). The letter of each is its position.
ORDER = (
    ("app:screening", "Covariate screening"),
    ("app:missing", "Missing data"),
    ("app:validation", "Extraction validation"),
    ("app:auditdiag", "Reporting audit diagnostics"),
    ("app:ordinal", "The ordinal family"),
    ("app:fourclass", "The four-class ordering"),
    ("app:ames", "Average marginal effects"),
    ("app:labels", "Cue label variants"),
    ("app:m2", "The latent-class-weighted model M2"),
    ("app:randpar", "Random parameters"),
    ("app:transfer", "Transferability"),
    ("app:benchmark", "Out-of-sample evaluation"),
)
LETTERS = {label: string.ascii_uppercase[i] for i, (label, _) in enumerate(ORDER)}

# Two sections are printed as the manuscript's appendices, under the titles below. The
# others are printed in the Supplementary material as subsections of its last section,
# and so is the independence test of the multinomial logit. Generated sentences name a
# table by a working letter from `LETTERS` (for example "Table E.4"); `fix_refs` maps
# each such name to its label and prints the number the table has in the manuscript or
# in the supplement.
KEEP = {"app:ordinal": "Model checks", "app:labels": "Cue tests"}
MOVED_EXTRA = (("app:iia", "Independence tests for the multinomial logit"),)
MOVED = tuple(label for label, _ in ORDER if label not in KEEP)
OLD_TABLES = {
    "A.1": "tab:a_screening", "B.1": "tab:a_missing", "B.2": "tab:a_mi",
    "C.1": "tab:a_kappa", "C.2": "tab:a_routes", "C.3": "tab:a_calibration",
    "C.4": "tab:a_altmodel", "D.1": "tab:a_auditdiag", "E.1": "tab:a_ordinal",
    "E.2": "tab:a_iia", "E.3": "tab:a_ordinalfamily", "E.4": "tab:a_shrinkage",
    "F.1": "tab:a_fourclass", "G.1": "tab:a_ames", "H.1": "tab:a_labelvariants",
    "H.2": "tab:a_power", "H.3": "tab:a_excluded", "I.1": "tab:a_m2",
    "J.1": "tab:a_randompar_models", "K.1": "tab:a_transfer", "K.2": "tab:a_subgroups",
    "L.1": "tab:a_cvfolds", "L.2": "tab:a_benchmark",
}
OLD_SECTIONS = {letter_: label for label, letter_ in LETTERS.items()}


def supp_macro(label: str) -> str:
    """The macro that carries the supplement number of a moved table or section."""
    if label in SUPP_MACROS:
        return SUPP_MACROS[label]
    kind = "Tab" if label.startswith("tab:") else "Sec"
    stem = label.split(":", 1)[1].removeprefix("a_")
    name = "".join(w.capitalize() for w in re.split(r"[_\W]+", stem) if w)
    # A TeX control word holds letters only, so a digit is spelled out (M2 -> MTwo).
    digits = ("Zero", "One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine")
    return "supp" + kind + "".join(digits[int(c)] if c.isdigit() else c for c in name)


def letter(label: str) -> str:
    return LETTERS[label]


# ---------------------------------------------------------------- table machinery

def _group(spec: str, i: int) -> tuple[str, int]:
    r"""The brace group starting at `i`, and the index just past it."""
    if i >= len(spec) or spec[i] != "{":
        return "", i
    depth, start = 0, i
    while i < len(spec):
        if spec[i] == "{":
            depth += 1
        elif spec[i] == "}":
            depth -= 1
            if depth == 0:
                return spec[start + 1:i], i + 1
        i += 1
    return spec[start + 1:], i


def ncols(colspec: str) -> int:
    r"""How many columns a specification declares.

    It has to be parsed rather than counted. Counting the letters r, l, c and p treats the
    l in `\linewidth` as a column, and stripping every brace group first loses the columns
    inside `*{4}{>{\centering\arraybackslash}X}`. So a group following >, <, @ or ! is a
    modifier and contributes nothing; p, m and b take a width argument and contribute one;
    `*{n}{sub}` contributes n times whatever sub contributes; and l, c, r, X and Y
    contribute one each.
    """
    i, n = 0, 0
    while i < len(colspec):
        ch = colspec[i]
        if ch in "><@!":
            _, i = _group(colspec, i + 1)
        elif ch == "*":
            count, i = _group(colspec, i + 1)
            sub, i = _group(colspec, i)
            n += int(count.strip() or 0) * ncols(sub)
        elif ch in "pmb":
            _, i = _group(colspec, i + 1)
            n += 1
        else:
            if ch in "lcrXY":
                n += 1
            i += 1
    return n


def headerline(head: str) -> str:
    """The gold header row, from a plain `a & b & c \\\\` header string.

    A cell that is already a \\multicolumn keeps its span and only has its text wrapped.
    """
    body = head.strip()
    if body.endswith("\\\\"):
        body = body[:-2]
    cells = re.split(r"(?<!\\)&", body)
    out = []
    for c in cells:
        c = c.strip()
        m = re.match(r"\\multicolumn\{(\d+)\}\{([^}]*)\}\{(.*)\}$", c)
        if m:
            out.append("\\multicolumn{%s}{%s}{\\hcell{%s}}" % (m.group(1), m.group(2),
                                                               m.group(3)))
        else:
            out.append("\\hcell{" + c + "}")
    return "\\headerrow\n" + " & ".join(out) + " \\\\"


def banded(rows: list[str]) -> list[str]:
    """Alternating cream rows, restarting at each panel or inner header.

    A panel row is a heading, not data, so it takes no stripe and the alternation begins
    again underneath it.
    """
    out, i = [], 0
    for r in rows:
        if r.lstrip().startswith(("\\addlinespace", "\\multicolumn", "\\midrule",
                                  "\\headerrow", "\\rowcolor")):
            out.append(r)
            i = 0
            continue
        if i % 2 == 1:
            out.append("\\altrow")
        out.append(r)
        i += 1
    return out


def panel(title: str, ncols_: int) -> str:
    """A labelled block inside one table, so that a table number stays a table number."""
    return (f"\\addlinespace\n\\rowcolor{{PanelBg}}\n"
            f"\\multicolumn{{{ncols_}}}{{l}}{{\\textbf{{{title}}}}} "
            f"\\\\\n\\addlinespace[2pt]")


def inner_header(head: str) -> str:
    """A gold header row inside the body, for a panel whose columns differ."""
    return "\\midrule\n" + headerline(head) + "\n\\midrule"


# The two appendices print each table as a float on one page. A long table that starts
# low on a page can print only its "continued" footer there, so the float is used wherever
# the table fits a page; the supplement keeps the page-breaking form.
AS_FLOAT = False


def _float_table(caption: str, label: str, colspec: str, head: str, rows: list[str],
                 note: str, hlcolour: str, size: str, tabcolsep: str) -> str:
    """A float in the style of the main-text tables (serif face, threeparttable, notes)."""
    env = "tabularx" if "X" in colspec else "tabular*"
    width = "{\\textwidth}"
    if env == "tabular*":
        colspec = "@{\\extracolsep{\\fill}}" + colspec
    cap = ("\\caption{\\hltint{%s}{%s}}" % (hlcolour, caption) if hlcolour
           else "\\caption{%s}" % caption)
    sep = ("\\setlength{\\tabcolsep}{%s}\n" % tabcolsep) if tabcolsep else ""
    notes = ("\\begin{tablenotes}[flushleft]\\footnotesize\n\\item \\textit{Note:} %s\n"
             "\\end{tablenotes}\n" % note) if note else ""
    return ("\\begin{table}[pos=htbp]\n\\tableserif\n\\centering\n\\begin{threeparttable}\n"
            + ("\\hllabelcolour{%s}\n" % hlcolour if hlcolour else "")
            + cap + "\n\\label{" + label + "}\n\\footnotesize\n"
            + "\\renewcommand{\\arraystretch}{1.15}\n" + sep
            + f"\\begin{{{env}}}{width}{{{colspec}}}\n\\toprule\n{headerline(head)}\n\\midrule\n"
            + "\n".join(banded(rows))
            + f"\n\\bottomrule\n\\end{{{env}}}\n" + notes
            + "\\end{threeparttable}\n\\end{table}\n\n")


def longtable(caption: str, label: str, colspec: str, head: str, rows: list[str],
              note: str = "", hlcolour: str = "HlM", size: str = "\\small",
              repeat_head: bool = True, tabcolsep: str = "", need: int = 0) -> str:
    r"""A table that may run past a page, in the style the main-text tables use.

    `note` is prose; it is printed under the table after "Note:" in italics. When
    `repeat_head` is false the table has panels with their own column headers, so a
    continuation page repeats only the table number rather than a header that may belong
    to a different panel.
    """
    n = ncols(colspec)
    if AS_FLOAT:
        return _float_table(caption, label, colspec, head, rows, note, hlcolour, size,
                            tabcolsep)
    foot = ""
    if note:
        foot = ("\n\\multicolumn{%d}{p{0.97\\linewidth}}{\\footnotesize "
                "\\textit{Note:} %s}\\\\" % (n, note))

    # The class sets float captions as the number on one line and the text below it,
    # flush left. A longtable caption goes through the caption package instead, which
    # would centre it in a 4in box after a colon, so the same layout is requested here,
    # locally, with the number in the highlight colour.
    pre = ("\\setlength{\\LTcapwidth}{\\textwidth}\n"
           "\\captionsetup{labelsep=newline,justification=raggedright,"
           "singlelinecheck=false,font=small}\n")
    if hlcolour:
        pre += ("\\ifdefined\\cleanbuild\\captionsetup{labelfont=bf}\\else"
                "\\captionsetup{labelfont={bf,color=%s}}\\fi\n" % hlcolour)
    cap = ("\\caption{\\hltint{%s}{%s}}" % (hlcolour, caption) if hlcolour
           else "\\caption{%s}" % caption)
    hdr = headerline(head)
    cont = ("\\multicolumn{%d}{l}{\\cellcolor{HeaderGold}\\textcolor{HeaderText}"
            "{\\textbf{\\tablename\\ \\thetable{} \\textit{(continued)}}}} \\\\" % n)
    if "X" in colspec:
        begin = f"\\begin{{xltabular}}{{\\textwidth}}{{{colspec}}}"
        end = "\\end{xltabular}"
    else:
        begin = f"\\begin{{longtable}}{{{colspec}}}"
        end = "\\end{longtable}"
    again = f"{cont}\n\\toprule\n{hdr}\n\\midrule" if repeat_head else f"{cont}\n\\toprule"
    if tabcolsep:
        pre += "\\setlength{\\tabcolsep}{%s}\n" % tabcolsep
    # A table starts on a new page unless about a third of the page is left. Line numbers
    # are suspended inside it: lineno's output routine breaks a long table's first page,
    # leaving only its "continued" footer there. The guard leaves the supplement, which
    # has no line numbers, unchanged.
    keep = ("\\needspace{0.33\\textheight}\n"
            "\\ifdefined\\endnolinenumbers\\begin{nolinenumbers}\\fi")
    return (f"{keep}\n{{{size}\n{pre}\\renewcommand{{\\arraystretch}}{{1.15}}\n"
            f"{begin}\n"
            f"{cap}\\label{{{label}}}\\\\\n"
            f"\\toprule\n{hdr}\n\\midrule\n\\endfirsthead\n"
            f"{again}\n\\endhead\n"
            f"\\midrule\n\\multicolumn{{{n}}}{{r}}{{\\textit{{Continued on next page}}}} \\\\\n"
            f"\\endfoot\n\\bottomrule{foot}\n\\endlastfoot\n"
            + "\n".join(banded(rows))
            + f"\n{end}}}\n\\ifdefined\\endnolinenumbers\\end{{nolinenumbers}}\\fi\n\n")


def section(label: str, body: str) -> str:
    if label in KEEP:
        return (f"\\section{{{KEEP[label]}}}\\label{{{label}}}\n"
                "\\setcounter{table}{0}\\setcounter{figure}{0}\\setcounter{equation}{0}\n\n"
                + body)
    # A moved appendix is a subsection of the supplement's last section, and its tables
    # continue the S numbering of the supplement.
    title = dict(ORDER + MOVED_EXTRA)[label]
    return f"\\subsection{{{title}}}\\label{{{label}}}\n\n" + body


def ci(lo: float, hi: float, places: int = 3) -> str:
    return f"[{fmt(lo, places)}, {fmt(hi, places)}]"


def joined(items: list[str]) -> str:
    items = list(items)
    if len(items) <= 1:
        return "".join(items)
    return ", ".join(items[:-1]) + " and " + items[-1]


def epp(counts: dict, parameters: int, stored: float) -> float:
    """Events per parameter in the smallest class, from the class counts, checked against
    the stored value, so the printed ratio is not a stored rounded ratio rounded again."""
    value = min(counts.values()) / parameters
    if abs(value - stored) > 6e-3:
        raise SystemExit(f"events per parameter {stored} does not follow from the counts "
                         f"({value:.4f})")
    return value


def bold_if(text: str, flag: bool) -> str:
    return ("\\textbf{" + text + "}") if flag else text


# The tables printed in the Supplementary material, by label, and the macro that carries
# the printed number of each. build_supplementary.py numbers them in the order they appear
# there and writes the macros to supp_labels.tex, which the manuscript reads in its preamble.
SUPP_MACROS = {
    "tab:a_actor": "suppActor",
    "tab:a_randompar": "suppRandCand",
    "tab:a_annual": "suppAnnual",
    "tab:a_shifter": "suppShifter",
    "tab:a_ppocoef": "suppPPOCoef",
    "tab:a_spatialalt": "suppSpatialAlt",
    "tab:a_perclass": "suppPerClass",
}


# The tables of the two appendices; every other table and section built here is printed
# in the supplement. Used where the full build of the parts is not at hand.
KEPT_TABLES = ("tab:a_ordinal", "tab:a_ordinalfamily", "tab:a_shrinkage",
               "tab:a_labelvariants", "tab:a_power", "tab:a_excluded")
STATIC_MOVED = (set(MOVED) | {label for label, _ in MOVED_EXTRA}
                | (set(OLD_TABLES.values()) - set(KEPT_TABLES)) | set(SUPP_MACROS))


def supp(label: str) -> str:
    """'Supplementary Table S<n>' for a table printed in the supplement."""
    return "Supplementary Table~\\" + SUPP_MACROS[label] + "{}"


# ---------------------------------------------------------------- A screening

DISPOSITION = {
    "included": "Included",
    "included: missingness indicator, exempt (prespec 3.2)": "Included, exempt indicator",
    "excluded: smallest severity cell below 10": "Excluded, event rule",
}


def app_screening(n: dict) -> str:
    d = n["appendix"]["screening_detail"]
    rows = []
    for c in d["candidates"]:
        ev = c.get("events_by_severity")
        median = c.get("median_by_severity")
        if ev:
            cells = " & ".join(str(req(ev, s)) for s in SEV3)
            smallest = str(req(c, "min_cell"))
        elif median:
            cells = " & ".join(f"\\textit{{{rnd(req(median, s), 3)}}}" for s in SEV3)
            smallest = "\\textit{n/a}"
        else:
            raise SystemExit(f"A: {c['candidate']} has neither counts nor medians")
        vif = f"{rnd(c['vif'], 2)}" if c.get("vif") is not None else "--"
        disp = DISPOSITION.get(c["disposition"])
        if disp is None:
            raise SystemExit(f"A: unknown disposition '{c['disposition']}'")
        rows.append(f"{tex(pretty(c['candidate']))} & {cells} & "
                    f"{smallest} & {vif} & {disp} \\\\")

    merged_fields = sorted({s.split(":")[0].strip() for s in d["stage_two_merges"]})
    n_cand = len(d["candidates"])
    text = (
        "This appendix lists every candidate covariate that entered the screening step, "
        "with the counts and the variance-inflation factor that the screening rules "
        "were applied to. The pre-specification fixed three conditions before any model "
        "was estimated. The first, a stated Safe System or confounder role, "
        "was met by every candidate defined in advance, so it removed none. The event rule admits a binary candidate only when each of the "
        f"three severity classes holds at least {d['min_events_rule']} riders with the "
        "attribute, and the collinearity rule admits a candidate only when its "
        f"variance-inflation factor is below {d['max_vif_rule']:g}. Indicators that record "
        "a field as not completed are exempt from the two count rules, because deleting such an "
        "indicator would move the affected records into the reference category rather "
        "than remove them from the model. The screening ran in two stages, and the second "
        "stage merged a level that broke the event rule with a neighboring level of the "
        f"same field. This happened for {joined(merged_fields)}, and the merged levels are "
        "the ones listed in Table~\\ref{tab:covariates} and in "
        f"Table~{letter('app:screening')}.1.\n\n")
    return section("app:screening", text + longtable(
        f"All {n_cand} screening candidates and their disposition.",
        "tab:a_screening",
        ">{\\raggedright\\arraybackslash}X rrrrr>{\\raggedright\\arraybackslash}p{3.3cm}",
        "Candidate & O & BC & KA & Smallest cell & VIF & Disposition \\\\",
        rows,
        note=("Counts are riders with the attribute in each severity class, and the "
              "smallest cell is the one the event rule tests. Rows in italics are "
              "continuous covariates, to which the event rule does not apply, and their "
              "cells give the median by severity class instead of a count. A dash in the "
              "variance-inflation column marks a candidate that the event rule excluded "
              "before the collinearity check.")))


# ---------------------------------------------------------------- B missing data

MISSING_FIELDS = (
    ("age_unknown", "Rider age",
     "Own indicator, rider age not recorded"),
    ("gender_unknown", "Rider gender",
     "Own indicator, rider gender not recorded"),
    ("ethnicity_unknown", "Rider ethnicity",
     "Pooled with the recorded other category in one level"),
    ("speed_limit_unknown", "Posted speed limit",
     "Own band, speed limit not recorded"),
    ("coordinates_missing", "Crash coordinates",
     "Own indicator; road density and poverty share take the city median, and the urban-area indicator takes the city majority, or the rural flag where the city has no located record"),
)

PREDICTOR_WORDS = {"year_c": "year", "nighttime": "night", "weekend": "weekend",
                   "intx_intersection_or_driveway": "intersection relation",
                   "road_not_city_street": "roadway class",
                   "large_city": "the large-city indicator"}


def app_missing(n: dict) -> str:
    mm = req(n, "sensitivity", "supporting", "data_sensitivities", "missingness_mechanism")
    groups = {g["group"]: g for g in n["covariate_table"]["groups"]}
    rows = []
    for key, field, treatment in MISSING_FIELDS:
        m = req(mm, "models", key)
        rates = " / ".join(pct(req(m, "missing_rate_by_severity", s)) for s in SEV3)
        rows.append(f"{field} & {req(m, 'n_missing')} & {rates} & "
                    f"{pfmt(req(m, 'severity_block_p'))} & {treatment} \\\\")

    def combined(group: str, column: str) -> int:
        level = next(l for l in groups[group]["levels"] if l["column"] == column)
        return level["n"]

    rows.append(f"Striking vehicle & -- & -- & -- & Combined level of "
                f"{combined('Striking vehicle', 'veh_other_unknown')} riders with other "
                "body types, exempt indicator \\\\")
    rows.append(f"Light condition & -- & -- & -- & Combined level of "
                f"{combined('Light condition', 'light_dark_or_unknown')} riders with dark "
                "conditions \\\\")
    rows.append(f"Poverty share, located crashes & {req(n, 'data', 'acs', 'poverty_missing')}"
                " & -- & -- & City median, or the overall median where the city has no "
                "located record \\\\")

    adjusters = [PREDICTOR_WORDS[p] for p in req(mm, "predictors")
                 if not p.startswith("sev_")]
    text = (
        "This appendix gives, for each field with unrecorded values, the number of riders "
        "missing it, the share missing within each severity class, and the treatment the "
        "model applies. No rider is dropped for an incomplete field. Rider age, rider "
        "gender and posted speed limit enter with their own level for an unrecorded "
        "value, and a crash without coordinates carries the coordinates-absent indicator. "
        "Unrecorded ethnicity is pooled with the recorded other category, because the "
        "recorded category alone broke the event rule. The two spatial covariates of a "
        "record without coordinates take the median of its city, or the overall median "
        "where the city has no located record, so they are imputed values for those "
        "records. The striking-vehicle and light fields pool unrecorded values with "
        "recorded ones, and only their combined counts are reported.\n\n"
        "The severity association in "
        f"Table~{letter('app:missing')}.1 is a likelihood-ratio test of severity class in "
        "a logistic model of missingness that also adjusts for "
        f"{joined(adjusters)}. A small $p$-value means that the field is not missing "
        "completely at random with respect to severity, and none of these models "
        "separates data missing at random from data missing not at random. The two "
        "spatial covariates are left out of these models because they are imputed for "
        "exactly the records without coordinates.\n\n")
    return section("app:missing", text + longtable(
        "Missing values by field and the treatment applied to each.",
        "tab:a_missing",
        "l r c c >{\\raggedright\\arraybackslash}X",
        "Field & Riders missing & Missing share O / BC / KA (\\%) & Severity $p$ & "
        "Treatment \\\\",
        rows,
        note=("The missing share is the percentage of riders in each coded severity "
              "class with the field not recorded. A dash marks a field whose unrecorded "
              "values are not counted apart from a recorded category."))
        + mi_block(n))


# ---------------------------------------------------------------- annual (supplement)

def annual_block(n: dict) -> str:
    """The annual composition of the sample, printed in the Supplementary material."""
    a = req(n, "sensitivity", "supporting", "annual")
    years = [str(y) for y in req(a, "years")]
    riders = req(a, "rider_rows_per_year")
    crashes = req(a, "distinct_crashes_per_year")
    sev = req(a, "severity_counts")
    share = req(a, "severity_share_percent")
    sev3 = req(n, "data", "sev3")
    n_rows = req(n, "data", "rider_rows")
    n_crash = req(n, "data", "distinct_crashes")

    cols = len(years) + 2
    rows = [panel("Panel A. Rider records and severity", cols)]
    rows.append("Rider records & " + " & ".join(str(riders[y]) for y in years)
                + f" & {n_rows} \\\\")
    rows.append("Distinct crashes & " + " & ".join(str(crashes[y]) for y in years)
                + f" & {n_crash} \\\\")
    for s in SEV3:
        cells = [f"{sev[y][s]} ({rnd(share[y][s], 1)})" for y in years]
        rows.append(f"{s}, $n$ (\\%) & " + " & ".join(cells)
                    + f" & {sev3[s]} ({rnd(100 * sev3[s] / n_rows, 1)}) \\\\")

    rows.append(panel("Panel B. Share of rider records with the attribute (\\%)", cols))
    binary = req(a, "binary_covariates_by_year")
    for cov in req(a, "covariates_reported"):
        if cov not in binary:
            continue
        e = binary[cov]
        rows.append(f"{tex(pretty(cov))} & "
                    + " & ".join(pct(req(e, y, "share")) for y in years)
                    + f" & {pct(req(e, 'all_years', 'share'))} \\\\")

    rows.append(panel("Panel C. Continuous covariates, median [quartile range]", cols))
    cont = req(a, "continuous_covariates_by_year")
    for cov in ("road_density_km_100m", "poverty_share"):
        e = req(cont, cov)
        rows.append(f"{tex(pretty(cov))} & "
                    + " & ".join(f"{rnd(e[y]['median'], 3)} [{rnd(e[y]['q1'], 3)}, {rnd(e[y]['q3'], 3)}]"
                                 for y in years + ["all_years"]) + " \\\\")

    # The reporting cities on either side of the 2024 break, from the composition check of
    # the temporal transferability test.
    cc = req(n, "transferability", "temporal", "2024", "city_composition")
    bp = req(cc, "breakpoint")
    before = [y for y in years if int(y) < bp]
    after = [y for y in years if int(y) >= bp]
    if (sum(riders[y] for y in before) != cc["n_before"]
            or sum(riders[y] for y in after) != cc["n_after"]):
        raise SystemExit("S: city composition does not match the annual rider counts")
    rows.append(panel(f"Panel D. Reporting cities before and after the {bp} break", cols))
    rows.append(f"Distinct reporting cities & \\multicolumn{{{len(before)}}}{{c}}"
                f"{{{cc['distinct_cities_before']}}} & \\multicolumn{{{len(after)}}}{{c}}"
                f"{{{cc['distinct_cities_after']}}} & \\\\")

    counts = [riders[y] for y in years]
    rising = all(b > a_ for a_, b in zip(counts, counts[1:]))
    empty = [(y, s) for y in years for s in SEV3 if sev[y][s] == 0]
    trend = (f"The number of rider records rises in every year, from {counts[0]} in "
             f"{years[0]} to {counts[-1]} in {years[-1]}." if rising else
             f"The number of rider records ranges from {min(counts)} to {max(counts)} a "
             f"year.")
    gap = ""
    if empty:
        y, s = empty[0]
        nxt = years[years.index(y) + 1]
        gap = (f" The {y} records include no rider coded {s}, which is why the break "
               f"between {y} and {nxt} is not among the temporal transferability tests of "
               "the manuscript.")
    text = (
        "How the sample is distributed over the reporting years, and how its composition "
        "changes from year to year, bears on pooling the years in one model. Counts are rider records, one for each e-scooter "
        f"rider, so the {n_rows} rider records come from {n_crash} crashes. "
        + trend + gap +
        " Table~\\ref{tab:a_annual} gives the severity mix of each year, the share "
        "of each year's records carrying each binary covariate of M1, and the median and "
        "quartile range of the two continuous covariates. Its last panel counts the "
        f"distinct cities that reported a record before and after the {bp} break, "
        f"{cc['distinct_cities_before']} against {cc['distinct_cities_after']}.\n\n")
    colspec = (">{\\raggedright\\arraybackslash}X"
               + "*{%d}{>{\\centering\\arraybackslash}p{1.45cm}}" % (len(years) + 1))
    return text + longtable(
        "Rider records, severity mix and covariate composition by year.",
        "tab:a_annual", colspec,
        "Row & " + " & ".join(years) + " & All \\\\",
        rows, size="\\footnotesize", need=20,
        note=("Percentages in Panel A are the shares of the three severity classes "
              "within a year. Percentages in Panel B are the shares of a year's rider "
              "records with the attribute, so the reference levels are not listed. The "
              "age reference band covers riders aged "
              + tex(next(g["reference"] for g in n["covariate_table"]["groups"]
                         if g["group"] == "Rider age")) + ". Panel~D counts the distinct "
              "city names on the records of each period, with the records that list no "
              "city counted as one name."))


# ---------------------------------------------------------------- C validation

ROUTE_TITLES = (
    ("route_a", "Route A, deterministic lexical rules"),
    ("route_b_reference", "Route B, language-model reference run"),
    ("route_b_vote_half", "Route B, majority of the repeated runs"),
    ("fusion_automated", "Automated fusion of the two routes"),
)


def alt_macros(alt: dict) -> tuple[float, float]:
    """Macro F1 of the reference and the alternative model, from the confusion counts."""
    pq = req(alt, "paired_llama_vs_qwen")
    out = []
    for model, stored in (("llama_reference", pq["macro_f1_llama"]),
                          ("qwen", pq["macro_f1_qwen"])):
        v = req(alt, "validation", model)
        per = {k: req(e, "unweighted") for k, e in v["per_label"].items()}
        out.append(macro_f1(per, v["labels_in_macro"], stored))
    return out[0], out[1]


def app_validation(n: dict) -> str:
    ex = n["extraction"]
    routes = ex["routes"]
    kappa = ex["fleiss_kappa_by_label"]
    pairwise = ex["pairwise_cohen_kappa"]
    sup = req(n, "sensitivity", "supporting", "extraction_intervals_calibration")
    boot = req(sup, "per_cue_intervals", "bootstrap")
    macro_ci = req(sup, "macro_f1_intervals", "routes")
    labels_cal = req(n, "sensitivity", "labels", "calibration")
    cal_set = req(n, "sensitivity", "labels", "settings", "calibration")
    # The human-coded set is built from counts, not from the stored description string.
    n_human = req(n, "extraction", "final_labels", "by_source", "human")
    n_b2 = req(n, "sensitivity", "data_extra", "text_counts", "batch2_human_labelled")
    if n_human != req(n, "extraction", "validation_n") + n_b2:
        raise SystemExit("human-coded narratives do not equal validation plus flagged")
    alt = req(n, "sensitivity", "alt_model")
    mask = req(n, "sensitivity", "masking")
    L = letter("app:validation")

    # C.1 coder agreement
    krows = []
    for label in sorted(kappa, key=lambda k: -kappa[k]):
        pw = req(pairwise, label)
        krows.append(f"{tex(cue_name(label))} & {rnd(kappa[label], 3)} & "
                     + " & ".join(f"{rnd(req(pw, k), 3)}" for k in ("1v2", "1v3", "2v3"))
                     + " \\\\")

    # C.2 route performance with intervals. The pre-crash cues are the labels Route A
    # reads, and the injury cues the rest.
    MECH_ALL = set(routes["route_a"]["per_label"])
    body = []
    for key, title in ROUTE_TITLES:
        body.append(panel(title, 9))
        per = routes[key]["per_label"]
        ints = req(sup, "per_cue_intervals", "routes", key)
        mech = [k for k in per if k in kappa and k in req(macro_ci, key, "labels")]
        for label in per:
            e = per[label]
            # Scores from the confusion counts, so a stored score is not rounded twice.
            pr, rc, f1 = prf(e)
            u = req(ints, label, "unweighted")
            if abs(u["f1"] - f1) > 6e-4:
                raise SystemExit(f"C: interval record for {key}.{label} has F1 {u['f1']}, "
                                 f"the counts give {f1:.4f}")
            lo, hi = req(u, "ci95", "f1")
            body.append(f"{tex(cue_name(label))} & {rnd(pr, 3)} & "
                        f"{rnd(rc, 3)} & {rnd(f1, 3)} & {ci(lo, hi)} & "
                        f"{int(e['tp'])} & {int(e['fp'])} & {int(e['fn'])} & "
                        f"{int(e['tn'])} \\\\")
        mu = req(macro_ci, key, "unweighted")
        lo, hi = req(mu, "macro_f1_mechanisms_only_ci95")
        mech_labels = [k for k in req(macro_ci, key, "labels") if k in per and k in MECH_ALL]
        body.append(f"\\textit{{Macro $F_1$, pre-crash cues}} & & & "
                    f"{rnd(macro_f1(per, mech_labels, req(mu, 'macro_f1_mechanisms_only')), 3)}"
                    f" & {ci(lo, hi)} & & & & \\\\")
        cues = routes[key].get("macro_f1_injury_cues")
        if cues is not None:
            inj = [k for k in per if k not in MECH_ALL]
            body.append(f"\\textit{{Macro $F_1$, injury cues}} & & & "
                        f"{rnd(macro_f1(per, inj, cues), 3)} & -- & & & & \\\\")
        if len(mech) == 0:
            raise SystemExit(f"C: no pre-crash cues for {key}")

    # C.3 calibration
    folds = req(cal_set, "folds")
    crows, few = [], []
    for label, c in labels_cal.items():
        crows.append(f"{tex(cue_name(label))} & {req(c, 'human_positives')} & "
                     f"{req(c, 'folds_used')} & {rnd(req(c, 'raw_vote_share_ece_weighted'), 4)} & "
                     f"{rnd(req(c, 'oof_ece_weighted'), 4)} & "
                     f"{rnd(req(c, 'raw_vote_share_brier_weighted'), 4)} & "
                     f"{rnd(req(c, 'oof_brier_weighted'), 4)} \\\\")
        if c["folds_used"] < folds:
            few.append((label, c["human_positives"], c["folds_used"]))

    # C.4 alternative model and masking
    pq = req(alt, "paired_llama_vs_qwen")
    vl = req(alt, "validation", "llama_reference", "per_label")
    vq = req(alt, "validation", "qwen", "per_label")
    arows = []
    for label in vl:
        # Scores and differences from the confusion counts of each model.
        a = prf(req(vl, label, "unweighted"))
        b = prf(req(vq, label, "unweighted"))
        d = req(pq, "per_label", label)
        if abs(d["f1_difference"] - (a[2] - b[2])) > 6e-4:
            raise SystemExit(f"C.4: F1 difference for {label} does not follow from the counts")
        lo, hi = d["f1_difference_ci95"]
        arows.append(f"{tex(cue_name(label))} & {rnd(a[0], 2)} & {rnd(a[1], 2)} & "
                     f"{rnd(a[2], 2)} & {rnd(b[0], 2)} & {rnd(b[1], 2)} & "
                     f"{rnd(b[2], 2)} & {fmt(a[2] - b[2], 3)} {ci(lo, hi)} & "
                     f"{pfmt(d['mcnemar_exact_p'])} \\\\")
    lo_l, hi_l = pq["macro_f1_llama_ci95"]
    lo_q, hi_q = pq["macro_f1_qwen_ci95"]
    lo, hi = pq["macro_f1_difference_ci95"]
    m_l, m_q = alt_macros(alt)
    arows.append("\\midrule\n\\textit{Macro $F_1$} & \\multicolumn{3}{c}{"
                 f"{rnd(m_l, 3)} {ci(lo_l, hi_l)}}} & \\multicolumn{{3}}{{c}}{{"
                 f"{rnd(m_q, 3)} {ci(lo_q, hi_q)}}} & "
                 f"{fmt(m_l - m_q, 3)} {ci(lo, hi)} & \\\\")

    sp = req(mask, "validation", "same_period_masked_rerun")
    comps = (
        ("Unmasked against masked rerun, same period",
         req(sp, "paired_unmasked_vs_masked_rerun"), "macro_f1_unmasked",
         "macro_f1_masked_rerun"),
        ("The same, narratives changed by masking",
         req(sp, "paired_unmasked_vs_masked_rerun_text_changed_only"), "macro_f1_unmasked",
         "macro_f1_masked_rerun"),
        ("Unmasked run against masked reference run",
         req(mask, "validation", "paired_unmasked_vs_masked"), "macro_f1_unmasked",
         "macro_f1_masked"),
        ("Masked rerun against masked reference run",
         req(sp, "paired_masked_rerun_vs_masked_reference"), "macro_f1_masked_rerun",
         "macro_f1_masked_reference"),
    )
    arows.append(panel("Panel B. Masked against unmasked narratives, macro $F_1$ on the "
                       "pre-crash cues", 9))
    arows.append(inner_header("Comparison & $n$ & \\multicolumn{2}{c}{First run} & "
                              "\\multicolumn{2}{c}{Second run} & "
                              "\\multicolumn{2}{c}{Difference [95\\% CI]} & \\\\"))
    # The full-sample comparison has confusion counts for both runs, so its two scores and
    # their difference are computed from them, as in Table C.2. The same-period reruns
    # have only their stored scores.
    mv = req(mask, "validation")
    full = req(mv, "paired_unmasked_vs_masked")
    counted = {}
    if mv["masked"]["n"] == mv["unmasked"]["n"] == full["n"]:
        for run, key in (("unmasked", "macro_f1_unmasked"), ("masked", "macro_f1_masked")):
            per = {k: req(e, "unweighted") for k, e in mv[run]["per_label"].items()}
            counted[key] = macro_f1(per, mv[run]["labels_in_macro"], full[key])
    for title, e, first, second in comps:
        l1, h1 = e[first + "_ci95"]
        l2, h2 = e[second + "_ci95"]
        lo, hi = e["macro_f1_difference_ci95"]
        first_val, second_val, dv = e[first], e[second], e["macro_f1_difference"]
        if e is full and counted:
            first_val, second_val = counted[first], counted[second]
            dv = first_val - second_val
        arows.append(f"{title} & {e['n']} & \\multicolumn{{2}}{{c}}{{{rnd(first_val, 3)}}} & "
                     f"\\multicolumn{{2}}{{c}}{{{rnd(second_val, 3)}}} & "
                     f"\\multicolumn{{2}}{{c}}{{{fmt(dv, 4)} "
                     f"{ci(lo, hi, 4)}}} & \\\\")

    sev = req(mask, "severity")
    arows.append(panel("Panel C. Cue block of the severity model, M0c against M1", 9))
    arows.append(inner_header("Label source & \\multicolumn{2}{c}{$\\chi^2$} & "
                              "\\multicolumn{2}{c}{d.f.} & \\multicolumn{2}{c}{$p$} & & \\\\"))
    for title, key in (("Masked narratives, reference run", "masked"),
                       ("Unmasked narratives", "unmasked")):
        t = req(sev, key, "lr_M0c_to_M1_cue_block")
        arows.append(f"{title} & \\multicolumn{{2}}{{c}}{{{rnd(t['lr_chi2'], 2)}}} & "
                     f"\\multicolumn{{2}}{{c}}{{{t['df']}}} & "
                     f"\\multicolumn{{2}}{{c}}{{{pfmt(t['p'])}}} & & \\\\")

    rr = req(sp, "reference_reproduction")
    ti = req(mask, "agreement_all_narratives", "text_identical")
    arows.append(panel("Panel D. Variation between runs with identical prompts", 9))
    arows.append(inner_header("Comparison & \\multicolumn{2}{c}{Decisions} & "
                              "\\multicolumn{2}{c}{Differing} & "
                              "\\multicolumn{2}{c}{Share (\\%)} & & \\\\"))
    arows.append("Masked rerun against reference run, validation narratives & "
                 f"\\multicolumn{{2}}{{c}}{{{rr['label_decisions']}}} & "
                 f"\\multicolumn{{2}}{{c}}{{{rr['decisions_differing_from_reference']}}} & "
                 f"\\multicolumn{{2}}{{c}}{{{pct(rr['share_differing'])}}} & & \\\\")
    arows.append("Unmasked against masked run, narratives masking leaves unchanged & "
                 f"\\multicolumn{{2}}{{c}}{{{ti['label_decisions']}}} & "
                 f"\\multicolumn{{2}}{{c}}{{{ti['decisions_differing']}}} & "
                 f"\\multicolumn{{2}}{{c}}{{{pct(ti['share_of_decisions_differing'])}}} & "
                 "& \\\\")

    st = req(alt, "settings")
    prov_q = req(alt, "provenance", "qwen")
    # The logged value carries its source in parentheses, which the prose states instead.
    quant_q = re.sub(r"\s*\(.*\)\s*$", "", req(prov_q, "quantization_logged", 0))
    ident = req(alt, "provenance", "prompt_identity_llama_vs_qwen")
    model_l = req(alt, "provenance", "llama_reference", "models_returned", 0)
    n_val = ex["validation_n"]

    few_text = ""
    if few:
        parts = [f"{tex(cue_name(l)).lower()}, with {hp} human positives and {fu} folds"
                 for l, hp, fu in few]
        few_text = (" A cue with few human positives is fitted on fewer folds, as for "
                    + joined(parts) + ".")

    text = (
        "This appendix reports the extraction validation in detail. A probability sample "
        f"of {n_val} narratives was coded independently by three human coders, and their "
        "majority vote, taken without a discussion step, is the reference against which "
        "every automated route is scored. Agreement among the coders is reported first and on its own, in "
        f"Table~{L}.1, because it involves no automated output. Fleiss' $\\kappa$ is "
        "computed over the three coders together, and each pairwise column is Cohen's "
        "$\\kappa$ for one pair of coders. "
        f"Table~{L}.2 then scores each automated route against the reference with "
        "precision, recall and $F_1$, the four confusion counts, and a 95\\% percentile "
        f"bootstrap interval for $F_1$ from {boot['resamples']} resamples of narratives. "
        "The scores are unweighted, so they describe the validation sample rather than "
        "the population of narratives.\n\n"
        + longtable("Agreement among the three human coders, per label.",
                    "tab:a_kappa", "Xrrrr",
                    "Label & Fleiss' $\\kappa$ & Coders 1 and 2 & Coders 1 and 3 & "
                    "Coders 2 and 3 \\\\", krows,
                    note=("Fleiss' $\\kappa$ measures agreement among the three coders, "
                          "each pairwise column gives Cohen's $\\kappa$ for one pair, and no "
                          "automated output enters this table."))
        + longtable("Performance of each automated route against the coders' majority "
                    "vote.",
                    "tab:a_routes",
                    ">{\\raggedright\\arraybackslash}X rrr c rrrr",
                    "Label & P & R & $F_1$ & $F_1$ 95\\% CI & TP & FP & FN & TN \\\\",
                    body, size="\\footnotesize", tabcolsep="3.5pt",
                    note=("P is precision and R is recall, each computed against the "
                          "coders' majority vote. TP, FP, FN and TN are the "
                          "true positive, false positive, false negative and true "
                          "negative counts, so any cell can be recomputed. Route A reads "
                          "the masked narrative and is scored on the pre-crash cues only. "
                          "The repeated-run route labels a cue positive when more than half "
                          "of the repeated runs are positive. The macro rows average $F_1$ "
                          "over the cues of the pass, and the injury-cue macro mean was not "
                          "bootstrapped, so it has no interval."))
        + "The vote shares of Route~B were calibrated per cue with isotonic regression, "
        f"cross-fitted over {folds} folds on the {n_human} human-coded narratives, the "
        f"{n_val} of the validation sample and {n_b2} further narratives flagged for "
        "human coding. Each "
        "human-coded narrative receives the calibrated value from the map fitted without "
        f"it, so the errors in Table~{L}.3 are out of fold. Flagged narratives enter with a "
        "weight of one and the other validation narratives with the inverse of their "
        "inclusion probability, so the errors describe the population of narratives. "
        f"Table~{L}.3 compares the expected calibration error and the Brier score of the "
        "raw vote share with those of the calibrated value." + few_text + "\n\n"
        + longtable("Out-of-fold calibration of the Route B vote shares, per cue.",
                    "tab:a_calibration", "Xrrrrrr",
                    "Cue & Human positives & Folds & ECE raw & ECE calibrated & Brier raw & "
                    "Brier calibrated \\\\", crows,
                    note=("ECE is the expected calibration error, and both measures are "
                          "weighted by the inverse inclusion probability and computed out "
                          "of fold, and lower values are better. The raw columns score the "
                          "uncalibrated vote share on the same narratives."))
        + "Two further checks concern the extraction itself, and the first replaces the "
        f"language model. The mechanism pass was rerun on the {n_val} validation "
        f"narratives with \\texttt{{{tex(st['alternative_model_requested'])}}} through the "
        f"same pinned provider, {tex(st['alternative_provider_pinned'])}, with prompt "
        f"version \\texttt{{{tex(st['prompt_version'])}}}, the same JSON schema, "
        f"temperature {st['temperature']:g} and seed {st['seed']}, and with provider "
        f"fallbacks disabled. All {ident['identical_prompt_sha256']} prompts were "
        f"identical, by hash, to those of the reference run with "
        f"\\texttt{{{tex(model_l)}}}, and the provider listed the endpoint as serving "
        f"\\texttt{{{tex(quant_q)}}} weights. "
        "The second check removes the masking from the mechanism pass. The mechanism pass was rerun on every "
        "unmasked narrative, and a masked rerun on the validation narratives was made in "
        "the same period, so the masking comparison is not confounded with the time "
        f"between runs. Panel~D of Table~{L}.4 gives the variation between runs with "
        "identical prompts, which sets the scale for both comparisons.\n\n"
        + longtable("Alternative language model, masking and rerun variation.",
                    "tab:a_altmodel",
                    ">{\\raggedright\\arraybackslash}X rrr rrr "
                    ">{\\centering\\arraybackslash}p{2.9cm} r",
                    "Label & P & R & $F_1$ & P & R & $F_1$ & "
                    "$\\Delta F_1$ [95\\% CI] & $p$ \\\\",
                    [panel("Panel A. "
                           + tex(model_l.split('/')[-1]) + " (first three columns) against "
                           + tex(st['alternative_model_requested'].split('/')[-1])
                           + " (next three)", 9)] + arows,
                    size="\\footnotesize", repeat_head=False,
                    note=("Panel A scores both models on the ten pre-crash cues of the "
                          f"{n_val} validation narratives, with $\\Delta F_1$ the first "
                          "model minus the second and $p$ the exact McNemar test of the "
                          "paired decisions. A difference in Panel B is the first run "
                          "minus the second, and every interval is a paired percentile "
                          f"bootstrap from {pq['n_boot']} resamples. One validation "
                          "narrative has no label in the masked rerun, which is why the "
                          "same-period comparisons hold one narrative fewer. Panel C tests "
                          "the five cues of M1 with labels from each run on every rider.")))

    return section("app:validation", text)


# ---------------------------------------------------------------- D audit diagnostics

def app_auditdiag(n: dict) -> str:
    af = req(n, "sensitivity", "audit_final")
    st = req(af, "settings")
    prim = req(af, "primary")
    dep = req(af, "dependence_sensitivity")
    nl = req(af, "narrative_length_sensitivity", "models")
    L = letter("app:auditdiag")

    def share(m: dict) -> str:
        s = m.get("model_implied_class_share")
        if not s:
            return "--"
        o = s["O"]
        return f"{rnd(o['mean'], 2)} {ci(o['ci95'][0], o['ci95'][1], 2)}"

    def lam(m: dict) -> str:
        if m.get("dependence_fixed_at") is not None:
            return f"{m['dependence_fixed_at']:g}"
        d = m["dependence_posterior"]
        return f"{rnd(d['mean'], 3)} {ci(d['ci95'][0], d['ci95'][1], 3)}"

    def row(title: str, m: dict, lam_text: str | None = None) -> str:
        dg, fit = m["diagnostics"], m["fit_against_raw_table"]
        return (f"{title} & {lam_text or lam(m)} & {dg['n_parameters_summarized']} & "
                f"{rnd(dg['max_r_hat'], 3)} & {dg['min_ess_bulk']} / {dg['min_ess_tail']} & "
                f"{dg['divergences']} & {rnd(fit['max_abs_pearson_residual'], 3)} & "
                f"{'Yes' if fit['reproduces_raw_table'] else 'No'} & {share(m)} \\\\")

    rows = [panel("Panel A. Dependence term fixed", 9),
            row("Moderate prior, primary", prim["moderate"]),
            row("Weak prior", prim["weak"])]
    for value, m in sorted(dep["fixed_grid_moderate"].items(), key=lambda kv: float(kv[0])):
        rows.append(row("Moderate prior", m, f"{float(value):g}"))
    rows.append(panel("Panel B. Dependence term estimated", 9))
    rows.append(row("Moderate prior", dep["estimated"]["moderate"]))
    rows.append(row("Weak prior", dep["estimated"]["weak"]))
    rows.append(panel("Panel C. Narrative length added, moderate prior, no dependence term",
                      9))
    length_titles = {
        "class_specific_with_prevalence_slopes": "Class-specific slopes, prevalence slopes",
        "class_specific_no_prevalence_slopes": "Class-specific slopes only",
        "class_common_with_prevalence_slopes": "Common slopes, prevalence slopes",
    }
    for key, m in nl.items():
        if m.get("prior_set") != "moderate":
            raise SystemExit(f"D: length model {key} is not on the moderate prior")
        rows.append(row(length_titles.get(key, tex(key.replace("_", " "))), m, "--"))

    # The primary length model and the slopes whose intervals exclude zero.
    lp = nl["class_specific_with_prevalence_slopes"]
    n_narr_ex = lp["n_narrative_slopes_excluding_zero"]
    prev_ex = [k for k, v in lp["prevalence_length_slopes"].items() if v["excludes_zero"]]
    n_narr = len(lp["narrative_length_slopes"])
    n_prev = len(lp["prevalence_length_slopes"])
    if n_narr + n_prev != lp["n_length_parameters"]:
        raise SystemExit("D: length slopes do not add up to the length parameters")
    slope_text = (
        "In the primary length model, with class-specific slopes and prevalence slopes, "
        f"{words(n_narr_ex)} of the {words(n_narr)} narrative slopes and "
        f"{words(len(prev_ex))} of the {words(n_prev)} prevalence slopes have a 95\\% "
        f"interval that excludes zero, {words(n_narr_ex + len(prev_ex))} of its "
        f"{words(lp['n_length_parameters'])} length parameters in all.")

    every = ([("the primary fit", prim["moderate"]), ("the weak-prior fit", prim["weak"])]
             + [(f"the fit with the dependence term fixed at {float(v):g}", m)
                for v, m in dep["fixed_grid_moderate"].items()]
             + [("the moderate-prior fit with the term estimated",
                 dep["estimated"]["moderate"]),
                ("the weak-prior fit with the term estimated", dep["estimated"]["weak"])]
             + [(f"the length model with {length_titles[k].lower()}", m)
                for k, m in nl.items()])
    unhealthy = [(t, m["diagnostics"]) for t, m in every if not m["diagnostics"]["healthy"]]
    criteria = ("a largest $\\hat{R}$ below 1.01, a smallest bulk effective sample size "
                "above 400 and no divergent transition")
    health_text = f"Every fit meets the convergence criteria of {criteria}."
    if unhealthy:
        parts = [f"{t}, with a largest $\\hat{{R}}$ of {rnd(d['max_r_hat'], 3)} and a smallest "
                 f"bulk effective sample size of {d['min_ess_bulk']}" for t, d in unhealthy]
        health_text = (f"The convergence criteria are {criteria}. Every fit meets them "
                       "except " + joined(parts) + ", whose values are therefore less "
                       "precise than the others.")

    fixed0 = [prim["moderate"], prim["weak"]]
    misfit0 = all(not m["fit_against_raw_table"]["reproduces_raw_table"] for m in fixed0)
    est = [dep["estimated"]["moderate"], dep["estimated"]["weak"]]
    fit_est = all(m["fit_against_raw_table"]["reproduces_raw_table"] for m in est)
    o_est = dep["estimated"]["moderate"]["model_implied_class_share"]["O"]
    length_fit = all(not m["fit_against_raw_table"]["reproduces_raw_table"]
                     for m in nl.values())

    reading = []
    if misfit0:
        reading.append("With the dependence term fixed at zero, the largest residual "
                       "exceeds that threshold under both priors.")
    if fit_est:
        reading.append(
            "With the term estimated, the observed table is reproduced under both priors, "
            "but the latent class shares are then weakly identified, and the O share "
            f"under the moderate prior has a 95\\% interval from {rnd(o_est['ci95'][0], 2)} to "
            f"{rnd(o_est['ci95'][1], 2)}.")
    if length_fit:
        reading.append("Adding narrative length as a predictor of the narrative indicator "
                       "does not bring the residuals under the threshold in any of the "
                       "three forms fitted.")

    text = (
        "This appendix reports the convergence and the fit of the Bayesian measurement "
        "model behind the reporting audit, for the primary specification and for each "
        f"sensitivity fit. Every fit used the {tex(st['sampler'])} implementation of the "
        f"No-U-Turn sampler with {st['chains']} chains, target acceptance "
        f"{st['target_accept']:g} and seed {st['seed']}. The primary fits ran "
        f"{st['tune']} tuning and {st['draws']} sampling iterations per chain, and every "
        f"sensitivity fit ran {st['sensitivity_fits']['tune']} of each. The dependence "
        "term is a log-scale boost on the cells where the coded level and the narrative "
        "level imply the same severity position. The fit check compares the table implied "
        "by the posterior with the observed cross-tabulation of Table~\\ref{tab:audit}, "
        "and a largest absolute Pearson residual above about 2 means that the measurement "
        "layer does not represent the observed table. " + health_text + "\n\n"
        + " ".join(reading) + " " + slope_text + " Table~"
        f"{L}.1 gives the diagnostics and the fit check of every fit. "
        + supp("tab:a_shifter") + " gives the recovery of a planted reporting shifter in "
        "simulated data, which is why the demographic and contextual shifters are not part "
        "of the reported audit.\n\n"
        + longtable("Convergence and fit of the reporting-audit measurement model.",
                    "tab:a_auditdiag",
                    ">{\\raggedright\\arraybackslash}X c r r c r r c "
                    ">{\\centering\\arraybackslash}p{2.3cm}",
                    "Specification & $\\lambda$ & Par. & $\\hat{R}$ & ESS bulk / tail & "
                    "Div. & $|r|_{\\max}$ & Fits table & O share [95\\% CI] \\\\",
                    rows, size="\\footnotesize",
                    note=("$\\lambda$ is the dependence term, fixed or with its posterior "
                          "mean and 95\\% interval, and a dash marks the length models, "
                          "which carry no dependence term. Par.\\ counts the summarized "
                          "parameters, $\\hat{R}$ is the largest potential scale reduction "
                          "factor, ESS gives the smallest bulk and tail effective sample "
                          "sizes, Div.\\ counts divergent transitions, and $|r|_{\\max}$ is "
                          "the largest absolute Pearson residual against the observed "
                          "table. The O share is the model-implied share of the latent O "
                          "class. A dash in that column marks a fit where the prevalence "
                          "parameter is not the class share and the share was not "
                          "computed.")))
    return section("app:auditdiag", text)


def shifter_block(n: dict) -> str:
    """The shifter-recovery simulation, printed in the Supplementary material."""
    sp = req(n, "audit", "shifter_power")
    text = (
        "The demographic and contextual reporting shifters are not part of the reported "
        "audit. A recovery simulation planted a shifter of "
        f"{sp['planted_effect']:g} on the latent scale in {sp['datasets']} simulated data "
        "sets of the observed size and refitted the model to each, with the result shown "
        "in Table~\\ref{tab:a_shifter}.\n\n")
    return text + longtable("Recovery of a planted reporting shifter in simulated data.",
                    "tab:a_shifter", ">{\\raggedright\\arraybackslash}X r",
                    "Quantity & Value \\\\",
                    [f"Planted shifter on the latent scale & {sp['planted_effect']:g} \\\\",
                     f"Simulated data sets & {sp['datasets']} \\\\",
                     f"Mean posterior mean of the shifter & {rnd(sp['mean_posterior_mean'], 3)} \\\\",
                     f"Ratio of that mean to the planted value & {sp['multiplicative_bias']:g} \\\\",
                     f"Data sets whose 95\\% interval covers the planted value & "
                     f"{sp['coverage'].replace('/', ' of ')} \\\\",
                     f"Data sets whose 95\\% interval excludes zero & "
                     f"{sp['power'].replace('/', ' of ')} \\\\",
                     f"Mean count of zero-valued shifters with an interval excluding zero & "
                     f"{rnd(sp['false_positives_of_47'], 2)} \\\\"],
                    note=("The simulation keeps the observed sample size and class "
                          "structure, plants one non-zero shifter, and holds the other "
                          "shifters at zero."))


# ---------------------------------------------------------------- E ordinal family

def _ordinal_parts(n: dict) -> tuple[str, str]:
    po = n["severity"]["proportional_odds"]
    mult = n["severity"]["multiplicity"]
    mnl = n["appendix"]["multinomial_robustness"]
    iia = n["severity"]["iia"]
    L = letter("app:ordinal")

    rows = [panel("Panel A. Tests of proportional odds", 5),
            f"Omnibus likelihood ratio & {rnd(po['omnibus_lr_chi2'], 2)} & "
            f"{po['omnibus_df']} & {pfmt(po['omnibus_p'])} & "
            f"{'Rejects' if po['rejects'] else 'Does not reject'} \\\\",
            f"Wald form, for comparison & {rnd(po['wald_form_chi2'], 2)} & "
            f"{po['omnibus_df']} & {pfmt(po['wald_form_p'])} & "
            f"{'Rejects' if po['wald_form_p'] < 0.05 else 'Does not reject'} \\\\",
            panel("Panel B. Selection under each multiplicity correction", 5),
            inner_header("Correction & Coefficients freed & $\\Delta$BIC & & "
                         "Selected model \\\\")]
    titles = {"uncorrected": "None", "benjamini_hochberg": "Benjamini and Hochberg",
              "holm": "Holm"}
    for name, entry in mult.items():
        rows.append(f"{titles.get(name, tex(name))} & {entry['n_freed']} & "
                    f"{fmt(entry['bic_difference'], 2)} & & "
                    f"{tex(entry['primary']).capitalize()} \\\\")

    fit = req(mnl, "multinomial_fit")
    cal = req(mnl, "small_hsiao_calibration")
    irows = []
    for idx, s in enumerate(SEV3):
        t = req(mnl, "iia_tests", s)
        sh = t["small_hsiao"]
        null = req(cal, "per_outcome", f"drop_{idx}")
        irows.append(f"{s} & {t['n_retained']} & {fmt(t['hausman_mcfadden_chi2'], 2)} & "
                     f"{pct(sh['reject_fraction_at_5pct'])} ({sh['usable']}) & "
                     f"{pct(null['reject_fraction'])} ({null['usable']}) \\\\")
    tests = [req(mnl, "iia_tests", s) for s in SEV3]
    dfs = {t["hausman_mcfadden_df"] for t in tests} | {t["small_hsiao"]["df"] for t in tests}
    if len(dfs) != 1:
        raise SystemExit(f"Table E.2: the IIA tests differ in degrees of freedom ({dfs})")
    iia_df = dfs.pop()
    n_neg = sum(bool(t["hausman_mcfadden_negative"]) for t in tests)
    if any(t["hausman_mcfadden_negative"] != (t["hausman_mcfadden_p"] is None) for t in tests):
        raise SystemExit("Table E.2: a negative Hausman-McFadden statistic carries a p-value")
    if n_neg == len(tests):
        neg_text = (f"is negative for all {words(len(tests))} outcomes, a known "
                    "finite-sample property of the test, so no $p$-value is defined for any "
                    "of them")
    elif n_neg:
        neg_text = (f"is negative for {words(n_neg)} of the {words(len(tests))} outcomes, "
                    "a known finite-sample property of the test, and no $p$-value is "
                    "defined for those")
    else:
        neg_text = "is positive for every outcome"
    neg_note = ("" if not n_neg else
                " A negative Hausman and McFadden statistic leaves no $p$-value defined, "
                "so none is given.")

    text = (
        "The omnibus likelihood-ratio test rejects proportional odds, and "
        f"{po['n_failing']} of the {po['omnibus_df']} covariates depart individually at the "
        "5\\% level (Panel~A of Table~\\ref{tab:a_ordinal}, beside the Wald form, which omits "
        "the cross-equation covariance). Panel~B gives the BIC difference of the partial "
        "proportional-odds model under each multiplicity correction. The pre-specification "
        "names the uncorrected set, so the ordered logit is primary. The multinomial logit "
        "fitted for comparison with the e-scooter literature, and its independence tests, are in "
        + "Supplementary Table~\\" + supp_macro("tab:a_iia") + "{}.\n\n"
        + longtable("Proportional-odds tests and selection under multiplicity corrections.",
                    "tab:a_ordinal", "Xrrrl",
                    "Test & $\\chi^2$ & d.f. & $p$ & Conclusion \\\\", rows,
                    repeat_head=False,
                    note=("A negative $\\Delta$BIC favors the partial proportional-odds "
                          "model over the ordered logit.")))
    iia_text = ("The unordered multinomial logit has "
        f"{fit['n_parameters']} parameters and a log-likelihood of "
        f"{fmt(fit['loglik'], 2)} on {fit['n']} riders. Its independence assumption was "
        "tested with the Hausman and McFadden test and the Small and Hsiao test, each "
        f"dropping one outcome at a time, as Table~{L}.2 shows. The Hausman and McFadden "
        f"statistic {neg_text}. The Small and Hsiao test was also run on data simulated "
        "with independence holding by construction, where a test at the 5\\% level should "
        "reject in about one split in twenty. It rejected in up to "
        f"{pct(iia['small_hsiao_worst_reject_rate'], 0)}\\% of the splits, so at this "
        "sample size its rejections do not separate a violation of independence from the "
        "behavior of the test itself.\n\n"
        + longtable("Tests of independence from irrelevant alternatives in the multinomial "
                    "logit.", "tab:a_iia",
                    ">{\\raggedright\\arraybackslash}X r"
                    "*{3}{>{\\centering\\arraybackslash}p{2.7cm}}",
                    "Outcome dropped & Riders kept & Hausman and McFadden $\\chi^2$ & "
                    "Small and Hsiao rejections (\\%) & Rejections under independence "
                    "(\\%) \\\\", irows,
                    note=(f"Both tests have {iia_df} degrees of freedom for every "
                          "outcome dropped." + neg_note + " The rejection columns give the "
                          "percentage of random sample splits rejecting at the 5\\% level, "
                          "with the number of usable splits in parentheses, on the observed "
                          "data and on data simulated under independence.")))
    return text, iia_text


def app_ordinal(n: dict) -> str:
    return section("app:ordinal", _ordinal_parts(n)[0] + ordinal_family_block(n))


def app_iia(n: dict) -> str:
    return section("app:iia", _ordinal_parts(n)[1])


# ---------------------------------------------------------------- F four-class

def app_fourclass(n: dict) -> str:
    d = n["appendix"]["four_class_detail"]
    fit = d["fit"]
    ames = d["average_marginal_effects"]
    order = d["order"]
    five = req(n, "sensitivity", "supporting", "data_sensitivities", "severity_classes",
               "five_class")
    epp5 = req(five, "events_per_parameter")
    rows = []
    for name in sorted(ames, key=lambda k: -abs(ames[k][order[-1]]["ame"])):
        e = ames[name]
        cells = []
        for level in order:
            lo, hi = e[level]["ci95"]
            cells.append(bold_if(f"{fmt(e[level]['ame'])} {ci(lo, hi)}",
                                 e[level]["excludes_zero"]))
        rows.append(f"{tex(pretty(name))} & " + " & ".join(cells) + " \\\\")

    smallest = epp5["smallest_class"]
    small_level = next(k for k, v in epp5["class_counts"].items() if v == smallest)
    five_text = (
        "The five-class ordering was also fitted as a sensitivity. It converged "
        f"{'without' if not five['separation']['suggests_separation'] else 'with'} a "
        f"separation flag, but its smallest class, {small_level}, holds {smallest} riders, "
        f"which leaves {rnd(epp(epp5['class_counts'], epp5['n_parameters'], epp5['events_per_parameter_smallest_class']), 3)} events per "
        "parameter. That is below the floor of one event per parameter fixed before "
        "fitting, so the five-class estimates are not interpreted."
        if not five.get("estimable") else
        "The five-class ordering was also fitted as a sensitivity and met the "
        "estimability rule fixed before fitting.")

    text = (
        "This appendix reports the four-class ordering of severity, "
        + joined(order) + ", estimated in the same ordered logit family and with the same "
        f"{len(ames)} covariates as M1. Its adjusted $\\rho^2$ is "
        f"{fmt(fit['adjusted_rho2'], 4)} with {fit['n_parameters']} parameters on "
        f"{fit['n']} riders, so once its parameters are paid for it fits worse than a "
        "model with thresholds only. " + five_text + " Table~"
        f"{letter('app:fourclass')}.1 gives the average marginal effect of each covariate "
        "on each outcome probability, ordered by the size of the effect on the most "
        "severe class.\n\n")
    return section("app:fourclass", text + longtable(
        "Average marginal effects in the four-class ordering.",
        "tab:a_fourclass",
        ">{\\raggedright\\arraybackslash}p{3.2cm}"
        + "*{%d}{>{\\centering\\arraybackslash}X}" % len(order),
        "Covariate & " + " & ".join(order) + " \\\\", rows,
        size="\\footnotesize", hlcolour="HlC",
        note=("Cells give the average marginal effect with its 95\\% interval. Bold marks "
              "an effect whose 95\\% interval excludes zero.")))


# ---------------------------------------------------------------- G AMEs

def app_ames(n: dict) -> str:
    ames = n["severity"]["ames_all"]
    order = sorted(ames, key=lambda k: -abs(ames[k]["KA"]["ame"]))
    n_excl = sum(1 for e in ames.values() if any(e[s]["excludes_zero"] for s in SEV3))
    kr = req(n, "sensitivity", "labels", "settings", "kr_draws")

    # The absent-coordinates KA interval ends next to zero, so its dependence on the
    # Krinsky-Robb seed is printed with the table rather than left to the result file.
    sc = req(n, "sensitivity", "ordinal_alternatives", "ordinal_family",
             "primary_kr_seed_check")
    by_seed = sc["ka_ci95_by_seed"]["coord_missing"]
    ok_seed = sc["ka_excludes_zero_by_seed"]["coord_missing"]
    base_seed = str(sc["seeds"][0])
    if by_seed[base_seed] != ames["coord_missing"]["KA"]["ci95"]:
        raise SystemExit("seed check does not reproduce the absent-coordinates interval")
    others = [s for s in by_seed if s != base_seed]
    n_ok = sum(1 for s in others if ok_seed[s])
    seed_note = (
        f"The KA interval for crash coordinates absent was also drawn with "
        f"{words(len(others))} further seeds of {kr:,} draws each, and it "
        + (f"includes zero under each of them" if n_ok == 0
           else f"excludes zero under {words(n_ok)} of them")
        + ", with upper bounds of "
        + joined([fmt(by_seed[s][1], 5) for s in others])
        + (", so it includes zero whichever seed is used."
           if n_ok == 0 and not ok_seed[base_seed]
           else ", so that interval is borderline."))

    rows = []
    for name in order:
        cells = []
        for level in SEV3:
            e = ames[name][level]
            lo, hi = e["ci95"]
            cells.append(bold_if(f"{fmt(e['ame'])} {ci(lo, hi)}", e["excludes_zero"]))
        rows.append(f"{tex(pretty(name))} & " + " & ".join(cells) + r" \\")

    text = (
        "This appendix gives the average marginal effects of the primary specification, "
        "which are the quantities the text reports. Each is the change in an outcome "
        "probability for a one-unit change in the covariate, averaged over the riders "
        "with the other covariates at their observed values. For a binary covariate the "
        "change is from zero to one. The 95\\% intervals come from "
        f"{kr:,} Krinsky and Robb draws of the full parameter vector, thresholds "
        "included. The coefficients behind them are in Table~\\ref{tab:model_estimates}, "
        f"and {n_excl} of the {len(order)} covariates have an interval that excludes zero "
        "on at least one outcome.\n\n")
    return section("app:ames", text + longtable(
        "Average marginal effects on each outcome probability.",
        "tab:a_ames",
        ">{\\raggedright\\arraybackslash}p{4.0cm}"
        "*{3}{>{\\centering\\arraybackslash}X}",
        "Covariate & " + " & ".join(SEV3) + " \\\\", rows,
        size="\\footnotesize", hlcolour="HlC",
        note=("Cells give the average marginal effect with its 95\\% interval, and the "
              "rows are ordered by the size of the effect on the KA probability. Bold "
              "marks an effect whose 95\\% interval excludes zero. " + seed_note)))


# ---------------------------------------------------------------- H label variants

VARIANTS = (
    ("i_final_labels_primary", "final_labels_primary", "Final labels, primary"),
    ("ii_llm_reference_labels", "llm_reference_labels", "Reference run on every rider"),
    ("ii_b_vote_share_majority_labels", "vote_share_majority_labels",
     "Majority of the repeated runs"),
    ("iii_plugin_calibrated_hybrid", "calibrated_hybrid_mean",
     "Calibrated plug-in, human labels kept"),
    ("iii_b_plugin_calibrated_all_rows", "calibrated_all_rows_mean",
     "Calibrated plug-in on every rider"),
    ("iii_c_plugin_raw_vote_share", "raw_vote_share_mean", "Uncalibrated vote share"),
    ("iv_draws_calibrated", "draws_calibrated_mean", "Draws from the calibrated probability"),
    ("iv_b_draws_outcome_informed", "draws_outcome_informed_mean",
     "Outcome-informed draws"),
)


# The label sets of Table H.1 that hold a hard 0/1 label for every rider, with the number
# of riders their shares are taken over, every rider of the model frame or the human-coded
# riders of the reduced model.
HARD_LABEL_SETS = {"final_labels_primary": "all", "llm_reference_labels": "all",
                   "vote_share_majority_labels": "all",
                   "human_subset_human_labels_n187": "human",
                   "human_subset_route_b_labels_n187": "human"}


def app_labels(n: dict) -> str:
    lab = req(n, "sensitivity", "labels")
    var = lab["variants"]
    prev = lab["prevalence"]
    st = lab["settings"]
    red = req(var, "v_human_subset_reduced")
    L = letter("app:labels")

    def prev_cells(key: str) -> str:
        p = req(prev, key)
        size = HARD_LABEL_SETS.get(key)
        if size is None:
            return " & ".join(pct(req(p, c)) for c in MECH5)
        # A hard-label share is a count over the riders, stored at four places. The count
        # is recovered and the share recomputed, so the printed percentage is not a
        # stored rounded share rounded again.
        size = req(n, "data", "rider_rows") if size == "all" else red["rows"]
        cells = []
        for c in MECH5:
            share = req(p, c)
            k = round(share * size)
            if abs(share - k / size) > 5.1e-5:
                raise SystemExit(f"H.1: {key}.{c} = {share} is not a count over {size}")
            cells.append(pct(k / size))
        return " & ".join(cells)

    rows = [panel(f"Panel A. Every rider, $n = {var['i_final_labels_primary']['n']}$", 9)]
    for key, pkey, title in VARIANTS:
        v = req(var, key)
        if "cue_block_lr" in v:
            t = v["cue_block_lr"]
            stat = f"$\\chi^2 = {rnd(t['lr_chi2'], 2)}$"
            p = t["p"]
            cv = fmt(req(v, "cross_validation", "mean_difference_M1_minus_M0c"), 4)
        else:
            t = req(v, "cue_block_test_D1_wald")
            stat = f"$F = {rnd(t['D1_F'], 3)}$"
            p = t["p"]
            cv = "--"
        rows.append(f"{title} & {stat} & {pfmt(p)} & {cv} & {prev_cells(pkey)} \\\\")

    rows.append(panel(f"Panel B. Human-coded riders only, $n = {red['rows']}$", 9))
    for key, pkey, title in (
            ("human_labels_admitted_cues", "human_subset_human_labels_n187",
             "Human labels, admitted cues"),
            ("human_labels_all_five_cues", "human_subset_human_labels_n187",
             "Human labels, all five cues"),
            ("route_b_labels_same_rows_admitted_cues", "human_subset_route_b_labels_n187",
             "Route B labels, admitted cues")):
        v = req(red, key)
        t = v["cue_block_lr"]
        rows.append(f"{title} & $\\chi^2_{{{t['df']}}} = {rnd(t['lr_chi2'], 2)}$ & "
                    f"{pfmt(t['p'])} & -- & {prev_cells(pkey)} \\\\")

    fmi = {}
    for key in ("iv_draws_calibrated", "iv_b_draws_outcome_informed"):
        vals = [req(var, key, "fraction_missing_information", c, "coefficient")
                for c in MECH5]
        fmi[key] = (min(vals), max(vals))
    cv_set = req(st, "cv")
    ep = req(red, "events_per_parameter")
    admitted = [cue_name(c).lower() for c in req(red, "admitted_cues")]

    text = (
        "Cue coefficients describe associations with a cue being documented, and each label "
        f"source carries its own error. Table~{L}.1 refits the cue block of M1 under each "
        f"alternative label set, with the test of the {len(MECH5)} cues against M0c, the "
        "cross-validated change in log loss and the prevalence of each cue. Three plug-in "
        "sets replace an automated rider's hard label with a probability, and two sets "
        f"pool {st['label_draws']} draws by Rubin's rules with the D1 Wald test. Draws from "
        "the calibrated probability ignore severity and so attenuate toward zero, and the "
        "outcome-informed draws check that attenuation (fraction of missing information "
        f"{rnd(fmi['iv_draws_calibrated'][0], 2)} to {rnd(fmi['iv_draws_calibrated'][1], 2)} "
        f"and {rnd(fmi['iv_b_draws_outcome_informed'][0], 2)} to "
        f"{rnd(fmi['iv_b_draws_outcome_informed'][1], 2)}).\n\n"
        f"Panel~B fits a reduced model on the {red['rows']} human-coded riders, admitting a "
        f"cue only when each class holds at least {st['reduced_model_min_events']} riders "
        f"with it, so only {joined(admitted)} enter "
        f"({rnd(epp(ep['class_counts'], ep['n_parameters'], ep['events_per_parameter_smallest_class']), 2)} "
        "events per parameter in the smallest class). The same model with Route~B labels is "
        "shown beside it.\n\n")
    return section("app:labels", text + longtable(
        "The cue block of M1 under alternative label sets.",
        "tab:a_labelvariants",
        ">{\\raggedright\\arraybackslash}X c r r rrrrr",
        "Label set & Test & $p$ & $\\Delta$CV & " + " & ".join(MECH5_SHORT[c] for c in MECH5)
        + " \\\\", rows, size="\\footnotesize",
        note=(f"The test is the likelihood-ratio $\\chi^2$ of the cue block, with "
              f"{len(MECH5)} degrees of freedom unless a subscript gives fewer, or for the "
              f"drawn sets the D1 Wald $F$ pooled over {st['label_draws']} draws. "
              f"$\\Delta$CV is the mean over {cv_set['repeats']} repeats of "
              f"{cv_set['folds']}-fold cross-validation of the M1 log loss minus the M0c "
              "log loss on the same fold, so a positive value means the cues raised the "
              "out-of-sample loss. It was not computed for the drawn sets or the reduced "
              "model, which hold a dash. The last five columns give the percentage of riders with "
              "the cue, or the mean probability for a plug-in or drawn set, for the "
              "sidewalk to roadway transition (SW), failure to yield (FTY), signal "
              "violation (SV), vehicle turning across (VTA) and lane positioning (LP). "
              "Draws from the calibrated probability alone ignore severity, so they "
              "attenuate any association between a cue and severity toward zero. The "
              "reduced model admits a covariate or cue only when each severity class "
              f"holds at least {st['reduced_model_min_events']} riders with it."))
        + power_block(n) + excluded_block(n))


# ---------------------------------------------------------------- I M2

def app_m2(n: dict) -> str:
    m2 = req(n, "sensitivity", "m2")
    st = m2["settings"]
    # Weight vectors exist only for the (coded level, narrative level) cells riders occupy.
    counts = req(n, "audit", "cross_tab_kabco", "counts")
    cells_n = sum(1 for row in counts.values() for v in row.values() if v > 0)
    L = letter("app:m2")

    specs = (
        ("M1, model-based errors", m2["M1"]["coefficients_model_based"],
         m2["M1"]["ames_model_based"], None, m2["expected_class_counts"]["coded"]),
        ("M1, rider-clustered sandwich", m2["M1"]["coefficients_sandwich"],
         m2["M1"]["ames_sandwich"], None, m2["expected_class_counts"]["coded"]),
        ("M2, expected likelihood, pooled", m2["M2_expected_likelihood"]["coefficients"],
         m2["M2_expected_likelihood"]["ames"], m2["M2_expected_likelihood"]["fmi_summary"],
         m2["expected_class_counts"]["mean_over_draws"]),
        ("M2, posterior-mean weights",
         m2["M2_expected_likelihood_at_posterior_mean_weights"]["coefficients"], None, None,
         m2["expected_class_counts"]["posterior_mean_weights"]),
        ("M2, coded-class weight", m2["M2_alt_i_coded_class_weight"]["coefficients"],
         m2["M2_alt_i_coded_class_weight"]["ames"], None, None),
        ("M2, modal reassignment", m2["M2_alt_ii_modal_reassignment"]["coefficients"],
         m2["M2_alt_ii_modal_reassignment"]["ames"], None,
         m2["M2_alt_ii_modal_reassignment"]["modal_counts"]),
        ("M2, weak-prior weights",
         m2["M2_expected_likelihood_audit_sensitivity"]["weak_prior_weights"]["coefficients"],
         None,
         m2["M2_expected_likelihood_audit_sensitivity"]["weak_prior_weights"]["fmi_summary"],
         None),
        ("M2, estimated-dependence weights",
         m2["M2_expected_likelihood_audit_sensitivity"]["estimated_dependence_weights"][
             "coefficients"], None,
         m2["M2_expected_likelihood_audit_sensitivity"]["estimated_dependence_weights"][
             "fmi_summary"], None),
    )
    covs = []
    for _, coefs, _, _, _ in specs:
        for name, e in coefs.items():
            if e["excludes_zero"] and name not in covs:
                covs.append(name)
    covs.sort(key=list(specs[0][1]).index)

    def counts(c):
        if not c:
            return "--"
        return " / ".join(f"{c[s]:g}" for s in SEV3)

    rows = [panel("Panel A. Coefficients, with standard errors in parentheses", 6)]
    for title, coefs, _, fmi, cls in specs:
        cells = [bold_if(f"{fmt(coefs[c]['coef'])} ({rnd(coefs[c]['se'], 3)})",
                         coefs[c]["excludes_zero"]) for c in covs]
        f = f"{rnd(fmi['median'], 3)} ({rnd(fmi['max'], 3)})" if fmi else "--"
        rows.append(f"{title} & " + " & ".join(cells) + f" & {f} & {counts(cls)} \\\\")
    rows.append(panel("Panel B. Average marginal effects on the KA probability, "
                      "with 95\\% intervals", 6))
    for title, _, ames, _, _ in specs:
        if ames is None:
            continue
        cells = []
        for c in covs:
            e = ames[c]["KA"]
            cells.append(bold_if(f"{fmt(e['ame'])} {ci(e['ci95'][0], e['ci95'][1])}",
                                 e["excludes_zero"]))
        rows.append(f"{title} & " + " & ".join(cells) + " & & \\\\")

    alt_i = m2["M2_alt_i_coded_class_weight"]
    colspec = (">{\\raggedright\\arraybackslash}p{3.4cm}"
               + "*{%d}{>{\\centering\\arraybackslash}X}" % len(covs)
               + ">{\\centering\\arraybackslash}p{1.5cm}>{\\centering\\arraybackslash}p{2.1cm}")
    text = (
        "Model M2 refits the M1 specification with each rider's outcome replaced by the "
        "posterior probabilities of the latent severity classes from the primary audit. "
        "The weight of rider $i$ on class $c$ is\n"
        "\\begin{equation}\n"
        "w_{ic} = \\frac{\\pi_c \\, p_K(k_i \\mid c) \\, p_N(n_i \\mid c)}"
        "{\\sum_{c'} \\pi_{c'} \\, p_K(k_i \\mid c') \\, p_N(n_i \\mid c')},\n"
        "\\label{eq:a_m2weight}\n\\end{equation}\n"
        "where $\\pi_c$ is the latent prevalence, $k_i$ the coded KABCO level and $n_i$ the "
        "narrative level, and M2 maximizes $\\sum_i \\sum_c w_{ic} \\log P(Y_i = c \\mid "
        "\\mathbf{x}_i)$. The weights depend on a rider only through the coded level and the "
        f"narrative level, so the {st['n_riders']} riders carry {cells_n} distinct weight "
        "vectors. Because the weights are a function of the coded level that M1 models, "
        f"Eq.~(\\ref{{eq:a_m2weight}}) defines M2 as a sensitivity whose reading depends on "
        "the measurement assumptions of the audit. The weights are recomputed at each of "
        f"{st['n_posterior_draws']} posterior draws of the audit, each fit uses a "
        "rider-clustered sandwich variance, and the fits are pooled by Rubin's rules. M1 "
        "is refitted with the same sandwich variance for comparison, and the model-based "
        "M1 is shown as well.\n\n"
        "Two alternative weight definitions are reported beside the expected likelihood. The coded-class weight keeps "
        "each rider's coded class as the outcome and weights it by the posterior "
        "probability of that class, and modal reassignment replaces the coded class with "
        "the most probable latent class and gives every rider a weight of one. The "
        "expected-likelihood fit is also repeated at the posterior-mean weights, and with "
        "weights from the weak-prior audit and from the audit with the dependence term "
        f"estimated. Table~{L}.1 shows every covariate whose 95\\% interval excludes zero "
        "in at least one of these specifications.\n\n")
    return section("app:m2", text + longtable(
        "The latent-class-weighted model M2 under each weight definition.",
        "tab:a_m2", colspec,
        "Specification & " + " & ".join(tex(pretty(c)) for c in covs)
        + " & FMI median (max) & O / BC / KA \\\\",
        rows, size="\\footnotesize", repeat_head=False,
        note=("Bold marks an estimate whose 95\\% interval excludes zero. Standard errors are "
              "rider-clustered sandwich errors except in the first row. FMI is the "
              "fraction of missing information across the posterior draws, given as the "
              "median over the coefficients with the maximum in parentheses. The class "
              "sizes are the coded counts for M1, the expected counts under the posterior "
              "weights, or the modal counts. The coded-class weights sum to "
              f"{alt_i['sum_of_weights']:g}, with an effective sample size of "
              f"{alt_i['effective_n']:g}. A dash marks a quantity that is not defined for "
              "the specification, and Panel~B omits the fits without marginal effects.")))


# ---------------------------------------------------------------- J random parameters

def app_randpar(n: dict) -> str:
    rp = req(n, "sensitivity", "random_parameters")
    proc = rp["procedure"]
    cand = rp["candidates"]
    summ = rp["screen_summary"]
    het = rp["heterogeneity"]
    final = het["final_model"]
    joint = rp["joint_model"]
    comp = rp["comparison"]

    all_conv = all(c["convergence"]["converged"] for c in cand.values())
    starts = all(c["convergence"]["starts_agree_loglik_within_0.01"] for c in cand.values())

    single = {k: v for k, v in comp.items() if k.startswith("single_random[")}
    best_single = min(single, key=lambda k: single[k]["bic"])
    best_name = best_single[len("single_random["):-1]

    def fitrow(title, f, lr=None):
        lrc = (f"{rnd(lr['lr_chi2'], 2)} & {lr['df']} & {pfmt(lr.get('p_chi_bar_squared', lr.get('p')))}"
               if lr else "-- & -- & --")
        return (f"{title} & {fmt(f['loglik'], 2)} & {f['n_parameters']} & {rnd(f['aic'], 1)} & "
                f"{rnd(f['bic'], 1)} & {lrc} \\\\")

    mrows = [panel("Panel A. Fit", 8),
             fitrow("Fixed-parameter M1", rp["fixed_model"]["fit"]),
             fitrow(f"Single random coefficient, {tex(pretty(best_name)).lower()}",
                    single[best_single], single[best_single]["lr_vs_fixed"]),
             fitrow(f"Joint, {len(joint['random'])} random coefficients", joint["fit"],
                    joint["lr_vs_fixed"]),
             fitrow("Final, joint plus shifters", final["fit"], final["lr_vs_fixed"]),
             panel("Panel B. Standard deviations and shifters", 8),
             inner_header("Parameter & \\multicolumn{2}{c}{Joint estimate (SE)} & Joint $p$ & "
                          "\\multicolumn{2}{c}{Final estimate (SE)} & Final $p$ & \\\\")]

    def pname(key: str) -> str:
        m = re.match(r"(sd|mean_shift|var_shift)\[([^|\]]+)(?:\|([^\]]+))?\]", key)
        kind, a, b = m.groups()
        if kind == "sd":
            return "SD, " + tex(pretty(a)).lower()
        what = "Mean shift" if kind == "mean_shift" else "Variance shift"
        return f"{what}, {tex(pretty(a)).lower()} by {tex(pretty(b)).lower()}"

    keys = [k for k in final["parameters"] if "[" in k]
    for k in keys:
        f = final["parameters"][k]
        j = joint["parameters"].get(k)
        jcell = (f"\\multicolumn{{2}}{{c}}{{{fmt(j['est'])} ({rnd(j['se'], 3)})}} & {pfmt(j['p'])}"
                 if j else "\\multicolumn{2}{c}{--} & --")
        mrows.append(f"{pname(k)} & {jcell} & "
                     f"\\multicolumn{{2}}{{c}}{{{fmt(f['est'])} ({rnd(f['se'], 3)})}} & "
                     f"{pfmt(f['p'])} & \\\\")

    fixed_bic = rp["fixed_model"]["fit"]["bic"]
    lower_aic = [t for t, f in (("joint", joint["fit"]), ("final", final["fit"]))
                 if f["aic"] < rp["fixed_model"]["fit"]["aic"]]
    higher_bic = [t for t, f in (("joint", joint["fit"]), ("final", final["fit"]))
                  if f["bic"] > fixed_bic]
    reading = ""
    if lower_aic == ["joint", "final"] and higher_bic == ["joint", "final"]:
        reading = (" The joint and final models have a lower AIC than the fixed M1 but a "
                   "higher BIC, so under the decision rule stated in the Methods the fixed "
                   "M1 stays primary.")
    if single[best_single]["bic"] < fixed_bic:
        reading += (f" The single random coefficient on "
                    f"{tex(pretty(best_name)).lower()} has the lowest BIC of all the "
                    "models.")

    n_het = het["n_tests"]
    text = (
        "This appendix reports the random-parameter robustness analysis summarized in the "
        "Results. Each of the "
        f"{len(cand)} covariates of M1 was given a normally distributed coefficient in "
        "turn, with the others fixed, and estimated by simulated maximum likelihood over "
        f"{proc['draws']['estimation']} scrambled Halton draws. A candidate was retained "
        "when the Wald test of its standard deviation and the likelihood-ratio test "
        f"against the fixed M1 were both significant at {pct(proc['alpha'], 0)}\\%, the "
        "second against a chi-bar-squared reference, because a standard deviation of zero "
        "lies on the boundary of the parameter space. " + supp("tab:a_randompar")
        + f" gives the result for every candidate, and {words(summ['n_retained'])} were "
        "retained. "
        + ("Every candidate model converged" if all_conv else
           "Not every candidate model converged")
        + (", and its log-likelihood agreed across the three starting values of the "
           "standard deviation." if starts else ".")
        + " The specification, the checks over draws and seeds, and the stability of "
        "the estimates are given in Supplementary Section S4.\n\n"
        + f"The {summ['n_retained']} retained coefficients were then made random together. "
        "Heterogeneity in their means and variances was tested one context variable at a "
        f"time against the {joined([tex(pretty(k)).lower() for k in proc['context_variables']])} "
        f"indicators, in {n_het} admissible tests of which "
        f"{het['n_tests_not_converged']} did not converge, and the shifters significant "
        f"at {pct(proc['alpha'], 0)}\\% entered a final model together. "
        "Table~\\ref{tab:a_randompar_models} compares the fit of the fixed M1, of the single random coefficient with the "
        "lowest BIC, of the joint model and of the final model, and gives the standard "
        "deviations and shifters of the last two." + reading + "\n\n"
        + longtable("Fit of the random-parameter models and their distribution "
                    "parameters.", "tab:a_randompar_models",
                    ">{\\raggedright\\arraybackslash}X rrrrrrr",
                    "Model & $\\log L$ & Par. & AIC & BIC & LR $\\chi^2$ & d.f. & $p$ \\\\",
                    mrows, size="\\footnotesize", repeat_head=False,
                    note=("Lower AIC and BIC mean a better penalized fit. The "
                          "likelihood-ratio tests compare each model with the fixed M1, "
                          "and $p$ uses the chi-bar-squared reference for standard "
                          "deviations on the boundary. A shifter is named by its random "
                          "coefficient and the context variable after the word by, and a "
                          "dash marks a parameter absent from the joint model.")))
    return section("app:randpar", text)


def randpar_candidates_block(n: dict) -> str:
    """The single random coefficients of every candidate, printed in the Supplementary
    material, where the candidate-selection text of Section S4 introduces it."""
    cand = req(n, "sensitivity", "random_parameters", "candidates")
    rows = []
    for name, c in cand.items():
        t = c["lr_vs_fixed"]
        rows.append(bold_if(tex(pretty(name)), c["retained"])
                    + f" & {fmt(c['mean'])} & {rnd(c['sd'], 3)} & {rnd(c['sd_se'], 3)} & "
                    f"{pfmt(c['sd_wald_p'])} & {rnd(t['lr_chi2'], 2)} & "
                    f"{pfmt(t['p_chi_bar_squared'])} & "
                    f"{'Yes' if c['retained'] else 'No'} \\\\")
    return longtable("Single random coefficients, one candidate at a time.",
                     "tab:a_randompar",
                     ">{\\raggedright\\arraybackslash}X rrrrrrc",
                     "Covariate & Mean & SD & SE of SD & Wald $p$ & LR $\\chi^2$ & "
                     "LR $p$ & Retained \\\\", rows, size="\\footnotesize",
                     note=("Mean and SD are the location and standard deviation of the "
                           "random coefficient, and the likelihood-ratio test compares the "
                           "model with one random coefficient against the fixed M1, with "
                           "its $p$ from the chi-bar-squared reference. Bold marks a "
                           "candidate that met both retention criteria."))


# ---------------------------------------------------------------- K transferability

def app_transfer(n: dict) -> str:
    tr = n["transferability"]
    det = req(n, "sensitivity", "supporting", "transferability_detail")
    cities = req(det, "large_city_definition", "cities_in_code")
    reps = req(det, "bootstrap", "replications")
    ua_pop = req(n, "sensitivity", "data_extra", "spatial_indicator",
                 "urbanized_area_indicator", "ua_population_threshold")
    L = letter("app:transfer")

    rows = []
    for year, e in sorted(tr["temporal"].items()):
        rows.append(f"Temporal, {year} break & {e['n_before']} & {e['n_after']} & "
                    f"{e['covariates_tested']} & {fmt(e['lr'], 2)} & {e['df']} & "
                    f"{pfmt(e['p_asymptotic'])} & {pfmt(e['p_bootstrap'])} \\\\")
    sp = tr["spatial"]
    rows.append(f"Spatial, {words(len(cities))} major cities & {sp['n_large_city']} & "
                f"{sp['n_rest']} & {sp['covariates_tested']} & {rnd(sp['lr_statistic'], 2)} & "
                f"{sp['df']} & {pfmt(sp['p_asymptotic'])} & {pfmt(sp['p_bootstrap'])} \\\\")

    srows = []
    side_names = {"before": "before", "after": "after", "large_city": "large cities",
                  "rest": "rest"}
    for key, split in det["splits"].items():
        title = (f"{key.split('_')[1]} break" if key.startswith("temporal")
                 else "Spatial")
        for side, g in split["groups"].items():
            sev = " / ".join(str(req(g, "severity_counts", s)) for s in SEV3)
            label = f"{title}, {side_names.get(side, side)}"
            if not split.get("testable", True):
                srows.append(f"{label} & {g['n']} & {sev} & \\multicolumn{{6}}{{l}}"
                             "{Not fitted, an outcome class is empty} \\\\")
                continue
            sep = g["separation"]
            ser = g["se_ratio_subgroup_over_pooled"]
            srows.append(f"{label} & {g['n']} & {sev} & {g['estimable_parameters']} & "
                         f"{rnd(epp(g['severity_counts'], g['estimable_parameters'], g['events_per_parameter_smallest_class']), 2)} & "
                         f"{'Yes' if g['converged'] else 'No'} & "
                         f"{rnd(sep['max_abs_coefficient'], 2)} & {rnd(sep['max_standard_error'], 2)} & "
                         f"{rnd(ser['max'], 2)} ({rnd(ser['median'], 2)}) \\\\")

    text = (
        "This appendix gives the transferability tests in full and the detail of every "
        "subgroup fit behind them. A split is tested on the covariates estimable within "
        "both of its sides, which the screening decides per split, so each test has its "
        "own covariate set and its own degrees of freedom. The temporal breaks split the "
        f"sample before and after a year, and the spatial split separates the "
        f"{words(len(cities))} major cities, {joined([tex(c) for c in cities])}, from "
        "the rest. Both $p$-values come from the same statistic, the asymptotic one from "
        f"the $\\chi^2$ reference and the other from {reps} parametric bootstrap "
        f"replications under the pooled model, as Table~{L}.1 shows.\n\n"
        + longtable("Transferability tests, per split.", "tab:a_transfer",
                    ">{\\raggedright\\arraybackslash}X rrrrrrr",
                    "Split & $n$ first & $n$ second & Covariates & LR & d.f. & $p$ asym. & "
                    "$p$ boot. \\\\", rows,
                    note=("The first group is the earlier period or the large cities. The "
                          "covariate count is the number estimable within both sides of "
                          "the split."))
        + f"Table~{L}.2 reports each side of each split with its severity counts, the "
        "number of estimated parameters, the events per parameter in its smallest class "
        "and convergence. It also gives the largest coefficient and standard error from "
        "the separation screen, and the ratio of the subgroup standard errors to the "
        "pooled ones on the same covariates. " + supp("tab:a_spatialalt") + " gives two "
        "alternative spatial splits through the same bootstrap test, by a 2020 urban area "
        f"of {ua_pop:,} or more and by the rural flag.\n\n"
        + longtable("Subgroup fits behind each transferability test.", "tab:a_subgroups",
                    ">{\\raggedright\\arraybackslash}X r c r r c r r c",
                    "Split and side & $n$ & O / BC / KA & Par. & EPP & Conv. & "
                    "$|\\hat\\beta|_{\\max}$ & SE$_{\\max}$ & SE ratio max (median) \\\\",
                    srows, size="\\footnotesize",
                    note=("Par.\\ counts the estimated parameters and EPP is the number "
                          "of events per parameter in the smallest severity class. The SE "
                          "ratio divides the standard error of a slope in the subgroup fit "
                          "by that of the same slope in the pooled fit on the same "
                          "covariates, and gives the largest ratio with the median in "
                          "parentheses.")))
    return section("app:transfer", text)


# ---------------------------------------------------------------- L out-of-sample

MODEL_NAMES = {"random_forest_1": "Random forest 1", "random_forest_2": "Random forest 2",
               "gradient_boosting_1": "Gradient boosting 1",
               "gradient_boosting_2": "Gradient boosting 2",
               "ordinal_logit": "Ordered logit"}


def app_benchmark(n: dict) -> str:
    cvf = req(n, "sensitivity", "supporting", "cv_fold_level")
    b = n["benchmark"]
    bd = req(n, "sensitivity", "supporting", "benchmark_detail")
    L = letter("app:benchmark")

    rows = [panel("Panel A. Log loss per rung", 7)]
    for rung, e in cvf["rungs"].items():
        rows.append(f"{rung} & {rnd(e['mean_log_loss'], 4)} & {rnd(e['sd_across_folds'], 4)} & "
                    f"{rnd(e['min_log_loss'], 4)} & {rnd(e['max_log_loss'], 4)} & "
                    f"{e['n_covariates']} & {e['n_folds_failed']} \\\\")
    rows.append(panel("Panel B. Paired differences between rungs", 7))
    rows.append(inner_header("Difference & Mean & SD & Min & Max & Folds first lower & "
                             "Repeats first lower \\\\"))
    n_folds = cvf["repeats"] * cvf["folds"]
    for key, e in cvf["paired_differences"].items():
        a, c = key.split("_minus_")
        rows.append(f"{a} minus {c} & {fmt(e['mean'], 4)} & {rnd(e['sd'], 4)} & "
                    f"{fmt(e['min'], 5)} & {fmt(e['max'], 4)} & "
                    f"{e['n_folds_first_better']} of {len(e['per_fold'])} & "
                    f"{e['n_repeats_first_better']} of {len(e['per_repeat_mean'])} \\\\")
    if any(len(e["per_fold"]) != n_folds for e in cvf["paired_differences"].values()):
        raise SystemExit("L: paired differences do not cover every fold")

    grid = req(bd, "grid")

    def config(name: str) -> str:
        if name == "ordinal_logit":
            return "M1 specification"
        family, idx = name.rsplit("_", 1)
        g = grid[family][int(idx) - 1]
        if family == "random_forest":
            depth = "no depth limit" if g["max_depth"] is None else f"depth {g['max_depth']}"
            return f"{g['n_estimators']} trees, {depth}, leaf {g['min_samples_leaf']}"
        return (f"{g['n_estimators']} trees, rate {g['learning_rate']:g}, depth "
                f"{g['max_depth']}")

    summ = req(bd, "summary_metrics_mean_sd_over_50_folds")
    metrics = (("balanced_accuracy", True), ("macro_f1", True), ("pr_auc_KA", True),
               ("brier_KA", False))
    best = {}
    for m, higher in metrics:
        places = 4 if m == "brier_KA" else 3
        vals = {k: rval(v[m]["mean"], places) for k, v in summ.items()}
        best[m] = max(vals.values()) if higher else min(vals.values())
    brows = []
    for name, v in summ.items():
        if rval(b["models"][name]["macro_f1"]["mean"], 4) != rval(v["macro_f1"]["mean"], 4):
            raise SystemExit(f"L: benchmark summary for {name} differs from the stored one")
        cells = []
        for m, _ in metrics:
            places = 4 if m == "brier_KA" else 3
            text_ = f"{v[m]['mean']:.{places}f} ({v[m]['sd']:.{places}f})"
            cells.append(bold_if(text_, rval(v[m]["mean"], places) == best[m]))
        brows.append(f"{MODEL_NAMES.get(name, tex(name))} & {config(name)} & "
                     + " & ".join(cells) + " \\\\")
    base = req(bd, "baseline")
    brows.append(f"\\midrule\nAlways predicting BC & Baseline & "
                 f"{rnd(base['balanced_accuracy'], 3)} & -- & -- & -- \\\\")

    res = req(bd, "resampling")
    inputs = req(bd, "inputs_to_models")

    text = (
        "This appendix reports the out-of-sample evaluation in detail. The log loss of "
        f"the specification ladder uses {cvf['repeats']} repeats of {cvf['folds']}-fold "
        f"stratified cross-validation with seed {cvf['seed']}. The folds depend only on "
        "the outcome and the seed, so every rung is scored on the same held-out riders and "
        f"the differences between rungs are paired by fold. Table~{L}.1 summarizes the "
        "fold-level log loss of each rung and the paired differences, and its first panel "
        "reproduces the means of Table~\\ref{tab:ladder}.\n\n"
        + longtable("Fold-level cross-validated log loss of the specification ladder.",
                    "tab:a_cvfolds", ">{\\raggedright\\arraybackslash}X rrrrrr",
                    "Rung & Mean & SD & Min & Max & Covariates & Failed folds \\\\",
                    rows, repeat_head=False,
                    note=("Log loss is the mean negative log-likelihood per rider of the "
                          "held-out fold, and lower is better. A difference is the log "
                          "loss of the first rung minus that of the second on the same "
                          "fold, so a negative value favors the first. The last two "
                          "columns count the folds and the repeats in which the first "
                          f"rung had the lower loss. The {n_folds} folds overlap across "
                          "repeats, so the standard deviation of the differences describes "
                          "their spread and is not a standard error."))
        + "The predictive benchmark compares four machine-learning configurations with "
        f"the ordered logit on the {inputs['n_covariates']} covariates of M1, over "
        f"{res['repeats']} repeats of {res['folds']}-fold stratified cross-validation with "
        f"its own seed, {res['seed']}, so its folds differ from those of Table~{L}.1. Each "
        "point of the hyperparameter grid is fitted as its own model on every fold and all "
        "are reported, so no setting is chosen on the evaluation folds. The cue covariates "
        "enter as the binary final labels, no probability is calibrated in the benchmark, "
        "and the classes are not reweighted. Table~"
        f"{L}.2 gives the summary metrics, and " + supp("tab:a_perclass") + " gives the "
        "precision, recall and $F_1$ of each class.\n\n"
        + longtable("Out-of-sample performance of each benchmark configuration.",
                    "tab:a_benchmark",
                    ">{\\raggedright\\arraybackslash}p{2.6cm}>{\\raggedright\\arraybackslash}X"
                    " cccc",
                    "Model & Configuration & Balanced accuracy & Macro $F_1$ & PR-AUC (KA) & "
                    "Brier (KA) \\\\", brows, size="\\footnotesize",
                    note=(f"Cells give the mean over the {n_folds} folds with the standard "
                          "deviation in parentheses. PR-AUC and the Brier score refer to "
                          "the KA class, and a lower Brier score is better. Leaf is the "
                          "minimum number of riders in a leaf. Bold marks the best value in "
                          "each column.")))
    return section("app:benchmark", text)


def perclass_block(n: dict) -> str:
    """The per-class benchmark metrics, printed in the Supplementary material."""
    bd = req(n, "sensitivity", "supporting", "benchmark_detail")
    res = req(bd, "resampling")
    per = req(bd, "per_class_metrics_mean_sd_over_50_folds")
    prow = []
    for name, v in per.items():
        prow.append(panel(MODEL_NAMES.get(name, tex(name)), 4))
        for s in SEV3:
            prow.append(f"{s} & " + " & ".join(
                f"{rnd(v[s][m]['mean'], 3)} ({rnd(v[s][m]['sd'], 3)})"
                for m in ("precision", "recall", "f1")) + " \\\\")
    never = [MODEL_NAMES.get(m, m).lower() for m in req(bd, "models_that_never_predict_KA")]
    text = (
        "The predictive benchmark of the manuscript compares four machine-learning "
        "configurations with the ordered logit over "
        f"{res['repeats']} repeats of {res['folds']}-fold stratified cross-validation, and "
        f"Table~{letter('app:benchmark')}.2 of the manuscript gives its summary metrics. "
        "Table~\\ref{tab:a_perclass} breaks the comparison down by severity class, with the "
        "precision, recall and $F_1$ of each configuration for O, BC and KA.\n\n")
    return text + longtable(
        "Precision, recall and $F_1$ per severity class for each benchmark configuration.",
        "tab:a_perclass",
        ">{\\raggedright\\arraybackslash}X"
        "*{3}{>{\\centering\\arraybackslash}p{3.4cm}}",
        "Class & Precision & Recall & $F_1$ \\\\", prow,
        size="\\footnotesize",
        note=("Cells give the mean over the folds with the standard deviation "
              f"in parentheses. Both {joined(never)} never predict KA on any "
              "fold, so their KA precision, recall and $F_1$ are zero."
              if never else "Cells give the mean over the folds with the "
              "standard deviation in parentheses."))


# ---------------------------------------------------------------- assembly

# ---------------------------------------------------------------- added tables
#
# The tables below extend appendices B, E and H, each appended after the tables an
# appendix already holds. ppo_coef_block, actor_block and spatial_alt_block are printed in
# the Supplementary material instead, by build_supplementary.py.

KA_KEY = ("poverty_share", "coord_missing", "age_unknown")
EXTRA_NAMES = {"eth_other": "Rider ethnicity other", "eth_unknown": "Ethnicity not recorded",
               "urban_area_200k": "Inside an urban area of 200,000 or more"}


def nice(name: str) -> str:
    return tex(EXTRA_NAMES.get(name, pretty(name)))


def mark(flag: bool) -> str:
    return "$\\bullet$" if flag else "$\\circ$"


def ame_cell(e: dict, places: int = 3) -> str:
    lo, hi = e["ci95"]
    point = e["ame"] if "ame" in e else e["estimate"]
    return bold_if(f"{fmt(point, places)} {ci(lo, hi, places)}", e["excludes_zero"])


def ame_stack(e: dict, places: int = 3) -> str:
    """ame_cell with the interval on its own line, for narrow columns."""
    lo, hi = e["ci95"]
    point = e["ame"] if "ame" in e else e["estimate"]
    return bold_if(f"{fmt(point, places)}\\newline {ci(lo, hi, places)}", e["excludes_zero"])


def best_flags(values: list[float], places: int) -> list[bool]:
    """True where a value is the lowest at its printed precision, ties included."""
    shown = [rval(v, places) for v in values]
    low = min(shown)
    return [s == low for s in shown]


SHORT = {"poverty_share": "poverty share", "coord_missing": "absent coordinates",
         "age_unknown": "unrecorded age", "gender_unknown": "unrecorded gender",
         "eth_unknown": "unrecorded ethnicity", "age_under25": "rider under 25",
         "eth_black": "rider Black", "eth_hispanic": "rider Hispanic"}


SHORT_ROW = {"poverty_share": "Poverty share", "coord_missing": "Coordinates absent",
             "age_unknown": "Rider age not recorded",
             "sidewalk_transition": "Cue, sidewalk transition",
             "vehicle_turning_across": "Cue, turning across"}


def listed(names: list[str]) -> str:
    return (joined([SHORT.get(c, nice(c).lower()) for c in names]) if names
            else "none")


FAMILY = (
    ("ordered_logit_M1", None, "Ordered logit, M1"),
    ("ordered_probit", None, "Ordered probit"),
    ("adjacent_category_logit", None, "Adjacent-category logit"),
    ("partial_proportional_odds_brant_noncrossing", "partial_proportional_odds_brant",
     "Partial proportional odds, Brant set"),
    ("partial_proportional_odds_holm_noncrossing", "partial_proportional_odds_holm",
     "Partial proportional odds, Holm set"),
    ("generalized_ordered_logit_noncrossing", "generalized_ordered_logit",
     "Generalized ordered logit"),
)


def noncrossing_check(models: dict) -> dict:
    """What the noncrossing refits change, computed rather than asserted.

    The note under the ordinal-family table says the refits change the log-likelihood by
    a stated amount and change no selection decision. Both halves are checked here: the
    sign of every BIC difference against the ordered logit, and the model chosen by AIC,
    BIC and cross-validated log loss with the unconstrained fits in place of the refits.
    A failed check stops the build.
    """
    ol = models["ordered_logit_M1"]["fit"]
    pairs = [(nc, un) for nc, un, _ in FAMILY if un]
    gaps = [req(models, nc, "loglik_gap_to_unconstrained") for nc, _ in pairs]
    neg = [req(models, un, "crossing", "rows_with_negative_middle_probability_at_estimate")
           for _, un in pairs]
    for nc, un in pairs:
        a = models[un]["fit"]["bic"] - ol["bic"]
        b = models[nc]["fit"]["bic"] - ol["bic"]
        if (a < 0) != (b < 0):
            raise SystemExit(f"noncrossing refit of {un} reverses its BIC comparison")

    def winner(use_unconstrained: bool, metric: str) -> str:
        best, val = None, None
        for nc, un, _ in FAMILY:
            key = un if (un and use_unconstrained) else nc
            m = models[key]
            v = (m["cross_validated_log_loss"]["mean_log_loss"] if metric == "cv"
                 else m["fit"][metric])
            if val is None or v < val:
                best, val = nc, v
        return best

    for metric in ("aic", "bic", "cv"):
        if winner(True, metric) != winner(False, metric):
            raise SystemExit(f"noncrossing refits change the {metric} selection")
    return {"max_gap": max(gaps), "neg_min": min(neg), "neg_max": max(neg)}


def ordinal_family_block(n: dict) -> str:
    oa = req(n, "sensitivity", "ordinal_alternatives")
    fam = req(oa, "ordinal_family")
    models = fam["models"]
    design = fam["design"]
    L = letter("app:ordinal")
    chk = noncrossing_check(models)
    n_ppo = sum(1 for k, _, _ in FAMILY if k.startswith("partial_proportional"))

    fits = [req(models, k, "fit") for k, _, _ in FAMILY]
    cvs = [req(models, k, "cross_validated_log_loss", "mean_log_loss") for k, _, _ in FAMILY]
    b_aic = best_flags([f["aic"] for f in fits], 1)
    b_bic = best_flags([f["bic"] for f in fits], 1)
    b_cv = best_flags(cvs, 3)
    rows = []
    for i, (key, _, title) in enumerate(FAMILY):
        f, ka = fits[i], req(models, key, "ka_ames")
        cue = any(req(ka, c, "excludes_zero") for c in MECH5)
        aic_s, bic_s, cv_s = f"{rnd(f['aic'], 1)}", f"{rnd(f['bic'], 1)}", f"{rnd(cvs[i], 3)}"
        rows.append(f"{title} & {fmt(f['loglik'], 2)} & {f['n_parameters']} & "
                    f"{bold_if(aic_s, b_aic[i])} & {bold_if(bic_s, b_bic[i])} & "
                    f"{bold_if(cv_s, b_cv[i])} & "
                    + " & ".join(mark(req(ka, c, "excludes_zero")) for c in KA_KEY)
                    + f" & {mark(cue)} \\\\")

    seed = req(fam, "primary_kr_seed_check")
    n_seed_ok = sum(1 for v in seed["ka_excludes_zero_by_seed"]["coord_missing"].values() if v)
    n_seeds = len(seed["seeds"])
    folds = req(models, "ordered_logit_M1", "cross_validated_log_loss")
    text = (
        f"Table~\\ref{{tab:a_ordinalfamily}} compares the ordered logit with "
        f"{words(len(FAMILY) - 1)} other ordinal models on the same "
        f"{len(design['covariates'])} covariates and {design['n']} riders, namely the "
        f"ordered probit, the adjacent-category logit, {words(n_ppo)} partial "
        "proportional-odds models freeing the uncorrected and the Holm sets, and the "
        "generalized ordered logit. For each it gives the fit, the log loss over the same "
        f"{folds['repeats']} repeats of {folds['folds']}-fold cross-validation, and whether "
        "the KA interval of poverty share, absent coordinates, unrecorded age or any of "
        f"the {words(len(MECH5))} cues excludes zero. " + supp("tab:a_ppocoef")
        + " gives the threshold-specific coefficients of the partial proportional-odds "
        "models.\n\n"
        + longtable("Fit and KA conclusions across the ordinal family.",
                    "tab:a_ordinalfamily",
                    ">{\\raggedright\\arraybackslash}X r r r r r c c c c",
                    "Model & LL & Par. & AIC & BIC & CV log loss & Pov. & Coord. & Age & "
                    "Cue \\\\", rows, size="\\footnotesize", tabcolsep="4pt", need=24,
                    note=(
                        "LL is the log-likelihood and Par.\\ the number of parameters, and "
                        f"the log loss is the mean over the {folds['n_folds_total']} folds. "
                        "Bold marks the lowest AIC, BIC and log loss. A filled circle marks "
                        "a KA interval that excludes zero and an open circle one that "
                        "includes it, for poverty share (Pov.), absent coordinates "
                        "(Coord.), unrecorded age (Age) and any of the five cues (Cue). "
                        "The unconstrained partial proportional-odds and generalized fits "
                        f"gave a negative middle-class probability on {chk['neg_min']} to "
                        f"{chk['neg_max']} riders. The rows shown are refits that keep "
                        "every probability valid, which changed the log-likelihood by at "
                        f"most {rnd(chk['max_gap'], 1)} and changed no selection decision. The "
                        "ordered logit interval for absent coordinates excludes zero under "
                        f"{words(n_seed_ok)} of {words(n_seeds)} Krinsky and Robb seeds."))
    )
    return text + shrinkage_block(n)


def ppo_coef_block(n: dict) -> str:
    """The partial proportional-odds coefficients, printed in the Supplementary material."""
    models = req(n, "sensitivity", "ordinal_alternatives", "ordinal_family", "models")
    sets =(("partial_proportional_odds_brant_noncrossing", "partial_proportional_odds_brant",
             "Panel A. Brant set, without correction"),
            ("partial_proportional_odds_holm_noncrossing", "partial_proportional_odds_holm",
             "Panel B. Holm set"))
    rows = []
    for nc, un, title in sets:
        m = models[nc]
        rows.append(panel(f"{title}, {m['n_freed']} covariates freed", 6))
        for cov in m["freed_covariates"]:
            t = req(m, "threshold_specific_coefficients", cov)
            c1, c2 = t["cut1_BC_or_KA_vs_O"], t["cut2_KA_vs_O_or_BC"]
            olc = req(models, un, "threshold_specific_coefficients", cov,
                      "ordered_logit_M1_coef")
            rows.append(f"{tex(pretty(cov))} & "
                        f"{bold_if(fmt(c1['coef']), c1['excludes_zero'])} & {rnd(c1['se'], 3)} & "
                        f"{bold_if(fmt(c2['coef']), c2['excludes_zero'])} & {rnd(c2['se'], 3)} & "
                        f"{fmt(olc)} \\\\")
    holm = models["partial_proportional_odds_holm_noncrossing"]
    # The marginal effects of unrecorded age under the Holm-set refit, beside M1, because
    # the Results read its KA effect from this model.
    m1_age = req(n, "severity", "ames_all", "age_unknown")
    rows += [panel("Panel C. Average marginal effect of unrecorded rider age with 95\\% "
                   "interval", 6),
             inner_header("Outcome & \\multicolumn{3}{c}{Partial proportional odds, Holm set} "
                          "& \\multicolumn{2}{c}{Ordered logit, M1} \\\\")]
    for s in SEV3:
        rows.append(f"P({s}) & \\multicolumn{{3}}{{c}}"
                    f"{{{ame_cell(req(holm, 'ames', 'age_unknown', s))}}} & "
                    f"\\multicolumn{{2}}{{c}}{{{ame_cell(req(m1_age, s))}}} \\\\")
    text = (
        "A partial proportional-odds model gives each freed covariate one coefficient per "
        "threshold. The first contrasts the BC and KA classes together against O, and the "
        "second contrasts KA against O and BC together. Two coefficients of opposite sign "
        "mean that the covariate moves probability toward both ends of the ordering, a "
        "pattern the ordered logit has to average into one slope. "
        f"Table~\\ref{{tab:a_ppocoef}} gives both coefficients and their standard errors for the "
        f"{models['partial_proportional_odds_brant_noncrossing']['n_freed']} covariates of "
        f"the Brant set and the {holm['n_freed']} of the Holm set, each from the refit "
        "that keeps every probability valid, beside the single ordered logit "
        "coefficient. Panel~C gives the average marginal effects of unrecorded rider age "
        "on the three outcomes under the Holm-set refit and under M1.\n\n")
    return text + longtable(
        "Threshold-specific coefficients of the freed covariates.", "tab:a_ppocoef",
        ">{\\raggedright\\arraybackslash}X r r r r r",
        "Covariate & BC or KA vs O & SE & KA vs O or BC & SE & Ordered logit \\\\",
        rows, size="\\footnotesize", repeat_head=True, need=16,
        note=("Coefficients are on the log-odds scale of the more severe side, and bold "
              "marks a coefficient or marginal effect whose 95\\% interval excludes zero. "
              "Standard errors come "
              "from the inverse Hessian at the constrained optimum, so for riders on the "
              "boundary of the constraint they are approximate. The marginal effects in "
              "Panel~C are discrete changes in the probability of each outcome, with "
              "Krinsky and Robb intervals."))


def shrinkage_block(n: dict) -> str:
    oa = req(n, "sensitivity", "ordinal_alternatives")
    sh = req(oa, "shrinkage")
    fam = req(oa, "ordinal_family", "models")
    firth = sh["firth"]
    braw = req(sh, "bayesian", "raw_coding")
    b2sd = req(sh, "bayesian", "continuous_per_2sd")
    rt = req(oa, "random_thresholds")
    sep = req(oa, "separation")
    st = req(oa, "settings", "bayes")
    L = letter("app:ordinal")
    cols = (("M1", req(fam, "ordered_logit_M1", "ka_ames")), ("Firth", firth["ka_ames"]),
            ("raw", braw["ka_ames"]), ("2sd", b2sd["ka_ames"]))

    rows = [panel("Panel A. Average marginal effect on KA with 95\\% interval", 5)]
    for c in KA_KEY + MECH5:
        rows.append(f"{tex(SHORT_ROW.get(c, pretty(c)))} & " + " & ".join(
            ame_stack(req(a, c)) if AS_FLOAT else ame_cell(req(a, c)) for _, a in cols)
                    + " \\\\")
    rows += [panel("Panel B. Fit and diagnostics", 5),
             f"Log-likelihood & {fmt(req(fam, 'ordered_logit_M1', 'fit', 'loglik'), 2)} & "
             f"{fmt(firth['fit']['loglik'], 2)} & {fmt(braw['loglik_at_posterior_mean'], 2)} & "
             f"{fmt(b2sd['loglik_at_posterior_mean'], 2)} \\\\",
             f"Largest absolute slope & {rnd(firth['max_abs_coefficient_M1'], 3)} & "
             f"{rnd(firth['max_abs_coefficient'], 3)} & {rnd(braw['max_abs_posterior_mean'], 3)} & "
             f"{rnd(b2sd['max_abs_posterior_mean'], 3)} \\\\",
             f"Mean absolute slope, ratio to M1 & -- & "
             f"{rnd(firth['mean_abs_shrinkage_of_slopes'], 3)} & -- & -- \\\\",
             f"Converged or healthy & Yes & {'Yes' if firth['convergence']['converged'] else 'No'}"
             f" & {'Yes' if braw['diagnostics']['healthy'] else 'No'} & "
             f"{'Yes' if b2sd['diagnostics']['healthy'] else 'No'} \\\\",
             f"Largest $\\hat R$ & -- & -- & {rnd(braw['diagnostics']['max_r_hat'], 3)} & "
             f"{rnd(b2sd['diagnostics']['max_r_hat'], 3)} \\\\",
             f"Smallest bulk ESS & -- & -- & "
             f"{braw['diagnostics']['min_ess_bulk']} & {b2sd['diagnostics']['min_ess_bulk']} \\\\",
             f"Divergent transitions & -- & -- & {braw['diagnostics']['divergences']} & "
             f"{b2sd['diagnostics']['divergences']} \\\\",
             panel("Panel C. Random thresholds", 5),
             inner_header("Variant & SD (SE) & $\\bar\\chi^2$ $p$ & $\\Delta$BIC & "
                          "Converged \\\\")]
    # A variant that did not converge has no estimates to report, so its row says so
    # instead of printing values from the point where the optimizer stopped.
    rt_variants = (("increment", "Gap between thresholds", ("sd_log_increment",)),
                   ("both", "Both thresholds and the gap",
                    ("sd_shift_both_thresholds", "sd_log_increment")))
    failed_rt = []
    for key, label, sd_keys in rt_variants:
        v = req(rt, "variants", key)
        if not v["convergence"]["converged"]:
            failed_rt.append((label, v["convergence"]))
            rows.append(f"{label} & \\multicolumn{{3}}{{c}}{{Did not converge}} & No \\\\")
            continue
        sds = v["random_sds"]
        rows.append(f"{label} & "
                    + ", ".join(f"{fmt(sds[k]['sd'])} ({rnd(sds[k]['se'], 3)})" for k in sd_keys)
                    + f" & {pfmt(v['lr_vs_fixed_M1']['p_chi_bar_squared'])} & "
                    f"{rnd(v['bic_change_vs_M1'], 1)} & Yes \\\\")
    rt_note = ""
    for label, conv in failed_rt:
        why = (", as its Hessian is not positive definite where the optimizer stopped"
               if not conv.get("hessian_positive_definite", True) else "")
        rt_note += (f" The variant with random terms on {label[0].lower() + label[1:]} did "
                    f"not converge{why}. Its estimates are therefore not reported.")
    rows += [panel("Panel D. Separation screen", 5),
             inner_header("Rung & Covariates & $|\\hat\\beta|_{\\max}$ & SE$_{\\max}$ & "
                          "Separation \\\\")]
    for rung in ("M0", "M0c", "M1"):
        s = sep[rung]
        found = s["suggests_separation_coef_or_se_above_10"] or \
            s["linear_programming_check"]["any_separation"]
        # The largest slope and standard error are read from the rung's estimates, the
        # values Table 4 prints, after checking that the screen found the same covariates.
        slopes = req(n, "severity", "rungs", rung, "coefficients", "slopes")
        big_b = max(slopes, key=lambda k: abs(slopes[k]["coef"]))
        big_se = max(slopes, key=lambda k: slopes[k]["se"])
        if (abs(abs(slopes[big_b]["coef"]) - s["max_abs_coefficient"]) > 6e-4
                or abs(slopes[big_se]["se"] - s["max_standard_error"]) > 6e-4):
            raise SystemExit(f"E: separation screen of {rung} does not match its estimates")
        rows.append(f"{rung} & {s['n_covariates']} & {rnd(abs(slopes[big_b]['coef']), 3)} & "
                    f"{rnd(slopes[big_se]['se'], 3)} & {'Found' if found else 'None'} \\\\")

    diff = sh["summary"]["differs_from_primary"]
    lost = sorted({c for v in diff.values() for c in v})
    changed = ("no KA interval changes under any shrinkage fit" if not lost else
               f"the KA intervals that change under a shrinkage fit are those for "
               f"{listed(lost)}")
    cue_hits = [c for v in sh["summary"]["ka_excludes_zero_by_model"].values()
                for c in v if c in MECH5]
    if cue_hits:
        raise SystemExit(f"Shrinkage prose: a cue KA interval excludes zero ({cue_hits})")
    text = (
        "Three fits check whether the estimates depend on the small O and KA classes. "
        "The Firth fit penalizes the likelihood to remove its first-order bias, and two "
        "Bayesian fits place a normal(0, 1) prior on every slope, in the coding of the model "
        "frame and with the continuous covariates per two standard deviations "
        f"({st['chains']} chains of {st['draws']:,} draws after {st['tune']:,} tuning steps). "
        "Panel~A of Table~\\ref{tab:a_shrinkage} gives the KA effects under each fit and "
        "Panel~B the diagnostics. Panel~C adds a normal random term to the thresholds, "
        "tested against the chi-bar-squared reference, and Panel~D gives the largest "
        "coefficient and standard error of each rung with a linear-programming separation "
        f"check. Against M1, {changed}, and no admitted cue interval excludes zero in any "
        "of these fits.\n\n")
    return text + longtable(
        "Shrinkage fits, random thresholds and the separation screen.", "tab:a_shrinkage",
        ">{\\raggedright\\arraybackslash}p{2.9cm} *{4}{>{\\centering\\arraybackslash}X}",
        "Quantity & M1 & Firth & Bayes, raw & Bayes, 2 SD \\\\",
        rows, size="\\footnotesize", repeat_head=False, tabcolsep="2.5pt",
        note=("Bold marks a KA effect whose 95\\% interval excludes zero. The Bayesian "
              "intervals are posterior percentiles and the point value is the posterior "
              "mean of the effect. In Panel~C the SD is that of the random term, $p$ uses "
              "the chi-bar-squared reference, and $\\Delta$BIC is the change against M1."
              + rt_note + " The ratio to M1 divides the mean absolute slope of the Firth fit by that of M1, "
              "so a value below one means the penalty shrinks the slopes on average."
              " ESS is the effective sample size of the posterior draws. The Firth log-likelihood is evaluated "
              "without its penalty, and the "
              "Bayesian one at the posterior mean. Separation is flagged when a "
              "coefficient or standard error exceeds 10 or the linear program finds a "
              "separating direction."))


def power_block(n: dict) -> str:
    pw = req(n, "sensitivity", "ordinal_alternatives", "cue_block_power")
    obs = pw["observed"]
    aa = req(n, "sensitivity", "data_extra", "actor_attribution")
    rule = req(aa, "event_rule_min_cell")
    # The pointer sentence says the split fails the rule, which holds only while at least
    # one split indicator falls short of it.
    if all(s["meets_event_rule"] for c in aa["cues"].values()
           for s in c["split_indicators"].values()):
        raise SystemExit("H: every actor split indicator meets the event rule")
    rows = []
    for b in pw["betas"]:
        g = req(pw, "grid", str(float(b)))
        rows.append(f"{rnd(b, 2)} & {rnd(g['odds_ratio'], 2)} & {pct(g['power'])} & "
                    f"{pct(g['mc_se'])} & {rnd(g['median_lr'], 2)} & "
                    f"{fmt(g['mean_estimated_cue_coefficient'])} & {g['n_ok']} \\\\")
    rows.append(panel(f"Interpolated point of {pct(0.8, 0)}\\% power", 7))
    rows.append(f"{rnd(pw['beta_at_power_0.8_interpolated'], 3)} & "
                f"{rnd(pw['odds_ratio_at_power_0.8'], 2)} & {pct(0.8)} & -- & -- & -- & -- \\\\")
    text = (
        "To show what the null can rule out, outcomes were simulated from M0c with a common "
        f"log-odds coefficient $\\beta$ planted on the {len(pw['cues'])} admitted cues, in "
        f"{pw['n_replications_per_beta']} replications at each value. The "
        f"{obs['cue_block_lr']['df']}-degree-of-freedom test rejects in "
        f"{pct(pw['size_at_beta_0'])}\\% of replications at $\\beta = 0$ and reaches 80\\% "
        f"power at $\\beta$ of about {rnd(pw['beta_at_power_0.8_interpolated'], 2)}, an odds "
        f"ratio of {rnd(pw['odds_ratio_at_power_0.8'], 2)} per cue "
        "(Table~\\ref{tab:a_power}). " + supp("tab:a_actor") + " gives the party that "
        "performed the action for failure to yield and signal violation, a split that "
        f"fails the {rule}-event rule.\n\n")
    return text + longtable(
        "Power of the cue block test by planted coefficient.", "tab:a_power",
        "*{7}{>{\\centering\\arraybackslash}X}",
        "$\\beta$ & Odds ratio & Rejections (\\%) & MC SE (\\%) & Median LR & "
        "Mean $\\hat\\beta$ & Replications \\\\",
        rows, size="\\footnotesize", hlcolour="HlD", need=22,
        note=("Rejections give the percentage of replications in which the cue block test "
              f"rejects at {pct(pw['alpha'], 0)}\\%, which is the size at $\\beta = 0$ and "
              "the power elsewhere, and MC SE is its Monte Carlo standard error. Mean "
              "$\\hat\\beta$ averages the estimated cue coefficients over replications. The "
              "last row gives the coefficient and odds ratio at which power reaches "
              "80\\%, interpolated linearly between the bracketing values of $\\beta$."))


def excluded_block(n: dict) -> str:
    """H.3: the post hoc refit with the cues the event rule kept out of M1."""
    ex = req(n, "sensitivity", "excluded_cues")
    ol = req(ex, "ordered_logit")
    cues = req(ex, "excluded_cues_fitted")
    focus = req(ol, "focus_ame", "covariate")
    ka = req(ol, "focus_ame", "KA")
    four = req(ol, "lr_four_vs_M1")
    nine = req(ol, "lr_nine_cues_vs_M0c")
    cv = req(ex, "cv")
    mll, pair = cv["mean_log_loss"], cv["paired"]

    def or_cell(e):
        return f"{rnd(e['odds_ratio'], 2)} {ci(e['or_ci95'][0], e['or_ci95'][1], 2)}"

    rows = [panel("Panel A. The four excluded cues added to M1 (model MX)", 8)]
    for c in cues:
        k = req(ex, "counts", c, "events_by_severity")
        e = req(ol, "coefficients", c)
        rows.append(f"{cue_name(c)} & {k['O']} & {k['BC']} & {k['KA']} & {fmt(e['coef'])} & "
                    f"{or_cell(e)} & {pfmt(req(ol, 'single_cue_lr_vs_M1', c, 'p'))} & "
                    f"{pfmt(req(ol, 'single_cue_holm_p', c))} \\\\")
    rows.append(panel(f"Panel B. {cue_name(focus)} under alternative fits of MX", 8))
    nl, fi, ba = req(ex, "narrative_length"), req(ex, "firth"), req(ex, "bayes")
    nk = req(ex, "excluding_fatal")
    rows += [
        f"Ordered logit & -- & -- & -- & {fmt(ol['coefficients'][focus]['coef'])} & "
        f"{or_cell(ol['coefficients'][focus])} & "
        f"{pfmt(ol['single_cue_lr_vs_M1'][focus]['p'])} & -- \\\\",
        f"With narrative length & -- & -- & -- & {fmt(nl['coefficient']['coef'])} & "
        f"{or_cell(nl['coefficient'])} & {pfmt(nl['lr_focus']['p'])} & -- \\\\",
        f"Without fatal riders & {nk['focus_cells']['O']} & {nk['focus_cells']['BC']} & "
        f"{nk['focus_cells']['KA']} & {fmt(nk['coefficient']['coef'])} & "
        f"{or_cell(nk['coefficient'])} & {pfmt(nk['lr_focus']['p'])} & -- \\\\",
        f"Firth-type penalty & -- & -- & -- & {fmt(fi['coef'])} & {or_cell(fi)} & -- & -- \\\\",
        f"Normal(0, 1) prior & -- & -- & -- & {fmt(ba['posterior_mean'])} & {or_cell(ba)} & "
        "-- & -- \\\\",
    ]
    swerve = req(ex, "counts", "swerve_loss_control", "events_by_severity")
    text = (
        "The event rule of Appendix~\\ref{app:screening} keeps five cues out of M1. In a "
        "post hoc refit (MX), specified after the primary results and after the counts of "
        "each cue by severity class were seen, the "
        f"{words(len(cues))} excluded cues that more than two riders carry are added to M1, "
        f"and dooring ({ex['dooring_left_out']}) stays out. Against M0c, all nine cues "
        f"together give $\\chi^2 = {rnd(nine['lr_chi2'], 2)}$ on {nine['df']} d.f. "
        f"($p = {pfmt(nine['p']).strip('$')}$). {cue_name(focus)} of either "
        "party is the one cue that survives Holm adjustment, and its estimate holds with "
        "narrative length as a control, under both shrinkage fits, and without the "
        f"{words(nk['n_fatal_removed'])} fatal riders, the most fully investigated crashes, "
        f"{words(nk['fatal_with_focus_cue'])} of whom carry the cue (Panel~B of "
        "Table~\\ref{tab:a_excluded}). "
        f"{cue_name('swerve_loss_control')} rests on {words(swerve['KA'])} KA rider and is "
        "not interpreted. The distraction or impairment label is among the least reliable "
        "of the ten (Table~\\ref{tab:validation}), and the refit was not repeated under the "
        "alternative label sets of Table~\\ref{tab:a_labelvariants}. MX does not improve "
        "cross-validated log loss over M0c "
        f"({fmt(mll['MX'], 3)} against {fmt(mll['M0c'], 3)}).\n\n")
    return text + longtable(
        "The cues excluded by the event rule, in a post hoc refit.", "tab:a_excluded",
        ">{\\raggedright\\arraybackslash}X rrr r c r r",
        "Cue or fit & O & BC & KA & Coef. & Odds ratio [95\\% CI] & LR $p$ & Holm $p$ \\\\",
        rows, size="\\footnotesize", hlcolour="HlD", need=16, repeat_head=False,
        note=("O, BC and KA count the riders with the cue in each coded class. Coefficients "
              "are on the log-odds scale of the ordered logit, positive toward KA. LR $p$ "
              "tests the cue against M1 alone, or against MX without it for the fit with "
              "narrative length or without fatal riders, and Holm $p$ adjusts it across the four "
              f"cues of Panel A. The fit without fatal riders holds {nk['n_riders']} riders, and its "
              f"counts are those of the cue in each class after the {nk['n_fatal_removed']} K riders "
              "are removed. "
              "The Firth-type and Bayesian fits are those of Table~\\ref{tab:a_shrinkage} "
              "applied to MX; for the prior the coefficient is the posterior mean and the "
              "interval the 95\\% credible interval. The analysis was specified after the "
              "primary results and is reported as exploratory."))


ACTOR_LEVELS = (("rider", "Rider"), ("driver", "Driver"), ("both", "Both"),
                ("unclear", "Unclear"), ("not recorded", "Not recorded"))
SPLIT_NAMES = (("rider_attributed", "Rider or both"), ("driver_attributed", "Driver or both"),
               ("unclear_attributed", "Unclear or not recorded"))


def actor_block(n: dict) -> str:
    """The actor attribution of two cues, printed in the Supplementary material."""
    aa = req(n, "sensitivity", "data_extra", "actor_attribution")
    rule = aa["event_rule_min_cell"]
    rows = [panel("Panel A. Attribution of cue-positive riders by severity class", 6)]
    for cue, c in aa["cues"].items():
        xt = c["crosstab_actor_by_severity"]
        tot = c["cue_positive_rows"]
        for key, title in ACTOR_LEVELS:
            if key not in xt:
                continue
            k = xt[key]
            nk = sum(k[s] for s in SEV3)
            rows.append(f"{tex(cue_name(cue))}, {title.lower()} & "
                        + " & ".join(str(k[s]) for s in SEV3)
                        + f" & {nk} & {rnd(100 * nk / tot, 1)} \\\\")
    rows += [panel("Panel B. Split indicators under the 10-event rule", 6),
             inner_header("Indicator & O & BC & KA & Smallest cell & Meets rule \\\\")]
    for cue, c in aa["cues"].items():
        for key, title in SPLIT_NAMES:
            s = c["split_indicators"][key]
            e = s["events_by_severity"]
            rows.append(f"{tex(cue_name(cue))}, {title.lower()} & "
                        + " & ".join(str(e[x]) for x in SEV3)
                        + f" & {s['smallest_cell']} & {'Yes' if s['meets_event_rule'] else 'No'} \\\\")
    rows += [panel("Panel C. Agreement of the reference run with the coders", 6),
             inner_header("Cue and narratives & Compared & Agree & Agreement (\\%) & "
                          "Cohen's $\\kappa$ & \\\\")]
    samples = (("batch1_validation_sample", "validation sample", True),
               ("pooled", "validation and flagged narratives", True))
    for cue, c in aa["cues"].items():
        ag = c["validation_agreement_llm_vs_coders"]
        rows_cue = list(samples)
        if cue == "signal_violation":
            rows_cue.append(("batch2_flagged_adjudication", "flagged narratives only", False))
        for key, title, with_kappa in rows_cue:
            g = ag[key]
            kap = f"{rnd(g['cohen_kappa'], 3)}" if with_kappa else "--"
            rows.append(f"{tex(cue_name(cue))}, {title} & {g['n_both_attributed']} & "
                        f"{g['n_agree']} & {rnd(g['percent_agreement'], 1)} & {kap} & \\\\")

    fty = aa["cues"]["failure_to_yield"]["split_indicators"]
    sv = aa["cues"]["signal_violation"]
    flagged_levels = sorted(sv["validation_agreement_llm_vs_coders"]
                            ["batch2_flagged_adjudication"]["confusion_rows_coders_cols_llm"])
    unc = fty["unclear_attributed"]["events_by_severity"]
    unclear_words = (f"has fewer than {words(rule)} in every class"
                     if max(unc.values()) < rule
                     else f"has {fty['unclear_attributed']['smallest_cell']} in its smallest class")
    n_sv_fail = sum(1 for v in sv["split_indicators"].values() if not v["meets_event_rule"])
    text = (
        "The pre-specification records, for failure to yield and signal violation, which "
        f"party the narrative says performed the action. These {words(len(aa['cues']))} "
        "cues are the ones either "
        "party can perform. The attribution comes from the coders' majority vote for the "
        "human-coded narratives and from the reference run for the rest, and a rider "
        "enters only when the final cue is positive. The split into rider-attributed, "
        "driver-attributed and unclear indicators was to enter M1 only if every indicator "
        f"held at least {rule} riders in each severity class. Panel~B of "
        "Table~\\ref{tab:a_actor} shows "
        "that the gate fails. For failure to yield the driver-attributed indicator has "
        f"{fty['driver_attributed']['events_by_severity']['KA']} KA riders and the unclear "
        f"indicator {unclear_words}, and for signal violation "
        + ("all three indicators fall short" if n_sv_fail == 3
           else f"{words(n_sv_fail)} of the three indicators fall short")
        + ". The split therefore does "
        "not enter M1, and the attribution is reported here as a cross-tabulation only.\n\n"
        "Panel~C compares the attribution of the reference run with the coders' majority "
        "vote on narratives where both code the cue positive. A narrative was flagged for "
        "human coding when the two automated routes disagreed and the repeated runs did "
        "not settle the label, so the flagged narratives are not a probability sample. In "
        "that set the coders attributed every signal violation to "
        f"the {listed_plain(flagged_levels)}, which leaves Cohen's $\\kappa$ without "
        "meaning there, and only the agreement rate is shown for it.\n\n")
    return text + longtable(
        "Attribution of failure to yield and signal violation.", "tab:a_actor",
        ">{\\raggedright\\arraybackslash}X r r r r r",
        "Cue and attribution & O & BC & KA & Riders & Share (\\%) \\\\",
        rows, size="\\footnotesize", repeat_head=False,
        note=("Panel~A counts cue-positive riders by the attribution recorded with the "
              "final label, and the share is of all cue-positive riders for the cue. In "
              "Panel~B a rider coded both enters the rider and the driver indicator. Panel~C "
              "counts narratives where the coders' majority vote and the reference run both "
              "code the cue positive and the majority attribution is not tied, and its "
              "rates are unweighted."))


def listed_plain(levels: list[str]) -> str:
    return joined([tex(x) for x in levels]) + (" alone" if len(levels) == 1 else "")


def mi_block(n: dict) -> str:
    mi = req(n, "sensitivity", "data_extra", "multiple_imputation")
    ue = req(n, "sensitivity", "data_extra", "unknown_ethnicity")
    ur = req(n, "sensitivity", "urbanized")
    L = letter("app:missing")
    comp = mi["requested_ka_comparison"]
    st = mi["settings"]
    rows = [panel("Panel A. Multiple imputation, effect on KA with 95\\% interval", 4)]
    for c, e in comp.items():
        rows.append(f"{nice(c)} & {ame_cell(e['primary_ka'])} & {ame_cell(e['mi_ka'])} & "
                    f"{rnd(e['mi_ka']['fmi'], 3)} \\\\")
    for c in ("age_unknown",):
        if c in comp:
            continue
        rows.append(f"{nice(c)} & {ame_cell(req(n, 'severity', 'ames_all', c, 'KA'))} & "
                    "Not in the model & -- \\\\")

    def sev(d: dict) -> str:
        return " / ".join(str(d[s]) for s in SEV3)

    def sig(names: list[str]) -> str:
        return listed(names)

    uf, upf = ue["fit"], ue["primary_fit"]
    uc = ue["coefficients"]["eth_unknown"]
    ulr = ue["lr_split_vs_primary"]
    rows += [panel("Panel B. Unrecorded ethnicity as its own indicator", 4),
             inner_header("Quantity & M1 & Refit & \\\\"),
             f"Riders O / BC / KA, other or not recorded & "
             f"{sev(ue['counts_by_severity']['eth_other_or_unknown_primary'])} & -- & \\\\",
             f"Riders O / BC / KA, other & -- & {sev(ue['counts_by_severity']['eth_other'])} & \\\\",
             f"Riders O / BC / KA, not recorded & -- & "
             f"{sev(ue['counts_by_severity']['eth_unknown'])} & \\\\",
             f"Coefficient, ethnicity not recorded (SE) & -- & {fmt(uc['coef'])} "
             f"({rnd(uc['se'], 3)}) & \\\\",
             f"Log-likelihood & {fmt(upf['loglik'], 2)} & {fmt(uf['loglik'], 2)} & \\\\",
             f"BIC & {rnd(upf['bic'], 2)} & {rnd(uf['bic'], 2)} & \\\\",
             f"Likelihood ratio against M1 & -- & {rnd(ulr['lr_chi2'], 2)} ({ulr['df']} d.f., "
             f"$p$ = {pfmt(ulr['p'])}) & \\\\",
             f"KA intervals excluding zero & {sig(ue['excludes_zero']['primary']['KA'])} & "
             f"{sig(ue['excludes_zero']['refit']['KA'])} & \\\\"]

    wi, wo, kr = ur["primary_with_indicator"], ur["without_indicator"], ur["kr_seed_check"]
    sc = ur["screening"]
    ins = sc["event_rule"]["indicator_equal_1_by_severity"]
    outs = sc["event_rule"]["indicator_equal_0_by_severity"]
    urc, ulr2 = wi["coefficient"], wo["lr_indicator"]
    seeds = kr["comparison"]["coord_missing"]
    n_seeds = len(kr["seeds"])
    rows += [panel(f"Panel C. Inside an urban area of "
                   f"{ur['settings']['ua_population_threshold']:,} or more", 4),
             inner_header("Quantity & M1 & Without & \\\\"),
             f"Riders O / BC / KA, inside & {sev(ins)} & -- & \\\\",
             f"Riders O / BC / KA, outside & {sev(outs)} & -- & \\\\",
             f"VIF in the screening design & "
             f"{rnd(sc['vif_rule_pooled_screening_design']['vif_indicator'], 2)} & -- & \\\\",
             f"Coefficient (SE) & {fmt(urc['coef'])} ({rnd(urc['se'], 3)}) & -- & \\\\",
             f"Effect on KA & {ame_cell(wi['ame']['KA'])} & -- & \\\\",
             f"Log-likelihood & {fmt(wi['fit']['loglik'], 2)} & "
             f"{fmt(wo['fit']['loglik'], 2)} & \\\\",
             f"BIC & {rnd(wi['fit']['bic'], 2)} & {rnd(wo['fit']['bic'], 2)} & \\\\",
             f"Likelihood ratio of the indicator & {rnd(ulr2['lr_chi2'], 2)} ({ulr2['df']} d.f., "
             f"$p$ = {pfmt(ulr2['p'])}) & -- & \\\\",
             f"KA intervals excluding zero & {sig(wi['excludes_zero']['KA'])} & "
             f"{sig(wo['excludes_zero']['KA'])} & \\\\",
             f"Seeds with the absent-coordinates KA interval excluding zero & "
             f"{seeds['primary_excludes_zero_seeds']} of {n_seeds} & "
             f"{seeds['without_indicator_excludes_zero_seeds']} of {n_seeds} & \\\\"]

    ch = ue["conclusion_changes"]
    ua_pop = ur["settings"]["ua_population_threshold"]
    rm = wo["conclusion_changes_when_removed"]["KA"]
    ua_gain, ua_lost = rm["gained"], rm["lost"]
    if ua_gain == ["coord_missing"] and not ua_lost:
        nz = kr["changed_ka_bound_nearest_zero_by_seed"]["coord_missing"]
        ua_sentence = (
            "Without the urban-area indicator, the KA interval for absent coordinates "
            f"excludes zero under {words(seeds['without_indicator_excludes_zero_seeds'])} "
            f"of the {words(n_seeds)} Krinsky and Robb seeds, and in M1 under "
            f"{words(seeds['primary_excludes_zero_seeds'])} of them. "
            + ("The upper bounds under the two models do not overlap across the seeds, so "
               "the indicator moves that borderline interval by more than the seeds do."
               if not nz["seed_ranges_overlap"] else
               "The upper bounds under the two models overlap across the seeds, so the "
               "change is within the variation of that borderline interval."))
    elif not ua_lost and not ua_gain:
        ua_sentence = "Removing the urban-area indicator changes no KA conclusion."
    else:
        ua_sentence = ("Without the urban-area indicator, the KA intervals that change "
                       f"are those of {listed(sorted(ua_lost + ua_gain))}.")
    eth_sentence = (
        f"With unrecorded ethnicity separated, the KA interval of {listed(ch['KA']['lost'])} "
        "no longer excludes zero. " if ch["KA"]["lost"] else
        "Separating unrecorded ethnicity changes no KA conclusion. ")
    text = (
        "This table gathers three sensitivities of the primary model M1. The first "
        "replaces the explicit "
        "unknown levels by imputation, the second separates unrecorded ethnicity, and the "
        "third removes the urban-area indicator. The imputation fills rider "
        "age, gender, ethnicity and "
        f"posted speed limit by chained equations, with {st['m']} completed data sets of "
        f"{st['cycles_per_chain']} cycles each. Age and speed limit are imputed by "
        f"predictive mean matching with {st['pmm_donors']} donors, gender by a binary "
        "logit and ethnicity by a multinomial logit, and every imputation model includes "
        "severity. M1 is refitted on each completed data set without the unknown "
        "indicators, and the effects are pooled by Rubin's rules. "
        "Panel~A of Table~\\ref{tab:a_mi} gives the pooled KA effects beside the primary ones with "
        "the fraction of missing information. The imputation assumes the fields are "
        "missing at random given the predictors, so missingness that depends on the "
        "unrecorded value itself is not repaired.\n\n"
        "The second sensitivity separates unrecorded ethnicity from the recorded other "
        "category that M1 pools it with, although neither level meets the event rule on "
        "its own. The third concerns the pre-specified indicator for a crash inside a 2020 "
        f"urban area of {ua_pop:,} or more. It meets the event rule and the variance "
        "inflation rule and so enters M1, and a record without coordinates takes the "
        "value held by at least half of the city's located records. Panel~B reports the "
        "refit "
        "beside M1, and Panel~C reports M1 beside the same model without the indicator. "
        + eth_sentence + ua_sentence + "\n\n")
    return text + longtable(
        "Multiple imputation, unrecorded ethnicity and the urban-area indicator.",
        "tab:a_mi",
        ">{\\raggedright\\arraybackslash}X *{2}{>{\\centering\\arraybackslash}p{3.6cm}} r",
        "Covariate & M1 & Imputed & FMI \\\\",
        rows, size="\\footnotesize", repeat_head=False,
        note=("FMI is the fraction of missing information of the pooled KA effect. Bold "
              "marks an effect whose 95\\% interval excludes zero. The imputed fits carry no "
              "unrecorded "
              "level for age, gender or speed limit, so the unrecorded age row has no "
              "imputed value and the other two are not shown. The other ethnicity row of Panel~A is compared with the other or "
              "not recorded level of M1. The seed row counts the Krinsky and Robb seeds "
              f"out of {n_seeds} under which the KA interval for absent coordinates "
              "excludes zero."))


def spatial_alt_block(n: dict) -> str:
    """The rural and urban-area spatial splits, printed in the Supplementary material."""
    sp = req(n, "sensitivity", "data_extra", "spatial_indicator")
    L = letter("app:transfer")
    total = len(req(n, "sensitivity", "urbanized", "primary_with_indicator", "columns"))
    ua_pop = req(sp, "urbanized_area_indicator", "ua_population_threshold")
    tests = (("ua200k_test", "inside_ua200k", "outside_ua200k",
              f"Inside against outside an urban area of {ua_pop:,} or more"),
             ("rural_vs_nonrural_test", "rural", "non_rural", "Rural against non-rural"))
    rows = []
    for key, a, b, title in tests:
        t = sp[key]
        rows.append(f"{title} & {t['n_' + a]} & {t['n_' + b]} & {t['covariates_tested']} & "
                    f"{rnd(t['lr_statistic'], 2)} & {t['df']} & {pfmt(t['p_bootstrap'])} \\\\")
    names = {"six_major_cities": "Six major cities", "other_urban": "Other urban",
             "rural": "Rural"}
    rows += [panel("Panel B. Severity mix of three spatial groups", 7),
             inner_header("Group & $n$ & O (\\%) & BC (\\%) & KA (\\%) & & \\\\")]
    for g, e in sp["three_way"]["groups"].items():
        rows.append(f"{names[g]} & {e['n']} & " + " & ".join(
            f"{rnd(e['share_percent'][s], 1)}" for s in SEV3) + " & & \\\\")
    rows.insert(0, panel("Panel A. Alternative spatial splits", 7))
    kept = [sp[k]["covariates_tested"] for k, _, _, _ in tests]
    text = (
        "The six-city split was defined without a pre-specified rule, so two other spatial "
        "splits were run through the same bootstrap test. The first separates crashes "
        f"inside a 2020 Census urban area of {ua_pop:,} or more from the rest, and the "
        "second "
        "uses the rural flag of the crash record. Each split screens the covariates for "
        "estimability on both sides, as in Table~"
        f"{L}.1 of the manuscript, so few covariates survive when one side is small. Panel~A of "
        "Table~\\ref{tab:a_spatialalt} gives both tests, and Panel~B gives the severity mix of the six "
        "major cities, the other urban records and the rural records.\n\n")
    return text + longtable(
        "Alternative spatial splits and the severity mix by area.", "tab:a_spatialalt",
        ">{\\raggedright\\arraybackslash}X r r r r r r",
        "Split & $n$ first & $n$ second & Covariates & LR & d.f. & $p$ boot. \\\\",
        rows, size="\\footnotesize", repeat_head=False, need=22,
        note=(f"With {words(min(kept))} and {words(max(kept))} of {total} covariates "
              "estimable on both sides, these tests do not test the M1 specification, and "
              "they are reported as descriptive checks. In Panel~A the first group holds "
              "the records inside the urban area, or the rural records. The six major "
              "cities take precedence in "
              "Panel~B, and the rural group holds the remaining records with the rural "
              "flag set."))


BUILDERS = {
    "app:screening": app_screening,
    "app:missing": app_missing,
    "app:validation": app_validation,
    "app:auditdiag": app_auditdiag,
    "app:ordinal": app_ordinal,
    "app:fourclass": app_fourclass,
    "app:ames": app_ames,
    "app:labels": app_labels,
    "app:m2": app_m2,
    "app:randpar": app_randpar,
    "app:transfer": app_transfer,
    "app:benchmark": app_benchmark,
    "app:iia": app_iia,
}

_REF_TAB = re.compile(r"(Tables?)[~ ]([A-L])\.(\d+)((?:,\s*Panel~[A-D])?)(\s+of the manuscript)?")
_REF_APP = re.compile(r"Appendix~([A-L])(\s+of the manuscript)?")


def build_parts(n: dict) -> dict:
    """Appendix and supplement text, with every working table name renumbered."""
    global AS_FLOAT
    AS_FLOAT = True
    try:
        kept = {label: BUILDERS[label](n) for label in KEEP}
    finally:
        AS_FLOAT = False
    moved_order = []
    for label, _ in ORDER:
        if label in KEEP:
            if label == "app:ordinal":
                moved_order.append("app:iia")
            continue
        moved_order.append(label)
    moved = {label: BUILDERS[label](n) for label in moved_order}
    kept_numbers = {}
    for i, label in enumerate(KEEP):
        L = string.ascii_uppercase[i]
        for k, m in enumerate(re.finditer(r"\\label\{(tab:a_[a-z0-9_]+)\}", kept[label]), 1):
            kept_numbers[m.group(1)] = f"{L}.{k}"
        kept_numbers[label] = L
    moved_tables = [m.group(1) for label in moved_order
                    for m in re.finditer(r"\\label\{(tab:a_[a-z0-9_]+)\}", moved[label])]
    parts = {"kept": kept, "moved": moved, "moved_order": moved_order,
             "kept_numbers": kept_numbers, "moved_tables": moved_tables}
    parts["kept_text"] = refs_to_supplement(
        fix_refs("".join(kept.values()), "manuscript", parts), parts)
    parts["moved_text"] = "".join(moved[label] for label in moved_order)
    return parts


def fix_refs(text: str, context: str, parts: dict) -> str:
    """Print each working table or section name with its printed number."""
    in_supp = set(parts["moved_tables"]) | set(SUPP_MACROS)
    kept_numbers = parts["kept_numbers"]

    def tab(m: re.Match) -> str:
        word, key, panel_, man = m.group(1), f"{m.group(2)}.{m.group(3)}", m.group(4), m.group(5)
        label = OLD_TABLES[key]
        if label in in_supp:
            if context == "supp":
                return f"{word}~\\ref{{{label}}}{panel_}"
            return f"Supplementary {word}~\\{supp_macro(label)}{{}}{panel_}"
        if context == "supp":
            return f"{word}~{kept_numbers[label]}{panel_} of the manuscript"
        return f"{word}~\\ref{{{label}}}{panel_}"

    def app(m: re.Match) -> str:
        label = OLD_SECTIONS[m.group(1)]
        if label in KEEP:
            if context == "supp":
                return f"Appendix~{kept_numbers[label]} of the manuscript"
            return f"Appendix~\\ref{{{label}}}"
        if context == "supp":
            return f"Section~\\ref{{{label}}}"
        return f"Supplementary Section~\\{supp_macro(label)}{{}}"

    return _REF_APP.sub(app, _REF_TAB.sub(tab, text))


def moved_labels(parts: dict) -> set[str]:
    """Every label whose table or section is printed in the supplement."""
    return set(parts["moved_order"]) | set(parts["moved_tables"]) | set(SUPP_MACROS)


def refs_to_supplement(text: str, parts: dict) -> str:
    r"""Point each \ref to a moved table or section at its supplement number.

    A \ref cannot cross from the manuscript to the supplement, so "Table~\ref{tab:a_m2}"
    becomes "Supplementary Table~\suppTabMTwo{}", and "Appendix~\ref{app:m2}" becomes
    "Supplementary Section~..." with the macro written by build_supplementary.py.
    """
    return _refs_to_supp(text, moved_labels(parts))


def refs_to_supplement_static(text: str) -> str:
    """refs_to_supplement for text written outside this module, such as the main tables."""
    return _refs_to_supp(text, STATIC_MOVED)


def _refs_to_supp(text: str, moved: set[str]) -> str:
    words_ = {"Appendix": "Section", "Appendices": "Sections", "Table": "Table",
              "Tables": "Tables"}

    def prefixed(m: re.Match) -> str:
        if m.group(2) not in moved:
            return m.group(0)
        return f"Supplementary {words_[m.group(1)]}~\\{supp_macro(m.group(2))}{{}}"

    text = re.sub(r"(Appendix|Appendices|Tables|Table)~\\ref\{([^}]+)\}", prefixed, text)
    return re.sub(r"\\ref\{([^}]+)\}",
                  lambda m: (f"\\{supp_macro(m.group(1))}{{}}" if m.group(1) in moved
                             else m.group(0)), text)


def main() -> int:
    n = load_numbers()
    print(f"  reading {numbers_path()}")
    built = build_parts(n)
    parts = [built["kept_text"]]
    text = ("% GENERATED by src/build_appendix.py from data/numbers.json.\n"
            "% Do not edit by hand; rerun the generator.\n\n"
            "\\appendix\n"
            "% Lettered appendices. Every table, figure and equation carries its letter and\n"
            "% restarts its count, as in Table A.1. The hyperref names are kept distinct\n"
            "% from the main-text tables so the links do not collide.\n"
            "\\renewcommand{\\thesection}{\\Alph{section}}\n"
            "\\renewcommand{\\thetable}{\\thesection.\\arabic{table}}\n"
            "\\renewcommand{\\thefigure}{\\thesection.\\arabic{figure}}\n"
            "\\renewcommand{\\theequation}{\\thesection.\\arabic{equation}}\n"
            "\\renewcommand{\\theHtable}{app.\\thesection.\\arabic{table}}\n"
            "\\renewcommand{\\theHfigure}{app.\\thesection.\\arabic{figure}}\n"
            "\\renewcommand{\\theHequation}{app.\\thesection.\\arabic{equation}}\n"
            "\\setcounter{section}{0}\n"
            "% A table starts on a new page when fewer than eight lines remain, so that its\n"
            "% caption and header are never left alone at the foot of a page. needspace\n"
            "% settles the page before measuring, which a bare \\pagegoal test does not.\n"
            "\\providecommand{\\apptableneed}{\\needspace{8\\baselineskip}}\n\n"
            + "".join(parts))
    (OUT / "appendix.tex").write_text(text, encoding="utf-8")
    print(f"  wrote appendix.tex ({len(text.splitlines())} lines, "
          f"{text.count(chr(92) + 'section{')} appendices)")
    for i, (label, title) in enumerate(KEEP.items()):
        print(f"    {string.ascii_uppercase[i]}  {label:16s} {title}")
    print(f"  {len(built['moved_order'])} sections printed in the supplement: "
          + ", ".join(built["moved_order"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
