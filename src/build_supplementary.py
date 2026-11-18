"""Generate the supplementary material from the artifacts it documents.

The supplement has five sections:

    S1  the extraction protocol: the model, the prompts and their versions, the logging
        schema, the repeated runs, rerun variation and the alternative model
    S2  the masking list and the complete lexical rule set, with per-cue match counts
    S3  the coding guide given to the three coders, and the actor attribution
    S4  the random-parameter specifications, with every single-coefficient candidate
    S5  additional tables: data-only tables referred to from the manuscript

Tables are numbered S1, S2, ... through the whole supplement, without restarting in each
section. The seven data-only tables are built
by functions of build_appendix.py, and main() writes
`results/supp_labels.tex`, a macro per such table holding its printed number,
which the manuscript reads in its preamble. So no S number is typed by hand anywhere.

S1 and S2 are generated from the modules that do the work rather than transcribed from
them, because a transcribed protocol is a protocol that drifts. S1 reads the extraction
log, takes the label of each (crash, pass) from the latest parsed call, and counts the
prompt versions behind the labels the paper uses. It then recomputes the SHA-256 of the
current prompt text with each narrative and compares it with the logged hash, so the prompt
text printed here is shown to be the text that produced the labels.

S3 is the coding guide, which exists as `docs/coding_guide.md` and was written for
the coders rather than for this document; it is included verbatim, followed by the
agreement of the automated actor attribution with the coders' majority vote.

S4 is read from the random-parameter results in numbers.json. It gives the specification,
the candidate screen, the checks and the stability results; the model comparison is a table
of the manuscript appendix and is not repeated here.

Run with:  python src/build_supplementary.py
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "results"

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_appendix import (build_parts, fix_refs, supp_macro,  # noqa: E402
                            SEV3, SUPP_MACROS, actor_block, alt_macros, bold_if,  # noqa: E402
                            annual_block, ci, inner_header, letter, longtable, panel,
                            perclass_block, ppo_coef_block, randpar_candidates_block,
                            shifter_block, spatial_alt_block)
from build_tables import (cue_name, fmt, load_numbers, pct, pfmt, pretty, req, rnd,  # noqa: E402
                          rval, tex, words)


def verbatim(text: str) -> str:
    """Monospaced, escaped, and broken by line.

    A blank line cannot carry a "\\\\": TeX answers "There's no line here to end". Blank
    lines become paragraph breaks and only non-empty lines get an explicit break.
    """
    def escape(s: str) -> str:
        return (s.replace("\\", "\\textbackslash{}")
                 .replace("{", "\\{").replace("}", "\\}")
                 .replace("_", "\\_").replace("$", "\\$")
                 .replace("&", "\\&").replace("%", "\\%")
                 .replace("#", "\\#").replace("^", "\\^{}")
                 .replace("~", "\\textasciitilde{}")
                 # A long pattern has no spaces, so it is given break points after its
                 # alternation bars and closing groups rather than running into the margin.
                 .replace("|", "|\\allowbreak{}").replace(")", ")\\allowbreak{}"))

    out, pending_break = [], False
    for line in text.split("\n"):
        if not line.strip():
            if out:
                out.append("")
            pending_break = False
            continue
        if pending_break:
            out[-1] += " \\\\"
        out.append(escape(line))
        pending_break = True
    # A declared boundary, so the style checkers can tell a reproduced artifact from
    # prose written for this paper.
    return ("\\begin{reproduced}\n"
            + "\n".join(out) + "\n\\end{reproduced}\n\n")


def joined(items) -> str:
    items = list(items)
    if len(items) <= 1:
        return "".join(items)
    return ", ".join(items[:-1]) + " and " + items[-1]


def sci(x: float) -> str:
    """A small number in scientific notation, typeset."""
    text = f"{x:.2g}"
    if "e" not in text:
        return text
    mant, exp = text.split("e")
    return f"${mant} \\times 10^{{{int(exp)}}}$"


def short_version(v: str) -> str:
    return v.split("-")[0]


# ---------------------------------------------------------------- S1

def _latest_parsed(log: Path) -> dict[tuple[int, str], dict]:
    """The latest parsed call per (crash, pass), which is the label the paper uses."""
    latest: dict[tuple[int, str], dict] = {}
    with log.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if not rec.get("parsed"):
                continue
            key = (int(rec["crash_id"]), rec["pass"])
            if key not in latest or rec["timestamp_utc"] > latest[key]["timestamp_utc"]:
                latest[key] = rec
    return latest


def s1(n: dict) -> str:
    """The extraction protocol, read from the log that the run produced."""
    import extract  # noqa: PLC0415

    log = DATA / "logs" / "extract_reference.jsonl"
    if not log.exists():
        raise SystemExit("S1: no extraction log; the protocol cannot be documented")
    latest = _latest_parsed(log)
    if not latest:
        raise SystemExit("S1: the extraction log holds no parsed call")

    # Recompute each call's prompt hash from the current prompt text and the narrative the
    # pass received, exactly as extract._call builds it.
    frame = extract._narratives(None)
    texts = {int(r.Crash_ID): (r.masked, r.narrative) for r in frame.itertuples()}
    systems = {"mechanism": extract.MECHANISM_SYSTEM, "cue": extract.CUE_SYSTEM}
    versions: dict[str, Counter] = {"mechanism": Counter(), "cue": Counter()}
    match: dict[str, int] = {"mechanism": 0, "cue": 0}
    for (cid, pass_name), rec in latest.items():
        versions[pass_name][rec["prompt_version"]] += 1
        masked, raw = texts[cid]
        text = masked if pass_name == "mechanism" else raw
        user = f"NARRATIVE:\n{text}"
        sha = hashlib.sha256((systems[pass_name] + "\n\n" + user).encode("utf-8")).hexdigest()
        match[pass_name] += sha == rec["prompt_sha256"]
    totals = {p: sum(c.values()) for p, c in versions.items()}

    first = next(iter(latest.values()))
    fields = [
        ("Model", first.get("model")),
        ("Provider, pinned with fallbacks disabled", first.get("provider")),
        ("Quantization, as reported by the provider",
         re.sub(r"\s*\(.*\)\s*$", "", str(first.get("quantization")))),
        ("Temperature, reference run", first.get("temperature")),
        ("Seed, reference run", first.get("seed")),
        ("Response format", extract.RESPONSE_FORMAT),
        ("Maximum output tokens", extract.MAX_TOKENS),
        ("Current prompt version", extract.PROMPT_VERSION),
    ]
    rows = [f"{tex(str(k))} & \\texttt{{{tex(str(v))}}} \\\\" for k, v in fields
            if v is not None]
    logged = sorted(first.keys())

    def version_text(pass_name: str) -> str:
        c = versions[pass_name]
        if len(c) == 1:
            return f"\\texttt{{{short_version(next(iter(c)))}}}"
        parts = [f"\\texttt{{{short_version(v)}}} ({c[v]})" for v in sorted(c)]
        return joined(parts)

    mech_all_current = (set(versions["mechanism"]) == {extract.PROMPT_VERSION})
    hash_text = (
        f"For the cue pass the recomputed hash matches the logged hash for "
        f"{match['cue']} of {totals['cue']} calls, and for the mechanism pass it matches "
        f"for {match['mechanism']} of {totals['mechanism']} calls.")
    if match["cue"] == totals["cue"]:
        hash_text += (" The injury-cue prompt text is therefore the same under every "
                      "version string, and the version bumps changed the mechanism "
                      "definitions only.")

    mask = req(n, "sensitivity", "masking")
    rr = req(mask, "validation", "same_period_masked_rerun", "reference_reproduction")
    ti = req(mask, "agreement_all_narratives", "text_identical")
    alt = req(n, "sensitivity", "alt_model")
    st = alt["settings"]
    pq = req(alt, "provenance", "qwen")
    endpoints = req(alt, "provenance", "qwen_endpoint_listing", "endpoints")
    used = next(e for e in endpoints if e["provider"] == st["alternative_provider_pinned"])
    others = [e for e in endpoints if e["provider"] != st["alternative_provider_pinned"]]
    other_text = ""
    if others:
        o = others[0]
        other_text = (
            f" The other listed endpoint, {tex(o['provider'])}, served "
            f"\\texttt{{{tex(o['quantization'])}}} weights and "
            + ("did not support" if not o["supports_structured_outputs"] else "supported")
            + " structured outputs, so it was not used.")
    macro_llama, macro_qwen = alt_macros(alt)

    text = (
        "\\section{Extraction protocol}\\label{sup:protocol}\n\n"
        "Route~B labels each narrative with one language model, one provider and one "
        "narrative per request, and every request returns JSON that conforms to a strict "
        "schema. The settings below are read from the log of the reference run rather than "
        "from the configuration that was meant to be used, so they describe the run that "
        "produced the labels in the paper.\n\n"
        "\\begin{tabular}{@{}ll@{}}\n\\toprule\n"
        "Setting & Value \\\\\n\\midrule\n" + "\n".join(rows)
        + "\n\\bottomrule\n\\end{tabular}\n\n"
        "\\subsection*{Prompt versions behind the labels}\n\n"
        "Each narrative is sent to the model in two separate passes. The mechanism pass receives the masked "
        "narrative and assigns the ten pre-crash cues, and the cue pass receives the "
        "unmasked narrative and assigns the five injury cues. The label used for each "
        "crash and pass is the latest parsed call in the log. The prompt version string "
        "is a single constant of the extraction script, so it advances for both passes "
        f"together. The {totals['mechanism']} mechanism labels come from version "
        f"{version_text('mechanism')}"
        + (", which is the current prompt" if mech_all_current else "")
        + f". The {totals['cue']} injury-cue labels come from calls logged under versions "
        f"{version_text('cue')}. To check what each call was sent, the SHA-256 of the "
        "current prompt text and the narrative was recomputed for every call and compared "
        "with the hash logged at the time. " + hash_text + "\n\n"
        "\\subsection*{Repeated runs and rerun variation}\n\n"
        f"The vote shares come from {words(len(extract.VOTE_SEEDS))} further runs of the "
        f"mechanism pass at temperature {extract.VOTE_TEMPERATURE:g}, with seeds "
        f"{joined(str(s) for s in extract.VOTE_SEEDS)}, under the same model, provider and "
        "prompt. The reference run itself is not exactly reproducible on this provider, "
        "although its temperature is zero and its seed is fixed. A masked rerun of the "
        f"mechanism pass on the validation narratives changed {rr['decisions_differing_from_reference']} "
        f"of {rr['label_decisions']} label decisions, or {pct(rr['share_differing'])}\\%. "
        f"On the {ti['narratives']} narratives that masking leaves unchanged, so that the "
        "masked and unmasked runs received identical prompts, "
        f"{ti['decisions_differing']} of {ti['label_decisions']} decisions differed, or "
        f"{pct(ti['share_of_decisions_differing'])}\\%. Rerun variation of this size sets "
        "the scale against which the alternative-model and masking comparisons in "
        f"Appendix~{letter('app:validation')} of the manuscript are read.\n\n"
        "\\subsection*{The alternative model}\n\n"
        f"The alternative model was \\texttt{{{tex(st['alternative_model_requested'])}}}, "
        f"run on the {pq['crashes_labelled']} validation narratives through "
        f"{tex(used['provider'])} with the provider pinned and fallbacks disabled, the "
        f"endpoint serving \\texttt{{{tex(used['quantization'])}}} weights with a context "
        f"length of {used['context_length']:,} tokens and support for structured outputs "
        f"and a seed. It used prompt version \\texttt{{{tex(st['prompt_version'])}}}, the "
        f"same JSON schema, temperature {st['temperature']:g} and seed {st['seed']}, and "
        f"its {pq['calls_in_scope_total']} calls returned {pq['records_failed_in_scope']} "
        "failures." + other_text + " The alternative model reached a macro $F_1$ on the ten pre-crash cues of "
        f"{rnd(macro_qwen, 3)}, against {rnd(macro_llama, 3)} for "
        "the reference model on the same narratives.\n\n"
        "\\subsection*{Mechanism pass prompt, version "
        f"\\texttt{{{short_version(extract.PROMPT_VERSION)}}}}}\n\n"
        "The system message is reproduced below. The user message is the word NARRATIVE "
        "followed by the masked narrative.\n\n"
        + verbatim(extract.MECHANISM_SYSTEM)
        + "\\subsection*{Injury-cue pass prompt}\n\n"
        "The system message is reproduced below. The user message is the word NARRATIVE "
        "followed by the unmasked narrative.\n\n"
        + verbatim(extract.CUE_SYSTEM)
        + "\\subsection*{Response schemas}\n\n"
        "The mechanism pass must return the scooter unit, the ten cue labels, the actor "
        "of failure to yield and of a signal violation, and a rationale. The cue pass "
        "must return the scooter unit, the five injury-cue labels and a rationale.\n\n"
        + verbatim("mechanism: " + ", ".join(extract.MECHANISM_SCHEMA["schema"]["required"])
                   + "\n\ncue: " + ", ".join(extract.CUE_SCHEMA["schema"]["required"]))
        + "\\subsection*{What each call records}\n\n"
        "Every call is logged as one JSON object, so that any label can be traced to the "
        "call that produced it. The fields are the following.\n\n"
        + verbatim(", ".join(logged))
        + "A rerun skips any call already logged with a parsed response, so the run is "
          "resumable and a partial failure does not change the labels already produced. "
          "The rationale field of the parsed response can quote phrases of a narrative, "
          "so it is to be removed from the logs that are released.\n\n")
    return text


# ---------------------------------------------------------------- S2

def s2() -> str:
    """The masking list and the lexical rules, from the modules that apply them."""
    from masking import MASK_PATTERNS  # noqa: PLC0415
    import route_a  # noqa: PLC0415

    rules = getattr(route_a, "RULES", None)
    if rules is None:
        raise SystemExit("S2: route_a has no RULES to document")

    counts = {}
    path = DATA / "route_a_labels.csv"
    if path.exists():
        import pandas as pd  # noqa: PLC0415
        frame = pd.read_csv(path)
        for col in frame.columns:
            if frame[col].dropna().isin([0, 1, True, False]).all():
                counts[col] = int(frame[col].fillna(0).astype(float).sum())

    blocks = []
    for cue in sorted(rules):
        rule = rules[cue]
        k = counts.get(cue)
        head = f"\\subsection*{{{tex(cue_name(cue))} (\\texttt{{{tex(cue)}}})}}"
        if k is not None:
            head += f"\n\nThe rule matches {k} narratives."
        actor = getattr(rule, "actor", "any")
        if actor != "any":
            head += (f" The match is attributed only when the acting unit is the "
                     f"{tex(actor)}.")
        head += "\n"
        parts = []
        for field_name in ("include", "exclude"):
            patterns = getattr(rule, field_name, ())
            if patterns:
                parts.append(f"{field_name}:\n" + "\n".join(map(str, patterns)))
        require_all = getattr(rule, "require_all", ())
        if require_all:
            parts.append("require all of:\n"
                         + "\n".join("  " + " | ".join(map(str, group))
                                      for group in require_all))
        blocks.append(head + "\n" + verbatim("\n\n".join(parts) if parts
                                              else "(no patterns)"))

    return ("\\section{Masking and the lexical rules}\\label{sup:lexical}\n\n"
            "\\subsection*{The masking list}\n\n"
            "Before the mechanism pass, every term below is replaced in the narrative, so "
            "that a pre-crash cue is not inferred from an injury outcome. The cue pass "
            "receives the unmasked text of the same narrative. Which text each pass received is verified against "
            "the logged prompt hashes, as Section~S1 describes.\n\n"
            + "".join(
                f"\\textbf{{{tex(category)}}}\n" + verbatim("\n".join(patterns))
                for category, patterns in sorted(MASK_PATTERNS.items()))
            + "\\subsection*{The rules}\n\n"
              "Route A is deterministic and applies the same rules to every narrative. "
              "It uses no term weighting and no trained "
              "classifier, so no information from the validation sample enters it. Every "
              "rule resolves which unit is the scooter from its description before "
              "attributing any action to it, because unit numbering is not consistent in "
              "these reports.\n\n"
            + "".join(blocks))


# ---------------------------------------------------------------- S3

def s3(n: dict) -> str:
    guide = ROOT / "docs" / "coding_guide.md"
    if not guide.exists():
        raise SystemExit("S3: coding_guide.md does not exist")
    text = guide.read_text(encoding="utf-8")
    from guide_examples import constructed  # noqa: PLC0415
    return ("\\section{The coding guide}\\label{sup:guide}\n\n"
            "This is the document the three coders worked from. It was written before any "
            "coding began and was not revised during it. It is reproduced as it was given "
            "to them, except that its worked examples, which quoted crash narratives, are "
            "replaced by constructed examples that make the same point and keep the same "
            "label, because the narrative text is not redistributed.\n\n"
            + verbatim(constructed(text)) + "\n\n" + s3_attribution(n, text) + actor_block(n))


def _guide_section(text: str, title_start: str) -> str:
    """The number of a guide section, read from its heading rather than typed."""
    import re
    m = re.search(r"^###\s+(\d+\.\d+)\s+" + re.escape(title_start), text, re.M)
    if not m:
        raise SystemExit(f"S3: no guide section starting '{title_start}'")
    return m.group(1)


def s3_attribution(n: dict, guide: str) -> str:
    """How far the reference run's attribution agrees with the coders' majority vote."""
    aa = req(n, "sensitivity", "data_extra", "actor_attribution", "cues")
    fty = req(aa, "failure_to_yield", "validation_agreement_llm_vs_coders")
    sv = req(aa, "signal_violation", "validation_agreement_llm_vs_coders")
    s_fty = _guide_section(guide, "Failure to yield")
    s_sv = _guide_section(guide, "Signal or stop-sign violation")
    b2 = sv["batch2_flagged_adjudication"]
    levels = sorted(b2["confusion_rows_coders_cols_llm"])
    if levels != ["rider"]:
        raise SystemExit("S3: the flagged signal-violation attributions are not all rider")

    def rate(g: dict) -> str:
        return (f"{g['n_agree']} of {g['n_both_attributed']} narratives "
                f"({rnd(g['percent_agreement'], 1)}\\%, Cohen's $\\kappa$ "
                f"{rnd(g['cohen_kappa'], 3)})")

    return (
        f"Sections {s_fty} and {s_sv} of the guide ask the coder, once failure to yield "
        "or a signal violation is marked, which party the narrative attributes it to. "
        "The reference run records the same attribution and is compared here with the "
        "coders' majority vote where both code the cue positive. For failure to yield the reference run "
        f"gave the coders' attribution in {rate(fty['batch1_validation_sample'])} of the "
        f"validation sample, and in {rate(fty['pooled'])} once the narratives flagged for "
        "human coding are added. For signal violation the agreement is "
        f"{rate(sv['batch1_validation_sample'])} and {rate(sv['pooled'])}. In the flagged "
        "narratives alone the coders attributed every signal violation to the rider, so "
        f"only the agreement there, {b2['n_agree']} of {b2['n_both_attributed']}, is "
        "informative. Table~\\ref{tab:a_actor} gives the attribution by severity class "
        "and the event-rule check that keeps the split out of the severity models.\n\n")


# ---------------------------------------------------------------- S4

def s4(n: dict) -> str:
    rp = req(n, "sensitivity", "random_parameters")
    proc = rp["procedure"]
    draws = proc["draws"]
    summ = rp["screen_summary"]
    het = rp["heterogeneity"]
    final = het["final_model"]
    joint = rp["joint_model"]
    comp = rp["comparison"]
    cs = rp["comparison_summary"]
    stab = rp["stability"]
    interp = rp["interpretation"]
    ame = proc["average_marginal_effects"]
    K = letter("app:randpar")
    fixed = rp["fixed_model"]

    def names(keys) -> str:
        return joined(tex(pretty(k)).lower() for k in keys)

    retained = summ["retained"]
    single = {k: v for k, v in comp.items() if k.startswith("single_random[")}
    best_single = min(single, key=lambda k: single[k]["bic"])
    best_name = best_single[len("single_random["):-1]

    conv = final["convergence"]
    mean_sh = het["retained_mean_shifters"]
    var_sh = het["retained_variance_shifters"]

    def shifters(d: dict, kind: str) -> list[str]:
        return [f"a {kind} shift of the {tex(pretty(k)).lower()} coefficient by "
                f"{tex(pretty(z)).lower()}" for k, zs in d.items() for z in zs]

    shifter_list = shifters(mean_sh, "mean") + shifters(var_sh, "variance")

    cand = rp["candidates"]
    by_sd = sorted(cand, key=lambda k: -cand[k]["sd"])
    holm = summ["lr_significant_after_holm"]
    holm_sd = (", and these have the largest estimated standard deviations, "
               + joined(f"{rnd(cand[k]['sd'], 3)}" for k in by_sd[:len(holm)])
               if set(by_sd[:len(holm)]) == set(holm) else "")
    ranges = final["stability_summary"]["sd_range"]
    widest = max(ranges, key=lambda k: ranges[k][1] - ranges[k][0])
    lo_, hi_ = ranges[widest]
    fp = final["parameters"][widest]
    least_stable = (f"The standard deviation of the {tex(pretty(widest[3:-1])).lower()} "
                    f"coefficient varies most across the settings, from {rnd(lo_, 3)} to "
                    f"{rnd(hi_, 3)}"
                    + (", and its standard error in the final model is larger than the "
                       "estimate." if fp["se"] > fp["est"] else "."))

    # Stability of the final model over draw counts and seeds.
    settings = final["stability"]
    sd_keys = list(next(iter(settings.values()))["sd"])
    sh_keys = list(next(iter(settings.values()))["shifters"])

    def setting_title(key: str) -> str:
        d, s = key.replace("draws", "").split("_seed")
        return f"{d} draws, seed {s}"

    cols = list(settings)
    srows = []
    srows.append("$\\log L$ & " + " & ".join(f"{fmt(settings[c]['loglik'], 2)}"
                                             for c in cols) + " \\\\")
    for k in sd_keys:
        name = k[3:-1]
        srows.append(f"SD, {tex(pretty(name)).lower()} & "
                     + " & ".join(f"{rnd(settings[c]['sd'][k], 3)}" for c in cols) + " \\\\")
    for k in sh_keys:
        kind, rest = k.split("[", 1)
        a, z = rest.rstrip("]").split("|")
        title = "Mean shift" if kind == "mean_shift" else "Variance shift"
        srows.append(f"{title}, {tex(pretty(a)).lower()} by {tex(pretty(z)).lower()} & "
                     + " & ".join(f"{fmt(settings[c]['shifters'][k])}" for c in cols)
                     + " \\\\")
    head = ("\\toprule\nDraws & "
            + " & ".join(c.replace("draws", "").split("_seed")[0] for c in cols)
            + " \\\\\nSeed & "
            + " & ".join(c.split("_seed")[1] for c in cols)
            + " \\\\\n\\midrule\n")
    # The caption and label sit in the first head only, so a page break repeats the
    # column header without defining the label a second time.
    stable_table = (
        "{\\small\n\\begin{longtable}{>{\\raggedright\\arraybackslash}p{6.2cm}"
        + "r" * len(cols) + "}\n"
        "\\caption{Final random-parameter model under each draw count and seed.}"
        "\\label{sup:tab_rpstability}\\\\\n" + head + "\\endfirsthead\n"
        f"\\multicolumn{{{len(cols) + 1}}}{{l}}{{\\textbf{{\\tablename\\ \\thetable{{}} "
        "\\textit{(continued)}}} \\\\\n" + head + "\\endhead\n"
        + "\n".join(srows) + "\n\\bottomrule\n\\end{longtable}}\n\n")

    ka = cs["final_model_ame_KA_key_covariates"]
    flips = [k for k, v in ka.items()
             if v["fixed_excludes_zero"] and (v["fixed"] > 0) != (v["random_parameter"] > 0)]
    flip_text = ""
    for k in flips:
        v = ka[k]
        flip_text += (f" The KA marginal effect of {tex(pretty(k)).lower()} changes sign, "
                      f"from {fmt(v['fixed'], 3)} in the fixed M1 to "
                      f"{fmt(v['random_parameter'], 3)} in the final model.")

    text = (
        "\\section{Random-parameter specifications}\\label{sup:randpar}\n\n"
        "\\subsection*{Specification}\n\n"
        "The random-parameter models are ordered logits for the three severity classes "
        "with a logistic error, in which selected coefficients follow a normal "
        "distribution across riders and all other coefficients are fixed. They start "
        f"from the {len(rp['covariates'])} covariates of M1 and the same threshold "
        "parameterization, with the first threshold followed by log increments. The models "
        "are estimated by simulated maximum likelihood over quasi-random draws. The draws are scrambled Halton "
        "sequences mapped to the standard normal by the inverse distribution function, "
        f"one set per rider record, with {draws['estimation']} draws for estimation and "
        f"checks at {joined(str(d) for d in draws['stability_checks'])} draws. Three "
        f"seeds, {joined(str(s) for s in draws['seeds'])}, change only the scramble of the "
        "sequence, the first is used for estimation, and every model compared under one "
        "setting shares the same block of draws. The likelihood is maximized by BFGS with "
        "an analytic gradient, the covariance is the inverse of a Hessian obtained by "
        "central differences of that gradient, and every model is started from standard "
        f"deviations of {joined(f'{v:g}' for v in proc['sigma_starting_values'])}. With no "
        "random coefficient the likelihood reproduces that of the fixed M1 to within "
        f"{sci(rp['verification']['fixed_special_case']['abs_gap'])}, and the analytic "
        "gradient agrees with central differences to a relative error of "
        f"{sci(rp['verification']['analytic_gradient']['max_rel_error_vs_central_difference'])}."
        "\n\n"
        "\\subsection*{Candidate selection}\n\n"
        f"Each of the {summ['n_candidates']} covariates was made random in turn, with the "
        "others fixed. A candidate was retained when the Wald test of its standard "
        f"deviation and the likelihood-ratio test against the fixed M1 were both "
        f"significant at {pct(proc['alpha'], 0)}\\%. The likelihood-ratio test uses the "
        "chi-bar-squared reference, an equal mixture of chi-squared distributions with "
        "zero and one degree of freedom, because a standard deviation of zero lies on the "
        f"boundary of the parameter space. {words(summ['n_retained']).capitalize()} candidates were retained, "
        f"namely {names(retained)}. With the naive chi-squared reference "
        f"{len(summ['retained_if_naive_chi2'])} would have been retained. At the "
        f"{pct(proc['alpha'], 0)}\\% level about {summ['expected_false_retentions_at_alpha']:g} "
        "false retentions are expected among this many candidates. Only "
        f"{names(summ['lr_significant_after_holm'])} remain significant after a Holm "
        "correction" + holm_sd + ". Every candidate model converged, and its "
        "log-likelihood agreed across the three starting values. "
        "Table~\\ref{tab:a_randompar} gives the result for each candidate, and "
        f"Table~{K}.1 of the manuscript compares the fit of the random-parameter "
        "models.\n\n"
        + randpar_candidates_block(n)
        + "\\subsection*{Heterogeneity in means and variances}\n\n"
        "For each retained coefficient, heterogeneity in the mean was modeled as a shift "
        "by a context variable and heterogeneity in the variance as an exponential "
        "scaling of the standard deviation. Each was tested by likelihood ratio against "
        "the joint model, one context variable at a time, with the "
        + joined(f"{tex(pretty(k)).lower()} ({v} riders)"
                 for k, v in rp["context_variable_counts"].items())
        + f" indicators as candidates. A test was not run when fewer than {proc['min_cell']} riders "
        "fell on either side of the split among those with the covariate, or when the "
        f"interaction was collinear with the design. Of the {het['n_tests']} admissible "
        f"tests, {het['n_tests_not_converged']} did not converge and were not counted as "
        f"significant, and about {het['expected_false_positives_at_alpha']:g} false "
        "positives are expected among them. The shifters significant at "
        f"{pct(proc['alpha'], 0)}\\% were {joined(shifter_list)}, and they entered the "
        "final model together.\n\n"
        "\\subsection*{Convergence and stability}\n\n"
        f"The final model converged with a largest absolute gradient of "
        f"{sci(conv['max_abs_gradient'])}, a positive definite Hessian with condition number "
        f"{rnd(conv['hessian_condition_number'], 1, comma=True)}, and the same log-likelihood from each "
        "starting value of the shifters. Across the draw counts and seeds, no candidate "
        "changed its retention decision, the candidate log-likelihoods moved by at most "
        f"{stab['max_abs_loglik_deviation']:g}, and the candidate standard deviations by "
        f"at most {stab['max_abs_sd_deviation']:g}. "
        "Table~\\ref{sup:tab_rpstability} gives the final model under each setting. " + least_stable + "\n\n"
        + stable_table
        + "\\subsection*{Comparison with the fixed-parameter model}\n\n"
        f"The fixed M1 has an AIC of {rnd(fixed['fit']['aic'], 1)} and a BIC of "
        f"{rnd(fixed['fit']['bic'], 1)}. The joint model, with the {len(joint['random'])} "
        f"retained coefficients random, has an AIC of {rnd(joint['fit']['aic'], 1)} and a BIC "
        f"of {rnd(joint['fit']['bic'], 1)}, and the final model has an AIC of "
        f"{rnd(final['fit']['aic'], 1)} and a BIC of {rnd(final['fit']['bic'], 1)}. The single "
        f"random coefficient on {tex(pretty(best_name)).lower()} has the lowest BIC of all, "
        f"{rnd(single[best_single]['bic'], 1)}. Under the rule stated in the Methods the fixed "
        "M1 stays primary, because neither the joint nor the final model has a lower BIC. "
        f"Of the {words(len(retained))} retained coefficients, "
        f"{words(len(interp['retained_random_also_failing_brant']))} "
        "also depart from proportional odds at the 5\\% level, and each has a BC share "
        "below the overall "
        "share among the riders with the attribute. A normal random coefficient on a binary "
        "indicator widens the latent distribution of those riders and moves probability "
        "out of the middle class, so the retained standard deviations and the "
        "proportional-odds departures largely describe one feature of the data. Marginal "
        f"effects were integrated over {ame['integration_draws']} draws per rider and their "
        f"intervals use {ame['krinsky_robb_draws']} Krinsky and Robb draws." + flip_text
        + " In the fixed M1 the KA marginal effect has an interval excluding zero for "
        f"{names(cs['ka_excludes_zero_fixed'])}. In the final model it has one for "
        f"{names(cs['ka_excludes_zero_random_parameter'])}.\n\n")
    return text


# ---------------------------------------------------------------- S5

def s5(n: dict) -> str:
    """Data-only tables that the manuscript refers to by their number here."""
    return ("\\section{Additional tables}\\label{sup:tables}\n\n"
            "The tables in this section hold data that the manuscript refers to but does "
            "not print. They give the composition of the sample by reporting year, the "
            "recovery of a planted reporting shifter in simulated data, the "
            "threshold-specific coefficients of the partial proportional-odds models, two "
            "alternative spatial splits, and the per-class metrics of the predictive "
            "benchmark. Each is generated by the same script and from the same results "
            "file as the appendix of the manuscript, and each is introduced by a short "
            "account of what it shows and how it was produced.\n\n"
            + annual_block(n) + shifter_block(n) + ppo_coef_block(n)
            + spatial_alt_block(n) + perclass_block(n))


# ---------------------------------------------------------------- S6 validation sample

def s6(n: dict) -> str:
    """The composition of the validation sample, stratum by stratum, in draw order.

    The pools and draws are read from the audit that sampling.draw_validation wrote. The
    pools are recomputed here from the reference-run labels and the stratum each sampled
    narrative was drawn under, without the random draw, so a stale audit stops the build.
    """
    import pandas as pd  # noqa: PLC0415
    from labels import MECHANISMS  # noqa: PLC0415

    audit = json.loads((DATA / "validation_sampling_audit.json").read_text(encoding="utf-8"))
    ids = pd.read_csv(DATA / "validation_ids.csv")
    ref = pd.read_csv(DATA / "reference_labels.csv")
    pilot = set(pd.read_csv(DATA / "pilot_ids.csv")["Crash_ID"].astype(int))
    base = pd.read_csv(DATA / "base_frame.csv", usecols=["Crash_ID"]).drop_duplicates()
    eligible = base.merge(ref, on="Crash_ID", how="inner", validate="one_to_one")
    eligible = eligible[~eligible["Crash_ID"].isin(pilot)]

    n_val, n_elig = audit["n"], audit["eligible_n"]
    sizes, fill_n = audit["stratum_sizes"], audit["random_fill_n"]
    door_n = audit["dooring_all_n"]
    if len(eligible) != n_elig or len(ids) != n_val:
        raise SystemExit("S6: eligible or sampled narratives do not match the audit")
    order = [m for m in MECHANISMS if m != "dooring"]
    if list(sizes) != order:
        raise SystemExit("S6: the audit strata are not in the draw order of labels.py")
    drawn_cues = sum(sizes[m]["drawn"] for m in order)
    if door_n + drawn_cues + fill_n != n_val:
        raise SystemExit("S6: the strata do not add up to the sample")
    fill_pool = n_elig - (n_val - fill_n)

    # Recompute each pool. The narratives removed before a stratum is drawn are the
    # dooring positives and those drawn by every earlier stratum.
    stratum = dict(zip(ids["Crash_ID"].astype(int), ids["stratum"]))
    removed = {c for c, s in stratum.items() if s == "dooring_all"}
    if len(removed) != door_n or int(eligible["dooring"].sum()) != door_n:
        raise SystemExit("S6: the dooring stratum does not match the audit")
    predicted = {}
    for m in order:
        pos = set(eligible.loc[eligible[m] == 1, "Crash_ID"].astype(int))
        predicted[m] = len(pos)
        if len(pos - removed) != sizes[m]["pool"]:
            raise SystemExit(f"S6: recomputed pool for {m} differs from the audit")
        removed |= {c for c, s in stratum.items() if s == f"positive_{m}"}
    if n_elig - len(removed) != fill_pool:
        raise SystemExit("S6: the random-fill pool does not follow from the audit")

    # Gold positives among the sampled narratives, from the confusion counts of the
    # reference run, which scores every cue on every validation narrative.
    per = req(n, "extraction", "routes", "route_b_reference", "per_label")
    gold = {m: int(req(per, m, "tp") + req(per, m, "fn")) for m in MECHANISMS}
    gold_file = pd.read_csv(DATA / "gold_labels.csv")
    gold_file = gold_file[gold_file["Crash_ID"].isin(ids["Crash_ID"])]
    if len(gold_file) != n_val or any(int(gold_file[m].sum()) != gold[m] for m in MECHANISMS):
        raise SystemExit("S6: gold positives differ between numbers.json and gold_labels.csv")

    # Overlap among the cue strata, from the reference-run labels of the sampled narratives.
    cue_drawn = ids[ids["stratum"].str.startswith("positive_")].merge(ref, on="Crash_ID")
    multi = int((cue_drawn[list(MECHANISMS)].sum(axis=1) > 1).sum())
    if len(cue_drawn) != drawn_cues:
        raise SystemExit("S6: cue-stratum narratives do not match the audit")

    probs = ids["inclusion_prob"]
    p_lo, p_hi = float(probs.min()), float(probs.max())
    if abs(p_lo - fill_n / fill_pool) > 1e-9:
        raise SystemExit("S6: the smallest inclusion probability is not the fill fraction")
    boot = req(n, "sensitivity", "supporting", "extraction_intervals_calibration",
               "per_cue_intervals", "bootstrap")
    unit = req(n, "sensitivity", "supporting", "extraction_intervals_calibration",
               "macro_f1_intervals", "bootstrap", "resampling_unit")
    if not unit.startswith("narrative"):
        raise SystemExit("S6: the bootstrap no longer resamples narratives")
    L = letter("app:validation")

    rows = [f"Dooring, every predicted positive & {door_n} & {door_n} & {door_n} & "
            f"{pct(1.0)} & {gold['dooring']} \\\\"]
    for m in order:
        e = sizes[m]
        rows.append(f"{tex(cue_name(m))} & {predicted[m]} & {e['pool']} & {e['drawn']} & "
                    f"{pct(e['drawn'] / e['pool'])} & {gold[m]} \\\\")
    rows.append(f"Random fill, narratives not yet drawn & -- & {fill_pool} & {fill_n} & "
                f"{pct(fill_n / fill_pool)} & -- \\\\")
    rows.append("\\midrule")
    rows.append(f"Validation sample & -- & {n_elig} & {n_val} & "
                f"{pct(n_val / n_elig)} & -- \\\\")

    fracs = [sizes[m]["drawn"] / sizes[m]["pool"] for m in order]
    text = (
        "\\section{Validation-sample composition}\\label{sup:valsample}\n\n"
        f"The {n_val} validation narratives were drawn with seed {audit['seed']} in strata "
        "of the Route~B reference-run labels, taking every predicted dooring positive, then "
        f"up to {audit['per_mechanism_target']} predicted positives at random for each "
        "other cue in turn, and filling the remaining places at random. "
        "Table~\\ref{sup:tab_valsample} gives each stratum in the order it was drawn, with "
        "its pool, the number drawn and the sampling fraction, which ranges from "
        f"{pct(min(fracs))}\\% to {pct(max(fracs))}\\% for the nine cue strata.\n\n")
    return text + longtable(
        "Composition of the validation sample by sampling stratum, in draw order.",
        "sup:tab_valsample",
        ">{\\raggedright\\arraybackslash}X r r r r r",
        "Stratum & Predicted positives & Pool & Drawn & Fraction (\\%) & "
        "Gold positives \\\\",
        rows, size="\\footnotesize", hlcolour="HlA", need=22,
        note=(
            f"The {n_elig} eligible narratives exclude the {audit['pilot_excluded_n']} "
            "narratives used to develop the coding guide. Predicted positives counts the "
            "eligible narratives the reference run labels positive for the cue, and the "
            "pool of the last row is every eligible narrative. The pool removes those "
            "already drawn by an earlier "
            "stratum, because each stratum was drawn without replacement from the "
            "narratives not yet in the sample. The fraction is the number drawn divided by "
            "the pool of the stratum. The strata "
            f"overlap, since {multi} of the {drawn_cues} narratives drawn for a cue are "
            "predicted positive for more than one cue, so a fraction is the sampling "
            "fraction of its stratum and not the inclusion probability of a narrative. The "
            "inverse-probability weights approximate that probability as one minus the "
            "product of the chances of being missed by every stratum a narrative belongs "
            "to, treating the strata as independent and the random fill at its realized "
            "fraction. Over the "
            f"{n_val} narratives it ranges from {rnd(p_lo, 3)} to {rnd(p_hi, 3)}. Gold "
            "positives counts the sampled narratives that the coders' majority vote labels "
            f"positive, which equals TP plus FN in Table~{L}.2 of the manuscript. The "
            f"bootstrap intervals of that table draw {boot['resamples']} resamples of the "
            f"{n_val} narratives with replacement and do not resample within strata. A "
            "dash marks a cell that does not apply."))


# ---------------------------------------------------------------- S7 data sensitivities

def _band_title(spec: dict) -> str:
    """The cut points of a band specification, from the names of its stored bands."""
    bands = list(spec["band_cells_by_severity"]) + [spec["reference_band"]]
    cuts = sorted(int(re.search(r"\d+", b).group()) for b in bands
                  if not b.startswith("under"))
    if len(cuts) != len(bands) - 1:
        raise SystemExit(f"S7: cannot read the cut points of {bands}")
    return "Bands cut at " + joined(str(c) for c in cuts)


def _or(o: dict) -> str:
    """An odds ratio and its interval, at one decimal when the ratio is 10 or more and at
    two below, with the bounds at the places of the ratio."""
    places = 1 if o["or"] >= 10 else 2
    lo, hi = o["ci95"]
    return f"{rnd(o['or'], places)} [{rnd(lo, places)}, {rnd(hi, places)}]"


def s7(n: dict) -> str:
    """Age form, complete case, ACS vintage and the missingness mechanism."""
    ds = req(n, "sensitivity", "supporting", "data_sensitivities")
    age = req(ds, "age")
    cc = req(ds, "complete_case")
    mm = req(ds, "missingness_mechanism")
    acs = req(ds, "acs_and_bike_facility", "acs_vintage")
    m1_ames = req(n, "severity", "ames_all")
    bands = req(age, "band_specifications")
    splines = req(age, "spline_specifications")
    primary_fit = req(bands, "primary_lt25_25to44_ge45", "fit")

    # Every block must sit on the current M1, so its primary fit is checked against the
    # M1 of the ACS block and its comparison values against the M1 marginal effects.
    m1_fit = req(acs, "primary", "fit")
    if (primary_fit["loglik"] != m1_fit["loglik"]
            or primary_fit["n_parameters"] != m1_fit["n_parameters"]):
        raise SystemExit("S7: the age block is not fitted on the current M1")
    comp = req(cc, "ka_comparison_with_primary")
    for k, v in comp.items():
        if v["primary_KA"]["ame"] != req(m1_ames, k, "KA", "ame"):
            raise SystemExit(f"S7: complete-case comparison for {k} is not the current M1")
    if req(acs, "primary", "poverty_share_ame", "KA", "ame") != req(
            m1_ames, "poverty_share", "KA", "ame"):
        raise SystemExit("S7: the ACS primary fit is not the current M1")

    cols = 7

    # Panel A, age forms.
    base = req(age, "no_age_terms_model", "fit")
    forms = [("No age terms, the nested base", base, None)]
    for key in ("primary_lt25_25to44_ge45", "alt_lt25_25to54_ge55",
                "alt_lt21_21to29_30to44_ge45"):
        e = bands[key]
        title = _band_title(e) + (", M1" if key.startswith("primary") else "")
        forms.append((title, e["fit"], e["lr_vs_no_age_terms"]))
    forms.append(("Linear, per ten years", splines["linear"]["fit"],
                  splines["linear"]["lr_vs_no_age_terms"]))
    for key in ("rcs_3_knots", "rcs_4_knots"):
        e = splines[key]
        forms.append((f"Spline, {len(e['knots_years'])} knots", e["fit"],
                      e["lr_vs_no_age_terms"]))
    rows = [panel("Panel A. Age forms, each against the model without age terms", cols)]
    for title, fit, lr in forms:
        tail = (f"{rnd(lr['lr_chi2'], 2)} & {lr['df']} & {pfmt(lr['p'])}" if lr
                else "-- & -- & --")
        rows.append(f"{title} & {fmt(fit['loglik'], 2)} & {fit['n_parameters']} & "
                    f"{rnd(fit['bic'], 1)} & {tail} \\\\")
    knots3 = req(splines, "rcs_3_knots", "knot_percentiles")
    knots4 = req(splines, "rcs_4_knots", "knot_percentiles")
    sep4 = req(splines, "rcs_4_knots", "separation")

    # Panel B, complete case against M1, KA effects.
    n_m1 = req(primary_fit, "n")
    cov_m1 = req(bands, "primary_lt25_25to44_ge45", "events_per_parameter", "n_covariates")
    cov_cc = req(cc, "events_per_parameter", "n_covariates")
    dropped = req(cc, "covariates_dropped_as_constant_indicators")
    if cov_cc + len(dropped) != cov_m1 or len(comp) != cov_cc:
        raise SystemExit("S7: complete-case covariates do not add up to M1")
    shown = [k for k in m1_ames if req(m1_ames, k, "KA", "excludes_zero")
             or req(cc, "ames").get(k, {}).get("KA", {}).get("excludes_zero")]
    flips = []
    for k, v in comp.items():
        a, b = v["primary_KA"], v["complete_case_KA"]
        same = (a["ame"] > 0) == (b["ame"] > 0)
        if same != v["same_sign"]:
            raise SystemExit(f"S7: stored sign flag for {k} disagrees with the effects")
        if not same:
            if a["excludes_zero"] or b["excludes_zero"]:
                raise SystemExit(f"S7: {k} changes sign with an interval excluding zero")
            flips.append(k)

    def ame_ci(e: dict) -> str:
        return f"{fmt(e['ame'], 3)} {ci(*e['ci95'])}"

    def span(text: str) -> str:
        return f"\\multicolumn{{3}}{{c}}{{{text}}}"

    rows += [panel("Panel B. Complete case against M1, KA marginal effects", cols),
             inner_header("Row & \\multicolumn{3}{c}{M1, KA [95\\% CI]} & "
                          "\\multicolumn{3}{c}{Complete case, KA [95\\% CI]} \\\\"),
             f"Riders & {span(n_m1)} & {span(req(cc, 'n'))} \\\\",
             f"Covariates & {span(cov_m1)} & {span(cov_cc)} \\\\"]
    for k in shown:
        a = req(m1_ames, k, "KA")
        b = req(cc, "ames").get(k, {}).get("KA")
        if b is None and k not in dropped:
            raise SystemExit(f"S7: {k} is neither in the complete-case fit nor dropped")
        rows.append(f"{tex(pretty(k))} & {span(ame_ci(a))} & "
                    f"{span(ame_ci(b) if b else 'dropped, constant')} \\\\")
    rows.append(f"KA effects that change sign & \\multicolumn{{6}}{{c}}{{{len(flips)} of "
                f"{len(comp)} compared}} \\\\")

    # Panel C, ACS vintage.
    def target(key: str) -> str:
        m = re.match(r"acs(\d{4})_(\d{4})_for_(.+)_crashes$", key)
        if not m:
            raise SystemExit(f"S7: unrecognized ACS variant {key}")
        years = re.findall(r"\d{4}", m.group(3))
        return f"the {joined(years)} crashes" if years else "every crash"

    def vintage(key: str) -> str:
        m = re.match(r"acs(\d{4})_(\d{4})_", key)
        return (f"{m.group(1)} to {m.group(2)}, "
                f"{target(key).removeprefix('the ').removesuffix(' crashes')}")

    acs_rows = [(f"{tex(req(acs, 'primary_vintage'))}, every crash, M1",
                 req(acs, "primary", "poverty_share_ame"))]
    variants = req(acs, "variants")
    if not all(k.startswith("acs" + req(acs, "sensitivity_vintage").replace(" to ", "_"))
               for k in variants):
        raise SystemExit("S7: an ACS variant is not of the sensitivity vintage")
    for key, e in variants.items():
        acs_rows.append((vintage(key), e["poverty_share_ame"]))
    rows += [panel("Panel C. Poverty-share marginal effects by ACS vintage", cols),
             inner_header("ACS estimates, assigned to & "
                          + " & ".join(f"{s} & 95\\% CI" for s in SEV3) + " \\\\")]
    for title, e in acs_rows:
        rows.append(f"{title} & " + " & ".join(
            f"{fmt(req(e, s, 'ame'), 3)} & {ci(*req(e, s, 'ci95'))}" for s in SEV3)
            + " \\\\")
    changed = [(target(k), e["rows_with_value_changed"]) for k, e in variants.items()]

    # Panel D, missingness mechanism.
    from build_appendix import MISSING_FIELDS, PREDICTOR_WORDS  # noqa: PLC0415
    rows += [panel("Panel D. Odds of an unrecorded field by severity class, against BC",
                   cols),
             inner_header("Field & Missing & OR, O [95\\% CI] & $p$ & "
                          "OR, KA [95\\% CI] & $p$ & Share (\\%) \\\\")]
    for key, field, _ in MISSING_FIELDS:
        m = req(mm, "models", key)
        shares = " / ".join(pct(req(m, "missing_rate_by_severity", s)) for s in SEV3)
        cells = []
        for s in ("sev_O", "sev_KA"):
            o = req(m, "severity_odds_ratios", s)
            cells.append(f"{_or(o)} & {pfmt(o['p'])}")
        rows.append(f"{field} & {req(m, 'n_missing')} & " + " & ".join(cells)
                    + f" & {shares} \\\\")
    adjusters = [PREDICTOR_WORDS[p] for p in req(mm, "predictors") if not p.startswith("sev_")]
    sep_drop = [(field, PREDICTOR_WORDS[p]) for key, field, _ in MISSING_FIELDS
                for p in req(mm, "models", key, "predictors_dropped_for_separation")]
    sep_text = "".join(f" The model for {f.lower()} leaves out {p}, which separates it."
                       for f, p in sep_drop)

    def ordinal(k: int) -> str:
        suffix = "th" if 10 <= k % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(k % 10, "th")
        return f"{k}{suffix}"

    B = letter("app:missing")
    text = (
        "\\section{Age form, complete cases, ACS vintage and missingness}"
        "\\label{sup:datasens}\n\n"
        "These checks examine how the data enter M1 rather than the narrative-derived "
        "cues. Table~\\ref{sup:tab_datasens} compares the forms of rider age, refits M1 on "
        "the riders with every demographic field recorded and coordinates present, "
        "assigns the poverty share from an earlier ACS vintage, and models which fields "
        "go unrecorded.\n\n")
    return text + longtable(
        "Age forms, complete-case refit, ACS vintage and the missingness mechanism.",
        "sup:tab_datasens",
        ">{\\raggedright\\arraybackslash}X r r r r r r",
        "Row & $\\log L$ & Parameters & BIC & LR & d.f. & $p$ \\\\",
        rows, size="\\footnotesize", repeat_head=False, hlcolour="HlA", need=30,
        tabcolsep="4pt",
        note=(
            "In Panel~A every model keeps the indicator for unrecorded age, and the linear "
            "and spline forms give riders without a recorded age the median recorded age. "
            "The restricted cubic splines place knots at the "
            f"{joined(ordinal(k) for k in knots3)} or the "
            f"{joined(ordinal(k) for k in knots4)} percentiles of recorded age, and the "
            "four-knot fit shows separation, with a coefficient of "
            f"{rnd(sep4['max_abs_coefficient'], 1)} in absolute value. Panel~B lists the "
            "covariates whose KA interval excludes zero in either fit, and the refit drops "
            f"{words(len(dropped))} indicators that are constant among complete cases. "
            "Every KA effect that changes sign has an interval including zero in both "
            "fits. In Panel~C the vintages correlate at "
            f"{rnd(req(acs, 'correlation_between_vintages_located_rows'), 3)} on located "
            "records, and the earlier vintage changes the poverty share of "
            + joined(f"{c} riders when assigned to {t}" for t, c in changed)
            + ". Panel~D adjusts for " + joined(adjusters) + "." + sep_text
            + " OR is the odds ratio against BC. The share is the percentage of riders "
            "with the field unrecorded in O, BC and KA, in that order, as in "
            f"Table~{B}.1 of the manuscript."))


def s8(n: dict) -> str:
    """Posted speed at the statutory line, and what the record holds about each unit."""
    ss = req(n, "sensitivity", "safe_system")
    sp = req(ss, "statutory_speed")
    d = req(sp, "descriptive")
    cov = req(ss, "record_coverage")
    veh = req(ss, "striking_vehicle_block", "lr_test")
    cols = 7

    def mix_cells(m: dict) -> str:
        return " & ".join(f"{req(m, s, 'count')} ({rnd(req(m, s, 'percent'), 1)})"
                          for s in SEV3)

    rows = [panel("Panel A. Riders by posted limit against the 35 mph line", cols)]
    lim = req(d, "statutory_limit_mph")
    rows.append(f"{lim} mph or less & {req(d, 'severity_mix_at_or_below', 'n')} & "
                f"{mix_cells(req(d, 'severity_mix_at_or_below'))} & & \\\\")
    rows.append(f"Above {lim} mph & {req(d, 'severity_mix_above', 'n')} & "
                f"{mix_cells(req(d, 'severity_mix_above'))} & & \\\\")

    def clear(e: dict) -> bool:
        # Bold only when the printed interval excludes zero; a bound stored as -0.0003
        # prints as 0.000, and a bold value beside it would contradict the cell.
        lo, hi = (float(rnd(x, 3)) for x in e["ci95"])
        return e["excludes_zero"] and (lo > 0 or hi < 0)

    def model_row(title: str, v: dict) -> tuple[str, str]:
        r = req(v, req(v, "reported"))
        c1, c2 = req(r, "cut1_BC_or_KA_vs_O"), req(r, "cut2_KA_vs_O_or_BC")
        coef = (f"{title} & {bold_if(fmt(c1['coef']), clear(c1))} & "
                f"{ci(*c1['ci95'])} & {bold_if(fmt(c2['coef']), clear(c2))} & "
                f"{ci(*c2['ci95'])} & {pfmt(req(r, 'difference_wald_p'))} & "
                f"{rnd(req(r, 'bic_minus_ordered_logit'), 1).replace('-', '$-$')} \\\\")
        a = req(r, "ames")
        ame = (f"{title} & " + " & ".join(
            f"{bold_if(fmt(a[s]['ame']), clear(a[s]))} & {ci(*a[s]['ci95'])}"
            for s in SEV3) + " \\\\")
        return coef, ame

    bands = req(sp, "prespecified_bands_holm_form")
    stat = req(sp, "statutory_line_holm_form")
    b_coef, b_ame = model_row("45 mph or above, pre-specified bands", bands)
    s_coef, s_ame = model_row(f"Above {lim} mph, exploratory", stat)
    rows += [panel("Panel B. Threshold-specific coefficients of the speed term", cols),
             inner_header("Speed term & Any injury & 95\\% CI & KA & 95\\% CI & "
                          "Difference $p$ & $\\Delta$BIC \\\\"),
             b_coef, s_coef,
             panel("Panel C. Average marginal effects of the speed term", cols),
             inner_header("Speed term & O & 95\\% CI & BC & 95\\% CI & KA & 95\\% CI \\\\"),
             b_ame, s_ame]

    names = {"Veh_Make_ID": "Make", "Veh_Mod_ID": "Model", "Veh_Body_Styl_ID": "Body style",
             "Veh_Mod_Year": "Model year"}
    rows += [panel("Panel D. Crashes with the field recorded, by unit", cols),
             inner_header("Field & E-scooter & Motor vehicle & & & & \\\\")]
    for k, title in names.items():
        rows.append(f"{title} & {req(cov, 'scooter_unit', k)} & "
                    f"{req(cov, 'striking_vehicle', k + '_2')} & & & & \\\\")

    reported = {k: req(v, "reported") for k, v in (("bands", bands), ("stat", stat))}
    crossing_b = req(bands, req(bands, "reported"), "ame_crossing", "draws_with_any_crossing")
    crossing_s = req(stat, req(stat, "reported"), "ame_crossing", "draws_with_any_crossing")
    draws = req(ss, "krinsky_robb", "draws")
    text = (
        "\\section{Posted speed at the statutory line and the record of each unit}"
        "\\label{sup:safesystem}\n\n"
        "Texas allows a motor-assisted scooter only on a street posted at "
        f"{lim} mph or less. The pre-specified speed bands put 35 and 40 mph in one band, "
        "so they cannot show whether severity differs on either side of that line. "
        "Table~\\ref{sup:tab_safesystem} reports an exploratory split at the line, beside "
        "the pre-specified 45 mph band, each freed in the partial proportional-odds form "
        "of the pre-specified Holm set, and what the crash record holds about the scooter "
        "and about the motor vehicle in the same crashes.\n\n")
    return text + longtable(
        "Posted speed at the statutory line, and the fields recorded for each unit.",
        "sup:tab_safesystem",
        ">{\\raggedright\\arraybackslash}X r r r r r r",
        "Posted limit & Riders & O, $n$ (\\%) & BC, $n$ (\\%) & KA, $n$ (\\%) & & \\\\",
        rows, size="\\footnotesize", repeat_head=False, hlcolour="HlM", need=30,
        tabcolsep="4pt",
        note=(
            f"Panel~A counts the {req(d, 'riders_with_recorded_limit')} riders with a "
            f"recorded posted limit; {req(d, 'above_limit_at_intersection_or_driveway')} of "
            f"the {req(d, 'riders_above_limit')} above the line crashed at an intersection "
            "or driveway. In Panels~B and~C the speed term and unrecorded rider age carry "
            "threshold-specific coefficients and every other M1 covariate a shared one. "
            "Any injury contrasts BC and KA with O, and KA contrasts KA with O and BC, so "
            "a positive coefficient favors the more severe side of that contrast. The "
            "difference $p$ is the Wald test that the two coefficients are equal, and "
            "$\\Delta$BIC is the change against the single-slope ordered logit. The "
            "pre-specified bands are reported from the "
            + ("fit constrained to keep every probability valid, because the unconstrained "
               "fit gives a negative middle-class probability on "
               f"{req(bands, 'rows_crossing_unconstrained')} riders"
               if reported["bands"] == "noncrossing" else "unconstrained fit")
            + "; the split at the line needs no constraint. Bold marks an interval that "
            f"excludes zero. Marginal effects use {rnd(draws, 0, comma=True)} Krinsky and "
            f"Robb draws, of which {crossing_b} and {crossing_s} give a negative "
            "middle-class probability for some rider under a counterfactual. The split at "
            "the line was chosen after the pre-specified results were known and is "
            "recorded as a departure from the pre-specification. Panel~D counts the "
            f"{req(cov, 'crashes')} modeled crashes in which the field is present and not "
            "unknown; in M1 the three body-type indicators of the motor vehicle together "
            f"give a likelihood-ratio $p$ of {pfmt(veh['p'])}."))


# The table labels of the supplement, beyond the moved tables.
OWN_TABLES = ("sup:tab_rpstability",)
# Tables appended after the moved ones, with the macro that carries each printed number.
OWN_MACROS = {
    "sup:tab_valsample": "suppValSample",
    "sup:tab_datasens": "suppDataSens",
    "sup:tab_safesystem": "suppSafeSystem",
}


def further(parts: dict) -> str:
    """The appendices printed here rather than in the manuscript, one subsection each."""
    return ("\\section{Further analyses}\\label{sup:further}\n\n"
            "The subsections below hold the diagnostics and sensitivity analyses that the "
            "manuscript states in a sentence and points to here. Each is generated by the "
            "same script and from the same results files as the manuscript.\n\n"
            + parts["moved_text"])


def resolve_refs(body: str, macro_labels: dict[str, str]) -> str:
    r"""Make every reference resolvable inside the supplement.

    A supplement macro such as \suppActor names a table of this document, so it becomes a
    \ref. A \ref to a label of the manuscript cannot cross documents, so it is replaced
    by the number the manuscript printed at its last compile (manuscript.aux).
    """
    for label, macro in macro_labels.items():
        body = body.replace(f"Supplementary Table~\\{macro}{{}}", f"Table~\\ref{{{label}}}")
        body = body.replace(f"Supplementary Section~\\{macro}{{}}", f"Section~\\ref{{{label}}}")
        body = body.replace(f"\\{macro}{{}}", f"\\ref{{{label}}}")
    defined = set(re.findall(r"\\label\{([^}]+)\}", body))
    aux = OUT / "manuscript.aux"
    printed = {}
    if aux.exists():
        printed = dict(re.findall(r"\\newlabel\{([^}]+)\}\{\{([^}]*)\}",
                                  aux.read_text(encoding="utf-8", errors="replace")))
    missing = set()

    def one(m: re.Match) -> str:
        label = m.group(1)
        if label in defined:
            return m.group(0)
        if label in printed:
            return printed[label]
        missing.add(label)
        return m.group(0)

    body = re.sub(r"\\ref\{([^}]+)\}", one, body)
    if missing and aux.exists():
        raise SystemExit("supplement: references to labels that neither the supplement nor "
                         f"the last manuscript compile defines: {sorted(missing)}")
    if missing:
        print(f"  NOTE: no manuscript.aux, so {len(missing)} reference(s) to the manuscript "
              "are left as \\ref and print as ??; compile the manuscript first")
    return body


def table_numbers(body: str, extra: tuple = ()) -> dict[str, int]:
    """The printed number of each table, from the order of the table labels in the body.

    Tables are numbered through the whole supplement, so the n-th captioned table is
    Table Sn. The count of captions is checked against the labels found, so a captioned
    table without a known label stops the build rather than shifting every number after it.
    """
    known = set(SUPP_MACROS) | set(OWN_TABLES) | set(OWN_MACROS) | set(extra)
    order = [m.group(1) for m in re.finditer(r"\\label\{([^}]+)\}", body)
             if m.group(1) in known]
    captions = len(re.findall(r"\\caption\{", body))
    if captions != len(order) or len(set(order)) != len(order):
        raise SystemExit(f"supplement: {captions} captions but table labels {order}")
    missing = (set(SUPP_MACROS) | set(OWN_MACROS)) - set(order)
    if missing:
        raise SystemExit(f"supplement: moved tables not printed: {sorted(missing)}")
    return {label: i + 1 for i, label in enumerate(order)}


def check_aux(numbers: dict[str, int]) -> None:
    """Compare with the numbers LaTeX printed at the last compile, when there is one."""
    aux = OUT / "supplementary.aux"
    if not aux.exists():
        return
    printed = dict(re.findall(r"\\newlabel\{([^}]+)\}\{\{S?([^}]*)\}", aux.read_text(
        encoding="utf-8", errors="replace")))
    for label, k in numbers.items():
        got = printed.get(label)
        if got is not None and got != str(k):
            print(f"  NOTE: {label} printed as S{got} at the last compile, now S{k}; "
                  "compile the supplement again")


def main() -> int:
    n = load_numbers()
    parts = build_parts(n)
    body = s1(n) + s2() + s3(n) + s4(n) + s5(n) + s6(n) + s7(n) + s8(n) + further(parts)
    body = fix_refs(body, "supp", parts)
    moved_macros = {label: supp_macro(label)
                    for label in list(parts["moved_order"]) + list(parts["moved_tables"])}
    body = resolve_refs(body, {**SUPP_MACROS, **OWN_MACROS, **moved_macros})
    text = ("% GENERATED by src/build_supplementary.py.\n"
            "% Do not edit by hand; rerun the generator.\n\n"
            "\\PassOptionsToPackage{table}{xcolor}\n"
            "\\documentclass[11pt,letterpaper]{article}\n"
            "\\usepackage[margin=1in]{geometry}\n"
            "\\usepackage[T1]{fontenc}\n\\usepackage[utf8]{inputenc}\n"
            "\\usepackage{newtxtext,newtxmath}\n\\usepackage{booktabs}\n"
            "\\usepackage{xcolor}\n\\usepackage{longtable}\n\\usepackage{array}\n"
            "\\usepackage{xltabular}\n\\usepackage{needspace}\n"
            "\\usepackage[labelfont=bf]{caption}\n\\usepackage[hidelinks]{hyperref}\n"
            "\\renewcommand{\\thesection}{S\\arabic{section}}\n"
            "% Tables are numbered S1, S2, ... through the whole supplement.\n"
            "\\renewcommand{\\thetable}{S\\arabic{table}}\n"
            "% The table style of the manuscript appendix, for the tables built by the same\n"
            "% functions. The supplement carries no highlight colours.\n"
            "\\def\\cleanbuild{}\n"
            "\\definecolor{HeaderGold}{HTML}{FFE78D}\n"
            "\\definecolor{HeaderText}{HTML}{3A2E00}\n"
            "\\definecolor{RowAlt}{HTML}{FFF8E1}\n"
            "\\definecolor{PanelBg}{HTML}{FFF0C2}\n"
            "\\newcommand{\\headerrow}{\\rowcolor{HeaderGold}}\n"
            "\\newcommand{\\hcell}[1]{\\textcolor{HeaderText}{\\textbf{#1}}}\n"
            "\\newcommand{\\altrow}{\\rowcolor{RowAlt}}\n"
            "\\newcommand{\\hltint}[2]{#2}\n"
            "\\newcommand{\\apptableneed}{\\needspace{8\\baselineskip}}\n"
            "\\newenvironment{reproduced}\n  {\\begin{quote}\\footnotesize\\ttfamily\\raggedright}\n  {\\end{quote}}\n"
            "\\title{Supplementary material\\\\"
            "\\large Reporting Discordance and Narrative-Derived Behavioral Cues in "
            "Electric Scooter Injury Surveillance Under a Safe Systems Lens}\n"
            "\\date{}\n\\author{}\n\n\\begin{document}\n\\maketitle\n\n"
            + body + "\\end{document}\n")
    (OUT / "supplementary.tex").write_text(text, encoding="utf-8")
    print(f"  wrote supplementary.tex ({len(text.splitlines())} lines)")

    numbers = table_numbers(body, tuple(parts["moved_tables"]))
    k_further = body[:body.index("\\label{sup:further}")].count("\\section{")
    lines = ["% GENERATED by src/build_supplementary.py.",
             "% Do not edit by hand; rerun the generator.",
             "% The printed number of each table of the Supplementary material that the",
             "% manuscript refers to. The supplement is a separate document, so these",
             "% numbers cannot be \\ref'd across; manuscript.tex reads this file instead."]
    for label, macro in {**SUPP_MACROS, **OWN_MACROS}.items():
        lines.append(f"\\newcommand{{\\{macro}}}{{S{numbers[label]}}}"
                     f"  % {label}")
    lines.append("% The appendices printed in the supplement: their tables and sections.")
    for label in parts["moved_tables"]:
        lines.append(f"\\newcommand{{\\{supp_macro(label)}}}{{S{numbers[label]}}}"
                     f"  % {label}")
    for i, label in enumerate(parts["moved_order"], 1):
        lines.append(f"\\newcommand{{\\{supp_macro(label)}}}{{S{k_further}.{i}}}"
                     f"  % {label}")
    (OUT / "supp_labels.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("  wrote supp_labels.tex")
    for label, k in sorted(numbers.items(), key=lambda kv: kv[1]):
        macros = {**SUPP_MACROS, **OWN_MACROS}
        macro = ("\\" + macros[label]) if label in macros else ""
        print(f"    Table S{k:<2d} {label:22s} {macro}")
    check_aux(numbers)
    return 0


if __name__ == "__main__":
    sys.exit(main())
