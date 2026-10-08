# Departures from the pre-specification

A running record of every point where the analysis departs from `prespec.md`, or where
the pre-specification was silent and a choice had to be made. Each entry states what
the pre-specification says, what was done, when, and why.

The pre-specification was written and frozen on 2026-09-14, before any model was
estimated. It was not backdated. This file exists so that a reader can tell which
choices were made in advance and which were made afterwards, without taking anyone's
word for it.

**Status of the numbers below.** Every entry was re-checked on 2026-10-05 against the
final labels (the coders' majority vote where it exists, the reference run elsewhere) and
the primary specification as it now stands, which includes the urbanized-area indicator
of entry 7. Where an entry first recorded preliminary figures on 2026-09-14, the decision
rule it describes is unchanged and only its figures are updated; the earlier figures are
noted where the comparison matters.

---

## 1. Section 8b, actor attribution: the split does not enter the models

*Amended 2026-10-05 on the final labels (`data/data_extra.json`, key
`actor_attribution`; in `numbers.json` under `sensitivity.data_extra.actor_attribution`).
The preliminary version of this entry, computed on 2026-09-14 from the reference
extraction run, is superseded and its counts are no longer used.*

**What section 8b says.** The two cues either party can perform (failure to yield the
right of way; signal or stop-sign violation) are re-estimated split into a
rider-attributed indicator and a driver-attributed indicator, with records coded
"both" entering each and records coded "unclear" entering a third indicator. "The split
enters only if each resulting indicator satisfies the 10-event rule in section 3;
otherwise the attribution is reported as a descriptive cross-tabulation only."

**Source of the attribution.** The final labels carry the attribution of their own
source. For a human-coded narrative (the validation sample and the narratives flagged for
human coding) it is the coders' majority vote, with no discussion step; for every other
narrative it is the reference extraction run. A record enters only when its final cue is
1. A cue-positive record with no attribution recorded counts as "unclear", as section 8b
directs. No attribution vote was tied, so no record carries an arbitrary attribution.

**What the attribution looks like.** Counts are of rider records, by severity class,
with "both" entering each of the first two rows as section 8b directs.

### failure_to_yield (170 cue-positive records)

| indicator | n | O | BC | KA | smallest cell | meets the 10-event rule |
|---|---|---|---|---|---|---|
| rider-attributed | 93 | 14 | 62 | 17 | 14 | yes |
| driver-attributed | 75 | 10 | 57 | 8 | 8 | no |
| unclear | 5 | 0 | 3 | 2 | 0 | no |

### signal_violation (89 cue-positive records)

| indicator | n | O | BC | KA | smallest cell | meets the 10-event rule |
|---|---|---|---|---|---|---|
| rider-attributed | 63 | 10 | 44 | 9 | 9 | no |
| driver-attributed | 27 | 4 | 19 | 4 | 4 | no |
| unclear | 0 | 0 | 0 | 0 | 0 | no |

Of the 170 records with a documented yield failure, 52.9 percent attribute it to the
rider alone, 42.4 percent to the driver alone, 1.8 percent to both and 2.9 percent are
unclear. For signal violation the shares are 69.7 percent rider, 29.2 percent driver and
1.1 percent both.

**What was done.** Neither cue's split enters the severity models. Both are reported as
descriptive cross-tabulations, which is the contingency section 8b names, and no split
model was estimated. For `failure_to_yield` the driver-attributed indicator has 8 events
in the KA class and the unclear indicator has none in O, so two of the three indicators
fail the rule. For `signal_violation` all three indicators fail it.

**Why this is no longer a drafting question.** The preliminary version of this entry
argued that the literal rule was unreachable for `failure_to_yield`, because only the
residual "unclear" indicator failed and that indicator can never reach ten events per
class. On the final labels the driver-attributed indicator fails on its own, so the
outcome is the same under the literal text and under any reading that sets the
"unclear" indicator aside. The rule decided the disposition without the drafting gap
having to be resolved.

**Agreement of the attribution with the coders.** Where the coders' majority and the
reference run both code a cue positive and the majority attribution is not tied, the
reference run gives the same attribution for `failure_to_yield` in 28 of 29 validation
narratives (96.6 percent, Cohen's kappa 0.933) and 34 of 37 when the narratives flagged
for human coding are pooled in (91.9 percent, kappa 0.844). For `signal_violation` the
figures are 23 of 25 (92.0 percent, kappa 0.740) and 28 of 31 pooled (90.3 percent, kappa
0.656). In the flagged set alone the coders used only "rider" for `signal_violation`, so
agreement there (5 of 6, 83.3 percent) is reported without a kappa. The flagged set was
selected for disagreement and is not a probability sample, and the agreement figures are
unweighted.

**What this costs the paper.** The claim that a cue mixing rider and driver actions
cannot be read as a rider behavioral lever, which is the reason section 8b exists, still
stands, and the cross-tabulation supports it, since driver-attributed records are 42.4
percent of documented yield failures. What is lost is the ability to say whether the
rider-attributed and driver-attributed versions carry different severity gradients. That
limitation is stated in the text rather than worked around.

---

## 2. Section 7, the Brant test: what happens when the omnibus rejects but no covariate does

*Preliminary — recorded 2026-09-14, before the adjudicated labels were available.*

**What section 7 says.** The omnibus test governs the model family: if it rejects at the
5 percent level the primary specification is a partial proportional-odds model, confirmed
by BIC against ordered logit; if it does not reject, ordered logit is primary. The
per-covariate tests govern which covariates receive category-specific coefficients.

**The gap.** Section 7 does not say what happens when the omnibus rejects but no
covariate fails individually. The omnibus is the sum of the per-covariate statistics, so
with thirty-five covariates it can reject on degrees of freedom alone while no single
covariate departs from proportional odds. A partial proportional-odds model would then
free nothing and be arithmetically identical to ordered logit, which makes the
BIC confirmation a comparison of a model against itself.

**What was decided, and when.** In that case ordered logit is reported as primary, on
the grounds that the two models are the same model and the simpler description of it is
the honest one; the omnibus rejection is reported in the text with the observation that
it is driven by degrees of freedom rather than by any identifiable covariate. This was
written into `src/severity.py` on 2026-09-14 **before** the test was run on the real
frame, and it is stated in the module's own docstring and in the test output, so the
rule cannot be re-chosen after seeing the result.

**What actually happened.** The question did not arise. On the final labels and the
35-covariate primary specification the omnibus rejects (likelihood ratio 74.68 on 35
degrees of freedom, p = 0.0001) *and* twelve covariates fail individually, so there is a
non-empty set to free. Ordered logit is nonetheless primary, for the separate reason in
entry 5: BIC does not confirm the partial proportional-odds model. On the preliminary
labels of 2026-09-14 the figures were 83.09 on 33 degrees of freedom with thirteen
failing covariates, and the decision was the same.

An earlier version of this entry reported that the omnibus did not reject
(chi-squared 35.40, p = 0.36) and that no covariate failed. That was the Wald form of the
statistic, which entry 4 explains is biased toward exactly that conclusion and is no
longer the decision rule.

---

## 3. Section 3.2, exempt indicators: the exemption covers both screening rules

*Recorded 2026-09-14, at the time the amendment was made.*

This is the amendment described in section 3.2 itself rather than a departure from it,
noted here so the record is in one place. The 10-event rule and the variance-inflation
rule would each have deleted the missingness indicators, which the analysis reports by
design. Deleting a missingness indicator does not remove those records from
the model; it pools them into the reference category, which is the specific reporting
defect the indicators exist to prevent. Section 3.2 was therefore amended to name an exempt set —
`speed_unknown`, `gender_unknown`, `age_unknown`, `coord_missing`, `veh_other_unknown` —
exempt from both rules. The three person-field unknowns correlate at 0.80 to 0.83 with
one another, which is why the variance-inflation rule had to be covered as well as the
event rule.

The amendment was made before any model was estimated. The exempt indicators are
reported with their variance-inflation factors in the screening table so a reader can
see exactly what was exempted and by how much it exceeded the threshold.

Two of these exempt indicators, `gender_unknown` and `age_unknown`, later turned out to
be among the strongest violations of proportional odds on the frame, and `coord_missing`
among them too (entry 4). Had the event rule been allowed to delete them, that would
never have been visible.

---

## 4. Section 7: the Brant test is computed as a likelihood-ratio test, not in the Wald form

*Preliminary — recorded 2026-09-14.*

**What section 7 says.** "The Brant test is run per covariate and as an omnibus test."

**What was done.** The proportional-odds assumption is tested by likelihood ratio
against the project's own partial proportional-odds estimator: the omnibus frees every
covariate at once, and each per-covariate test frees one. The classical Brant Wald
statistic is computed and reported alongside for comparability with the literature, but
it does not decide anything.

**Why.** The Brant statistic in its usual form fits the two binary cumulative logits
separately and tests the equality of their slopes using `V11 + V22` as the variance of
the difference. The two fits use the same rows and are positively correlated, so the
correct variance subtracts twice the cross-fit covariance. Omitting it inflates the
denominator, shrinks the statistic, and makes the test conservative — and the direction
of that bias is toward retaining ordered logit, which is the conclusion the test is
supposed to adjudicate. A test must not be biased toward its own favoured answer.

On this frame the two forms do not merely differ, they disagree completely (final labels,
35-covariate primary specification; the preliminary figures of 2026-09-14 were 83.09 and
35.40 on 33 degrees of freedom):

| | statistic | df | p | covariates failing individually |
|---|---|---|---|---|
| likelihood ratio | 74.68 | 35 | 0.0001 | 12 |
| Wald, classical Brant form | 30.54 | 35 | 0.683 | 0 |

The gap is not only the missing covariance term. Several of the covariates involved are
sparse indicators, and for those the Wald statistic fails in a second and worse way: as
a coefficient grows, its standard error grows faster, so the Wald statistic falls toward
zero exactly when the effect is strongest. `gender_unknown` showed it plainly on the
preliminary labels of 2026-09-14 (figures not re-derived here) — a Wald
statistic of 0.014, p = 0.91, against a likelihood ratio of 9.78, p = 0.002. The partial
proportional-odds fit puts its coefficient at -1.47 on the first threshold and +1.10 on
the second. Those are opposite signs, and a proportional-odds model cannot represent
them: it averages the two into -1.30 and is wrong at both thresholds.

That pattern is substantively sensible rather than a numerical accident. Records with an
unknown rider gender concentrate in the no-injury class (17 of 26) and are also somewhat
over-represented among the most severe relative to the middle class. A covariate that
pushes probability mass toward both ends of the ordering is precisely what proportional
odds cannot accommodate, and it is what a missingness indicator should look like if
reporting is least complete both for trivial crashes and for catastrophic ones.

**Checks performed before accepting the likelihood-ratio result.** The partial
proportional-odds estimator warm-starts from the ordered-logit solution, so any slack in
the ordered-logit fit's own convergence would masquerade as evidence against
proportional odds. Fitting the partial proportional-odds model with nothing freed on the
full M1 covariate set reproduces the ordered-logit log-likelihood to 7e-06, a spurious
likelihood ratio of 0.0000. Every per-covariate fit and the omnibus fit converged, with
a maximum absolute gradient of 4.5e-06 at the omnibus optimum.

---

## 5. Section 7: no multiplicity correction, and this choice changes the primary model

*Preliminary — recorded 2026-09-14. **This entry records a choice made after seeing its
consequences.** Read it with that in mind.*

**What section 7 says.** Covariates that fail the Brant test at the 5 percent level
receive category-specific coefficients. Thirty-five covariates are tested. The
pre-specification says nothing about multiplicity.

**Why it matters here.** Under the null, one or two of thirty-five tests fail by chance
at 5 percent. Three defensible rules give three different sets, and — unusually — they
do not agree on the primary model:

| rule | covariates freed | BIC difference against ordered logit | primary model |
|---|---|---|---|
| uncorrected at 5 percent | 12 | +14.3 | ordered logit |
| Benjamini-Hochberg | 9 | +8.8 | ordered logit |
| Holm | 2 | -14.6 | partial proportional odds |

(Final labels, 35-covariate primary specification. The preliminary figures of 2026-09-14
were 13, 9 and 2 covariates with BIC differences of +17.2, +8.5 and -15.4; the pattern and
the decision are unchanged.)

The reason Holm flips the result is not that Holm is the better rule. Holm keeps only the
two strongest violations, so the likelihood gain per added parameter is at its highest,
and BIC rewards that. Freeing more covariates buys more log-likelihood but not enough per
parameter. The tension is between two different questions: which covariates violate
proportional odds, and which model predicts best.

**What was decided.** The pre-specification names a 5 percent threshold and no
correction, so the uncorrected set is used and ordered logit is primary. The other two
rules are reported as a sensitivity.

**Why this is the defensible branch despite having been chosen after the fact.** The
uncorrected rule is the literal pre-specified text. Switching to Holm would be a change
of rule, made after seeing that the change flips the answer, in the direction of a more
elaborate model — which is the exact pattern a pre-specification exists to prevent.
Following the frozen text is the only branch here that is not a post-hoc choice, and the
fact that it was verified afterwards to be the conservative one does not make it less
so.

**What is owed to the reader.** That the primary specification is not robust to the
multiplicity rule is a real limitation and is stated in the manuscript, with the table
above, rather than being resolved silently in favour of whichever model the text
preferred. The proportional-odds assumption is violated on this frame under any of the
three rules; what is uncertain is only whether relaxing it earns its parameters.

**This rule is now fixed for the adjudicated run.** The adjudicated labels will change
the cue coefficients and may change which covariates fail. The rule above — uncorrected
at 5 percent, BIC confirmation, sensitivity reported under Holm and Benjamini-Hochberg —
is applied mechanically to those results and is not revisited.

---

## 6. Section 7: the Small-Hsiao test is reported with its calibration check

*Recorded 2026-09-14; figures updated 2026-10-05 to the final labels and the 35-covariate
primary specification (`appendix.multinomial_robustness` in `data/numbers.json`).*

**What section 7 says.** The multinomial logit is reported in the appendix as a
robustness specification "with Hausman-McFadden and Small-Hsiao tests of the
independence of irrelevant alternatives."

**What was done.** Both tests are computed and reported. The Small-Hsiao result is
reported as a distribution over random splits, beside a calibration check of the test on
data where independence holds by construction. Because the test does not hold its
nominal level at this sample size, its rejections are not read as evidence about
independence in either direction. The departure from section 7 is this added check and
the reading it imposes, not the omission of the test.

**Why.** Two things surfaced, and the second settles it.

*The test depends on its random split.* Small-Hsiao partitions the sample at random, so
a single run reports the split as much as the data. Across twelve splits on the frame of
2026-09-14 the p-value for removing BC ran from 0.0000 to 0.9998. Any conclusion an
author wanted was available by choosing a seed. The implementation now runs twenty-five
splits and reports the distribution.

*The test does not hold its nominal level at this sample size.* That is measurable
rather than arguable. Data generated from a multinomial logit, so that independence
holds by construction, at the same shape as the real frame (522 rows, 35 covariates, 72
multinomial parameters, the observed class shares) is rejected by the test in 7 of 20
usable splits when O is removed, 6 of 8 when BC is removed and 8 of 20 when KA is
removed, or 35, 75 and 40 percent. A nominal 5 percent test should reject in about one
split in twenty. The multinomial logit carries 72 parameters and the test fits it on
random halves of about 261 rows, so the asymptotics have not arrived. The check is
`small_hsiao_calibration` in `src/severity_appendix.py`, with seed 20260914.

On 2026-09-14 the same implementation was also run on 3,000 simulated rows with 5
covariates and 12 multinomial parameters, where it rejected in 1 of 20, 1 of 20 and 0 of
20 splits, close to nominal. That control run is not stored in a result file; it is
recorded here as it was observed at the time. It indicates that the problem is the frame
rather than the code.

**What this means for the appendix.** On the real frame the test rejects in 19 of 25, 19
of 21 usable and 13 of 25 splits when O, BC and KA are removed (76.0, 90.5 and 52.0
percent); the preliminary figures of 2026-09-14 were 21 of 25, 18 of 23 and 13 of 25.
These rates are of the same order as those the test produces on data where independence
is true by construction, so they carry no information about independence. Reporting them
as a rejection would have put a false claim in the paper. Table E.2 gives the observed
rejection rates beside the calibration rates, and the appendix text states that at this
sample size the rejections do not separate a violation of independence from the behavior
of the test itself.

**The Hausman-McFadden test is reported as computed.** On the final labels it returns a
negative statistic for all three outcomes (-33.89 removing O, -13.09 removing BC, -1.29
removing KA, each on 36 degrees of freedom), so no p-value is defined; the preliminary
run had two negative statistics and 6.28 with p = 1.00 removing BC. A negative statistic
is a known finite-sample property of the test, arising because the difference of the two
covariance matrices need not be positive definite; McFadden reads it as evidence that
independence has not been violated. It is neither clipped to zero nor hidden.

**What is lost.** The pre-specification asked for evidence on independence of irrelevant
alternatives, and the honest answer is that this sample cannot supply it from these two
tests. This matters less than it would elsewhere. The multinomial logit is a robustness
specification that discards the ordering of the outcome, which is the property the
primary model is chosen to respect, so it was never a candidate for primary regardless of
how the tests came out.

---

## 7. Section 2.1, the urbanized-area indicator: omitted from screening, then added as specified

*Recorded 2026-10-05. Code: `src/model_frame.py` (`urban_area_200k`), `src/severity.py`;
reverse sensitivity in `data/urbanized_sensitivity.json` (`numbers.json`,
`sensitivity.urbanized`); the pre-addition outputs are kept in
`data/backup_pre_urbanized/`.*

**What section 2.1 says.** "Urbanized" is a roadway and environment candidate, coded 1 if
the crash point falls inside a 2020 Census Urban Area of population 200,000 or more, with
a city lookup where coordinates are absent. Every candidate in section 2 is screened
under the section 3 rules.

**What happened.** The column was not built into `model_frame.csv` in the first runs,
so it never reached screening. The omission was found on 2026-10-05, when
`src/data_extra.py` built the indicator for a spatial check. It was a departure by
omission, not a decision.

**What was done.** The pre-specification, not the results, decides whether a candidate
enters, so the indicator was built as section 2.1 defines it and screened under the
section 3 rules like every other candidate. It is 1 for 404 records (62 O, 282 BC, 60 KA)
and 0 for 118 (15 O, 81 BC, 22 KA), meets the 10-event rule, and has a variance inflation
factor of 1.25, so it is included. Records without coordinates take the majority value of
their city's located records, or the crash-record rural flag where the city cannot be
placed, which is the city lookup section 2.1 names. Every analysis that depends on the
covariate set was then rerun (M0 has 28 covariates, M0c 30, M1 35).

**What the addition changes.** The indicator is not itself associated with severity in M1
(coefficient -0.325, SE 0.256, p = 0.20; KA average marginal effect -0.043 [-0.117,
0.019]). Ordered logit stays primary, the cue block stays null (p = 0.977), no
transferability test rejects, and poverty share's KA interval excludes zero
(-0.212 [-0.345, -0.091]). The one conclusion that changes is absent crash coordinates:
its KA interval in M1 now includes zero under all four Krinsky-Robb seeds
(-0.080 [-0.147, 0.014]); before the addition it excluded zero under one of the four.
The model without the indicator is reported as a reverse sensitivity.

**Why this branch.** Leaving the indicator out because its omission was found late would
have kept a specification the pre-specification does not describe. Adding it with the
results in view carries its own risk, which is why the decision rests only on the section 3
screen, applied mechanically, and why both fits are reported.

---

## 8. The spatial transferability split: six named cities, set after the pre-specification

*Recorded 2026-10-05. No file dates the writing of the split definition apart from
the project's working log, where the transferability module is described between entries dated
2026-09-14 and 2026-09-16.*

**What the pre-specification says.** Nothing. `prespec.md` (sections 1 to 9) defines the
outcome, the candidates, the inclusion rules, the ladder, the model family and the cue
measurement. It does not define a transferability test or a spatial split. The analysis
plan asked for a spatial test of "urbanized versus rest" without fixing the
definition.

**What was done.** `transferability.spatial_test` splits the sample into the six cities
Austin, Houston, San Antonio, Dallas, Fort Worth and El Paso (by `City_ID`) against every
other record. The definition was written into the code after the pre-specification was
frozen and before the test was first run, and it is the one the main-text table reports
(257 against 265 records, 17 covariates estimable on both sides, bootstrap p = 0.108).
The same six cities define the `large_city` context variable of the random-parameter
analysis (entry 9). The split is not the urbanized-area indicator of section 2.1.

**The checks.** Because the six-city definition was chosen without a pre-specified rule,
two alternatives were run through the identical bootstrap test. Inside against outside a
2020 urban area of 200,000 or more gives 404 against 118 records, 6 covariates on both
sides, LR 7.30 on 8 degrees of freedom, bootstrap p = 0.511. The CRIS rural flag gives 63
against 459 records, 2 covariates, LR 0.74 on 4, bootstrap p = 0.938. Neither rejects,
but with 6 and 2 of 35 covariates estimable on both sides they do not test the M1
specification, and they are reported as descriptive checks only. The six-city test is
reported as exploratory.

**Note for the record.** An earlier `split` label in `transferability.json`
read "five largest reporting cities against the rest" while the code and every count used
six cities. The label was corrected in `src/transferability.py` on 2026-10-05 and the
stored file now reads "six most populous cities against the rest"; the six-city
definition is the one that was always run.

---

## 9. The random-parameter selection rule was set after the pre-specification

*Recorded 2026-10-05. The rule is stored verbatim in every random-parameter result file
under `procedure`; no file dates its writing separately from the runs.*

**What the pre-specification says.** Nothing. `prespec.md` does not mention random
parameters. The analysis plan sketched a rule in which the candidates would
be all narrative cues and the three strongest structured effects, tested one at a time
and then jointly, under normal and triangular distributions.

**What was done.** The rule in `src/random_parameters.py` was fixed before any
random-parameter model was fitted, and it differs from the plan's sketch in three ways.
Every one of the 35 M1 covariates is a candidate, made random one at a time, so the
candidate set is not chosen from estimated effect sizes, which "the three strongest
structured effects" would have required. A candidate is retained only if both the Wald
test of its standard deviation and the likelihood-ratio test against the fixed-parameter
M1, with the chi-bar-squared reference, are significant at 5 percent. Only the normal
mixing distribution is used. Heterogeneity in the means and variances is then tested one
context variable at a time (night, the six-city indicator of entry 8, and posted speed of
45 mph or more).

**Why it is recorded.** The rule was chosen after the pre-specification was frozen, so a
reader cannot verify from the pre-specification alone that it was not chosen with results
in view. It retained 9 of 35 candidates, and the paper reports the random-parameter model
as a robustness analysis, not as the primary specification.

---

## 10. Posted speed at the statutory line: an exploratory split added after the results

*Recorded 2026-10-06. Result file `data/safe_system_evidence.json`, written by
`src/safe_system_evidence.py`.*

**What the pre-specification says.** Section 2.1 codes posted speed in three bands, 30 mph
or less (reference), 35 to 40 mph, and 45 mph and over, with an unrecorded level.

**What was done.** Texas Transportation Code 551.352(a) allows a motor-assisted scooter
only on a street posted at 35 mph or less. The pre-specified bands put 35 and 40 mph in
one band, so they cannot show whether severity differs on either side of the line the law
draws. After the pre-specified results were known, the two recorded bands were replaced
by one indicator for a posted limit above 35 mph, keeping the unrecorded level, and the
model was refitted in the same partial proportional-odds form as the pre-specified Holm
set, with unrecorded age and the speed term freed. The pre-specified bands were refitted
in the same form beside it.

**Why it is recorded.** The split was chosen with the pre-specified results in view and
for its policy meaning, so it is reported as an exploratory sensitivity. The primary
specification and every pre-specified result are unchanged.
