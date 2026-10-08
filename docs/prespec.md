# Covariate pre-specification

**Frozen:** 2026-09-14, before any model was estimated.
**Scope of the modeled sample:** police-recorded crashes in Texas CRIS, 2021 through 2025, in which an electric scooter was one unit and a motor vehicle was the second unit.

This document fixes the candidate covariate set, the coding of every candidate, the rules that decide whether a candidate enters a model, and the rules that merge sparse categories. It is dated and committed before estimation so that no covariate is selected, merged, or dropped on the basis of an estimated coefficient. Any deviation from this document is recorded in `docs/prespec_deviations.md` with its reason and is reported in the manuscript.

---

## 1. Outcome

**Primary outcome.** Crash severity from `Crash_Sev_ID`, collapsed to three ordered classes:

| Class | CRIS levels |
|---|---|
| O | Not Injured |
| BC | Possible Injury; Non-Incapacitating Injury |
| KA | Incapacitating Injury; Killed |

Records with `Crash_Sev_ID = Unknown` are excluded, and the count excluded is reported.

**Secondary outcome.** The four-class ordering (O, C, B, KA) is estimated in the ordinal primary form and reported in the appendix. The five-class KABCO ordering is not estimated; the reason (the number of fatalities) is stated in the text with its count.

---

## 2. Candidate covariate set

Every candidate below is defined here. Reference categories are marked "(ref)".

### 2.1 Roadway and environment

| Candidate | Coding | Source field |
|---|---|---|
| Speed-limit band | 30 mph or less (ref); 35 to 40 mph; 45 mph and over; unknown | `Crash_Speed_Limit`, with `-1` and missing mapped to unknown |
| Weekend | 1 if Saturday or Sunday | `Day_of_Week` |
| Lighting | daylight, dusk, or dawn (ref); dark lighted; dark not lighted or unknown | `Light_Cond_ID` |
| Intersection relation | non-intersection (ref); intersection or intersection-related; driveway access | `Intrsct_Relat_ID` |
| Traffic control | none, other, or unknown (ref); signal; stop sign | `Traffic_Cntl_ID` |
| Road class | city street (ref); state, US, interstate, FM, or county; non-trafficway | `Hwy_Sys_ID` / road designation fields |
| Urbanized | 1 if the crash point falls inside a 2020 Census Urban Area with population 200,000 or more; city-lookup fallback where coordinates are absent | 2020 UA shapefile |

### 2.2 Spatial context (built from public sources)

| Candidate | Coding | Source |
|---|---|---|
| Bike facility on nearest segment | 1 if the roadway segment nearest the crash point carries a cycle facility | OpenStreetMap via `osmnx`, extraction date recorded |
| Drivable centerline length within 100 m | kilometers, continuous | OpenStreetMap |
| Block-group poverty share | share of individuals below poverty among those for whom poverty status is determined; continuous | ACS 5-year 2019 to 2023, joined through TIGER 2023 block groups |
| Coordinate missing | 1 if the record has no usable latitude and longitude | CRIS |

Bike-facility presence within 50 m, 100 m, and 200 m buffers is computed and reported as a sensitivity; it is not the primary form.

### 2.3 Rider demographics

| Candidate | Coding | Source field |
|---|---|---|
| Gender | male (ref); female; unknown | `Prsn_Gndr_ID` |
| Ethnicity | White (ref); Black; Hispanic; other (Asian, other, American Indian); unknown | `Prsn_Ethnicity_ID` |
| Age | 25 to 54 (ref); under 25; 55 and over; unknown | `Prsn_Age` |

Unknown is an explicit category for all three. It is never pooled into the reference group. A complete-case sensitivity and a chained-equations multiple-imputation sensitivity (m = 20) are reported.

### 2.4 Striking vehicle and collision configuration

| Candidate | Coding | Source field |
|---|---|---|
| Striking-vehicle body type | car (ref); SUV; pickup, truck, or van; other or unknown | secondary-unit body style |
| Collision manner | angle (ref); single motor vehicle straight; turning-involved; same-direction; other | `FHE_Collsn_ID` |

### 2.5 Officer-coded contributing factors (M0c layer)

Binary indicators derived from `Contrib_Factr_1_ID`, `Contrib_Factr_1_ID_2`, `Contrib_Factr_2_ID`, and `Othr_Factr_ID`:

rider failed to yield; driver failed to yield; driver inattention; disregard of signal or stop; speed-related; driveway entry or exit; wrong side or wrong way.

Each indicator is reported with its count.

### 2.6 Narrative-derived pre-crash cues (M1 layer)

The ten mechanism labels, each a binary indicator: wrong way; sidewalk transition; driveway or alley; failure to yield; signal violation; swerve or loss of control; dooring; distraction or impairment; vehicle turning across; lane positioning.

The cues are non-exclusive binaries. No cue is omitted as a reference category. Each cue enters or is excluded under the rules in section 3.

---

## 3. Inclusion rules

A candidate enters a specification if and only if all three conditions hold.

1. **Event count.** The smallest cell formed by the indicator and a severity class contains at least 10 observations. For a categorical variable, this applies to every non-reference category after the merging rules in section 4.
2. **Collinearity.** The variance inflation factor in the pooled design matrix is below 5.
3. **Conceptual role.** The variable is a Safe Systems lever or an established confounder in the police-record severity literature, and that role is stated.

No candidate is dropped, retained, or re-coded on the basis of the sign, magnitude, or significance of its estimated coefficient.

Appendix Table A1 of the manuscript lists every candidate, its counts by severity class, its VIF, and its disposition under these rules.

### 3.1 Order of application

The merging rules in section 4 are applied first, and the rules in section 3 are then tested on the merged categories. Where a merge is conditional on a cell count, the condition is evaluated, the merge applied, and the count re-tested, until section 4 offers no further merge.

### 3.2 Indicators exempt from the inclusion rules

The following indicators are retained in every specification at any cell count and at any variance inflation factor:

`speed_unknown`, `gender_unknown`, `age_unknown`, `coord_missing`, and the residual striking-vehicle level `veh_other_unknown`, the majority of whose records carry a body style coded Unknown.

Where ethnicity's sparse "other" level merges with "unknown" under section 4, the resulting `eth_other_or_unknown` level carries substantive categories as well as unrecorded ones. It is therefore **not** exempt and is screened on its merits like any other covariate; its counts and VIF are reported in Appendix Table A1.

These are not substantive covariates. Each exists so that records with an unrecorded value are not pooled into a reference category, which is the reporting defect the analysis sets out to avoid. Deleting such an indicator does not remove the affected records from the model; it silently assigns them to the reference group and attributes the reference group's mean to them. The exemption therefore covers both the event-count rule and the collinearity rule: unknown values in police records co-occur by their nature, because an officer who leaves one person field blank commonly leaves the others blank as well, so a collinearity threshold calibrated for substantive covariates does not govern them.

The coefficients on these indicators are reported in the tables and are **not interpreted**. What the analysis says about missingness is carried by the complete-case sensitivity and the multiple-imputation sensitivity in section 2.3, not by these coefficients. The co-occurrence of the three person-field unknown indicators is reported as a cross-tabulation, so that a reader can see how far they describe the same records.

---

## 4. Merging rules for sparse categories (fixed here)

Applied in this order, and only until the 10-event rule in section 3 is satisfied.

- **Lighting.** "dark not lighted", "dark unknown lighting", "other", and "unknown" merge into one level. If that level then falls below 10 events in a severity class it merges with "dark, lighted" into a single **dark** level, so that the contrast the variable carries is daylight against dark rather than daylight against a residue. The merged level is named "dark or unknown lighting".
- **Ethnicity.** Asian, American Indian, and other merge into "other". If "other" then still falls below 10 events in a severity class, it merges with "unknown" to form "other or unknown", and the merged level is exempt under section 3.2. Ethnicity recorded as unknown is never merged into White, Black, or Hispanic.
- **Age.** The bands are under 25, 25 to 54 (reference), and 55 and over. If "55 and over" falls below 10 events in a severity class the threshold moves to 50 and over; if that still falls below 10 it moves to 45 and over. The chain stops there: a threshold below 45 would no longer describe an older-rider group and the band would be retained at 45 and over with its counts reported whatever they are. Every step taken is reported with the counts that triggered it. The under-25 threshold does not move and is supported by cited sources.
- **Striking-vehicle body type.** Two-door and four-door cars are one category. Truck, van, and truck tractor merge with pickup. Motorcycle, bus, emergency vehicles, and unlisted merge into "other or unknown".
- **Collision manner.** The categories are assigned in this order, and the first that matches wins: (i) **same-direction** for any configuration whose CRIS label begins "Same Direction", including its turning variants; (ii) **turning-involved** for any remaining configuration naming a left turn, right turn, or U-turn; (iii) **single motor vehicle straight**; (iv) **angle** for the remaining angle configurations; (v) **other**. Same direction takes precedence over turning because the relative-path geometry, not the presence of a turn, is what the category is meant to capture. If a resulting level falls below 10 events in a severity class it merges into "other", and if "other" then still falls below 10 it merges into the "angle" reference.
- **Road class.** State, US, interstate, farm-to-market, and county roads merge into one non-city-street trafficway category. If "non-trafficway" falls below 10 events in a severity class it merges into that same category, and the merged level is named "not a city street".
- **Traffic control.** Marked lanes, crosswalk, and all other coded controls merge into the "none, other, or unknown" reference; signal and stop sign remain separate.
- **Intersection relation.** If "driveway access" falls below 10 events in a severity class it merges with "intersection or intersection-related" into "intersection or driveway access", because both describe a conflict point rather than open roadway.
- **Officer-coded contributing factors.** These are individually coded binaries and are not merged with one another. A factor that fails the event rule is excluded and reported as excluded with its counts. The threshold is not moved to admit a factor that falls short of it.

---

## 5. Exclusions fixed in advance, with the reason

| Excluded | Reason |
|---|---|
| Weather condition | Insufficient variation; the count in the dominant category is stated in the text |
| Surface condition | Insufficient variation; the count in the dominant category is stated in the text |
| Helmet use | Almost every record is coded "Not Applicable"; the count is stated |
| Alcohol test result | A result is present in a small number of records; the count is stated |
| Drug test result | As above |
| Time-of-day bins in the severity models | Lighting carries the same information and is the established covariate; nighttime is retained in the audit's reporting-context vector |

Nighttime is retained in the reporting-discordance audit's reporting-context vector. In the severity models, lighting replaces nighttime unless the VIF rule permits both, in which case lighting is used and the choice is stated.

---

## 6. Specification ladder

| Model | Contents |
|---|---|
| M0 | Structured covariates: roadway and environment, spatial context, demographics, striking vehicle and collision configuration |
| M0c | M0 plus officer-coded contributing factors |
| M1 | M0c plus the ten narrative-derived pre-crash cues |
| M2 | M1 under the expected-likelihood latent-severity formulation |

Explanatory gain along the ladder is reported as the nested likelihood-ratio test, adjusted rho-squared, AIC, BIC, and repeated stratified cross-validated log-loss (10 repeats by 5 folds, with fold-level values shown).

---

## 7. Model family

Ordered logit is fitted first. The Brant test is run per covariate and as an omnibus test.

- If the omnibus test rejects at the 5 percent level, the primary specification is a partial proportional-odds model in which only the covariates that fail the Brant test receive category-specific coefficients, and the primary is confirmed by BIC against ordered logit.
- If the omnibus test does not reject, ordered logit is the primary specification.

The multinomial logit is reported in the appendix as a robustness specification with Hausman-McFadden and Small-Hsiao tests of the independence of irrelevant alternatives. Average marginal effects for all three outcome probabilities, with 1,000 Krinsky-Robb draws, are the interpretive basis. All intervals are at the 95 percent level.

---

## 8. Narrative-cue measurement fixed in advance

The primary severity models use the final adjudicated hard 0/1 cue labels. Four sensitivities are pre-specified:

1. LLM-only labels without adjudication.
2. Calibrated cue probabilities as plug-in regressors.
3. Fifty label draws from the calibrated probabilities, combined by Rubin's rules.
4. Human labels on the validation subset in a reduced model.

---

## 8b. Actor attribution for the two shared cues (pre-specified sensitivity)

Two of the ten cues describe an action that either party can perform: failure to yield the right of way, and signal or stop-sign violation. The primary models use the cue as locked, a binary indicator that the narrative documents such an action by either party. In addition, the extraction schema and the human coding instrument record which party the narrative attributes the action to (scooter rider; motor-vehicle driver; both; unclear), and this attribution is validated in the same exercise as the labels.

Pre-specified sensitivity: the primary specification is re-estimated with each of these two cues split into a rider-attributed indicator and a driver-attributed indicator, with records coded "both" entering each and records coded "unclear" entering a third indicator. The split enters only if each resulting indicator satisfies the 10-event rule in section 3; otherwise the attribution is reported as a descriptive cross-tabulation only. The result is reported in the appendix beside the primary. The reason for pre-specifying it is that a cue combining rider and driver actions cannot be read as a rider behavioral lever, and the distribution of attribution is not known before coding.

## 9. Sample-size reporting

A table reports the count of each narrative cue by severity class and the events-per-parameter ratio for every specification in the ladder.

---

*End of pre-specification. Frozen 2026-09-14.*
