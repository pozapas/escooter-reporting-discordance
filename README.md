# Replication materials

Replication materials for the paper "Reporting Discordance and Narrative-Derived Behavioral
Cues in Electric Scooter Injury Surveillance Under a Safe Systems Lens", on KABCO-narrative
reporting discordance and narrative-derived pre-crash cues in police-recorded e-scooter
crashes in Texas, 2021 to 2025.

## What is here

| Path | Contents |
|---|---|
| `docs/prespec.md` | The dated pre-specification: outcome, candidate covariates, inclusion and merge rules, specification ladder, model family and cue measurement. |
| `docs/prespec_deviations.md` | Every departure from the pre-specification, with its date and reason. |
| `docs/coding_guide.md` | The written guide the three coders used (see below for its worked examples). |
| `src/` | The full analysis pipeline, from the raw extract to every number, table and figure. |
| `data/` | Derived data and the outputs of every analysis step. |
| `data/logs/` | Per-call extraction logs, reduced to model, provider, settings, prompt version, prompt hash and labels. |
| `data/SCHEMA.md` | The fields of the Texas CRIS extract the pipeline reads. |
| `results/tables/` | The main-text tables, as generated LaTeX. |
| `results/figures/` | The figures. |
| `results/appendix_tables/`, `results/supplementary_tables/` | Every appendix and supplementary table, as generated LaTeX, one file per table. |
| `requirements.txt` | The Python environment (Python 3.12). |
| `LICENSE` | The MIT License, under which the code is released. |

`data/numbers.json` is the single source of every number the paper prints. Each entry
records the analysis output it was read from. The generated tables use the table macros of
the manuscript preamble (`\hcell`, `\altrow`, `\hltint` and the like), so they are
meant to be read or included in a document that defines them.

## What is not here, and why

* **The Texas CRIS extract.** The crash records come from the Texas Crash Records
  Information System, which TxDOT releases under terms that restrict onward distribution.
  The narrative text is therefore not redistributed in any form, and neither are the
  person-level frames built from the extract (`base_frame.csv`, `model_frame.csv`), which
  hold coordinates, age, gender and ethnicity per rider.
* **Free text that could quote a narrative.** The extraction logs are released without the
  model's free-text rationale, and the coders' exports without their free-text notes.
* **The narrative quotations in the coding guide.** The guide the coders used illustrates
  each rule with short quotations from crash narratives. In `docs/coding_guide.md` each
  quotation is replaced by a constructed example that makes the same point and receives the
  same label (`src/guide_examples.py`). The rules, labels and reasoning are unchanged.
* **Credentials and caches.** API keys are read from a local `.env` file that is not
  released. Downloaded public data (OpenStreetMap tiles, Census files) are re-fetched by
  the scripts.

## Obtaining the crash extract

Request from TxDOT, through the CRIS query and data request service, all crashes from
2017 to 2025 whose unit records flag an e-scooter (`E_Scooter_ID = Yes`), in wide format
with the crash, primary unit, primary person, secondary unit and secondary person fields
and the investigator narrative. `data/SCHEMA.md` lists the fields the pipeline reads.
Save the extract as `data/E_Scooter_IDYes2017_2025a.xlsx`. The pipeline keeps the 2021 to
2025 crashes in which the second unit is a motor vehicle. The released crash identifiers
(`Crash_ID`) let a reader confirm that a new extract covers the same records.

## Public inputs fetched by the scripts

* OpenStreetMap features, through the Overpass API (`osm_download.py`, `osm_features.py`).
  `data/osm_features.csv` holds the snapshot used in the paper, extracted on
  14 September 2026.
* American Community Survey table B17021, five-year estimates for 2019 to 2023 and
  2017 to 2021, and TIGER 2023 block groups for Texas (`acs_features.py`). A Census API key
  is read from `CENSUS_API_KEY`.
* 2020 Census urban areas: the shapefile `tl_2020_us_uac20.shp` (with its companion files)
  and the Census urban-area list `ua_list_all.txt`, placed in `data/shapefiles/`.
* The language-model extraction runs through OpenRouter (`extract.py`), with the key read
  from `OPENROUTER_API_KEY`. The released logs and labels let every later step run without
  repeating it.

## Running the pipeline

Create the environment with `pip install -r requirements.txt`, then run each step from the
repository root with `python src/<script>.py`. Steps 1 and 2 need the crash extract;
every later step can start from the released files in `data/`, except where it reads
`base_frame.csv` (step 1) or `model_frame.csv` (step 5), which are rebuilt from the
extract.

1. **Base frame.** `build_base.py`.
2. **Samples, extraction and labels.** `sampling.py` (the frozen guide-development and
   validation samples), `masking.py`, `extract.py`, `route_a.py`, `fuse.py`,
   `validation_metrics.py`, `build_final_labels.py`.
3. **Area features.** `osm_download.py`, `osm_features.py`, `acs_features.py`.
4. **Reporting audit.** `audit_indicator.py`, `audit_model.py`, `audit_final.py`,
   `audit_recovery.py`, `audit_power.py`.
5. **Severity models.** `model_frame.py`, `severity.py`, `severity_appendix.py`,
   `transferability.py`, `benchmark.py`.
6. **Sensitivity analyses.** `label_sensitivity.py`, `m2_expected_likelihood.py`,
   `random_parameters.py`, `ordinal_alternatives.py`, `alt_model_compare.py`,
   `masking_compare.py`, `data_extra.py`, `urbanized_sensitivity.py`,
   `supporting_analyses.py`, `safe_system_evidence.py`, and `excluded_cues.py` (the post hoc
   refit with the cues the event rule excludes).
7. **Numbers, tables and figures.** `build_numbers.py` writes `data/numbers.json`; then
   `build_tables.py` writes `results/tables/`, `build_figures_main.py` writes
   `results/figures/`, and `build_appendix.py` and `build_supplementary.py` write the
   appendix and the supplementary material to `results/`. `build_tables.py` and
   `build_appendix.py` read only released files. `build_numbers.py` and
   `build_figures_main.py` also read `model_frame.csv` and `base_frame.csv`, and
   `build_supplementary.py` reads the narratives in `base_frame.csv` and the extraction logs
   as `extract.py` writes them, before their reduction for release.

`test_severity.py` checks the ordinal model code on simulated data with known truth, and
`verify_masking.py` checks from the logs which text each extraction pass received. Random
seeds are fixed in each script and recorded in its output file.

## License

The code is released under the MIT License; see `LICENSE`.
