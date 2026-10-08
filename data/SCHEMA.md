# Dataset Schema — E-Scooter Crash Data (Texas CRIS)

| Attribute | Value |
|-----------|-------|
| **Source** | Texas Crash Record Information System (CRIS), filtered for E-Scooter crashes |
| **File** | `E_Scooter_IDYes2017_2025a.xlsx` |
| **Records** | 527 crashes (all confirmed E-Scooter involved, `E_Scooter_ID = Yes`) |
| **Time Span** | 2021–2025 |
| **Unit of Analysis** | Crash-level with linked unit (vehicle/scooter) and person records |
| **Geography** | Texas (county, city, GPS coordinates) |
| **Structure** | Wide format — crash + primary unit + person + secondary unit + secondary person in one row |

## Key Variables

| Variable | Type | Values / Range |
|----------|------|----------------|
| Crash_Sev_ID | Categorical | Killed (11), Incapacitating Injury (71), Non-Incapacitating (260), Possible Injury (103), Not Injured (77) |
| Prsn_Injry_Sev_ID | Categorical | Killed (9), Incapacitating (68), Non-Incapacitating (253), Possible (103), Not Injured (57) |
| Death_Cnt / Tot_Injry_Cnt | Numeric | Deaths: 0–1; Total injuries: 0–4 (mean 0.87) |
| Sus_Serious_Injry_Cnt | Numeric | Suspected serious injury count |
| Prsn_Age | Numeric | Age 5–97, mean 27.2 years, median 23 — skewed young |
| Prsn_Gndr_ID | Categorical | Male (374), Female (125), Unknown (6) |
| Prsn_Ethnicity_ID | Categorical | White (195), Black (129), Hispanic (127), Asian (29) |
| Crash_Speed_Limit | Numeric | Mean 30.3 mph, range –1 to 70 mph |
| Road_Type_ID | Categorical | 2-Lane 2-Way (216), 4+ Lane Divided (135), 4+ Lane Undivided (62) |
| Rural_Urban_Type_ID | Categorical | Urbanized 200k+ (73), Large Urban 50–199k (16), Small Urban (11), Rural (8) |
| Wthr_Cond_ID | Categorical | Weather at time of crash |
| Light_Cond_ID | Categorical | Lighting condition |
| Surf_Cond_ID | Categorical | Road surface condition |
| Traffic_Cntl_ID | Categorical | Traffic control type |
| Intrsct_Relat_ID | Categorical | Intersection relationship |
| FHE_Collsn_ID | Categorical | First harmful event / collision type |
| Contrib_Factr_1_ID / _2_ID | Categorical | Contributing factors (primary and secondary unit) |
| Prsn_Alc_Rslt_ID / Prsn_Drg_Rslt_ID | Categorical | Alcohol/drug test result |
| Prsn_Helmet_ID | Categorical | Helmet use (mostly "Not Applicable" — indicates scooter riders) |
| Latitude / Longitude | Numeric | Crash GPS location |
| Investigator_Narrative | Free text | Crash narrative |
| Day_of_Week / Crash_Date / Crash_Time | Temporal | Full temporal detail |
| Secondary unit variables (_2 suffix) | Mixed | Characteristics of the other involved party |

## Notable Characteristics
- **All 527 records are confirmed E-Scooter crashes** — highly specific dataset
- Strong demographic detail: age, gender, ethnicity enable equity analysis
- Geospatial coordinates available for spatial analysis
- Links scooter rider to the opposing vehicle unit in same row
- Texas-only but covers urban, suburban, and rural contexts
- Relatively small n=527 — suitable for logistic regression, survival models, association analysis