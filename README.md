#  Socioeconomic Use Case Example: Evaluating the Mapping from ZfKD Format to OMOP CDM

## :bookmark: Table of Contents
- [Paper](#page_facing_up-Paper)
- [Repository Overview](#bar_chart-Repository-Overview)
- [Setup and Configuration](#gear-Setup-and-Configuration)
- [OMOP Implementation Specifics](#hammer_and_wrench-OMOP-Implementation-Specifics)
- [Directory Structure](#file_folder-Directory-Structure)
- [Required data sources for socioeconomic information](#earth_africa-Required-data-sources-for-socioeconomic-information)
- [References](#books-References)
- [License](#scroll-License)

<br><br>
## :page_facing_up: Paper
Titel: *From National to International: Evaluation of OMOP CDM Mapping for German Cancer Registry Data Using a Socioeconomic Use Case*


- This repository contains the analytical/statistical part of the study described in the paper.
- The ETL pipeline for transforming German Cancer Registry data from the ZfKD format to the OMOP Common Data Model (CDM) is maintained in a separate repository: 

If you use this work, please cite our paper as:


<br><br>
## :bar_chart: Repository Overview
This repository contains the implementation of an analysis exploring socioeconomic influences on patients diagnosed with Diffuse Large B-Cell Lymphoma (DLBCL).

Patient data is extracted from a PostgreSQL database using a SQL script. The database stores data in the OMOP Common Data Model (CDM) format. Relevant socioeconomic data is obtained from publicly available online sources and mapped to the extracted OMOP data. The analysis focuses on the German population.

Specifically, the analysis includes:

- Descriptive statistics
- Kaplan-Meier survival estimates
- Cox Proportional Hazards (CoxPH) model results
- Sunburst Visualization

Since the analysis is based on data stored in the OMOP CDM, it can easily be adapted to new datasets or other entities within the OMOP structure with minimal modifications.

A key focus was assessing the quality of the mapping process (from ZfKD format to OMOP CDM). For validation, we used the publication [German perspective on the impact of socioeconomic status in diffuse large B-cell lymphoma](https://www.nature.com/articles/s41408-024-01158-9) as a benchmark and aimed to compare results accordingly.


<br><br>
## :gear: Setup and Configuration
Before running the analysis, please complete the following preparation steps to set up the environment and required data.

### Preparation

1. The OMOP CDM is stored in a PostgreSQL database. Connection parameters are configured in the config.json file.
> Note: Before running the analysis, update the values in config.json to match your database settings.
2. Create the following empty folders: (1) src/data, (2) src/data/socioeconomic_info, (3) src/results.
3. Save Excel Files from data sources (see below in section "Required data sources for socioeconomic information") in src/data/socioeconomic_info.

### Run the Analysis
This repository includes a development container to simplify setup and ensure a consistent environment. You can use it to run the project without manually installing dependencies.

Once the preparation steps are completed, run the analysis with the following command:
```python
python main.py 
```


<br><br>
## :hammer_and_wrench: OMOP Implementation Specifics
The `provider_id` in the `Provider` table is composed as follows in our mapping:

`provider_id = <district_id> + "_" + <federal_state>`
- <district_id> = person's place of residence (in german: Kreis-ID e.g. 1001 for Flensburg, 9572 for Erlangen-Höchstadt)
- <federal_state> = place of treatment

Since the `district_id` is required for our analysis, it is extracted in the first step of the cohort definition `prepare_DLBCL_SES_cohort` located in `utils/diffuse_large_b_cell_lymphoma.py`.

> **Note:** If your OMOP CDM stores the `district_id` elsewhere, please adjust the logic accordingly. Ensure that only valid district_ids are used in your analysis. 


<br><br>
## :file_folder: Directory Structure
1. Create the following empty folders: (1) src/data, (2) src/data/socioeconomic_info, (3) src/results
2. Get csv from data sources (see below in section "Required data sources for socioeconomic information")
   
(*) need to be added
```text
├── src
    ├── data (*)
        └── socioeconomic_info (*)
            ├── district_premature_mortality.csv (*)
            ├── gisd_federal_state_district.csv (*)
            ├── one_year_death_probabilities_general.csv (*)
            ├── plz_certified_hospital_center_germany.csv (*)
            └── plz_district_mapping.csv (*)
    ├── results (*)
    ├── sql
        └── diffuse_large_b_cell_lymphoma.sql
    ├── utils
        ├── data_loader.py
        ├── sunburst_dlbcl.py
        └── diffuse_large_b_cell_lymphoma.py
    ├── main.py
    └── validierung_dlbcl_paper.ipynb
├── config.json    
├── README.md
└── requirements.txt
```


<br><br>
## :earth_africa: Required data sources for socioeconomic information
The socioeconomic data used in this project refers to publicly available statistics on the German population. Detailed information about the data files can be found in the associated publication (see Citation)

| File name                                     | Columns                                                               | Seperator | Source (accessed August 07, 2025) |
|-----------------------------------------------|-----------------------------------------------------------------------|-----------|-----------------------------------|
| `district_premature_mortality.csv`            | district_id; year; sex; premature_mortality                           | semicolon | https://www.inkar.de/ Vorzeitige Sterblichkeit Frauen und Männer (category: SDG-Indikatoren für Kommunen) |
| `gisd_federal_state_district.csv`             | district_id, district_name, year, gisd_score, gisd_5, gisd_10, gisd_k | comma     | https://github.com/robert-koch-institut/German_Index_of_Socioeconomic_Deprivation_GISD/blob/main/GISD_Release_aktuell/Bund/GISD_Bund_Kreis.csv |
| `one_year_death_probabilities_general.csv`    | year; age; sex; one_year_death_prob_general                           | semicolon | https://www.mortality.org/Country/Country?cntr=DEUTNP Age-Specific Death Rates: 1x1 |
| `plz_certified_hospital_center_germany.csv`   | plz_certified_hospital_center                                         | semicolon | https://www.oncomap.de/cnetworks/cnoncos?selectedCountries=[Deutschland]&selectedCerttypes=[DKG]&showMap=1 |
| `plz_district_mapping.csv`                    | plz; district_id                                                      | semicolon | https://opendata.rhein-kreis-neuss.de/explore/dataset/nrw-postleitzahlen/table/ plz = post code, district_id = Kreis code |


<br><br>
## :books: References
As we reimplemented the analysis from **Ghandili, S., Dierlamm, J., Bokemeyer, C. et al.**, we would like to acknowledge the following publication:
> **Ghandili, S., Dierlamm, J., Bokemeyer, C. et al.**  
> *A German perspective on the impact of socioeconomic status in diffuse large B-cell lymphoma.*  
> Blood Cancer Journal (2024), 14:174  
> :book: [DOI: 10.1038/s41408-024-01158-9](https://doi.org/10.1038/s41408-024-01158-9)


<br><br>
## :scroll: License
This project is licensed under the Apache License, Version 2.0.
