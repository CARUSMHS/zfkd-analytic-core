import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import logrank_test
from utils import data_loader as lo

def prepare_DLBCL_SES_cohort(cohort):
    # 1. split district_id and register_id
    cohort["provider_id"] = cohort["provider_id"].astype(str)
    cohort["district_id"] = cohort["provider_id"].str[:-2]
    cohort["register_id"] = cohort["provider_id"].str[-2:]
    cohort["bundesland_id"] = cohort["provider_id"].str[:-5]

    # 2. replace gender_concept_id
    cohort['sex'] = cohort['gender_concept_id'].apply(lambda x: 1 if x == 8507 else 0)

    # 3. keep only patients in the age of 18-99  
    cohort["diagnosis_date"] = pd.to_datetime(cohort["diagnosis_date"])
    cohort["diagnosis_year"] = cohort["diagnosis_date"].dt.year
    cohort["age"] = cohort["diagnosis_year"] - cohort["year_of_birth"]
    cohort = cohort[(cohort["age"] > 17) & (cohort["age"] < 100)]
    cohort["age_60_plus"] = (cohort["age"] >= 60).astype(int)
    print(f"cohort size with valid age range: {cohort.shape}")
    
    
    # clean up dataset
    # remove 2025 due to too few values
    # Kaplan-Meier curves for register_id 10 show significant deviation – exclude to ensure comparability
    cohort = cohort[~cohort["diagnosis_year"].isin([2025])]
    cohort = cohort[~cohort["register_id"].isin(["10", "11", "12"])]
                    
    # 4. keep only observation times between 93 and 1827 days. If observation time exceeds 1827 days, cap it at 1827.
    cohort["observation_end_date"] = pd.to_datetime(cohort["observation_end_date"])
    cohort["survival_time"] = (cohort["observation_end_date"] - cohort["diagnosis_date"]).dt.days
    cohort = cohort[(cohort["survival_time"] > 92)]
    mask = cohort["survival_time"] > 1827
    cohort.loc[mask, "survival_time"] = 1827
    cohort.loc[mask, "death_flag"] = 0
    cohort = cohort.rename(columns={"death_flag": "survival_status"})
    print(f"cohort size with valid observation time: {cohort.shape}")

    # 5. keep only valid district_id
    cohort["district_id"] = cohort["district_id"].astype(int)
    cohort = cohort[cohort["district_id"] != 17000]
    print(f"cohort size with valid district_id: {cohort.shape}")

    # 6. add gisd score
    # GISD Score: 0 - 1 (1 = highest level of deprivation)
    gisd = pd.read_csv('src/data/socioeconomic_info/gisd_federal_state_district.csv')
    # gisd only available until 2021
    cohort["join_year"] = cohort["diagnosis_year"].apply(lambda x: 2021 if x > 2021 else x)
    cohord_gisd = pd.merge(
        cohort,
        gisd,
        left_on=["join_year", "district_id"],
        right_on=["year", "district_id"],
        how="inner"
    )
    cohord_gisd = cohord_gisd.drop(columns=['district_name', 'year', 'gisd_5', 'gisd_10', 'gisd_k', 'join_year'])
    print(f"control: cohort size after join with gisd: {cohord_gisd.shape}")

    # 7. add gisd SES (socio-economic status) -> low (5th), middle (2th - 4th), high (1th) 
    # GISD Quintile: 1 - 5 (5 = highest level of deprivation)
    cohord_gisd["gisd_quintile"] = pd.qcut(cohord_gisd["gisd_score"], q=5, labels=False) + 1
    cohord_gisd["ses"] = cohord_gisd["gisd_quintile"].map({
        1: "high",
        2: "middle",
        3: "middle",
        4: "middle",
        5: "low"
    })
    cohord_gisd = cohord_gisd.drop(columns=['gisd_quintile'])
    print(cohord_gisd.groupby("ses").size())
    
    # 8. Background mortality by age, sex, and year in the general population of Germany
    # source: https://www.mortality.org/Country/Country?cntr=DEUTNP
    # death rates only available until 2020 (because of corona: 2020 not included)
    cohord_gisd["join_year"] = cohord_gisd["diagnosis_year"].apply(lambda x: 2019 if x > 2019 else x)
    one_year_death_general = pd.read_csv('src/data/socioeconomic_info/one_year_death_probabilities_general.csv', sep=';')
    cohord_gisd = pd.merge(
        cohord_gisd,
        one_year_death_general,
        left_on=["join_year", "age", "sex"],
        right_on=["year", "age", "sex"],
        how="inner"
    )
    cohord_gisd = cohord_gisd.drop(columns=['year', 'join_year'])
    print(f"control: cohort size after join with one_year_prob: {cohord_gisd.shape}")
    
    # 9.Sex-specific premature mortality rate (deaths per 1,000 individuals below age 70) at the district level
    # source: Inkar(https://www.inkar.de/) -> premature mortality rate for men and women (category: SDG-Indikatoren für Kommunen)
    # premature mortality only available until 2017
    cohord_gisd["join_year"] = cohord_gisd["diagnosis_year"].apply(lambda x: 2017 if x > 2017 else x)
    premature_mortality = pd.read_csv('src/data/socioeconomic_info/district_premature_mortality.csv', sep=';')
    cohord_gisd = pd.merge(
        cohord_gisd,
        premature_mortality,
        left_on=["district_id", "join_year", "sex"],
        right_on=["district_id", "year", "sex"],
        how="inner"
    )
    cohord_gisd["premature_mortality"] = cohord_gisd["premature_mortality"].str.replace(',', '.')
    cohord_gisd = cohord_gisd.drop(columns=['year', 'join_year'])
    print(f"control: cohort size after join with premature_mortality: {cohord_gisd.shape}")
    
    # 10. At least one certified hospital center present in the district
    # source: https://www.oncomap.de/cnetworks/cnoncos?selectedCountries=[Deutschland]&selectedCerttypes=[DKG]&showMap=1
    certified_hospital_center = pd.read_csv('src/data/socioeconomic_info/plz_certified_hospital_center_germany.csv')
    plz_district = pd.read_csv('src/data/socioeconomic_info/plz_district_mapping.csv', sep=';')
    certified_hospital_center = pd.merge(
        certified_hospital_center,
        plz_district,
        left_on=["plz_certified_hospital_center"],
        right_on=['plz'],
        how="inner"
    )
    certified_hospital_center = certified_hospital_center.drop(columns=['plz'])
    certified_hospital_center['district_id'] = certified_hospital_center['district_id'].astype(int)
    cohord_gisd['certified_hospital_center'] = cohord_gisd['district_id'].isin(certified_hospital_center['district_id']).astype(int)
    print(f"control: cohort size after join with plz_certified_hospital_center: {cohord_gisd.shape}")
    
    # needed for sunburst visualisation
    lo.delete_table('analytics_diffuse_large_b_cell_lymphoma_prepared')
    lo.df_import(cohort,'analytics_diffuse_large_b_cell_lymphoma_prepared')
    
    # 11. clean up df
    cohord_gisd = cohord_gisd[[
            'person_id', 
            'bundesland_id',
            'register_id',
            'district_id',
            'sex',
            'diagnosis_year', 
            'age', 
            'age_60_plus', 
            'survival_status', 
            'survival_time',
            'gisd_score',
            'ses',
            'one_year_death_prob_general',
            'premature_mortality',
            'certified_hospital_center']]
    cohord_gisd[["bundesland_id", "register_id"]] = cohord_gisd[["bundesland_id", "register_id"]].astype(int)
    cohord_gisd[["one_year_death_prob_general", "premature_mortality"]] = cohord_gisd[["one_year_death_prob_general", "premature_mortality"]].astype(float)
    
    return cohord_gisd

def descriptive_analysis(cohort):
    # ------------------------------
    # 1. Continuous variables
    # ------------------------------
    continuous_vars = ["age", "survival_time", "gisd_score", "one_year_death_prob_general", "premature_mortality"]

    def summarize_continuous_by_group(df, group_col, vars_):
        grouped_summary = {}
        for group in df[group_col].unique():
            desc = df[df[group_col] == group][vars_].describe().T
            desc["SES"] = group
            grouped_summary[group] = desc
        desc_all = df[vars_].describe().T
        desc_all["SES"] = "All"
        return pd.concat([desc_all] + list(grouped_summary.values())).reset_index().rename(columns={"index": "Variable"})

    cont_summary = summarize_continuous_by_group(cohort, "ses", continuous_vars)
    print("\n### Descriptive statistics – continuous variables ###")
    print(cont_summary.to_string(index=False))

    # ------------------------------
    # 2. Categorical variables
    # ------------------------------
    categorical_vars = ["sex", "diagnosis_year", "age_60_plus", "survival_status", "certified_hospital_center"]

    def summarize_categorical_by_group(df, group_col, cat_vars):
        summaries = []
        for var in cat_vars:
            for group in ["All"] + list(df[group_col].dropna().unique()):
                subset = df if group == "All" else df[df[group_col] == group]
                counts = subset[var].value_counts(dropna=False).sort_index()
                props = subset[var].value_counts(normalize=True, dropna=False).sort_index()
                summary = pd.DataFrame({
                    "Variable": var,
                    "Category": counts.index,
                    "Count": counts.values,
                    "Proportion": props.values,
                    "SES": group
                })
                summaries.append(summary)
        return pd.concat(summaries, ignore_index=True)

    cat_summary = summarize_categorical_by_group(cohort, "ses", categorical_vars)
    print("\n### Descriptive statistics – categorical variables ###")
    print(cat_summary.to_string(index=False))
    

def kaplan_meier_analysis(cohort):
    kmf = KaplanMeierFitter()
    plt.figure(figsize=(10, 6))

    colors = {'low': '#cc2a36', 'middle': '#8b8589', 'high': '#005582'}
    legend_order = ['low', 'middle', 'high']

    # Years converted to days
    time_points = [365 * i for i in range(1, 6)]
    year_labels = ['1 year', '2 years', '3 years', '4 years', '5 years']

    survival_counts = {}  # Estimated absolute survival count
    at_risk_counts = {} # Observed number at risk
    
    for group in cohort['ses'].unique():
        mask = cohort['ses'] == group
        T = pd.to_numeric(cohort[mask]['survival_time'], errors='coerce')
        E = pd.to_numeric(cohort[mask]['survival_status'], errors='coerce')

        kmf.fit(T, event_observed=E, label=str(group))
        kmf.plot_survival_function(ci_show=False, color=colors.get(group, None))

        # --- 1. Estimated absolute survival count ---
        survival_prob = kmf.survival_function_.asof(time_points).squeeze()
        n_group = mask.sum()
        abs_surv = (survival_prob * n_group).round().astype(int)
        abs_surv.index = year_labels
        survival_counts[group] = abs_surv
        
         # --- 2. Observed number at risk ---
        at_risk = []
        for t in time_points:
            try:
                prev_time = kmf.event_table.index[kmf.event_table.index.get_indexer([t], method='ffill')[0]]
                at_risk.append(kmf.event_table.loc[prev_time, 'at_risk'])
            except IndexError:
                at_risk.append(np.nan)
        at_risk_series = pd.Series(at_risk, index=year_labels)
        at_risk_counts[group] = at_risk_series
    
    handles, labels = plt.gca().get_legend_handles_labels()
    label_to_handle = dict(zip(labels, handles))
    ordered_handles = [label_to_handle[g] for g in legend_order]

    plt.grid(True, linestyle='--', alpha=0.5)
    plt.yticks(np.linspace(0, 1, 5), ['0%', '25%', '50%', '75%', '100%'])
    plt.xticks(time_points, year_labels)

    plt.xlabel("Time (years)", fontsize=12)
    plt.ylabel("Overall survival (%)", fontsize=12)
    plt.legend(handles=ordered_handles, labels=legend_order, title="SES")

    # --- Add table below plot (number at risk) ---    
    at_risk_df = pd.DataFrame(at_risk_counts, index=year_labels)
    formatted_cell_text = []
    for group in ['low', 'middle', 'high']:
        row_values = []
        for v in at_risk_df[group].values:
            if np.isnan(v):
                row_values.append("")
            else:
                n = int(v)
                if n >= 10000:
                    row_values.append(format(n, ","))
                else:
                    row_values.append(str(n))
        formatted_cell_text.append(row_values)
    cell_text = np.array(formatted_cell_text)

    table = plt.table(
        cellText=cell_text,
        rowLabels=['low', 'middle', 'high'],
        colLabels=year_labels,
        cellLoc='center',
        rowLoc='center',
        colLoc='center',
        loc='bottom',
        bbox=[0, -0.5, 1, 0.3]  # x, y, width, height
    )
    plt.text(
        0.02, -0.18,              # x, y coordinates
        "Number at Risk",
        ha='left',                 # horizontal alignment
        va='bottom',               # vertical alignment
        fontsize=12,
        transform=plt.gca().transAxes
    )

    # 2. Logrank-Test ('low' vs 'high')
    group1 = cohort[cohort['ses'] == 'low']
    group2 = cohort[cohort['ses'] == 'high']

    results = logrank_test(
        group1['survival_time'],
        group2['survival_time'],
        event_observed_A=group1['survival_status'],
        event_observed_B=group2['survival_status']
    )
    
    p_val = results.p_value
    if p_val < 0.001:
        p_text = "p < 0.001"
    else:
        p_text = f"p < {p_val:.3f}"
    
    plt.text(
        0.02, 0.08,
        p_text,
        ha='left',
        va='top',
        fontsize=12,
        transform=plt.gca().transAxes
    )
    
    plt.subplots_adjust(left=0.15, bottom=0.2)
    plt.tight_layout()
    plt.savefig("src/results/km_ses_with_risk_table.png", dpi=300)
    plt.close()
    
    print("=== Logrank-Test: low vs high ===")
    print(results.summary)

    survival_df = pd.DataFrame(survival_counts, index=year_labels)
    at_risk_df = pd.DataFrame(at_risk_counts, index=year_labels)

    print("\n=== Estimated absolute survival count ===")
    print(survival_df)

    print("\n=== Observed number at risk ===")
    print(at_risk_df)
    
def cox_model_analysis(cohort):
    def run_cox_model(cohort, description):
        cohort = cohort.copy()
        cohort['ses'] = pd.Categorical(cohort['ses'], categories=['low', 'middle', 'high'], ordered=True)
        dummies = pd.get_dummies(cohort['ses'], drop_first=True)  # drops 'low', keeps 'middle' and 'high'
        cohort = pd.concat([cohort, dummies], axis=1)
        counts_by_ses = cohort.groupby('ses')['survival_status'].agg(['count', 'sum']).rename(columns={'count': 'n_ses', 'sum': 'events_ses'})
        cohort =cohort.drop(columns=['ses']) 

        # dummy variables for categorical features: 'diagnosis_year' and 'register_id'
        dummy_parts = []
        if 'diagnosis_year' in cohort.columns:
            dummies_year = pd.get_dummies(cohort['diagnosis_year'], prefix='year', drop_first=True)
            dummy_parts.append(dummies_year)
            cohort = cohort.drop(columns=['diagnosis_year'])
            
        if 'register_id' in cohort.columns:
            dummies_state = pd.get_dummies(cohort['register_id'], prefix='state', drop_first=True)
            dummy_parts.append(dummies_state)
            cohort = cohort.drop(columns=['register_id'])
        cohort = pd.concat([cohort] + dummy_parts, axis=1)
        
        cph = CoxPHFitter()
        cph.fit(cohort, duration_col='survival_time', event_col='survival_status', cluster_col='district_id')

        summary = cph.summary.loc[['middle', 'high']][['exp(coef)', 'exp(coef) lower 95%', 'exp(coef) upper 95%']]
        summary.columns = ['HR', 'CI_lower', 'CI_upper']
        summary['Subgroup'] = description
        summary['Variable'] = summary.index
        summary['n'] = summary['Variable'].map(counts_by_ses['n_ses'])
        summary['events'] = summary['Variable'].map(counts_by_ses['events_ses'])
        summary.reset_index(drop=True, inplace=True)
        return summary
    
    results = []

    cohort = cohort.drop(columns=['person_id', 'bundesland_id'])
    # unadjusted
    cohort_unadj = cohort.drop(columns=['register_id', 'sex', 'diagnosis_year', 'age', 'age_60_plus', 'gisd_score', 'one_year_death_prob_general', 'premature_mortality', 'certified_hospital_center'])
    results.append(run_cox_model(cohort_unadj, "Unadjusted"))
    
    # adjusted
    cohort_adj = cohort.drop(columns=['premature_mortality'])
    results.append(run_cox_model(cohort_adj, "Adjusted"))

    # adjusted + comorbidity
    cohort_comorbid = cohort.copy()
    results.append(run_cox_model(cohort_comorbid, "Adjusted + comorbidity"))

    # subgroups by time period
    results.append(run_cox_model(cohort[cohort['diagnosis_year'].between(2014, 2019)], "Adjusted + comorbidity, 2014–2019"))
    results.append(run_cox_model(cohort[cohort['diagnosis_year'].between(2020, 2024)], "Adjusted + comorbidity, 2020–2024"))

    # subgroups by region
    results.append(run_cox_model(cohort[~cohort['register_id'].isin([2, 4, 11])], "Adjusted + comorbidity, excl. city registries"))
    results.append(run_cox_model(cohort[~cohort['register_id'].isin([12, 13, 14, 15, 16])], "Adjusted + comorbidity, excl. East Germany"))
    
    final_df = pd.concat(results, ignore_index=True)
    print(final_df)
    
    final_df['HR_str'] = final_df.apply(
        lambda r: f"{r['HR']:.2f} ({r['CI_lower']:.2f}–{r['CI_upper']:.2f})", axis=1
    )

    final_df['N'] = final_df['n'].apply(
        lambda x: format(int(x), ",") if int(x) >= 10000 else str(int(x))
    )
    final_df['Events'] = final_df['events'].apply(
        lambda x: format(int(x), ",") if int(x) >= 10000 else str(int(x))
    )

    fig, ax = plt.subplots(figsize=(14, len(final_df) * 0.5))

    # plot (forestplot)
    y_pos = np.arange(len(final_df))
    ax.errorbar(final_df['HR'], y_pos, 
                xerr=[final_df['HR'] - final_df['CI_lower'], final_df['CI_upper'] - final_df['HR']],
                fmt='o', color='black')
    ax.axvline(x=1.0, color='grey', linestyle='--')

    ax.set_yticks(y_pos)
    ax.set_yticklabels(final_df['Subgroup'] + " - " + final_df['Variable'], fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel('Hazard Ratio (95% CI)', fontsize=10, fontweight='bold')

    plt.subplots_adjust(right=0.5, top=0.9)
    
    # table to the right of the plot
    table_data = final_df[['HR_str', 'N', 'Events']].values

    table = plt.table(cellText=table_data,
                    cellLoc='center',
                    colLoc='center',
                    bbox=[1.05, 0, 0.7, 1],
                    edges='horizontal')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.3, 1.3)

    # table titles
    fig.text(0.08, 0.92, 'Subgroup', ha='center', fontsize=10, fontweight='bold')
    fig.text(0.56, 0.92, 'Hazard Ratio', ha='center', fontsize=10, fontweight='bold')
    fig.text(0.64, 0.92, 'N', ha='center', fontsize=10, fontweight='bold')
    fig.text(0.74, 0.92, 'Events', ha='center', fontsize=10, fontweight='bold')

    plt.savefig("src/results/cox_forest_ses_with_table.png", dpi=300, bbox_inches='tight')
    plt.close()