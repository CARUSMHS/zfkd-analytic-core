from utils import data_loader as lo
from utils import diffuse_large_b_cell_lymphoma as DLBCL
from utils import sunburst_dlbcl as sunburst
import pandas as pd
import numpy as np

RESTRICTED_EXCL = [10, 11, 12]
QUINTILE_TO_SES = {1: "high", 2: "middle", 3: "middle", 4: "middle", 5: "low"}

def assign_ses_fixed_cutoffs(cohort, excl_registers=RESTRICTED_EXCL):
    restr = cohort.loc[~cohort["register_id"].isin(excl_registers), "gisd_score"]
    bins = np.quantile(restr, [0, .2, .4, .6, .8, 1.0])
    bins[0], bins[-1] = -np.inf, np.inf
    quintile = pd.cut(cohort["gisd_score"], bins=bins, labels=[1, 2, 3, 4, 5]).astype(int)
    out = cohort.copy()
    out["ses"] = quintile.map(QUINTILE_TO_SES)
    return out


# diffuse large B-cell lymphoma (DLBCL) cohort
def analyize_DLBCL():
    lo.create_cohort('diffuse_large_b_cell_lymphoma')
    cohort = lo.load_cohort('diffuse_large_b_cell_lymphoma')
    print(f"cohort size from database: {cohort.shape}")

    cohort = DLBCL.prepare_DLBCL_SES_cohort(cohort)
    cohort.to_csv("src/data/DLBCL_cohort.csv", index=False)

    # Quintile cut-offs were derived once from the full cohort and applied unchanged to all subsets.
    cohort_extended = assign_ses_fixed_cutoffs(cohort)
    cohort_restricted = cohort_extended[~cohort_extended["register_id"].isin(RESTRICTED_EXCL)]
    print(f"restricted: {cohort_restricted.shape}, extended: {cohort_extended.shape}")

    DLBCL.descriptive_analysis(cohort_restricted)
    DLBCL.kaplan_meier_analysis(cohort_restricted)
    DLBCL.cox_model_analysis(cohort_restricted, cohort_extended=cohort_extended)

    # sunburst visualizations
    sunburst.plot_sunburst_dynamics()
    sunburst.plot_sunburst_regimen()
    
if __name__ == "__main__":
    analyize_DLBCL()
