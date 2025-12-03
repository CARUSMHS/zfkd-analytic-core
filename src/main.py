from utils import data_loader as lo
from utils import diffuse_large_b_cell_lymphoma as DLBCL
from utils import sunburst_dlbcl as sunburst

# diffuse large B-cell lymphoma (DLBCL) cohort
def analyize_DLBCL():
    lo.create_cohort('diffuse_large_b_cell_lymphoma')
    cohort = lo.load_cohort('diffuse_large_b_cell_lymphoma')
    print(f"cohort size from database: {cohort.shape}")

    cohort = DLBCL.prepare_DLBCL_SES_cohort(cohort)
    cohort.to_csv("src/data/DLBCL_cohort.csv", index=False)

    DLBCL.descriptive_analysis(cohort)
    DLBCL.kaplan_meier_analysis(cohort)
    DLBCL.cox_model_analysis(cohort)
    
    # sunburst visualizations
    sunburst.plot_sunburst_dynamics()
    sunburst.plot_sunburst_regimen()
    
if __name__ == "__main__":
    analyize_DLBCL()
