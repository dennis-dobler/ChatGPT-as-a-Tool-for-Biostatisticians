#Coverage probability simulation 1
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.special import expit
from statsmodels.stats.proportion import proportion_confint
import ace_tools as tools

np.random.seed(123)

# Settings: 10 combinations of (n_cases, n_controls, exposure prevalence)
settings = [
    (20, 20, 0.2), (25, 25, 0.3), (30, 30, 0.4), (35, 35, 0.5), (40, 40, 0.6),
    (20, 30, 0.3), (30, 20, 0.4), (25, 35, 0.5), (35, 25, 0.6), (40, 20, 0.5)
]

def simulate_coverage(n_cases, n_controls, p_exposure, n_sim=5000, alpha=0.05):
    covered = 0
    for _ in range(n_sim):
        exposure_cases = np.random.binomial(1, p_exposure, n_cases)
        exposure_controls = np.random.binomial(1, p_exposure, n_controls)
        
        y = np.concatenate([np.ones(n_cases), np.zeros(n_controls)])
        x = np.concatenate([exposure_cases, exposure_controls])
        X = sm.add_constant(x)
        
        try:
            model = sm.Logit(y, X).fit(disp=0)
            conf_int = model.conf_int(alpha=alpha).loc[1]  # coefficient for exposure
            if conf_int[0] <= 0 <= conf_int[1]:  # Check if 0 is within CI
                covered += 1
        except Exception:
            continue  # In case of convergence issues
        
    return covered / n_sim

results = []
for n_cases, n_controls, p_exp in settings:
    coverage = simulate_coverage(n_cases, n_controls, p_exp)
    results.append({
        "n_cases": n_cases,
        "n_controls": n_controls,
        "p_exposure": p_exp,
        "coverage_probability": coverage
    })

df_results = pd.DataFrame(results)
tools.display_dataframe_to_user(name="Coverage Probability Results", dataframe=df_results)
