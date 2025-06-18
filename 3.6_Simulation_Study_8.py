#Coverage probability simulation 8
import numpy as np
import pandas as pd
from statsmodels.api import Logit, add_constant
from scipy.special import expit
from scipy.stats import norm
import ace_tools as tools

# Simulation parameters
n_simulations = 5000
sample_sizes = [(n_cases, n_controls) for n_cases in range(10, 25, 2) for n_controls in range(10, 25, 2)][:10]
true_beta = 0.0  # Null hypothesis: no effect
alpha = 0.05

results = []

for n_cases, n_controls in sample_sizes:
    coverage_count = 0

    for _ in range(n_simulations):
        # Generate exposure for controls and cases under null (no association)
        p_exposed_controls = np.random.uniform(0.1, 0.9)
        exposure_controls = np.random.binomial(1, p_exposed_controls, size=n_controls)
        exposure_cases = np.random.binomial(1, p_exposed_controls, size=n_cases)

        # Stack data
        y = np.concatenate([np.ones(n_cases), np.zeros(n_controls)])  # 1 = case, 0 = control
        x = np.concatenate([exposure_cases, exposure_controls])
        X = add_constant(x)

        try:
            model = Logit(y, X).fit(disp=0)
            beta_hat = model.params[1]
            se = model.bse[1]

            ci_lower = beta_hat - norm.ppf(1 - alpha/2) * se
            ci_upper = beta_hat + norm.ppf(1 - alpha/2) * se

            if ci_lower <= true_beta <= ci_upper:
                coverage_count += 1
        except:
            continue

    coverage_probability = coverage_count / n_simulations
    results.append({
        "n_cases": n_cases,
        "n_controls": n_controls,
        "coverage_probability": coverage_probability
    })

results_df = pd.DataFrame(results)
tools.display_dataframe_to_user(name="Coverage Probability Results", dataframe=results_df)
