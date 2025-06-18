#Coverage probability simulation 10
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import norm
import itertools
import ace_tools as tools

# Simulation settings
n_simulations = 5000
alpha = 0.05
z_crit = norm.ppf(1 - alpha / 2)

# Sample size settings
sample_sizes = [(cases, controls) for cases, controls in zip(range(20, 30, 1), range(30, 40, 1))][:10]

# Null hypothesis: log(OR) = 0 => OR = 1
true_log_or = 0
true_or = np.exp(true_log_or)

# Simulate function
def simulate_coverage(cases, controls, p_exposure_cases, p_exposure_controls):
    coverages = []
    for _ in range(n_simulations):
        exposure_cases = np.random.binomial(1, p_exposure_cases, cases)
        exposure_controls = np.random.binomial(1, p_exposure_controls, controls)

        y = np.concatenate([np.ones(cases), np.zeros(controls)])
        x = np.concatenate([exposure_cases, exposure_controls])

        try:
            model = sm.Logit(y, sm.add_constant(x)).fit(disp=0)
            log_or = model.params[1]
            se_log_or = model.bse[1]
            ci_lower = log_or - z_crit * se_log_or
            ci_upper = log_or + z_crit * se_log_or
            coverage = (ci_lower <= true_log_or <= ci_upper)
        except:
            coverage = False  # model failed to converge

        coverages.append(coverage)

    return np.mean(coverages)

# Run simulation for 10 settings
results = []
marginal_probs = np.linspace(0.1, 0.9, 10)  # varying exposure probabilities

for i, (cases, controls) in enumerate(sample_sizes):
    p = marginal_probs[i]
    coverage = simulate_coverage(cases, controls, p, p)
    results.append({
        "Cases": cases,
        "Controls": controls,
        "P(Exposure|Case)": p,
        "P(Exposure|Control)": p,
        "Empirical Coverage": coverage
    })

results_df = pd.DataFrame(results)
tools.display_dataframe_to_user(name="Empirical Coverage Results", dataframe=results_df)
