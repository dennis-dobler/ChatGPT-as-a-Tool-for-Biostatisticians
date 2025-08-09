# python code for coverage prob 13
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import fisher_exact

# Ensure reproducibility
np.random.seed(42)

def simulate_case_control(n_cases, n_controls, p):
    """Simulate exposure for given cases and controls under true OR=1."""
    # Number of exposed cases (a) and exposed controls (c) from Binomial distribution
    a = np.random.binomial(n_cases, p)
    c = np.random.binomial(n_controls, p)
    b = n_cases - a  # unexposed cases
    d = n_controls - c  # unexposed controls
    return a, b, c, d

def estimate_or_logistic(a, b, c, d):
    """Estimate odds ratio using logistic regression."""
    n_cases = a + b
    n_controls = c + d
    # Handle extreme cases where logistic regression may not converge (OR → 0 or ∞)
    if (a == 0 and c > 0) or (c == n_controls and a < n_cases):
        return 0.0   # OR approaches 0 (no exposed cases or all controls exposed)
    if (c == 0 and a > 0) or (a == n_cases and c < n_controls):
        return np.inf  # OR approaches ∞ (no exposed controls or all cases exposed)
    if (a == 0 and c == 0) or (a == n_cases and c == n_controls):
        return 1.0   # No exposure variation between groups, OR = 1
    # Prepare data for logistic regression
    exposure = [1]*a + [0]*b + [1]*c + [0]*d
    outcome  = [1]*(a+b) + [0]*(c+d)  # 1=case, 0=control
    X = sm.add_constant(exposure)
    model = sm.Logit(outcome, X)
    result = model.fit(disp=False)
    # Odds ratio = exp(beta) for exposure
    return float(np.exp(result.params[1]))

# Define simulation scenarios
scenarios = [
    {"n_cases": 10, "n_controls": 10, "p": 0.1},
    {"n_cases": 10, "n_controls": 10, "p": 0.5},
    {"n_cases": 30, "n_controls": 30, "p": 0.1},
    {"n_cases": 30, "n_controls": 30, "p": 0.5},
    {"n_cases": 10, "n_controls": 30, "p": 0.1},
    {"n_cases": 10, "n_controls": 30, "p": 0.5},
    {"n_cases": 30, "n_controls": 10, "p": 0.1},
    {"n_cases": 30, "n_controls": 10, "p": 0.5},
    {"n_cases": 20, "n_controls": 20, "p": 0.3},
    {"n_cases": 20, "n_controls": 30, "p": 0.3},
]

# Simulation parameters
n_sim = 5000
results = []

# Run simulations for each scenario
for scen in scenarios:
    n_cases, n_controls, p = scen["n_cases"], scen["n_controls"], scen["p"]
    count_in_ci = 0  # counter for CIs that include OR=1
    for sim in range(n_sim):
        # Simulate 2x2 table for this simulation run
        a, b, c, d = simulate_case_control(n_cases, n_controls, p)
        or_est = estimate_or_logistic(a, b, c, d)  # (Logistic OR estimate, not used further in coverage calc)
        # Exact test inversion: Fisher's exact two-sided p-value
        _, p_value = fisher_exact([[a, b], [c, d]], alternative='two-sided')
        if p_value >= 0.05:
            count_in_ci += 1  # 95% CI includes OR=1 (null not rejected at 5% level)
    coverage = count_in_ci / n_sim
    results.append({
        "Cases": n_cases, 
        "Controls": n_controls, 
        "Exposure Probability": p, 
        "Empirical Coverage": coverage
    })

# Display summary table of empirical coverage for each scenario
df = pd.DataFrame(results)
print(df.to_string(index=False, float_format="%.3f"))
