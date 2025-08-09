# python code for coverage prob 12
import numpy as np
from math import log, inf
from scipy.stats import nchypergeom_fisher
from scipy.optimize import brentq

np.random.seed(42)  # Fixed seed for reproducibility

# Function to compute exact 95% CI for OR by inverting Fisher’s exact test
def exact_ci_or(a, b, c, d):
    ncases = a + b
    ncontrols = c + d
    M = ncases + ncontrols
    total_exposed = a + c
    # Handle degenerate cases with no variation in exposure
    if total_exposed == 0 or (b + d) == 0:
        return 0.0, inf  # CI = [0, ∞] (includes all OR)
    # Initialize bounds
    lower, upper = None, None
    if a == 0 and c > 0:       # zero cases exposed -> OR estimate 0
        lower = 0.0
    if c == 0 and a > 0:       # zero controls exposed -> OR estimate ∞
        upper = inf
    # Define tail probability functions for given OR (psi)
    def lower_tail_p(psi):
        # P(X >= a) under OR=psi (conditional on margins)
        if psi <= 0:
            return -0.025  # treat psi->0 limit
        # CDF for X <= a-1, then 1-CDF gives P(X >= a)
        cdf = nchypergeom_fisher.cdf(a-1, M, ncases, total_exposed, psi) if a > 0 else 0.0
        return (1 - cdf) - 0.025
    def upper_tail_p(psi):
        # P(X <= a) under OR=psi
        if psi <= 0:
            return 1.0 - 0.025
        cdf = nchypergeom_fisher.cdf(a, M, ncases, total_exposed, psi)
        return cdf - 0.025

    # Solve for lower bound (if not already determined)
    if lower is None:
        psi_lo, psi_hi = 1e-9, 1.0
        # Increase psi_hi until sign change (target function crosses 0)
        while lower_tail_p(psi_hi) < 0 and psi_hi < 1e6:
            psi_hi *= 10
        lower = 0.0 if lower_tail_p(psi_hi) < 0 else brentq(lower_tail_p, psi_lo, psi_hi)
    # Solve for upper bound (if not already determined)
    if upper is None:
        psi_lo, psi_hi = 1e-9, 1.0
        while upper_tail_p(psi_hi) > 0 and psi_hi < 1e6:
            psi_hi *= 10
        upper = inf if upper_tail_p(psi_hi) > 0 else brentq(upper_tail_p, psi_lo, psi_hi)
    return lower, upper

# Simulation parameters
sample_sizes = [20, 25, 30, 35, 40]
prevalences = [0.1, 0.5]
n_reps = 5000

sim_results = {}  # dict to store results for each scenario
coverage_counts = {}  # to accumulate coverage counts

for n in sample_sizes:
    for p in prevalences:
        key = (n, n, p)  # (cases, controls, prevalence)
        logOR_vals = []
        ci_lower_vals = []
        ci_upper_vals = []
        cover_count = 0
        for _ in range(n_reps):
            # Generate data under OR=1
            a = np.random.binomial(n, p)  # exposed cases
            c = np.random.binomial(n, p)  # exposed controls
            b = n - a
            d = n - c
            # Log-OR estimate
            if a == 0 and c == 0:
                log_or = None  # undefined (no exposure in either group)
            elif a == 0 and c > 0:
                log_or = -inf
            elif c == 0 and a > 0:
                log_or = inf
            else:
                log_or = log((a * d) / (b * c))
            # Exact 95% CI for OR
            L, U = exact_ci_or(a, b, c, d)
            logL = -inf if L == 0 else log(L)
            logU = inf if U == inf else log(U)
            # Save results
            logOR_vals.append(log_or)
            ci_lower_vals.append(logL)
            ci_upper_vals.append(logU)
            # Check if true log-OR (0) is within CI
            if logL <= 0 <= logU:
                cover_count += 1
        sim_results[key] = {
            'logOR': np.array(logOR_vals),
            'logCI_lower': np.array(ci_lower_vals),
            'logCI_upper': np.array(ci_upper_vals)
        }
        coverage_counts[key] = cover_count

# Example: first 5 replicates for scenario (20 cases, 20 controls, p=0.1)
scenario_key = (20, 20, 0.1)
print(f"Example simulation results for scenario {scenario_key}:")
for i in range(5):
    lo_val = sim_results[scenario_key]['logCI_lower'][i]
    hi_val = sim_results[scenario_key]['logCI_upper'][i]
    or_ci_low = 0.0 if lo_val == -inf else np.exp(lo_val)
    or_ci_high = inf if hi_val == inf else np.exp(hi_val)
    print(f" Replicate {i+1}: logOR_est = {sim_results[scenario_key]['logOR'][i]}, "
          f"95% CI (OR) = [{or_ci_low:.3f}, {or_ci_high:.3f}]")
