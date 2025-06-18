#Coverage probability simulation 3
import numpy as np
import pandas as pd
import statsmodels.api as sm
from tqdm import tqdm
import ace_tools as tools

np.random.seed(42)

def simulate_coverage(n, pi_y, pi_x, n_sim=5000):
    coverages = []
    n_cases = int(n * pi_y)
    n_controls = n - n_cases
    for _ in range(n_sim):
        y = np.array([1]*n_cases + [0]*n_controls)
        x = np.random.binomial(1, pi_x, size=n)
        try:
            model = sm.Logit(y, sm.add_constant(x)).fit(disp=0)
            beta = model.params[1]
            se = model.bse[1]
            ci_lower = beta - 1.96 * se
            ci_upper = beta + 1.96 * se
            coverages.append(ci_lower <= 0 <= ci_upper)
        except:
            coverages.append(False)
    return np.mean(coverages)

# Settings
settings = [
    (20, 0.3), (20, 0.5), (20, 0.7),
    (25, 0.3), (25, 0.5), (25, 0.7),
    (30, 0.4), (35, 0.5), (40, 0.4), (40, 0.6)
]

results = []
for n, pi_y in tqdm(settings):
    coverage = simulate_coverage(n, pi_y, pi_x=0.5)
    results.append({'n': n, 'pi_y': pi_y, 'coverage': coverage})

df_results = pd.DataFrame(results)
tools.display_dataframe_to_user(name="Coverage Probability Results", dataframe=df_results)
