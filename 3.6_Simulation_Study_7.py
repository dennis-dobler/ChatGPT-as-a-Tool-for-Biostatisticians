#Coverage probability simulation 7
import numpy as np
import statsmodels.api as sm

def simulate_coverage(n_cases, n_controls, p_exp_control, n_sim=5000):
    covered = 0
    for _ in range(n_sim):
        x_cases = np.random.binomial(1, p_exp_control, size=n_cases)
        x_controls = np.random.binomial(1, p_exp_control, size=n_controls)

        y = np.concatenate([np.ones(n_cases), np.zeros(n_controls)])
        x = np.concatenate([x_cases, x_controls])

        X = sm.add_constant(x)
        model = sm.Logit(y, X)
        try:
            result = model.fit(disp=0)
            beta1 = result.params[1]
            se = result.bse[1]
            ci_lower = beta1 - 1.96 * se
            ci_upper = beta1 + 1.96 * se
            if ci_lower <= 0 <= ci_upper:
                covered += 1
        except:
            continue  # Skip failed fits

    return covered / n_sim
