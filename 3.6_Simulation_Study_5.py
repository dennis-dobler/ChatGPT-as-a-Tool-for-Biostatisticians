#Coverage probability simulation 5
# Modify to catch convergence and singular matrix errors and skip those iterations
def simulate_coverage_robust(n, p_exposure, n_sim=5000):
    covered = 0
    valid_runs = 0
    for _ in range(n_sim):
        try:
            n_cases = n // 2
            n_controls = n - n_cases
            exposure_cases = np.random.binomial(1, p_exposure, size=n_cases)
            exposure_controls = np.random.binomial(1, p_exposure, size=n_controls)

            exposure = np.concatenate([exposure_cases, exposure_controls])
            disease = np.array([1] * n_cases + [0] * n_controls)
            X = sm.add_constant(exposure)
            model = Logit(disease, X).fit(disp=0)
            ci = model.conf_int(alpha=0.05)
            ci_exposure = ci[1]  # second parameter: exposure
            if ci_exposure[0] <= 0 <= ci_exposure[1]:
                covered += 1
            valid_runs += 1
        except:
            continue
    return covered / valid_runs if valid_runs > 0 else np.nan

# Re-run with robust function
results_robust = []
for n, p in sample_settings:
    coverage = simulate_coverage_robust(n, p)
    results_robust.append({"Sample Size": n, "Exposure Probability": p, "Empirical Coverage": coverage})

df_results_robust = pd.DataFrame(results_robust)
tools.display_dataframe_to_user(name="Robust Coverage Probability Results", dataframe=df_results_robust)
