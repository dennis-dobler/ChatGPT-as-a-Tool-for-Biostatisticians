






#Coverage probability simulation 9
# Add check for size of confidence interval to avoid index errors
results = []

for (n_case, n_control), p_exp in zip(sample_sizes, marginal_probs):
    cover_count = 0
    valid_runs = 0

    for _ in range(n_sim_reduced):
        exp_cases = np.random.binomial(1, p_exp, size=n_case)
        exp_controls = np.random.binomial(1, p_exp, size=n_control)

        y = np.array([1] * n_case + [0] * n_control)
        x = np.concatenate([exp_cases, exp_controls])

        try:
            model = Logit(y, add_constant(x)).fit(disp=0)
            ci = model.conf_int()
            if ci.shape[0] > 1:  # Ensure CI for exposure coefficient exists
                lower, upper = ci[1]
                if lower <= true_log_odds_ratio <= upper:
                    cover_count += 1
                valid_runs += 1
        except (ValueError, PerfectSeparationError, np.linalg.LinAlgError):
            continue

    coverage = cover_count / valid_runs if valid_runs > 0 else np.nan
    results.append({
        'n_case': n_case,
        'n_control': n_control,
        'p_exposure': p_exp,
        'coverage': coverage,
        'valid_runs': valid_runs
    })

df_results = pd.DataFrame(results)
tools.display_dataframe_to_user(name="Coverage Probability Results", dataframe=df_results)
