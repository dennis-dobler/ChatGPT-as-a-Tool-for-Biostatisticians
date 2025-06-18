#Coverage probability simulation 6
# Reduce simulation time for feasibility in current environment: use 1000 runs instead of 5000
n_sim = 1000

# Re-run simulation with reduced number of simulations
results = []
for n_cases, n_controls in sample_settings:
    coverage = simulate_case_control(n_cases, n_controls)
    results.append({
        "n_cases": n_cases,
        "n_controls": n_controls,
        "coverage_probability": coverage
    })

df_results = pd.DataFrame(results)
tools.display_dataframe_to_user(name="Coverage Probability Results", dataframe=df_results)
