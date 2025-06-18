#Coverage probability simulation 2
simulate_coverage <- function(n_cases, n_controls, p_exp, reps = 5000) {
  cover <- replicate(reps, {
    exposure_cases <- rbinom(n_cases, 1, p_exp)
    exposure_controls <- rbinom(n_controls, 1, p_exp)
    exposure <- c(exposure_cases, exposure_controls)
    outcome <- c(rep(1, n_cases), rep(0, n_controls))

    fit <- suppressWarnings(glm(outcome ~ exposure, family = binomial))
    if (any(is.na(coef(summary(fit))))) return(NA)

    beta_hat <- coef(fit)[2]
    se_beta <- summary(fit)$coefficients[2, 2]
    ci_lower <- beta_hat - qnorm(0.975) * se_beta
    ci_upper <- beta_hat + qnorm(0.975) * se_beta

    return(ci_lower <= 0 & ci_upper >= 0)
  })
  mean(na.omit(cover))
}
