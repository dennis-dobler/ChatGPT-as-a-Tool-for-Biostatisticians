# Hui-Walter model with logistic parametrization, Wald CIs, and Bootstrap CIs
# Comprehensive example

#-------------------------------
# 1. Setup: Observed counts
#-------------------------------

# Observed counts from example
counts <- list(
  pop1 = c(n11 = 50, n10 = 20, n01 = 15, n00 = 100),
  pop2 = c(n11 = 30, n10 = 10, n01 = 25, n00 = 200)
)

#-------------------------------
# 2. Logistic (expit) function
#-------------------------------
expit <- function(x) {
  1 / (1 + exp(-x))
}

#-------------------------------
# 3. Log-likelihood function (logistic parametrization)
#-------------------------------
loglik_logit <- function(par, counts) {
  # Transform parameters to probability scale
  pi1 <- expit(par[1])
  pi2 <- expit(par[2])
  Se1 <- expit(par[3])
  Sp1 <- expit(par[4])
  Se2 <- expit(par[5])
  Sp2 <- expit(par[6])
  
  # Probabilities for each test outcome in each population
  # Population 1
  p11_1 <- pi1 * Se1 * Se2 + (1 - pi1) * (1 - Sp1) * (1 - Sp2)
  p10_1 <- pi1 * Se1 * (1 - Se2) + (1 - pi1) * (1 - Sp1) * Sp2
  p01_1 <- pi1 * (1 - Se1) * Se2 + (1 - pi1) * Sp1 * (1 - Sp2)
  p00_1 <- pi1 * (1 - Se1) * (1 - Se2) + (1 - pi1) * Sp1 * Sp2
  
  # Population 2
  p11_2 <- pi2 * Se1 * Se2 + (1 - pi2) * (1 - Sp1) * (1 - Sp2)
  p10_2 <- pi2 * Se1 * (1 - Se2) + (1 - pi2) * (1 - Sp1) * Sp2
  p01_2 <- pi2 * (1 - Se1) * Se2 + (1 - pi2) * Sp1 * (1 - Sp2)
  p00_2 <- pi2 * (1 - Se1) * (1 - Se2) + (1 - pi2) * Sp1 * Sp2
  
  # Add small constant to avoid log(0)
  eps <- 1e-10
  
  # Log-likelihood
  ll <- 0
  ll <- ll + counts$pop1["n11"] * log(p11_1 + eps)
  ll <- ll + counts$pop1["n10"] * log(p10_1 + eps)
  ll <- ll + counts$pop1["n01"] * log(p01_1 + eps)
  ll <- ll + counts$pop1["n00"] * log(p00_1 + eps)
  
  ll <- ll + counts$pop2["n11"] * log(p11_2 + eps)
  ll <- ll + counts$pop2["n10"] * log(p10_2 + eps)
  ll <- ll + counts$pop2["n01"] * log(p01_2 + eps)
  ll <- ll + counts$pop2["n00"] * log(p00_2 + eps)
  
  return(-ll)  # because optim does minimization
}

#-------------------------------
# 4. Initial values (logit scale)
#-------------------------------
# logit(0.5) = 0
# logit(0.8) ≈ 1.386
# logit(0.9) ≈ 2.197

init_logit <- c(0, 0, 1.386, 2.197, 1.386, 2.197)

#-------------------------------
# 5. MLE estimation with optim
#-------------------------------
fit_logit <- optim(
  par = init_logit,
  fn = loglik_logit,
  counts = counts,
  method = "BFGS",
  control = list(trace = 1, maxit = 1000)
)

# Extract MLE estimates on probability scale
param_names <- c("pi1", "pi2", "Se1", "Sp1", "Se2", "Sp2")
mle_estimates <- expit(fit_logit$par)

cat("\nEstimated parameters (MLE, probability scale):\n")
for (i in 1:6) {
  cat(param_names[i], ": ", mle_estimates[i], "\n")
}

#-------------------------------
# 6. Hessian-based Wald 95% CIs
#-------------------------------
library(numDeriv)

# Compute Hessian
hess_logit <- hessian(func = loglik_logit, x = fit_logit$par, counts = counts)

# Invert Hessian to get variance-covariance matrix
vcov_logit <- solve(hess_logit)

# Standard errors on logit scale
se_logit <- sqrt(diag(vcov_logit))

# Wald 95% CI on logit scale
logit_CI_lower <- fit_logit$par - 1.96 * se_logit
logit_CI_upper <- fit_logit$par + 1.96 * se_logit

# Transform to probability scale
expit_CI_lower <- expit(logit_CI_lower)
expit_CI_upper <- expit(logit_CI_upper)

cat("\nWald 95% CIs (probability scale):\n")
for (i in 1:6) {
  cat(param_names[i], ": ", mle_estimates[i], 
      " (", expit_CI_lower[i], ", ", expit_CI_upper[i], ")\n")
}

#-------------------------------
# 7. Parametric Bootstrap 95% CIs
#-------------------------------
n_boot <- 1000
boot_estimates <- matrix(NA, nrow = n_boot, ncol = 6)

set.seed(123)  # for reproducibility

for (b in 1:n_boot) {
  # Use MLE estimates to simulate new data
  pi1 <- mle_estimates[1]
  pi2 <- mle_estimates[2]
  Se1 <- mle_estimates[3]
  Sp1 <- mle_estimates[4]
  Se2 <- mle_estimates[5]
  Sp2 <- mle_estimates[6]
  
  # Probabilities of each cell
  probs_pop1 <- c(
    pi1 * Se1 * Se2 + (1 - pi1) * (1 - Sp1) * (1 - Sp2),
    pi1 * Se1 * (1 - Se2) + (1 - pi1) * (1 - Sp1) * Sp2,
    pi1 * (1 - Se1) * Se2 + (1 - pi1) * Sp1 * (1 - Sp2),
    pi1 * (1 - Se1) * (1 - Se2) + (1 - pi1) * Sp1 * Sp2
  )
  
  probs_pop2 <- c(
    pi2 * Se1 * Se2 + (1 - pi2) * (1 - Sp1) * (1 - Sp2),
    pi2 * Se1 * (1 - Se2) + (1 - pi2) * (1 - Sp1) * Sp2,
    pi2 * (1 - Se1) * Se2 + (1 - pi2) * Sp1 * (1 - Sp2),
    pi2 * (1 - Se1) * (1 - Se2) + (1 - pi2) * Sp1 * Sp2
  )
  
  # Simulate new counts
  n_pop1 <- sum(counts$pop1)
  n_pop2 <- sum(counts$pop2)
  
  new_counts_pop1 <- rmultinom(1, size = n_pop1, prob = probs_pop1)
  new_counts_pop2 <- rmultinom(1, size = n_pop2, prob = probs_pop2)
  
  new_counts <- list(
    pop1 = c(n11 = new_counts_pop1[1], n10 = new_counts_pop1[2], n01 = new_counts_pop1[3], n00 = new_counts_pop1[4]),
    pop2 = c(n11 = new_counts_pop2[1], n10 = new_counts_pop2[2], n01 = new_counts_pop2[3], n00 = new_counts_pop2[4])
  )
  
  # Re-fit model to bootstrap sample
  boot_fit <- optim(
    par = init_logit,
    fn = loglik_logit,
    counts = new_counts,
    method = "BFGS",
    control = list(maxit = 1000)
  )
  
  # Store bootstrap estimates on probability scale
  boot_estimates[b, ] <- expit(boot_fit$par)
}

# Compute bootstrap percentile CIs
boot_CIs <- apply(boot_estimates, 2, quantile, probs = c(0.025, 0.975))

cat("\nBootstrap 95% CIs (probability scale):\n")
for (i in 1:6) {
  cat(param_names[i], ": ", mle_estimates[i], 
      " (", boot_CIs[1,i], ", ", boot_CIs[2,i], ")\n")
}

