# ----------------------------------------------------------
# Hui-Walter Latent Class Model: Two Tests, Two Populations
# - MLE Estimation
# - Logistic Parametrization
# - Confidence Intervals via Hessian
# - Bootstrap Confidence Intervals
# ----------------------------------------------------------

# Load packages (base R is sufficient for this script)

# --- 1. Simulate Example Data ---

set.seed(123)

# True parameters
true_prev <- c(0.3, 0.6)  # Prevalences in populations 1 and 2
true_Se1 <- 0.85
true_Sp1 <- 0.90
true_Se2 <- 0.80
true_Sp2 <- 0.95

# Sample sizes
n1 <- 500
n2 <- 500

# Function to simulate test results
simulate_population <- function(prev, Se1, Sp1, Se2, Sp2, n) {
  D <- rbinom(n, 1, prev)
  T1 <- ifelse(D == 1, rbinom(n, 1, Se1), rbinom(n, 1, 1 - Sp1))
  T2 <- ifelse(D == 1, rbinom(n, 1, Se2), rbinom(n, 1, 1 - Sp2))
  table(factor(T1, levels = 0:1), factor(T2, levels = 0:1))
}

# Simulate for both populations
tab1 <- simulate_population(true_prev[1], true_Se1, true_Sp1, true_Se2, true_Sp2, n1)
tab2 <- simulate_population(true_prev[2], true_Se1, true_Sp1, true_Se2, true_Sp2, n2)

# Combine observed counts
counts <- c(as.vector(tab1), as.vector(tab2))

# --- 2. Define Log-Likelihood Function ---

loglik_fn <- function(par, counts) {
  # Logistic parametrization
  prev1 <- plogis(par[1])
  prev2 <- plogis(par[2])
  Se1 <- plogis(par[3])
  Sp1 <- plogis(par[4])
  Se2 <- plogis(par[5])
  Sp2 <- plogis(par[6])
  
  # Probabilities for population 1
  P11_1 <- prev1 * Se1 * Se2 + (1 - prev1) * (1 - Sp1) * (1 - Sp2)
  P10_1 <- prev1 * Se1 * (1 - Se2) + (1 - prev1) * (1 - Sp1) * Sp2
  P01_1 <- prev1 * (1 - Se1) * Se2 + (1 - prev1) * Sp1 * (1 - Sp2)
  P00_1 <- prev1 * (1 - Se1) * (1 - Se2) + (1 - prev1) * Sp1 * Sp2
  
  # Probabilities for population 2
  P11_2 <- prev2 * Se1 * Se2 + (1 - prev2) * (1 - Sp1) * (1 - Sp2)
  P10_2 <- prev2 * Se1 * (1 - Se2) + (1 - prev2) * (1 - Sp1) * Sp2
  P01_2 <- prev2 * (1 - Se1) * Se2 + (1 - prev2) * Sp1 * (1 - Sp2)
  P00_2 <- prev2 * (1 - Se1) * (1 - Se2) + (1 - prev2) * Sp1 * Sp2
  
  probs <- c(P11_1, P10_1, P01_1, P00_1, P11_2, P10_2, P01_2, P00_2)
  
  # Log-likelihood
  ll <- sum(counts * log(probs))
  return(-ll)  # Negative for minimization
}

# --- 3. Fit the Model (MLE) ---

# Starting values (on logit scale)
start <- qlogis(c(0.5, 0.5, 0.8, 0.9, 0.8, 0.9))

# Optimization
fit <- optim(start, loglik_fn, counts = counts, hessian = TRUE, method = "BFGS")

# Back-transform estimates
estimates <- plogis(fit$par)
names(estimates) <- c("Prev1", "Prev2", "Se1", "Sp1", "Se2", "Sp2")

# Display estimates
cat("\n--- MLE Estimates ---\n")
print(estimates)

# --- 4. Confidence Intervals via Hessian ---

# Variance-covariance matrix
vcov <- solve(fit$hessian)

# Standard errors on logit scale
se_logit <- sqrt(diag(vcov))

# Delta method for 95% CI
logit_est <- fit$par
lower <- plogis(logit_est - 1.96 * se_logit)
upper <- plogis(logit_est + 1.96 * se_logit)

# Combine results
ci_table <- data.frame(Estimate = estimates, Lower = lower, Upper = upper)

cat("\n--- 95% CI via Hessian ---\n")
print(ci_table)

# --- 5. Bootstrap Confidence Intervals ---

bootstrap_fn <- function(counts, B = 1000) {
  n_patterns <- length(counts) / 2
  n1 <- sum(counts[1:4])
  n2 <- sum(counts[5:8])
  
  boot_est <- matrix(NA, B, 6)
  
  for (b in 1:B) {
    # Resample
    boot_counts1 <- rmultinom(1, n1, prob = counts[1:4] / n1)
    boot_counts2 <- rmultinom(1, n2, prob = counts[5:8] / n2)
    boot_counts <- c(boot_counts1, boot_counts2)
    
    # Fit model
    boot_fit <- try(optim(start, loglik_fn, counts = boot_counts, method = "BFGS"), silent = TRUE)
    if (inherits(boot_fit, "try-error")) next
    
    boot_est[b, ] <- plogis(boot_fit$par)
  }
  
  # Remove NAs
  boot_est <- boot_est[complete.cases(boot_est), ]
  
  # Percentile CIs
  boot_ci <- apply(boot_est, 2, quantile, probs = c(0.025, 0.975))
  
  list(boot_estimates = boot_est, boot_ci = boot_ci)
}

# Run bootstrap (adjust B as desired)
set.seed(123)
boot_result <- bootstrap_fn(counts, B = 500)

cat("\n--- Bootstrap 95% CI ---\n")
boot_result$boot_ci

# ----------------------------------------------------------
# End of Script
# ----------------------------------------------------------
