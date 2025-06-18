# Hui-Walter paradigm: 2 populations, 2 tests
# Comprehensive R script with:
# - Logistic reparametrization
# - Point estimates
# - Hessian-based CIs
# - Bootstrap CIs

# --- Setup ---

library(tidyverse)

# --- Step 1: Simulate example data ---

# True parameters
Se_A <- 0.90
Sp_A <- 0.95
Se_B <- 0.85
Sp_B <- 0.92
pi1  <- 0.60
pi2  <- 0.20

# Function to compute P(test outcome combo | parameters)
get_probs <- function(pi, Se_A, Sp_A, Se_B, Sp_B) {
  probs <- matrix(0, nrow = 2, ncol = 2)
  
  probs[1,1] <- pi * Se_A * Se_B + (1 - pi) * (1 - Sp_A) * (1 - Sp_B)
  probs[1,2] <- pi * Se_A * (1 - Se_B) + (1 - pi) * (1 - Sp_A) * Sp_B
  probs[2,1] <- pi * (1 - Se_A) * Se_B + (1 - pi) * Sp_A * (1 - Sp_B)
  probs[2,2] <- pi * (1 - Se_A) * (1 - Se_B) + (1 - pi) * Sp_A * Sp_B
  
  return(probs)
}

# Simulate data
set.seed(123)

N1 <- 1000
N2 <- 1000

probs1 <- get_probs(pi1, Se_A, Sp_A, Se_B, Sp_B)
probs2 <- get_probs(pi2, Se_A, Sp_A, Se_B, Sp_B)

counts1 <- as.vector(rmultinom(1, N1, prob = as.vector(probs1)))
counts2 <- as.vector(rmultinom(1, N2, prob = as.vector(probs2)))

counts <- c(counts1, counts2)

cat("Observed counts (Population 1):\n")
print(matrix(counts1, 2, 2))
cat("\nObserved counts (Population 2):\n")
print(matrix(counts2, 2, 2))

# --- Step 2: Define logistic log-likelihood ---

loglik <- function(theta, counts) {
  pi1 <- plogis(theta[1])
  pi2 <- plogis(theta[2])
  Se_A <- plogis(theta[3])
  Sp_A <- plogis(theta[4])
  Se_B <- plogis(theta[5])
  Sp_B <- plogis(theta[6])
  
  p1 <- get_probs(pi1, Se_A, Sp_A, Se_B, Sp_B)
  p2 <- get_probs(pi2, Se_A, Sp_A, Se_B, Sp_B)
  
  ll <- sum(counts[1:4] * log(as.vector(p1))) +
    sum(counts[5:8] * log(as.vector(p2)))
  
  return(-ll)  # for minimization
}

# --- Step 3: MLE estimation ---

theta_init <- qlogis(c(0.5, 0.5, 0.8, 0.8, 0.8, 0.8))

fit <- optim(theta_init, loglik, counts = counts, method = "BFGS", hessian = TRUE)

estimate_prob <- plogis(fit$par)
names(estimate_prob) <- c("pi1", "pi2", "Se_A", "Sp_A", "Se_B", "Sp_B")

cat("\n--- Point estimates (MLE): ---\n")
print(estimate_prob)

# --- Step 4: Hessian-based 95% CIs ---

vcov_mat <- solve(fit$hessian)
se_logit <- sqrt(diag(vcov_mat))

lower_logit <- fit$par - 1.96 * se_logit
upper_logit <- fit$par + 1.96 * se_logit

lower_prob <- plogis(lower_logit)
upper_prob <- plogis(upper_logit)

results_hessian_CI <- data.frame(
  Parameter = names(estimate_prob),
  Estimate = estimate_prob,
  Lower_95_CI = lower_prob,
  Upper_95_CI = upper_prob
)

cat("\n--- Hessian-based 95% CIs (Wald): ---\n")
print(results_hessian_CI)

# --- Step 5: Bootstrap 95% CIs ---

# Simulate individual-level data for bootstrap
sim_data <- function(N, probs) {
  test_combos <- expand.grid(A = c(1, 0), B = c(1, 0))
  sample_idx <- sample(1:4, size = N, replace = TRUE, prob = as.vector(probs))
  df <- test_combos[sample_idx, ]
  return(df)
}

df1_orig <- sim_data(N1, probs1)
df2_orig <- sim_data(N2, probs2)

# Helper function to fit model
fit_model_logistic <- function(counts) {
  theta_init <- qlogis(c(0.5, 0.5, 0.8, 0.8, 0.8, 0.8))
  fit <- optim(theta_init, loglik, counts = counts, method = "BFGS")
  return(plogis(fit$par))
}

# Bootstrap loop
n_boot <- 1000
boot_estimates <- matrix(NA, nrow = n_boot, ncol = 6)
colnames(boot_estimates) <- names(estimate_prob)

set.seed(12345)
pb <- txtProgressBar(min = 0, max = n_boot, style = 3)

for (b in 1:n_boot) {
  df1_boot <- df1_orig[sample(1:N1, N1, replace = TRUE), ]
  df2_boot <- df2_orig[sample(1:N2, N2, replace = TRUE), ]
  
  counts1 <- table(factor(df1_boot$A, levels = c(1, 0)),
                   factor(df1_boot$B, levels = c(1, 0)))
  counts2 <- table(factor(df2_boot$A, levels = c(1, 0)),
                   factor(df2_boot$B, levels = c(1, 0)))
  
  counts_boot <- c(as.vector(counts1), as.vector(counts2))
  
  est <- tryCatch({
    fit_model_logistic(counts_boot)
  }, error = function(e) rep(NA, 6))
  
  boot_estimates[b, ] <- est
  setTxtProgressBar(pb, b)
}

close(pb)

boot_estimates <- boot_estimates[complete.cases(boot_estimates), ]

boot_CI <- apply(boot_estimates, 2, quantile, probs = c(0.025, 0.975))

cat("\n--- Bootstrap 95% CIs (percentile): ---\n")
print(t(boot_CI))

# --- Compare to true values ---

true_vals <- c(pi1, pi2, Se_A, Sp_A, Se_B, Sp_B)
names(true_vals) <- names(estimate_prob)

cat("\n--- True parameter values: ---\n")
print(true_vals)

