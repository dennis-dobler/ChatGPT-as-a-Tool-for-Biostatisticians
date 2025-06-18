# We'll use optim() for optimization; no special packages needed
#
#
loglik_hui_walter <- function(params, counts) {
  # Parameters to estimate
  Se1 <- params[1] # Sensitivity Test A
  Sp1 <- params[2] # Specificity Test A
  Se2 <- params[3] # Sensitivity Test B
  Sp2 <- params[4] # Specificity Test B
  pi1 <- params[5] # Prevalence Population 1
  pi2 <- params[6] # Prevalence Population 2
  
  # Check constraints
  if (any(c(Se1, Sp1, Se2, Sp2, pi1, pi2) < 0) || any(c(Se1, Sp1, Se2, Sp2, pi1, pi2) > 1)) {
    return(Inf)
  }
  
  # Pattern probabilities
  prob_pattern <- function(Se1, Sp1, Se2, Sp2, pi) {
    c(
      (1 - pi) * Sp1 * Sp2 + pi * (1 - Se1) * (1 - Se2),  # 00
      (1 - pi) * Sp1 * (1 - Sp2) + pi * (1 - Se1) * Se2,  # 01
      (1 - pi) * (1 - Sp1) * Sp2 + pi * Se1 * (1 - Se2),  # 10
      (1 - pi) * (1 - Sp1) * (1 - Sp2) + pi * Se1 * Se2   # 11
    )
  }
  
  p1 <- prob_pattern(Se1, Sp1, Se2, Sp2, pi1)
  p2 <- prob_pattern(Se1, Sp1, Se2, Sp2, pi2)
  
  # Observed counts
  counts1 <- counts[1:4]
  counts2 <- counts[5:8]
  
  # Log-likelihood
  -sum(counts1 * log(p1)) - sum(counts2 * log(p2))
}
#
#
# Test pattern order: 00, 01, 10, 11
counts <- c(80, 10, 20, 90, 50, 30, 10, 110)  # Pop1 + Pop2
#
#
# Initial values: Se1, Sp1, Se2, Sp2, pi1, pi2
# start_vals <- c(0.8, 0.9, 0.85, 0.95, 0.4, 0.6)
#
start_vals <- c(0.3, 0.5, 0.90, 0.9, 0.9, 0.9)

fit <- optim(start_vals, loglik_hui_walter, counts = counts, method = "L-BFGS-B",
             lower = rep(0.001, 6), upper = rep(0.999, 6), control = list(fnscale = 1))

# Print results
params <- fit$par
names(params) <- c("Se_TestA", "Sp_TestA", "Se_TestB", "Sp_TestB", "Prev_Pop1", "Prev_Pop2")
params
#
# Results from ChatGPT
#
# Se_TestA   = 0.88  
# Sp_TestA   = 0.92  
# Se_TestB   = 0.90  
# Sp_TestB   = 0.95  
# Prev_Pop1  = 0.55  
# Prev_Pop2  = 0.75  
#
#
library(numDeriv)

# Use fit from previous `optim()` call
# Recalculate Hessian at the MLE
hess <- hessian(func = loglik_hui_walter, x = fit$par, counts = counts)

# Invert to get variance-covariance matrix
vcov_mat <- solve(hess)

# Standard errors
se <- sqrt(diag(vcov_mat))

# 95% Confidence intervals
lower <- fit$par - 1.96 * se
upper <- fit$par + 1.96 * se

# Combine into a result data frame
result <- data.frame(
  Parameter = c("Se_TestA", "Sp_TestA", "Se_TestB", "Sp_TestB", "Prev_Pop1", "Prev_Pop2"),
  Estimate = fit$par,
  SE = se,
  CI_Lower = pmax(lower, 0),   # Clip lower to 0 (probabilities can't be < 0)
  CI_Upper = pmin(upper, 1)    # Clip upper to 1
)

print(result)
#
# Bootstrap confidence intervals
#
library(dplyr)

# Observed counts (from earlier)
# Patterns: 00,01,10,11 for Pop1, then Pop2
#
patterns <- matrix(c(0,0, 0,1, 1,0, 1,1), ncol=2, byrow=TRUE)
pop_labels <- rep(c("Pop1", "Pop2"), each=4)
total_counts <- rep(counts, times=1)
#
# Ich muss hier parameter setzen.
#
bootstrap_hw <- function(counts, B = 1000) {
#  on.exit(browser())
  param_mat <- matrix(NA, nrow = B, ncol = 6)
  
  for (b in 1:B) {
    # Resample from multinomial per population
    set.seed(b)  # for reproducibility
    
    # Separate Pop1 and Pop2 counts
    n1 <- sum(counts[1:4])
    n2 <- sum(counts[5:8])
    
    boot1 <- rmultinom(1, n1, prob = counts[1:4] / n1)
    boot2 <- rmultinom(1, n2, prob = counts[5:8] / n2)
    
    boot_counts <- c(boot1, boot2)
    
    # Fit model to bootstrap sample
    fit_b <- tryCatch({
      optim(
        #   par = c(0.8, 0.9, 0.8, 0.9, 0.5, 0.6),
            par=start_vals,
            fn = loglik_hui_walter,
            counts = boot_counts,
            method = "L-BFGS-B",
            lower = rep(0.001, 6),
            upper = rep(0.999, 6),
            control = list(fnscale = 1))
    }, error = function(e) NULL)
    
    # Store parameters if converged
    if (!is.null(fit_b) && fit_b$convergence == 0) {
      param_mat[b, ] <- fit_b$par
    }
  }
  
  # Remove failed iterations
  param_mat <- param_mat[complete.cases(param_mat), ]
  
  # Column names
  colnames(param_mat) <- c("Se_TestA", "Sp_TestA", "Se_TestB", "Sp_TestB", "Prev_Pop1", "Prev_Pop2")
  
  return(param_mat)
}
#
bootstrap_hw(counts,3)
#
# Run bootstrap
#
set.seed(123)
boot_results <- bootstrap_hw(counts, B = 1000)
#
# Summarise results
#
param_summary <- apply(boot_results, 2, function(x) {
  c(Estimate = mean(x), 
    Lower = quantile(x, 0.025), 
    Upper = quantile(x, 0.975))
})

t(round(t(param_summary), 3))
#
#
# Update the parametrization to a logistic parametrization of all parameters
#
loglik_logit_param <- function(theta, counts) {
  # Transform to probability scale using logistic function
  logistic <- function(x) 1 / (1 + exp(-x))
  
  Se1 <- logistic(theta[1])
  Sp1 <- logistic(theta[2])
  Se2 <- logistic(theta[3])
  Sp2 <- logistic(theta[4])
  pi1 <- logistic(theta[5])
  pi2 <- logistic(theta[6])
  
  # Pattern probabilities (same as before)
  prob_pattern <- function(Se1, Sp1, Se2, Sp2, pi) {
    c(
      (1 - pi) * Sp1 * Sp2 + pi * (1 - Se1) * (1 - Se2),  # 00
      (1 - pi) * Sp1 * (1 - Sp2) + pi * (1 - Se1) * Se2,  # 01
      (1 - pi) * (1 - Sp1) * Sp2 + pi * Se1 * (1 - Se2),  # 10
      (1 - pi) * (1 - Sp1) * (1 - Sp2) + pi * Se1 * Se2   # 11
    )
  }
  
  p1 <- prob_pattern(Se1, Sp1, Se2, Sp2, pi1)
  p2 <- prob_pattern(Se1, Sp1, Se2, Sp2, pi2)
  
  counts1 <- counts[1:4]
  counts2 <- counts[5:8]
  
  # Return negative log-likelihood
  -sum(counts1 * log(p1)) - sum(counts2 * log(p2))
}
#
# optimize in logit space
#
# Initial values in logit scale
logit <- function(p) log(p / (1 - p))

start_vals <- logit(c(0.8, 0.9, 0.85, 0.95, 0.5, 0.6))

fit_logit <- optim(start_vals, loglik_logit_param, counts = counts, method = "BFGS")

# Transform estimates back to probability scale
logistic <- function(x) 1 / (1 + exp(-x))
params_prob <- logistic(fit_logit$par)
names(params_prob) <- c("Se_TestA", "Sp_TestA", "Se_TestB", "Sp_TestB", "Prev_Pop1", "Prev_Pop2")
print(params_prob)
#
# update boostrap function for logit parametrization
#
bootstrap_hw_logit <- function(counts, B = 1000, seed = 123) {
  set.seed(seed)
  
  # Logistic transform
  logistic <- function(x) 1 / (1 + exp(-x))
  logit <- function(p) log(p / (1 - p))
  
  param_mat <- matrix(NA, nrow = B, ncol = 6)
  
  for (b in 1:B) {
    # Separate Pop1 and Pop2
    n1 <- sum(counts[1:4])
    n2 <- sum(counts[5:8])
    
    # Bootstrap resampling from multinomial
    boot1 <- rmultinom(1, n1, prob = counts[1:4] / n1)
    boot2 <- rmultinom(1, n2, prob = counts[5:8] / n2)
    boot_counts <- c(boot1, boot2)
    
    # Fit model with logit parametrization
    fit_b <- tryCatch({
      optim(par = logit(c(0.8, 0.9, 0.85, 0.95, 0.5, 0.6)),
            fn = loglik_logit_param,
            counts = boot_counts,
            method = "BFGS")
    }, error = function(e) NULL)
    
    # Store if successful
    if (!is.null(fit_b) && fit_b$convergence == 0) {
      param_mat[b, ] <- logistic(fit_b$par)  # back-transform to probability scale
    }
  }
  
  param_mat <- param_mat[complete.cases(param_mat), ]
  colnames(param_mat) <- c("Se_TestA", "Sp_TestA", "Se_TestB", "Sp_TestB", "Prev_Pop1", "Prev_Pop2")
  
  return(param_mat)
}
#
boot_logit_results <- bootstrap_hw_logit(counts, B = 1000)

# Summary: Mean + Percentile-based 95% CI
param_summary <- apply(boot_logit_results, 2, function(x) {
  c(Estimate = mean(x), 
    Lower = quantile(x, 0.025), 
    Upper = quantile(x, 0.975))
})

t(round(t(param_summary), 3))
#
# classical calculation of CIs
#
#
# Logistic and logit helpers
logistic <- function(x) 1 / (1 + exp(-x))
logit <- function(p) log(p / (1 - p))

# Initial logit-scale estimates
start_vals <- logit(c(0.8, 0.9, 0.85, 0.95, 0.5, 0.6))

# Fit the model
fit_logit <- optim(par = start_vals, 
                   fn = loglik_logit_param, 
                   counts = counts,
                   method = "BFGS", 
                   hessian = TRUE)


# Fit the model
fit_logit <- optim(par = start_vals, 
                   fn = loglik_logit_param, 
                   counts = counts,
                   method = "BFGS", 
                   hessian = TRUE)
library(numDeriv)

# If you didn't use `hessian = TRUE`, compute it like this:
# hess <- hessian(loglik_logit_param, x = fit_logit$par, counts = counts)
hess <- fit_logit$hessian

# Invert the Hessian to get the variance-covariance matrix
vcov_mat <- solve(hess)

# Standard errors on logit scale
se_logit <- sqrt(diag(vcov_mat))

# Point estimates on probability scale
est_prob <- logistic(fit_logit$par)
#
# Defining the output
#
# Delta method standard errors on probability scale
se_prob <- se_logit * est_prob * (1 - est_prob)

# Confidence intervals
lower <- est_prob - 1.96 * se_prob
upper <- est_prob + 1.96 * se_prob

# Clip to [0, 1] bounds
lower <- pmax(lower, 0)
upper <- pmin(upper, 1)

# Organize into a data frame
ci_hessian <- data.frame(
  Parameter = c("Se_TestA", "Sp_TestA", "Se_TestB", "Sp_TestB", "Prev_Pop1", "Prev_Pop2"),
  Estimate = round(est_prob, 3),
  SE = round(se_prob, 4),
  CI_Lower = round(lower, 3),
  CI_Upper = round(upper, 3)
)

print(ci_hessian)


