# Load required packages
library(stats4)  # for mle
library(numDeriv)  # for Hessian
library(boot)  # for bootstrap

# Simulate example data
# True parameters
Se1_true <- 0.85
Se2_true <- 0.90
Sp1_true <- 0.95
Sp2_true <- 0.92
pi1_true <- 0.3
pi2_true <- 0.6
N1 <- 500
N2 <- 500

# Function to simulate 2x2 table for one population
simulate_table <- function(N, pi, Se1, Se2, Sp1, Sp2) {
  D <- rbinom(N, 1, pi)
  T1 <- ifelse(D == 1, rbinom(N, 1, Se1), rbinom(N, 1, 1 - Sp1))
  T2 <- ifelse(D == 1, rbinom(N, 1, Se2), rbinom(N, 1, 1 - Sp2))
  table(factor(T1, levels = 0:1), factor(T2, levels = 0:1))
}

# Simulate data for both populations
tab1 <- simulate_table(N1, pi1_true, Se1_true, Se2_true, Sp1_true, Sp2_true)
tab2 <- simulate_table(N2, pi2_true, Se1_true, Se2_true, Sp1_true, Sp2_true)

# Convert tables to counts
counts1 <- as.vector(tab1)
counts2 <- as.vector(tab2)

# Negative log-likelihood with logistic reparam
nll <- function(logit_Se1, logit_Se2, logit_Sp1, logit_Sp2, logit_pi1, logit_pi2) {
  # Transform back
  Se1 <- plogis(logit_Se1)
  Se2 <- plogis(logit_Se2)
  Sp1 <- plogis(logit_Sp1)
  Sp2 <- plogis(logit_Sp2)
  pi1 <- plogis(logit_pi1)
  pi2 <- plogis(logit_pi2)
  
  # For each population, compute cell probabilities
  probs <- function(pi) {
    c(
      pi * (1-Se1)*(1-Se2) + (1-pi) * Sp1 * Sp2,
      pi * (1-Se1)*Se2    + (1-pi) * Sp1 * (1-Sp2),
      pi * Se1*(1-Se2)    + (1-pi) * (1-Sp1) * Sp2,
      pi * Se1*Se2        + (1-pi) * (1-Sp1) * (1-Sp2)
    )
  }
  
  p1 <- probs(pi1)
  p2 <- probs(pi2)
  
  # Log-likelihood
  ll1 <- sum(counts1 * log(p1))
  ll2 <- sum(counts2 * log(p2))
  
  return( - (ll1 + ll2) )  # negative log-likelihood
}

# Initial values on logit scale
init_vals <- rep(0, 6)  # logit(0.5) = 0

# Fit model using mle
fit <- mle(nll, start = list(
  logit_Se1 = init_vals[1], 
  logit_Se2 = init_vals[2], 
  logit_Sp1 = init_vals[3], 
  logit_Sp2 = init_vals[4], 
  logit_pi1 = init_vals[5], 
  logit_pi2 = init_vals[6]
), method = "BFGS")

# Extract estimates and transform back
summary(fit)
estimates_logit <- coef(fit)
estimates <- plogis(estimates_logit)
names(estimates) <- c("Se1", "Se2", "Sp1", "Sp2", "pi1", "pi2")
print(estimates)

# CI via Hessian
hessian_mat <- vcov(fit)
se_logit <- sqrt(diag(hessian_mat))
ci_lower <- plogis(estimates_logit - 1.96 * se_logit)
ci_upper <- plogis(estimates_logit + 1.96 * se_logit)

ci_hessian <- data.frame(
  Parameter = names(estimates),
  Estimate = estimates,
  Lower95 = ci_lower,
  Upper95 = ci_upper
)
print(ci_hessian)

# Bootstrap function
bootstrap_fn <- function(data, indices) {
  # resample counts with replacement
  counts1_boot <- data[indices[1:4]]
  counts2_boot <- data[indices[5:8]]
  
  nll_boot <- function(logit_Se1, logit_Se2, logit_Sp1, logit_Sp2, logit_pi1, logit_pi2) {
    Se1 <- plogis(logit_Se1)
    Se2 <- plogis(logit_Se2)
    Sp1 <- plogis(logit_Sp1)
    Sp2 <- plogis(logit_Sp2)
    pi1 <- plogis(logit_pi1)
    pi2 <- plogis(logit_pi2)
    
    probs <- function(pi) {
      c(
        pi * (1-Se1)*(1-Se2) + (1-pi) * Sp1 * Sp2,
        pi * (1-Se1)*Se2    + (1-pi) * Sp1 * (1-Sp2),
        pi * Se1*(1-Se2)    + (1-pi) * (1-Sp1) * Sp2,
        pi * Se1*Se2        + (1-pi) * (1-Sp1) * (1-Sp2)
      )
    }
    
    p1 <- probs(pi1)
    p2 <- probs(pi2)
    
    ll1 <- sum(counts1_boot * log(p1))
    ll2 <- sum(counts2_boot * log(p2))
    
    return( - (ll1 + ll2) )
  }
  
  fit_boot <- mle(nll_boot, start = list(
    logit_Se1 = 0, logit_Se2 = 0, logit_Sp1 = 0, logit_Sp2 = 0, logit_pi1 = 0, logit_pi2 = 0
  ), method = "BFGS")
  
  return( plogis(coef(fit_boot)) )
}

# Prepare data for boot
boot_data <- c(counts1, counts2)

# Run bootstrap
set.seed(123)
boot_out <- boot(boot_data, statistic = bootstrap_fn, R = 200)

# Bootstrap CIs
boot_ci <- boot.ci(boot_out, type = "perc", index = 1:6)
print(boot_ci)
