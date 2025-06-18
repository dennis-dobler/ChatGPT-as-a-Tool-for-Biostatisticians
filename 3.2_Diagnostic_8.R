# ==========================================
# Hui-Walter Latent Class Analysis - FULL EXAMPLE
# Two tests, two populations
# Logistic parametrization
# Confidence intervals via Hessian and Bootstrap
# ==========================================

# Load required libraries
library(numDeriv)  # for Hessian
library(boot)      # for bootstrap

# ==========================================
# 1️⃣ Simulate Data
# ==========================================

# True parameters
true_params <- list(
  pi1 = 0.3,
  pi2 = 0.6,
  Se1 = 0.85,
  Sp1 = 0.90,
  Se2 = 0.80,
  Sp2 = 0.95
)

# Function to simulate one population
simulate_population <- function(N, pi, Se1, Sp1, Se2, Sp2) {
  D <- rbinom(N, 1, pi)
  T1 <- ifelse(D == 1, rbinom(N, 1, Se1), rbinom(N, 1, 1 - Sp1))
  T2 <- ifelse(D == 1, rbinom(N, 1, Se2), rbinom(N, 1, 1 - Sp2))
  table(factor(T1, levels=0:1), factor(T2, levels=0:1))
}

# Sample sizes
N1 <- 500
N2 <- 500

# Simulate both populations
set.seed(123)
table1 <- simulate_population(N1, true_params$pi1, true_params$Se1, true_params$Sp1, true_params$Se2, true_params$Sp2)
table2 <- simulate_population(N2, true_params$pi2, true_params$Se1, true_params$Sp1, true_params$Se2, true_params$Sp2)

# Convert tables to vector: (00, 01, 10, 11)
counts1 <- as.vector(table1)
counts2 <- as.vector(table2)
counts <- c(counts1, counts2)

cat("Observed counts per pattern:\n")
print(matrix(counts, nrow=4, byrow=FALSE, dimnames=list(c("00","01","10","11"), c("Pop1","Pop2"))))

# ==========================================
# 2️⃣ Log-Likelihood Function
# ==========================================

loglik_fun <- function(theta, counts) {
  # Unpack parameters
  pi1 <- 1 / (1 + exp(-theta[1]))
  pi2 <- 1 / (1 + exp(-theta[2]))
  Se1 <- 1 / (1 + exp(-theta[3]))
  Sp1 <- 1 / (1 + exp(-theta[4]))
  Se2 <- 1 / (1 + exp(-theta[5]))
  Sp2 <- 1 / (1 + exp(-theta[6]))
  
  # Probabilities for Pop1
  P1 <- numeric(4)
  P1[1] <- pi1 * (1-Se1)*(1-Se2) + (1-pi1) * Sp1 * Sp2
  P1[2] <- pi1 * (1-Se1)*Se2     + (1-pi1) * Sp1 * (1-Sp2)
  P1[3] <- pi1 * Se1 * (1-Se2)   + (1-pi1) * (1-Sp1) * Sp2
  P1[4] <- pi1 * Se1 * Se2       + (1-pi1) * (1-Sp1) * (1-Sp2)
  
  # Probabilities for Pop2
  P2 <- numeric(4)
  P2[1] <- pi2 * (1-Se1)*(1-Se2) + (1-pi2) * Sp1 * Sp2
  P2[2] <- pi2 * (1-Se1)*Se2     + (1-pi2) * Sp1 * (1-Sp2)
  P2[3] <- pi2 * Se1 * (1-Se2)   + (1-pi2) * (1-Sp1) * Sp2
  P2[4] <- pi2 * Se1 * Se2       + (1-pi2) * (1-Sp1) * (1-Sp2)
  
  # Total log-likelihood
  logL <- sum(counts[1:4] * log(P1)) + sum(counts[5:8] * log(P2))
  
  return(-logL)  # Negative for minimization
}

# ==========================================
# 3️⃣ Parameter Estimation (MLE)
# ==========================================

# Initial values (logit scale)
init_theta <- rep(0, 6)

# Run optimizer
fit <- optim(init_theta, loglik_fun, counts=counts, method="BFGS", hessian=TRUE, control=list(maxit=10000))

# Estimated parameters
theta_hat <- fit$par

# Back-transform to probability scale
pi1_hat <- 1 / (1 + exp(-theta_hat[1]))
pi2_hat <- 1 / (1 + exp(-theta_hat[2]))
Se1_hat <- 1 / (1 + exp(-theta_hat[3]))
Sp1_hat <- 1 / (1 + exp(-theta_hat[4]))
Se2_hat <- 1 / (1 + exp(-theta_hat[5]))
Sp2_hat <- 1 / (1 + exp(-theta_hat[6]))

estimates <- c(pi1=pi1_hat, pi2=pi2_hat, Se1=Se1_hat, Sp1=Sp1_hat, Se2=Se2_hat, Sp2=Sp2_hat)
cat("\nMLE Estimates:\n")
print(estimates)

# ==========================================
# 4️⃣ Confidence Intervals via Hessian
# ==========================================

# Variance-covariance matrix
vcov <- solve(fit$hessian)

# Standard errors on logit scale
se_theta <- sqrt(diag(vcov))

# Delta method: transform to probability scale
logit_to_prob <- function(theta, se_theta) {
  p <- 1 / (1 + exp(-theta))
  dp_dtheta <- p * (1 - p)
  se_p <- dp_dtheta * se_theta
  ci_lower <- p - 1.96 * se_p
  ci_upper <- p + 1.96 * se_p
  return(c(estimate=p, lower=ci_lower, upper=ci_upper))
}

# Apply to all parameters
cis_hessian <- t(sapply(1:6, function(i) logit_to_prob(theta_hat[i], se_theta[i])))
rownames(cis_hessian) <- c("pi1", "pi2", "Se1", "Sp1", "Se2", "Sp2")

cat("\nConfidence intervals via Hessian (Delta method):\n")
print(cis_hessian)

# ==========================================
# 5️⃣ Bootstrap Confidence Intervals
# ==========================================

bootstrap_fun <- function(counts, indices) {
  # Resample counts within each population
  boot_counts <- counts
  for (pop in 1:2) {
    idx <- (1:4) + (pop-1)*4
    boot_sample <- rmultinom(1, sum(counts[idx]), counts[idx] / sum(counts[idx]))
    boot_counts[idx] <- boot_sample
  }
  
  # Refit model
  fit_boot <- optim(init_theta, loglik_fun, counts=boot_counts, method="BFGS", control=list(maxit=10000))
  theta_b <- fit_boot$par
  
  # Return back-transformed estimates
  return(c(
    1 / (1 + exp(-theta_b[1])),  # pi1
    1 / (1 + exp(-theta_b[2])),  # pi2
    1 / (1 + exp(-theta_b[3])),  # Se1
    1 / (1 + exp(-theta_b[4])),  # Sp1
    1 / (1 + exp(-theta_b[5])),  # Se2
    1 / (1 + exp(-theta_b[6]))   # Sp2
  ))
}

# Run bootstrap
n_boot <- 1000
cat("\nRunning bootstrap... (may take a moment)\n")
set.seed(456)
boot_results <- replicate(n_boot, bootstrap_fun(counts, NULL))

# Compute percentile bootstrap CIs
boot_cis <- apply(boot_results, 1, function(x) quantile(x, c(0.025, 0.975)))
rownames(boot_cis) <- c("2.5%", "97.5%")
colnames(boot_cis) <- c("pi1", "pi2", "Se1", "Sp1", "Se2", "Sp2")

cat("\nBootstrap percentile confidence intervals:\n")
print(boot_cis)

# ==========================================
# DONE 🚀
# ==========================================
