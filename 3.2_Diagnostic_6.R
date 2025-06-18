### Hui-Walter model with logistic reparametrization
### MLE estimation + Hessian-based CIs + Bootstrap CIs

# -----------------------
# 1. Simulate data
# -----------------------

set.seed(123)

# True parameters
Se1_true <- 0.9;  Sp1_true <- 0.95
Se2_true <- 0.8;  Sp2_true <- 0.9
pi1_true <- 0.3
pi2_true <- 0.7

# Simulate population function
simulate_population <- function(n, pi, Se1, Sp1, Se2, Sp2) {
  D <- rbinom(n, 1, pi)
  T1 <- ifelse(D == 1, rbinom(n, 1, Se1), rbinom(n, 1, 1 - Sp1))
  T2 <- ifelse(D == 1, rbinom(n, 1, Se2), rbinom(n, 1, 1 - Sp2))
  table(T1, T2)
}

n <- 1000

tab1 <- simulate_population(n, pi1_true, Se1_true, Sp1_true, Se2_true, Sp2_true)
tab2 <- simulate_population(n, pi2_true, Se1_true, Sp1_true, Se2_true, Sp2_true)

cat("Table 1:\n"); print(tab1)
cat("Table 2:\n"); print(tab2)

# -----------------------
# 2. Log-likelihood function with logistic parametrization
# -----------------------

loglik_hw_logistic <- function(theta) {
  # Transform back to (0,1)
  Se1 <- plogis(theta[1])
  Sp1 <- plogis(theta[2])
  Se2 <- plogis(theta[3])
  Sp2 <- plogis(theta[4])
  pi1 <- plogis(theta[5])
  pi2 <- plogis(theta[6])
  
  # Population 1
  ll1 <- 0
  for (t1 in 0:1) {
    for (t2 in 0:1) {
      p_pos <- pi1 * (ifelse(t1==1, Se1, 1-Se1)) * (ifelse(t2==1, Se2, 1-Se2))
      p_neg <- (1-pi1) * (ifelse(t1==1, 1-Sp1, Sp1)) * (ifelse(t2==1, 1-Sp2, Sp2))
      p_tot <- p_pos + p_neg
      count <- tab1[as.character(t1), as.character(t2)]
      ll1 <- ll1 + count * log(p_tot)
    }
  }
  
  # Population 2
  ll2 <- 0
  for (t1 in 0:1) {
    for (t2 in 0:1) {
      p_pos <- pi2 * (ifelse(t1==1, Se1, 1-Se1)) * (ifelse(t2==1, Se2, 1-Se2))
      p_neg <- (1-pi2) * (ifelse(t1==1, 1-Sp1, Sp1)) * (ifelse(t2==1, 1-Sp2, Sp2))
      p_tot <- p_pos + p_neg
      count <- tab2[as.character(t1), as.character(t2)]
      ll2 <- ll2 + count * log(p_tot)
    }
  }
  
  return(-(ll1 + ll2))
}

# -----------------------
# 3. MLE estimation
# -----------------------

# Logit transform function
logit <- function(p) log(p / (1 - p))

# Initial values on logit scale
init_theta <- c(
  logit(0.8),  # Se1
  logit(0.9),  # Sp1
  logit(0.7),  # Se2
  logit(0.85), # Sp2
  logit(0.5),  # pi1
  logit(0.5)   # pi2
)

# Run optim with Hessian
result <- optim(init_theta, loglik_hw_logistic, method = "BFGS", hessian = TRUE, control = list(maxit=10000))

# Extract estimates on probability scale
theta_hat <- result$par
est <- plogis(theta_hat)
names(est) <- c("Se1", "Sp1", "Se2", "Sp2", "pi1", "pi2")

cat("\nMLE estimates:\n")
print(est)

# -----------------------
# 4. Hessian-based CIs
# -----------------------

# Invert Hessian
hess <- result$hessian
vcov_theta <- solve(hess)
se_theta <- sqrt(diag(vcov_theta))

# Delta method
se_prob <- se_theta * est * (1 - est)

# 95% CIs
lower_hess <- est - 1.96 * se_prob
upper_hess <- est + 1.96 * se_prob

# Result table
ci_hessian <- data.frame(Estimate = est, SE = se_prob, Lower = lower_hess, Upper = upper_hess)
rownames(ci_hessian) <- names(est)

cat("\nHessian-based 95% CIs:\n")
print(ci_hessian)

# -----------------------
# 5. Bootstrap CIs
# -----------------------

n_boot <- 1000
boot_results <- matrix(NA, nrow = n_boot, ncol = 6)
colnames(boot_results) <- c("Se1", "Sp1", "Se2", "Sp2", "pi1", "pi2")

n1_total <- sum(tab1)
n2_total <- sum(tab2)

tab1_vec <- as.vector(tab1)
tab2_vec <- as.vector(tab2)

for (b in 1:n_boot) {
  if (b %% 50 == 0) cat("Bootstrap sample", b, "\n")
  
  # Parametric bootstrap: resample counts
  new_tab1 <- matrix(rmultinom(1, n1_total, prob = tab1_vec / n1_total), nrow = 2, byrow = TRUE)
  new_tab2 <- matrix(rmultinom(1, n2_total, prob = tab2_vec / n2_total), nrow = 2, byrow = TRUE)
  
  tab1 <- new_tab1
  tab2 <- new_tab2
  
  # Re-fit model
  res_b <- try(optim(init_theta, loglik_hw_logistic, method = "BFGS", control = list(maxit=10000)), silent = TRUE)
  
  if (class(res_b) != "try-error") {
    theta_hat_b <- res_b$par
    boot_results[b, ] <- plogis(theta_hat_b)
  }
}

# Remove failed runs
boot_results <- boot_results[complete.cases(boot_results), ]

# Bootstrap percentile CIs
boot_ci <- apply(boot_results, 2, function(x) quantile(x, probs = c(0.025, 0.975)))

cat("\nBootstrap 95% CIs (percentile):\n")
print(t(boot_ci))
