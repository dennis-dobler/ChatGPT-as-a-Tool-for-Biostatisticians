# Load required packages
library(numDeriv)  # for Hessian
library(MASS)      # for mvrnorm if needed in extensions

# 1️⃣ Simulate example data ---------------------------------------

# True parameters (on probability scale)
true_pi <- c(0.3, 0.6)  # Prevalence in pop 1 and 2
true_Se <- c(0.85, 0.90)  # Sensitivity of test 1 and 2
true_Sp <- c(0.80, 0.75)  # Specificity of test 1 and 2

# Sample sizes
N <- c(500, 500)

# Function to simulate one population
simulate_population <- function(N, pi, Se, Sp) {
  D <- rbinom(N, 1, pi)
  T1 <- ifelse(D == 1, rbinom(N, 1, Se[1]), rbinom(N, 1, 1 - Sp[1]))
  T2 <- ifelse(D == 1, rbinom(N, 1, Se[2]), rbinom(N, 1, 1 - Sp[2]))
  data.frame(D, T1, T2)
}

# Simulate two populations
set.seed(123)
pop1 <- simulate_population(N[1], true_pi[1], true_Se, true_Sp)
pop2 <- simulate_population(N[2], true_pi[2], true_Se, true_Sp)

# Tabulate observed counts (4 cells per population)
tabulate_counts <- function(data) {
  xtabs(~ T1 + T2, data)
}

counts1 <- tabulate_counts(pop1)
counts2 <- tabulate_counts(pop2)

print(counts1)
print(counts2)

# 2️⃣ Log-likelihood with logistic parametrization -------------------

# Logistic (logit) and inverse
logit <- function(p) log(p / (1 - p))
inv_logit <- function(x) 1 / (1 + exp(-x))

# Log-likelihood function
loglik <- function(par) {
  # Unpack parameters
  pi1 <- inv_logit(par[1])
  pi2 <- inv_logit(par[2])
  Se1 <- inv_logit(par[3])
  Sp1 <- inv_logit(par[4])
  Se2 <- inv_logit(par[5])
  Sp2 <- inv_logit(par[6])
  
  # Loop over populations
  ll <- 0
  for (k in 1:2) {
    pi_k <- ifelse(k == 1, pi1, pi2)
    counts_k <- ifelse(k == 1, counts1, counts2)
    
    for (t1 in 0:1) {
      for (t2 in 0:1) {
        # Probabilities given D=1
        p11 <- (Se1^t1) * ((1 - Se1)^(1 - t1)) * 
          (Se2^t2) * ((1 - Se2)^(1 - t2))
        # Probabilities given D=0
        p00 <- ((1 - Sp1)^t1) * (Sp1^(1 - t1)) * 
          ((1 - Sp2)^t2) * (Sp2^(1 - t2))
        # Marginal probability
        P_obs <- pi_k * p11 + (1 - pi_k) * p00
        # Contribution to log-likelihood
        n_obs <- counts_k[t1 + 1, t2 + 1]
        ll <- ll + n_obs * log(P_obs)
      }
    }
  }
  return(-ll)  # return NEGATIVE log-likelihood for minimization
}

# 3️⃣ Fit model -----------------------------------------------------

# Initial values on logit scale
init_par <- rep(0, 6)  # corresponds to probability 0.5

# Optimization
fit <- optim(init_par, loglik, hessian = TRUE, method = "BFGS", control = list(maxit = 10000))

# 4️⃣ Extract estimates ---------------------------------------------

# MLE estimates on probability scale
est <- inv_logit(fit$par)
names(est) <- c("pi1", "pi2", "Se1", "Sp1", "Se2", "Sp2")
print(est)

# 5️⃣ CI from Hessian -----------------------------------------------

# Variance-covariance matrix
vcov_mat <- solve(fit$hessian)

# Delta method to transform to probability scale
se_logit <- sqrt(diag(vcov_mat))
ci_lower <- inv_logit(fit$par - qnorm(0.975) * se_logit)
ci_upper <- inv_logit(fit$par + qnorm(0.975) * se_logit)

# Summary table
results <- data.frame(
  Parameter = names(est),
  Estimate = est,
  CI_lower = ci_lower,
  CI_upper = ci_upper
)
print(results)

# 6️⃣ Bootstrap -----------------------------------------------------

# Bootstrap function
bootstrap_fit <- function(B = 1000, seed = 1234) {
  set.seed(seed)
  boot_estimates <- matrix(NA, nrow = B, ncol = 6)
  
  for (b in 1:B) {
    # Resample each population with replacement
    pop1_boot <- pop1[sample(nrow(pop1), replace = TRUE), ]
    pop2_boot <- pop2[sample(nrow(pop2), replace = TRUE), ]
    
    # Recompute counts
    counts1_boot <- tabulate_counts(pop1_boot)
    counts2_boot <- tabulate_counts(pop2_boot)
    
    # New loglik function using local copies of counts
    loglik_boot <- function(par) {
      pi1 <- inv_logit(par[1])
      pi2 <- inv_logit(par[2])
      Se1 <- inv_logit(par[3])
      Sp1 <- inv_logit(par[4])
      Se2 <- inv_logit(par[5])
      Sp2 <- inv_logit(par[6])
      
      ll <- 0
      for (k in 1:2) {
        pi_k <- ifelse(k == 1, pi1, pi2)
        counts_k <- ifelse(k == 1, counts1_boot, counts2_boot)
        
        for (t1 in 0:1) {
          for (t2 in 0:1) {
            p11 <- (Se1^t1) * ((1 - Se1)^(1 - t1)) * 
              (Se2^t2) * ((1 - Se2)^(1 - t2))
            p00 <- ((1 - Sp1)^t1) * (Sp1^(1 - t1)) * 
              ((1 - Sp2)^t2) * (Sp2^(1 - t2))
            P_obs <- pi_k * p11 + (1 - pi_k) * p00
            n_obs <- counts_k[t1 + 1, t2 + 1]
            ll <- ll + n_obs * log(P_obs)
          }
        }
      }
      return(-ll)
    }
    
    # Fit bootstrap sample
    fit_boot <- optim(init_par, loglik_boot, method = "BFGS", control = list(maxit = 10000))
    
    # Store estimates on probability scale
    boot_estimates[b, ] <- inv_logit(fit_boot$par)
    
    if (b %% 100 == 0) cat("Bootstrap iteration:", b, "\n")
  }
  
  # Return bootstrap matrix
  return(boot_estimates)
}

# Run bootstrap
boot_results <- bootstrap_fit(B = 1000)

# Compute bootstrap percentile CIs
boot_CI <- apply(boot_results, 2, quantile, probs = c(0.025, 0.975))

# Format bootstrap summary
boot_summary <- data.frame(
  Parameter = c("pi1", "pi2", "Se1", "Sp1", "Se2", "Sp2"),
  Estimate = est,
  Boot_CI_lower = boot_CI[1, ],
  Boot_CI_upper = boot_CI[2, ]
)

print(boot_summary)

# Done! -----------------------------------------------------------
