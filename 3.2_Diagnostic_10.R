# Load necessary libraries
library(stats4)
library(numDeriv)

# --- Data Input: Counts per test outcome per population ---

# Example synthetic data:
# Format: c(n_++, n_+-, n_-+, n_--)
# You should replace these with YOUR observed counts

# Population 1 (high prevalence)
counts1 <- c(50, 10, 15, 25)

# Population 2 (low prevalence)
counts2 <- c(10, 5, 20, 65)

# --- Model: Negative Log-Likelihood ---

negloglik <- function(
    logit_pi1, logit_pi2,
    logit_se1, logit_sp1,
    logit_se2, logit_sp2
) {
  # Transform parameters to (0,1) via logistic
  pi1 <- plogis(logit_pi1)
  pi2 <- plogis(logit_pi2)
  se1 <- plogis(logit_se1)
  sp1 <- plogis(logit_sp1)
  se2 <- plogis(logit_se2)
  sp2 <- plogis(logit_sp2)
  
  # Outcome probabilities (4 categories)
  prob_cat <- function(pi) {
    c(
      pi * se1 * se2 + (1 - pi) * (1 - sp1) * (1 - sp2), # ++
      pi * se1 * (1 - se2) + (1 - pi) * (1 - sp1) * sp2, # +-
      pi * (1 - se1) * se2 + (1 - pi) * sp1 * (1 - sp2), # -+
      pi * (1 - se1) * (1 - se2) + (1 - pi) * sp1 * sp2  # --
    )
  }
  
  p1 <- prob_cat(pi1)
  p2 <- prob_cat(pi2)
  
  # Add small epsilon to avoid log(0)
  eps <- 1e-12
  
  -(
    sum(counts1 * log(p1 + eps)) +
      sum(counts2 * log(p2 + eps))
  )
}

# --- Initial parameter guesses ---

init <- c(
  logit(0.5), # logit_pi1
  logit(0.2), # logit_pi2
  logit(0.9), # logit_se1
  logit(0.9), # logit_sp1
  logit(0.9), # logit_se2
  logit(0.9)  # logit_sp2
)

# --- Fit model with MLE ---

fit <- optim(
  par = init,
  fn = negloglik,
  method = "BFGS",
  hessian = TRUE
)

# --- Extract parameter estimates ---

# Transform back from logits
estimates <- plogis(fit$par)
names(estimates) <- c(
  "pi1", "pi2",
  "Se1", "Sp1",
  "Se2", "Sp2"
)

print("MLE Estimates:")
print(estimates)

# --- Compute 95% CI via Hessian ---

# Get variance-covariance matrix
vcov <- solve(fit$hessian)
se <- sqrt(diag(vcov))

# Delta method: CI on probability scale
ci_lower <- plogis(fit$par - 1.96 * se)
ci_upper <- plogis(fit$par + 1.96 * se)

# Display CIs
results <- data.frame(
  Parameter = names(estimates),
  Estimate = estimates,
  CI_lower = ci_lower,
  CI_upper = ci_upper
)

print("95% CI via Hessian:")
print(results)

# --- Bootstrap for robust CIs ---

set.seed(123)
B <- 1000 # Number of bootstrap iterations
boot_results <- matrix(NA, nrow = B, ncol = length(estimates))

for (b in 1:B) {
  # Bootstrap resample counts via multinomial
  boot_counts1 <- as.vector(rmultinom(1, sum(counts1), prob = counts1 / sum(counts1)))
  boot_counts2 <- as.vector(rmultinom(1, sum(counts2), prob = counts2 / sum(counts2)))
  
  # Define new likelihood with bootstrapped data
  negloglik_boot <- function(
    logit_pi1, logit_pi2,
    logit_se1, logit_sp1,
    logit_se2, logit_sp2
  ) {
    pi1 <- plogis(logit_pi1)
    pi2 <- plogis(logit_pi2)
    se1 <- plogis(logit_se1)
    sp1 <- plogis(logit_sp1)
    se2 <- plogis(logit_se2)
    sp2 <- plogis(logit_sp2)
    
    prob_cat <- function(pi) {
      c(
        pi * se1 * se2 + (1 - pi) * (1 - sp1) * (1 - sp2),
        pi * se1 * (1 - se2) + (1 - pi) * (1 - sp1) * sp2,
        pi * (1 - se1) * se2 + (1 - pi) * sp1 * (1 - sp2),
        pi * (1 - se1) * (1 - se2) + (1 - pi) * sp1 * sp2
      )
    }
    
    p1 <- prob_cat(pi1)
    p2 <- prob_cat(pi2)
    
    eps <- 1e-12
    
    -(
      sum(boot_counts1 * log(p1 + eps)) +
        sum(boot_counts2 * log(p2 + eps))
    )
  }
  
  # Re-fit to bootstrap sample
  boot_fit <- try(optim(
    par = fit$par,
    fn = negloglik_boot,
    method = "BFGS"
  ), silent = TRUE)
  
  # Store if successful
  if (inherits(boot_fit, "try-error")) next
  
  boot_results[b, ] <- plogis(boot_fit$par)
}

# Remove failed iterations
boot_results <- boot_results[complete.cases(boot_results), ]

# Compute bootstrap CIs
boot_ci_lower <- apply(boot_results, 2, quantile, probs = 0.025)
boot_ci_upper <- apply(boot_results, 2, quantile, probs = 0.975)

# Display bootstrap CIs
boot_results_df <- data.frame(
  Parameter = names(estimates),
  Estimate = estimates,
  Boot_CI_lower = boot_ci_lower,
  Boot_CI_upper = boot_ci_upper
)

print("Bootstrap 95% CIs:")
print(boot_results_df)

# --- Helper: Logit and inverse logit ---

logit <- function(p) log(p / (1 - p))
