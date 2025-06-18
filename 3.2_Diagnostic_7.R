# =====================================================
# Hui-Walter Latent Class Model - FULL EXAMPLE IN R
# =====================================================

# Load required packages
library(tidyverse)
library(boot)

# =====================================================
# STEP 1: Simulate data
# =====================================================

# True parameter values
true_params <- list(
  Se1 = 0.85,
  Sp1 = 0.90,
  Se2 = 0.80,
  Sp2 = 0.95,
  pi1 = 0.30,
  pi2 = 0.60
)

# Function to simulate one population
simulate_population <- function(n, pi, Se1, Sp1, Se2, Sp2) {
  D <- rbinom(n, 1, pi)
  T1 <- ifelse(D == 1, rbinom(n, 1, Se1), rbinom(n, 1, 1 - Sp1))
  T2 <- ifelse(D == 1, rbinom(n, 1, Se2), rbinom(n, 1, 1 - Sp2))
  data.frame(D, T1, T2)
}

# Simulate populations
set.seed(123)

n1 <- 1000
n2 <- 1000

pop1 <- simulate_population(n1, true_params$pi1, true_params$Se1, true_params$Sp1, true_params$Se2, true_params$Sp2) %>%
  mutate(pop = 1)
pop2 <- simulate_population(n2, true_params$pi2, true_params$Se1, true_params$Sp1, true_params$Se2, true_params$Sp2) %>%
  mutate(pop = 2)

# Combine data
data_all <- bind_rows(pop1, pop2)

# Create contingency table
table_data <- data_all %>%
  group_by(pop, T1, T2) %>%
  summarise(count = n()) %>%
  ungroup()

cat("\n--- Contingency table of test outcomes ---\n")
print(table_data)

# =====================================================
# STEP 2: Define log-likelihood for Hui-Walter model
# =====================================================

loglik <- function(params, table_data) {
  Se1 <- params[1]
  Sp1 <- params[2]
  Se2 <- params[3]
  Sp2 <- params[4]
  pi1 <- params[5]
  pi2 <- params[6]
  
  logL <- 0
  
  for (k in 1:2) {
    pi_k <- ifelse(k == 1, pi1, pi2)
    
    sub_table <- table_data %>% filter(pop == k)
    
    for (i in 0:1) {
      for (j in 0:1) {
        count_ij <- sub_table %>% filter(T1 == i, T2 == j) %>% pull(count)
        if (length(count_ij) == 0) count_ij <- 0
        
        # Probability of (i,j) pattern
        p_ij <- pi_k * 
          (ifelse(i == 1, Se1, 1 - Se1)) * 
          (ifelse(j == 1, Se2, 1 - Se2)) +
          (1 - pi_k) * 
          (ifelse(i == 1, 1 - Sp1, Sp1)) * 
          (ifelse(j == 1, 1 - Sp2, Sp2))
        
        p_ij <- max(p_ij, 1e-10)  # Avoid log(0)
        
        logL <- logL + count_ij * log(p_ij)
      }
    }
  }
  
  return(-logL)  # Negative log-likelihood
}

# =====================================================
# STEP 3: Fit model (maximum likelihood estimation)
# =====================================================

# Initial parameter guesses
init_params <- c(0.8, 0.8, 0.8, 0.8, 0.5, 0.5)

# Optimization
fit <- optim(par = init_params,
             fn = loglik,
             table_data = table_data,
             method = "L-BFGS-B",
             lower = rep(0.01, 6),
             upper = rep(0.99, 6),
             hessian = TRUE)

# Extract estimates
est_params <- fit$par
names(est_params) <- c("Se1", "Sp1", "Se2", "Sp2", "pi1", "pi2")

# Print estimates
cat("\n--- Parameter estimates (MLE) ---\n")
print(est_params)

# =====================================================
# STEP 4: Wald confidence intervals (from Hessian)
# =====================================================

# Compute standard errors
se_params <- sqrt(diag(solve(fit$hessian)))

# Wald CIs
wald_ci <- tibble(
  parameter = names(est_params),
  estimate = est_params,
  lower = est_params - 1.96 * se_params,
  upper = est_params + 1.96 * se_params
)

cat("\n--- Wald confidence intervals ---\n")
print(wald_ci)

# =====================================================
# STEP 5: Bootstrap for robust confidence intervals
# =====================================================

# Bootstrap function
bootstrap_fn <- function(data, indices) {
  data_boot <- data[indices, ]
  
  table_boot <- data_boot %>%
    group_by(pop, T1, T2) %>%
    summarise(count = n()) %>%
    ungroup()
  
  fit_boot <- optim(par = init_params,
                    fn = loglik,
                    table_data = table_boot,
                    method = "L-BFGS-B",
                    lower = rep(0.01, 6),
                    upper = rep(0.99, 6))
  
  return(fit_boot$par)
}

# Prepare data for bootstrapping
data_long <- data_all

# Run bootstrap
set.seed(123)
boot_res <- boot(data = data_long, statistic = bootstrap_fn, R = 500)

# Compute bootstrap percentile CIs
boot_ci <- tibble(
  parameter = names(est_params),
  estimate = est_params,
  lower = apply(boot_res$t, 2, quantile, probs = 0.025),
  upper = apply(boot_res$t, 2, quantile, probs = 0.975)
)

cat("\n--- Bootstrap confidence intervals ---\n")
print(boot_ci)

# =====================================================
# END OF SCRIPT
# =====================================================
