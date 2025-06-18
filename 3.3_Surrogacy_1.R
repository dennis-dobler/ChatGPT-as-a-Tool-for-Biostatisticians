# Load required libraries
library(pROC)
library(survival)
library(ggplot2)

# ---------- Basic Functions ----------

# Function: Mutual Information for 2x2 table
compute_MI_2x2 <- function(table2x2) {
  N <- sum(table2x2)
  p_joint <- table2x2 / N
  p_S <- rowSums(p_joint)
  p_T <- colSums(p_joint)
  
  MI <- 0
  for (s in 1:2) {
    for (t in 1:2) {
      p_st <- p_joint[s, t]
      if (p_st > 0) {
        MI <- MI + p_st * log2(p_st / (p_S[s] * p_T[t]))
      }
    }
  }
  return(MI)
}

# Function: Mutual Information from multinomial table (binned scores)
compute_MI_multinomial <- function(table_bin_T) {
  N <- sum(table_bin_T)
  p_joint <- table_bin_T / N
  p_bin <- rowSums(p_joint)
  p_T <- colSums(p_joint)
  
  MI <- 0
  for (b in 1:nrow(p_joint)) {
    for (t in 1:ncol(p_joint)) {
      p_bt <- p_joint[b, t]
      if (p_bt > 0) {
        MI <- MI + p_bt * log2( p_bt / (p_bin[b] * p_T[t]) )
      }
    }
  }
  return(MI)
}

# Function: Entropy of binary outcome
compute_entropy_binary <- function(p1) {
  if (p1 == 0 || p1 == 1) {
    return(0)
  }
  return( -p1 * log2(p1) - (1 - p1) * log2(1 - p1) )
}

# ---------- PART 1: MI from 2x2 Table ----------

# Example 2x2 table
table_2x2 <- matrix(c(30, 10,
                      10, 50),
                    nrow = 2, byrow = TRUE)
cat("=== MI from 2x2 Table ===\n")
print(table_2x2)
MI_2x2 <- compute_MI_2x2(table_2x2)
cat("Mutual Information (bits) =", MI_2x2, "\n\n")

# ---------- PART 2: MI vs Threshold on ROC Curve ----------

# Simulate data for ROC
set.seed(123)
n <- 1000
T_true <- rbinom(n, 1, 0.4)
S_score <- 0.8 * T_true + rnorm(n, 0, 1)

# Compute ROC
roc_obj <- roc(T_true, S_score, quiet = TRUE)
thresholds <- roc_obj$thresholds
n_thresh <- length(thresholds)

MI_values <- numeric(n_thresh)

# Loop over thresholds
for (i in 1:n_thresh) {
  thresh <- thresholds[i]
  S_binary <- ifelse(S_score >= thresh, 1, 0)
  table_bin <- table(S_binary, T_true)
  
  # Ensure full 2x2
  full_table <- matrix(0, nrow = 2, ncol = 2)
  rownames(full_table) <- c("0", "1")
  colnames(full_table) <- c("0", "1")
  full_table[rownames(table_bin), colnames(table_bin)] <- table_bin
  
  MI_values[i] <- compute_MI_2x2(full_table)
}

# Plot MI vs threshold
df_MI_ROC <- data.frame(Threshold = thresholds, MutualInformation = MI_values)

ggplot(df_MI_ROC, aes(x = Threshold, y = MutualInformation)) +
  geom_line(color = "blue", size = 1.2) +
  labs(title = "MI vs Threshold (ROC Curve)", x = "Threshold", y = "Mutual Information (bits)") +
  theme_minimal()

# ---------- PART 3: MI from Logistic Regression ----------

# Simulate logistic regression data
set.seed(123)
n <- 2000
X1 <- rbinom(n, 1, 0.5)
X2 <- rbinom(n, 1, 0.5)
lin_pred <- -1 + 1.5 * X1 + 2 * X2 + 0.5 * X1 * X2
prob_T1 <- 1 / (1 + exp(-lin_pred))
T_logistic <- rbinom(n, 1, prob_T1)

# Fit logistic regression
log_model <- glm(T_logistic ~ X1 + X2 + I(X1*X2), family = binomial)
predicted_prob <- predict(log_model, type = "response")

# Binning predicted probabilities
n_bins <- 10
bins <- cut(predicted_prob, breaks = quantile(predicted_prob, probs = seq(0,1,length.out = n_bins+1)),
            include.lowest = TRUE, labels = FALSE)

table_bin_log <- table(bins, T_logistic)
MI_logistic <- compute_MI_multinomial(table_bin_log)

# Compute R²_info
p1_log <- mean(T_logistic)
entropy_T_log <- compute_entropy_binary(p1_log)
R2_info_log <- MI_logistic / entropy_T_log

cat("=== MI from Logistic Regression ===\n")
cat("Mutual Information (bits) =", MI_logistic, "\n")
cat("Entropy of T =", entropy_T_log, "\n")
cat("R²_info =", R2_info_log, "\n\n")

# ---------- PART 4: MI from Cox Regression ----------

# Simulate survival data
set.seed(123)
n <- 2000
X1 <- rbinom(n, 1, 0.5)
X2 <- rnorm(n, 0, 1)
linpred <- 0.8 * X1 + 1.2 * X2
baseline_hazard <- 0.05
time <- rexp(n, rate = baseline_hazard * exp(linpred))
censoring_time <- runif(n, 0, 10)
event <- ifelse(time <= censoring_time, 1, 0)
time_obs <- pmin(time, censoring_time)

# Fit Cox model
cox_model <- coxph(Surv(time_obs, event) ~ X1 + X2)
risk_score <- predict(cox_model, type = "lp")

# Choose t0
t0 <- 5
base_surv <- survfit(cox_model)
S0_t0 <- summary(base_surv, times = t0)$surv
if (is.na(S0_t0)) {
  closest_idx <- which.min(abs(base_surv$time - t0))
  S0_t0 <- base_surv$surv[closest_idx]
}

# Individual survival probabilities at t0
S_i_t0 <- S0_t0 ^ exp(risk_score)

# Binary outcome: survived past t0?
T_binary_surv <- as.integer(time_obs > t0)

# Binning risk score
bins_cox <- cut(risk_score, breaks = quantile(risk_score, probs = seq(0,1,length.out = n_bins+1)),
                include.lowest = TRUE, labels = FALSE)

table_bin_cox <- table(bins_cox, T_binary_surv)
MI_cox <- compute_MI_multinomial(table_bin_cox)

# Compute R²_info
p1_surv <- mean(T_binary_surv)
entropy_T_surv <- compute_entropy_binary(p1_surv)
R2_info_cox <- MI_cox / entropy_T_surv

cat("=== MI from Cox Regression at t0 =", t0, "===\n")
cat("Mutual Information (bits) =", MI_cox, "\n")
cat("Entropy of T =", entropy_T_surv, "\n")
cat("R²_info =", R2_info_cox, "\n\n")
