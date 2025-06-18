# ====================================================
# Comprehensive Script: Mutual Information (MI) Tools
# Topics:
# ✅ MI for 2x2 table
# ✅ MI vs threshold for ROC curve
# ✅ MI for logistic regression
# ✅ MI for Cox regression (landmark time)
# ====================================================

# ---- Load required packages ----
library(pROC)      # ROC curve
library(survival)  # Cox regression
library(survminer) # (optional) Kaplan-Meier plots

# ---- MI Function for Contingency Table ----
compute_MI <- function(tbl, logbase = 2) {
  N <- sum(tbl)
  p_joint <- tbl / N
  p_row <- rowSums(tbl) / N
  p_col <- colSums(tbl) / N
  
  MI <- 0
  for (i in 1:nrow(tbl)) {
    for (j in 1:ncol(tbl)) {
      if (p_joint[i,j] > 0) {
        MI <- MI + p_joint[i,j] * log(p_joint[i,j] / (p_row[i] * p_col[j]), base = logbase)
      }
    }
  }
  return(MI)
}

# ====================================================
# Part 1: MI for a 2x2 Table
# ====================================================
cat("\n===== Part 1: MI for a 2x2 Table =====\n")

# Example 2x2 table
tbl_2x2 <- matrix(c(30, 10,
                    20, 40), nrow = 2, byrow = TRUE)
rownames(tbl_2x2) <- c("S=0", "S=1")
colnames(tbl_2x2) <- c("Y=0", "Y=1")
print(tbl_2x2)

# Compute MI
MI_2x2 <- compute_MI(tbl_2x2)
cat("Mutual Information (2x2 table):", MI_2x2, "bits\n")

# ====================================================
# Part 2: MI vs Threshold for ROC Curve
# ====================================================
cat("\n===== Part 2: MI vs Threshold for ROC Curve =====\n")

# Simulate example data
set.seed(123)
n <- 500
Y <- rbinom(n, 1, 0.4)
S <- 0.5 * Y + rnorm(n)

# Compute ROC
roc_obj <- roc(Y, S)
thresholds <- roc_obj$thresholds

# Compute MI across thresholds
MI_values <- numeric(length(thresholds))

for (i in seq_along(thresholds)) {
  t <- thresholds[i]
  S_pred <- ifelse(S >= t, 1, 0)
  tbl <- table(S_pred, Y)
  if (nrow(tbl) < 2) tbl <- rbind(tbl, c(0,0))
  if (ncol(tbl) < 2) tbl <- cbind(tbl, c(0,0))
  MI_values[i] <- compute_MI(tbl)
}

# Plot MI vs Threshold
plot(thresholds, MI_values, type = "l", lwd = 2, col = "blue",
     xlab = "Threshold", ylab = "Mutual Information (bits)",
     main = "MI vs Threshold on ROC Curve")
grid()

# ====================================================
# Part 3: MI for Logistic Regression
# ====================================================
cat("\n===== Part 3: MI for Logistic Regression =====\n")

# Simulate logistic regression data
set.seed(123)
n <- 1000
X1 <- rnorm(n)
X2 <- rbinom(n, 1, 0.3)
linpred <- 0.5 * X1 + 1.0 * X2
prob <- 1 / (1 + exp(-linpred))
Y <- rbinom(n, 1, prob)

# Fit logistic model
logit_model <- glm(Y ~ X1 + X2, family = binomial)
p_hat <- predict(logit_model, type = "response")

# Bin predicted probabilities
n_bins <- 20
bins <- cut(p_hat, breaks = quantile(p_hat, probs = seq(0,1,length.out = n_bins+1)), include.lowest = TRUE)
tbl_logit <- table(bins, Y)
tbl_mat_logit <- as.matrix(tbl_logit)

# Compute MI
MI_logit <- compute_MI(tbl_mat_logit)
cat("Mutual Information (logistic regression):", MI_logit, "bits\n")

# ====================================================
# Part 4: MI for Cox Regression (Landmark Time)
# ====================================================
cat("\n===== Part 4: MI for Cox Regression (Landmark Time) =====\n")

# Simulate survival data
set.seed(123)
n <- 1000
X1 <- rnorm(n)
X2 <- rbinom(n, 1, 0.4)
linpred <- 0.5 * X1 + 1.0 * X2
T_event <- -log(runif(n)) / (0.1 * exp(linpred))
Censoring_time <- rexp(n, rate = 0.05)
Time <- pmin(T_event, Censoring_time)
Event <- as.numeric(T_event <= Censoring_time)

# Fit Cox model
cox_model <- coxph(Surv(Time, Event) ~ X1 + X2)
eta <- predict(cox_model, type = "lp")

# Define landmark time t*
t_star <- 5
Event_before_tstar <- ifelse(Time <= t_star & Event == 1, 1, 0)

# Bin linear predictor
n_bins <- 20
bins_cox <- cut(eta, breaks = quantile(eta, probs = seq(0,1,length.out = n_bins+1)), include.lowest = TRUE)
tbl_cox <- table(bins_cox, Event_before_tstar)
tbl_mat_cox <- as.matrix(tbl_cox)

# Compute MI
MI_cox <- compute_MI(tbl_mat_cox)
cat("Mutual Information (Cox model, event before t* =", t_star, "):", MI_cox, "bits\n")

