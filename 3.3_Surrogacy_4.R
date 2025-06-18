# Load required libraries
library(pROC)        # For ROC curves
library(survival)    # For Cox regression
library(survminer)   # For survival analysis
library(ggplot2)     # For nice plots

# ======== 1️⃣ Example: MI for 2x2 Table ========

compute_mi_table <- function(a, b, c, d) {
  N <- a + b + c + d
  p_st <- matrix(c(a, b, c, d), nrow = 2) / N
  p_s <- rowSums(p_st)
  p_t <- colSums(p_st)
  
  MI <- 0
  for (s in 1:2) {
    for (t in 1:2) {
      if (p_st[s,t] > 0) {
        MI <- MI + p_st[s,t] * log2(p_st[s,t] / (p_s[s] * p_t[t]))
      }
    }
  }
  return(MI)
}

# Example table
a <- 50; b <- 10; c <- 5; d <- 35
mi_2x2 <- compute_mi_table(a, b, c, d)
cat("MI for 2x2 table:", mi_2x2, "bits\n\n")

# ======== 2️⃣ MI across thresholds of score ========

# Simulate example data
set.seed(123)
n <- 200
score <- rnorm(n, mean = ifelse(runif(n) > 0.5, 1.5, 0))
label <- ifelse(score + rnorm(n) > 1, 1, 0)

# ROC object
roc_obj <- roc(label, score, direction = "<")
thresholds <- roc_obj$thresholds

# MI per threshold
mi_values <- numeric(length(thresholds))

for (i in seq_along(thresholds)) {
  thresh <- thresholds[i]
  pred <- ifelse(score >= thresh, 1, 0)
  
  a <- sum(pred == 1 & label == 1)
  b <- sum(pred == 1 & label == 0)
  c <- sum(pred == 0 & label == 1)
  d <- sum(pred == 0 & label == 0)
  
  mi_values[i] <- compute_mi_table(a, b, c, d)
}

# Plot MI vs threshold
plot(thresholds, mi_values, type = "l", lwd = 2,
     xlab = "Threshold", ylab = "Mutual Information (bits)",
     main = "MI along ROC curve")

# ======== 3️⃣ MI in Logistic Regression ========

# Simulate data
set.seed(123)
n <- 500
x1 <- rnorm(n)
x2 <- rbinom(n, 1, 0.5)
linpred <- 0.5 * x1 + 1.0 * x2
p <- 1 / (1 + exp(-linpred))
y <- rbinom(n, 1, p)

# Fit logistic regression
logit_model <- glm(y ~ x1 + x2, family = binomial)

# Predicted probabilities
p_hat <- predict(logit_model, type = "response")

# Marginal prevalence
p_Y <- mean(y)

# Compute MI
mi_logit <- mean(
  y * log2(p_hat / p_Y) +
    (1 - y) * log2((1 - p_hat) / (1 - p_Y))
)

cat("MI for Logistic Regression:", mi_logit, "bits\n\n")

# ======== 4️⃣ MI in Cox Regression ========

# Simulate survival data
set.seed(123)
n <- 300
x1 <- rnorm(n)
x2 <- rbinom(n, 1, 0.5)
hazard <- exp(0.7 * x1 + 1.2 * x2)
time <- rexp(n, rate = hazard)
censoring <- rexp(n, rate = 0.1)
event <- as.numeric(time <= censoring)
observed_time <- pmin(time, censoring)

# Fit Cox model
cox_model <- coxph(Surv(observed_time, event) ~ x1 + x2)

# Predict risk score (linear predictor)
risk_score <- predict(cox_model, type = "lp")

# Choose a fixed time t_star (say t=2) → compute survival probability
t_star <- 2
surv_fit <- survfit(cox_model)
# For simplicity, we can binarize event at t_star
event_at_t <- as.numeric(observed_time <= t_star & event == 1)

# ROC on risk score
roc_obj_cox <- roc(event_at_t, risk_score, direction = "<")
thresholds_cox <- roc_obj_cox$thresholds
mi_values_cox <- numeric(length(thresholds_cox))

# Compute MI across thresholds
for (i in seq_along(thresholds_cox)) {
  thresh <- thresholds_cox[i]
  pred <- ifelse(risk_score >= thresh, 1, 0)
  
  a <- sum(pred == 1 & event_at_t == 1)
  b <- sum(pred == 1 & event_at_t == 0)
  c <- sum(pred == 0 & event_at_t == 1)
  d <- sum(pred == 0 & event_at_t == 0)
  
  mi_values_cox[i] <- compute_mi_table(a, b, c, d)
}

# Plot MI vs threshold for Cox model
plot(thresholds_cox, mi_values_cox, type = "l", lwd = 2,
     xlab = "Threshold (Risk Score)", ylab = "Mutual Information (bits)",
     main = "MI along ROC (Cox model at t=2)")

