# Load libraries
library(dplyr)
library(survival)

# ---- PART 1: MI for 2x2 Table ----

cat("\n--- MI for 2x2 Table ---\n")

table_2x2 <- matrix(c(30, 10, 5, 55), nrow = 2, byrow = TRUE)
rownames(table_2x2) <- c("S=0", "S=1")
colnames(table_2x2) <- c("T=0", "T=1")

joint_p <- table_2x2 / sum(table_2x2)
p_s <- rowSums(joint_p)
p_t <- colSums(joint_p)

mi <- 0
for (s in 1:2) {
  for (t in 1:2) {
    p_joint <- joint_p[s, t]
    if (p_joint > 0) {
      mi <- mi + p_joint * log2(p_joint / (p_s[s] * p_t[t]))
    }
  }
}

cat("Mutual Information (2x2 Table) =", mi, "bits\n")

# ---- PART 2: MI across thresholds (ROC-like) ----

cat("\n--- MI across thresholds ---\n")

set.seed(123)
n <- 200
score <- rnorm(n)
label <- ifelse(score + rnorm(n, sd=1) > 0, 1, 0)

thresholds <- sort(unique(score), decreasing = TRUE)

compute_mi <- function(tab) {
  joint_p <- tab / sum(tab)
  p_s <- rowSums(joint_p)
  p_t <- colSums(joint_p)
  
  mi <- 0
  for (s in 1:2) {
    for (t in 1:2) {
      p_joint <- joint_p[s, t]
      if (p_joint > 0) {
        mi <- mi + p_joint * log2(p_joint / (p_s[s] * p_t[t]))
      }
    }
  }
  return(mi)
}

mi_values <- numeric(length(thresholds))
for (i in seq_along(thresholds)) {
  thr <- thresholds[i]
  pred_class <- ifelse(score >= thr, 1, 0)
  tab <- table(pred_class, label)
  
  tab_full <- matrix(0, nrow = 2, ncol = 2)
  tab_full[as.numeric(rownames(tab)) + 1, as.numeric(colnames(tab)) + 1] <- tab
  
  mi_values[i] <- compute_mi(tab_full)
}

# Plot MI vs threshold
plot(thresholds, mi_values, type = "l", xlab = "Threshold", ylab = "Mutual Information (bits)",
     main = "MI vs Threshold (MI curve)")

# ---- PART 3: MI for Logistic Regression ----

cat("\n--- MI for Logistic Regression ---\n")

set.seed(123)
n <- 1000
X <- rnorm(n)
beta0 <- -0.5
beta1 <- 1.2
p_T_given_X <- 1 / (1 + exp(-(beta0 + beta1 * X)))
T_bin <- rbinom(n, size = 1, prob = p_T_given_X)

logit_model <- glm(T_bin ~ X, family = binomial)
p_hat <- predict(logit_model, type = "response")

p_T <- mean(T_bin)
H_T <- - (p_T * log2(p_T) + (1 - p_T) * log2(1 - p_T))
H_T_given_X <- mean(- (p_hat * log2(p_hat) + (1 - p_hat) * log2(1 - p_hat)))
MI_logit <- H_T - H_T_given_X

cat("Mutual Information (Logistic Regression) =", MI_logit, "bits\n")

# ---- PART 4: MI for Cox Regression (Event Indicator) ----

cat("\n--- MI for Cox Regression ---\n")

set.seed(123)
n <- 1000
X <- rnorm(n)
baseline_hazard <- 0.1
beta <- 1.0
hazard <- baseline_hazard * exp(beta * X)
T_surv <- rexp(n, rate = hazard)
Censoring_time <- rexp(n, rate = 0.05)
Time_obs <- pmin(T_surv, Censoring_time)
Event <- as.numeric(T_surv <= Censoring_time)

cox_model <- coxph(Surv(Time_obs, Event) ~ X)
lp <- predict(cox_model, type = "lp")

# Approximate MI using logistic regression on event indicator
logit_model_event <- glm(Event ~ X, family = binomial)
p_hat_event <- predict(logit_model_event, type = "response")

p_Event <- mean(Event)
H_Event <- - (p_Event * log2(p_Event) + (1 - p_Event) * log2(1 - p_Event))
H_Event_given_X <- mean(- (p_hat_event * log2(p_hat_event) + (1 - p_hat_event) * log2(1 - p_hat_event)))
MI_cox <- H_Event - H_Event_given_X

cat("Mutual Information (Cox Regression, Event Indicator) =", MI_cox, "bits\n")

