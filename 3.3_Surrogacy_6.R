# --- LIBRARIES ---
library(survival)
library(rms)
library(survAUC)
# --- MI FUNCTION for 2x2 TABLE ---
compute_mi <- function(tbl) {
  p_joint <- tbl / sum(tbl)
  p_s <- rowSums(p_joint)
  p_t <- colSums(p_joint)
  
  mi <- 0
  for (i in 1:2) {
    for (j in 1:2) {
      if (p_joint[i,j] > 0) {
        mi <- mi + p_joint[i,j] * log2(p_joint[i,j] / (p_s[i] * p_t[j]))
      }
    }
  }
  return(mi)
}
# --- MI across thresholds (ROC) ---
compute_mi_roc <- function(score, label) {
  thresholds <- sort(unique(score))
  mi_vec <- numeric(length(thresholds))
  
  for (k in 1:length(thresholds)) {
    thr <- thresholds[k]
    pred <- ifelse(score >= thr, 1, 0)
    tbl <- table(pred, label)
    if (all(dim(tbl) == c(2,2))) {
      mi_vec[k] <- compute_mi(tbl)
    } else {
      mi_vec[k] <- NA
    }
  }
  
  return(data.frame(threshold=thresholds, MI=mi_vec))
}
# --- EXAMPLE: Logistic Regression MI ---
set.seed(123)
n <- 500
x <- rbinom(n, 1, 0.4)
z <- rnorm(n)
linpred <- -1 + 1.5 * x + 0.5 * z
prob <- plogis(linpred)
y <- rbinom(n, 1, prob)
fit_logistic <- glm(y ~ x + z, family=binomial)
p_hat <- predict(fit_logistic, type="response")
p1 <- mean(y)
H_T <- - (p1 * log2(p1) + (1-p1) * log2(1-p1))
H_T_given_X <- mean( - (p_hat * log2(p_hat) + (1 - p_hat) * log2(1 - p_hat)) )
MI_logistic <- H_T - H_T_given_X
cat("== Logistic Regression ==\n")
cat("Mutual Information (bits):", MI_logistic, "\n\n")
# --- EXAMPLE: Cox Regression MI ---
set.seed(123)
hazard <- exp(0.8 * x + 0.4 * z)
time <- rexp(n, rate = hazard)
censoring <- rexp(n, rate = 0.1)
status <- as.numeric(time <= censoring)
time <- pmin(time, censoring)
cox_fit <- coxph(Surv(time, status) ~ x + z)
t0 <- median(time)
sf <- survfit(cox_fit, newdata = data.frame(x=x, z=z))
s_prob <- summary(sf, times=t0)$surv
if (length(s_prob) != n) {
  lp <- predict(cox_fit, type="lp")
  basehaz_df <- basehaz(cox_fit, centered=FALSE)
  base_S_t0 <- exp(- approx(basehaz_df$time, basehaz_df$hazard, t0, rule=2)$y )
  s_prob <- base_S_t0 ^ exp(lp)
}
event_t0 <- as.numeric(time <= t0 & status == 1)
p1 <- mean(event_t0)
H_T <- - (p1 * log2(p1) + (1-p1) * log2(1-p1))
H_T_given_X <- mean( - ( (1-s_prob) * log2(1-s_prob) + s_prob * log2(s_prob) ) )
MI_cox <- H_T - H_T_given_X
cat("== Cox Regression ==\n")
cat("Mutual Information (bits) at t0 =", t0, ":", MI_cox, "\n\n")
# --- MI across thresholds (ROC) example ---
cat("== MI-ROC curve ==\n")
roc_result <- compute_mi_roc(score = linpred, label = y)
plot(roc_result$threshold, roc_result$MI, type="l", col="blue", lwd=2,
     xlab="Threshold", ylab="Mutual Information (bits)", main="MI across thresholds")

