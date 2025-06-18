# ---- 1. Mutual Information for 2x2 table ----
table <- matrix(c(30, 10,
                  20, 40), nrow = 2, byrow = TRUE)

joint_prob <- table / sum(table)
px <- rowSums(joint_prob)
py <- colSums(joint_prob)

MI_2x2 <- 0
for (i in 1:2) {
  for (j in 1:2) {
    if (joint_prob[i, j] > 0) {
      MI_2x2 <- MI_2x2 + joint_prob[i, j] * log2(joint_prob[i, j] / (px[i] * py[j]))
    }
  }
}
cat("MI for 2x2 table (bits):", MI_2x2, "\n")

# ---- 2. MI across ROC thresholds ----
library(pROC)

set.seed(123)
score <- runif(100)
label <- rbinom(100, 1, 0.5)

roc_obj <- roc(label, score)

compute_MI <- function(threshold, score, label) {
  pred <- ifelse(score >= threshold, 1, 0)
  table <- table(pred, label)
  if (nrow(table) < 2 || ncol(table) < 2) return(NA)
  
  joint_prob <- table / sum(table)
  px <- rowSums(joint_prob)
  py <- colSums(joint_prob)
  
  MI <- 0
  for (i in 1:2) {
    for (j in 1:2) {
      if (joint_prob[i, j] > 0) {
        MI <- MI + joint_prob[i, j] * log2(joint_prob[i, j] / (px[i] * py[j]))
      }
    }
  }
  return(MI)
}

thresholds <- roc_obj$thresholds
MI_values <- sapply(thresholds, compute_MI, score = score, label = label)

plot(thresholds, MI_values, type = "l", xlab = "Threshold", ylab = "Mutual Information (bits)", main = "MI vs Threshold")

# ---- 3. Logistic regression MI ----
df <- data.frame(
  X = rnorm(100),
  Y = rbinom(100, 1, 0.5)
)

logit_model <- glm(Y ~ X, family = binomial, data = df)

LL_null <- logLik(glm(Y ~ 1, family = binomial, data = df))
LL_model <- logLik(logit_model)

MI_logit <- as.numeric(LL_model - LL_null) / log(2)
cat("Logistic regression MI (bits):", MI_logit, "\n")

# ---- 4. Cox regression MI ----
library(survival)

df_surv <- data.frame(
  time = rexp(100, rate = 0.1),
  status = rbinom(100, 1, 0.7),
  X = rnorm(100)
)

cox_model <- coxph(Surv(time, status) ~ X, data = df_surv)

LL_null_cox <- cox_model$loglik[1]
LL_model_cox <- cox_model$loglik[2]

MI_cox <- (LL_model_cox - LL_null_cox) / log(2)
cat("Cox regression MI (bits):", MI_cox, "\n")

