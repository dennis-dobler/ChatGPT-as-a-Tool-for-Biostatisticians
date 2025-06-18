### Comprehensive R Script: Information-Theoretic Surrogacy & MI

# 1. Mutual Information for 2x2 Table
compute_mi <- function(tab) {
  n <- sum(tab)
  pxy <- tab / n
  px <- rowSums(pxy)
  py <- colSums(pxy)
  mi <- 0
  for (i in 1:nrow(pxy)) {
    for (j in 1:ncol(pxy)) {
      if (pxy[i,j] > 0) {
        mi <- mi + pxy[i,j] * log(pxy[i,j] / (px[i] * py[j]))
      }
    }
  }
  return(mi / log(2))  # bits
}

# Example
table <- matrix(c(30, 10, 20, 40), nrow=2)
compute_mi(table)

# 2. MI Across ROC Curve
set.seed(123)
n <- 200
scores <- runif(n)
labels <- rbinom(n, 1, plogis(3 * scores - 1))

library(ROCR)
pred <- prediction(scores, labels)
perf <- performance(pred, "tpr", "fpr")
thresholds <- perf@alpha.values[[1]]

mi_values <- sapply(thresholds, function(th) {
  preds <- ifelse(scores >= th, 1, 0)
  table2x2 <- table(preds, labels)
  if (nrow(table2x2) == 2 & ncol(table2x2) == 2) {
    compute_mi(as.matrix(table2x2))
  } else {
    NA
  }
})

plot(thresholds, mi_values, type="l", xlab="Threshold", ylab="Mutual Information (bits)", main="MI across ROC thresholds")

# 3. MI from Logistic Regression

# Data
df <- data.frame(x = rnorm(n), y = labels)
model <- glm(y ~ x, family = binomial, data = df)

# Compute MI
p_hat <- predict(model, type="response")
eps <- 1e-12
p_hat <- pmin(pmax(p_hat, eps), 1 - eps)

p_y1 <- mean(df$y)
H_Y <- -p_y1 * log(p_y1) - (1-p_y1) * log(1-p_y1)
H_Y_given_X <- -mean(df$y * log(p_hat) + (1 - df$y) * log(1 - p_hat))

MI_logistic <- (H_Y - H_Y_given_X) / log(2)
MI_logistic

# 4. MI from Cox Regression

library(survival)

# Simulate survival data
set.seed(123)
x <- rnorm(n)
time <- rexp(n, rate = exp(-0.5 + 0.7 * x))
status <- rbinom(n, 1, 0.8)

df_surv <- data.frame(time, status, x)

# Cox model
cox_model <- coxph(Surv(time, status) ~ x, data = df_surv)

lrt <- cox_model$loglik[2] - cox_model$loglik[1]
MI_cox <- (2 * lrt) / n / log(2)
MI_cox

# Done!
