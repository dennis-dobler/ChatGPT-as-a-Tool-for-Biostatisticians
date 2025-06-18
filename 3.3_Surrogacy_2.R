# ---- MI for 2x2 table ----
compute_mi <- function(tab) {
  N <- sum(tab)
  p_xy <- tab / N
  p_x <- rowSums(p_xy)
  p_y <- colSums(p_xy)
  mi <- 0
  for (i in 1:2) {
    for (j in 1:2) {
      if (p_xy[i,j] > 0) {
        mi <- mi + p_xy[i,j] * log2(p_xy[i,j] / (p_x[i] * p_y[j]))
      }
    }
  }
  return(mi)
}

# Example 2x2 table
table2x2 <- matrix(c(50, 10, 5, 35), nrow = 2, byrow = TRUE)
MI_2x2 <- compute_mi(table2x2)
cat("MI 2x2 table:", MI_2x2, "\n")

# ---- MI across ROC thresholds ----
set.seed(123)
score <- rnorm(200)
label <- ifelse(score + rnorm(200) > 0, 1, 0)

thresholds <- seq(min(score), max(score), length.out = 100)
MI_values <- sapply(thresholds, function(th) {
  pred <- ifelse(score >= th, 1, 0)
  tab <- table(pred, label)
  # Pad
  tab <- as.matrix(tab)
  if (nrow(tab) < 2) tab <- rbind(tab, c(0,0))
  if (ncol(tab) < 2) tab <- cbind(tab, c(0,0))
  compute_mi(tab)
})

plot(thresholds, MI_values, type = "l", lwd = 2, col = "blue",
     xlab = "Threshold", ylab = "Mutual Information (bits)",
     main = "MI across ROC thresholds")

# ---- Logistic regression MI ----
set.seed(123)
x <- rbinom(200, 1, 0.5)
y <- rbinom(200, 1, plogis(0.5 * x))
model_full <- glm(y ~ x, family = binomial)
model_null <- glm(y ~ 1, family = binomial)

LL_full <- logLik(model_full)
LL_null <- logLik(model_null)

MI_logreg <- (LL_full - LL_null) / log(2) / length(y)
cat("MI logistic regression (bits per obs):", MI_logreg, "\n")

# ---- Cox regression MI ----
library(survival)
set.seed(123)
x <- rbinom(200, 1, 0.5)
time <- rexp(200, rate = 0.1 * exp(0.5 * x))
status <- rbinom(200, 1, 0.9)

cox_full <- coxph(Surv(time, status) ~ x)
LL_full <- cox_full$loglik[2]
LL_null <- cox_full$loglik[1]

MI_cox <- (LL_full - LL_null) / log(2) / length(x)
cat("MI Cox regression (bits per obs):", MI_cox, "\n")
