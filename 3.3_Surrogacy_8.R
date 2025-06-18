# Comprehensive R Script for Information-Theoretic Surrogacy

# Load necessary libraries
library(pROC)
library(survival)

# Function to compute mutual information for a 2x2 table
mutual_information <- function(tbl) {
  N <- sum(tbl)
  p <- tbl / N
  p_row <- rowSums(p)
  p_col <- colSums(p)
  
  mi <- 0
  for (i in 1:nrow(tbl)) {
    for (j in 1:ncol(tbl)) {
      if (p[i,j] > 0) {
        mi <- mi + p[i,j] * log(p[i,j] / (p_row[i] * p_col[j]))
      }
    }
  }
  return(mi / log(2))  # in bits
}

##### 1️⃣ MI for a 2x2 table #####
cat("\n==== MI for 2x2 Table ====\n")

tbl <- matrix(c(30, 10, 20, 40), nrow=2, byrow=TRUE)
rownames(tbl) <- c("S=1", "S=0")
colnames(tbl) <- c("T=1", "T=0")
print(tbl)

mi_2x2 <- mutual_information(tbl)
cat("Mutual Information (bits):", mi_2x2, "\n")

##### 2️⃣ MI across thresholds in a ROC curve #####
cat("\n==== MI vs Threshold (ROC Curve) ====\n")

# Simulate data for ROC
set.seed(123)
score <- runif(1000)
class <- rbinom(1000, 1, score)

# Compute ROC
roc_obj <- roc(class, score)
thresholds <- roc_obj$thresholds

# MI at each threshold
mi_values <- sapply(thresholds, function(thresh) {
  pred <- ifelse(score >= thresh, 1, 0)
  tbl <- table(factor(pred, levels=c(1,0)), factor(class, levels=c(1,0)))
  if (length(tbl) < 4) { tbl <- matrix(c(tbl, 0), nrow=2) }
  mutual_information(tbl)
})

# Plot MI vs threshold
plot(thresholds, mi_values, type='l', xlab="Threshold", ylab="Mutual Information (bits)", main="MI vs Threshold (ROC)")

##### 3️⃣ MI from logistic regression #####
cat("\n==== MI from Logistic Regression ====\n")

# Simulate logistic regression data
set.seed(123)
x <- rnorm(1000)
y <- rbinom(1000, 1, plogis(0.5*x))

# Fit models
model_full <- glm(y ~ x, family=binomial)
model_null <- glm(y ~ 1, family=binomial)

# Compute MI
logLik_full <- logLik(model_full)
logLik_null <- logLik(model_null)
N <- length(y)

mi_logistic <- (as.numeric(logLik_full) - as.numeric(logLik_null)) / N / log(2)
cat("Mutual Information (bits) - Logistic Regression:", mi_logistic, "\n")

##### 4️⃣ MI from Cox regression #####
cat("\n==== MI from Cox Regression ====\n")

# Simulate survival data
set.seed(123)
n <- 1000
x <- rnorm(n)
time <- rexp(n, exp(0.5 * x))
event <- rbinom(n, 1, 0.8)

# Fit Cox models
cox_full <- coxph(Surv(time, event) ~ x)
cox_null <- coxph(Surv(time, event) ~ 1)

# Compute MI
logLik_full <- cox_full$loglik[2]
logLik_null <- cox_full$loglik[1]

mi_cox <- (logLik_full - logLik_null) / n / log(2)
cat("Mutual Information (bits) - Cox Regression:", mi_cox, "\n")

##### DONE #####
cat("\n==== END OF SCRIPT ====\n")

