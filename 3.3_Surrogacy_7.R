# Load required packages
library(survival)
library(dplyr)
library(tibble)
library(ggplot2)
library(Matrix)
library(infotheo)  # for mutual information estimation
library(rms)       # for survival modeling (optional but helpful)

# --- 1️⃣ Simulate data ---

set.seed(123)

n <- 1000

# Simulate binary risk factor and continuous surrogate
X1 <- rbinom(n, 1, 0.4)
X2 <- rnorm(n, mean = 0, sd = 1)

# Simulate binary outcome (logistic regression)
logit_p <- -0.5 + 1.2 * X1 + 0.8 * X2
p <- 1 / (1 + exp(-logit_p))
Y_bin <- rbinom(n, 1, p)

# Simulate survival outcome (Cox regression)
# Hazard depends on same covariates
lambda <- 0.1
beta1 <- 0.8
beta2 <- 0.5

linpred <- beta1 * X1 + beta2 * X2
event_time <- rexp(n, rate = lambda * exp(linpred))
censoring_time <- rexp(n, rate = 0.05)

time <- pmin(event_time, censoring_time)
status <- as.numeric(event_time <= censoring_time)

# --- 2️⃣ Fit logistic regression (binary outcome) ---

log_model <- glm(Y_bin ~ X1 + X2, family = binomial())

summary(log_model)

# --- 3️⃣ Fit Cox regression (survival outcome) ---

cox_model <- coxph(Surv(time, status) ~ X1 + X2)
summary(cox_model)

# --- 4️⃣ Mutual Information Calculation (binary outcome) ---

# Discretize continuous risk scores
# Here we use predicted probabilities from logistic model as "surrogate"

pred_prob <- predict(log_model, type = "response")

# Discretize predicted probability into bins
# For MI estimation, discretization is needed
bins <- 5
pred_prob_disc <- discretize(pred_prob, disc = "equalfreq", nbins = bins)

# Mutual information between discretized surrogate and true outcome
MI_bin <- mutinformation(pred_prob_disc, Y_bin)

# --- 5️⃣ Connection to deviance ---

# Deviance of null model
null_model <- glm(Y_bin ~ 1, family = binomial())
D_null <- deviance(null_model)

# Deviance of fitted model
D_model <- deviance(log_model)

# Pseudo-R2 (McFadden)
pseudo_R2 <- 1 - (D_model / D_null)

# --- 6️⃣ Normalize MI ---

# Entropy of outcome H(Y)
H_Y <- entropy(discretize(Y_bin, disc = "equalwidth", nbins = 2))

# Normalized MI (fraction of uncertainty explained)
normalized_MI <- MI_bin / H_Y

# --- 7️⃣ Results ---

cat("Mutual Information (binary outcome, logistic regression):", MI_bin, "nats\n")
cat("Entropy of outcome H(Y):", H_Y, "nats\n")
cat("Normalized MI (fraction of uncertainty explained):", normalized_MI, "\n")
cat("Pseudo-R2 (McFadden):", pseudo_R2, "\n")

# --- 8️⃣ Mutual Information for Cox regression (optional, more complex) ---

# For Cox models, there is no direct MI formula — but:

# One option is to compute MI between estimated risk scores and survival status at a fixed time point
# For demonstration, let's use 2-year survival probability as a surrogate

# First, compute linear predictor from Cox model
cox_lp <- predict(cox_model, type = "lp")

# Now choose a landmark time, e.g., t_star = 2 years
t_star <- 2

# Estimate baseline survival at t_star
surv_fit <- survfit(cox_model)
baseline_surv <- summary(surv_fit, times = t_star)$surv

# Individual survival probability at t_star:
surv_prob <- baseline_surv ^ exp(cox_lp)

# Convert to "event" at t_star (1 = died before t_star, 0 = survived)
event_t_star <- as.numeric(time <= t_star & status == 1)

# Discretize survival probability
surv_prob_disc <- discretize(surv_prob, disc = "equalfreq", nbins = bins)

# MI between risk score and event indicator
MI_cox <- mutinformation(surv_prob_disc, event_t_star)

# Entropy of event indicator
H_event <- entropy(discretize(event_t_star, disc = "equalwidth", nbins = 2))

# Normalized MI
normalized_MI_cox <- MI_cox / H_event

# --- Results for Cox ---

cat("Mutual Information (Cox regression at t =", t_star, "):", MI_cox, "nats\n")
cat("Entropy of event indicator:", H_event, "nats\n")
cat("Normalized MI (Cox):", normalized_MI_cox, "\n")
