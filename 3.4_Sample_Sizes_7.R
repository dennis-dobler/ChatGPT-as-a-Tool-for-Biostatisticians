library(survival)
library(survminer)
library(dplyr)
library(ggplot2)


set.seed(42)
n_per_group <- 166
n <- 2 * n_per_group

# Group assignment
group <- rep(c("control", "treatment"), each = n_per_group)

# Covariates
age <- pmin(pmax(rnorm(n, mean = 65, sd = 10), 40), 90)
ecog <- sample(0:3, n, replace = TRUE, prob = c(0.2, 0.4, 0.3, 0.1))
tumor_location <- sample(c("head", "body/tail"), n, replace = TRUE)
ca19_9 <- rlnorm(n, meanlog = 5, sdlog = 1)

# Hazard rates
lambda_control <- log(2) / 12
lambda_treatment <- log(2) / 18
lambda <- ifelse(group == "control", lambda_control, lambda_treatment)

# Linear predictor
lp <- 0.03 * (age - 65) +
  0.4 * ecog +
  0.3 * ifelse(tumor_location == "body/tail", 1, 0) +
  0.0005 * (ca19_9 - mean(ca19_9))

hazard <- lambda * exp(lp)
surv_time <- rexp(n, rate = hazard)

# Dropout and censoring
dropout_rate <- 0.15
dropout <- rbinom(n, 1, dropout_rate)
dropout_time <- runif(n, 0, 24)
follow_up <- 24
obs_time <- pmin(surv_time, ifelse(dropout == 1, dropout_time, follow_up))
event <- as.numeric(surv_time <= obs_time)

# Combine into data frame
df <- data.frame(
  time = obs_time,
  event = event,
  group = factor(group),
  age = age,
  ecog = ecog,
  tumor_location = factor(tumor_location),
  ca19_9 = ca19_9
)


for (score in sort(unique(df$ecog))) {
  g <- ggsurvplot(
    survfit(Surv(time, event) ~ group, data = df %>% filter(ecog == score)),
    data = df %>% filter(ecog == score),
    risk.table = FALSE,
    pval = TRUE,
    title = paste("ECOG", score),
    legend.title = "Group",
    palette = c("#E69F00", "#56B4E9")
  )
  print(g)
}

cox_unadj <- coxph(Surv(time, event) ~ group, data = df)
summary(cox_unadj)

cox_adj <- coxph(Surv(time, event) ~ group + age + ecog + tumor_location + ca19_9, data = df)
summary(cox_adj)

# Extract results
get_results <- function(model) {
  data.frame(
    variable = rownames(summary(model)$coefficients),
    HR = round(exp(coef(model)), 3),
    CI = paste0("[", round(exp(confint(model))[,1], 2), ", ", round(exp(confint(model))[,2], 2), "]"),
    p = signif(summary(model)$coefficients[,5], 3)
  )
}

tab_unadj <- get_results(cox_unadj)
tab_adj <- get_results(cox_adj)

# Merge into one table
full_table <- merge(tab_unadj, tab_adj, by = "variable", all = TRUE, suffixes = c(" unadj", " adj"))
print(full_table, row.names = FALSE)



