# Install if not already installed
install.packages(c("survival", "survminer", "dplyr"))

library(survival)
library(survminer)
library(dplyr)


# Assuming your dataset is called `data` and includes:
# - time: survival time
# - status: 1 = event, 0 = censored
# - group: "control" / "treatment"
# - ecog: 0 / 1 / 2


data <- read.csv("04_simulated_pancreas_data.csv")
data$status<-data$event
data$time<-data$observed_time

# Plot KM curves stratified by treatment within each ECOG group
for (ecog_level in unique(data$ecog)) {
  cat("Plotting ECOG =", ecog_level, "\n")
  
  subset_data <- filter(data, ecog == ecog_level)
  
  fit <- survfit(Surv(time, status) ~ group, data = subset_data)
  
  ggsurvplot(
    fit,
    data = subset_data,
    risk.table = TRUE,
    pval = TRUE,
    title = paste("Kaplan-Meier: ECOG", ecog_level),
    legend.title = "Treatment Group",
    xlab = "Time (months)",
    ylab = "Survival Probability"
  )
}


# Encode group as factor if not already
data$group <- factor(data$group, levels = c("control", "treatment"))

# Unadjusted model
cox_unadj <- coxph(Surv(time, status) ~ group, data = data)

# Adjusted model (age, sex, ecog, ca199, stage)
cox_adj <- coxph(Surv(time, status) ~ group + age + sex + ecog + ca199 + stage, data = data)

# Summary tables
summary(cox_unadj)
summary(cox_adj)

# Combine results into a table
library(broom)

unadj_tbl <- tidy(cox_unadj, exponentiate = TRUE, conf.int = TRUE) %>%
  select(term, estimate, conf.low, conf.high, p.value) %>%
  rename(
    Variable = term,
    `HR unadj.` = estimate,
    `95% - CI lower` = conf.low,
    `95% - CI upper` = conf.high,
    `p-value` = p.value
  )

adj_tbl <- tidy(cox_adj, exponentiate = TRUE, conf.int = TRUE) %>%
  select(term, estimate, conf.low, conf.high, p.value) %>%
  rename(
    Variable = term,
    `HR adj.` = estimate,
    `adj. 95%-CI lower` = conf.low,
    `adj. 95%-CI upper` = conf.high,
    `adj. p-value` = p.value
  )

# Merge tables
final_tbl <- full_join(unadj_tbl, adj_tbl, by = "Variable")
print(final_tbl)
