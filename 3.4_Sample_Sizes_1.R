# Load required packages
library(survival)
library(survminer)
library(dplyr)

# Assuming you already have a data frame 'df' with:
# time, event, treatment, age, sex_female, ecog, ca199

# Example: if importing from CSV
df <- read.csv("01_simulated_survival_data_with_covariates.csv")


# Plot one KM curve per ECOG group
ecog_levels <- sort(unique(df$ecog))

for (e in ecog_levels) {
  df_sub <- df %>% filter(ecog == e)
  fit <- survfit(Surv(time, event) ~ treatment, data = df_sub)
  g <- ggsurvplot(
    fit,
    data = df_sub,
    risk.table = TRUE,
    pval = TRUE,
    legend.title = "Treatment",
    legend.labs = c("Control", "Treatment"),
    title = paste("Kaplan-Meier Curve for ECOG", e)
  )
  print(g)
}


cox_unadj <- coxph(Surv(time, event) ~ treatment, data = df)
summary(cox_unadj)

cox_adj <- coxph(Surv(time, event) ~ treatment + age + sex_female + factor(ecog) + log(ca199), data = df)
summary(cox_adj)

library(broom)

# Unadjusted results
tidy_unadj <- tidy(cox_unadj, exponentiate = TRUE, conf.int = TRUE)

# Adjusted results
tidy_adj <- tidy(cox_adj, exponentiate = TRUE, conf.int = TRUE)

# Merge and format
results <- full_join(
  tidy_unadj %>% select(term, HR_unadj = estimate, CI_low_unadj = conf.low, CI_high_unadj = conf.high, p_unadj = p.value),
  tidy_adj %>% select(term, HR_adj = estimate, CI_low_adj = conf.low, CI_high_adj = conf.high, p_adj = p.value),
  by = "term"
)

# Optional: rename variables
results$term <- recode(results$term,
                       "treatment" = "treatment",
                       "age" = "age",
                       "sex_female" = "sex (female)",
                       "factor(ecog)1" = "ecog 1",
                       "factor(ecog)2" = "ecog 2",
                       "log(ca199)" = "log(CA19-9)"
)

print(results)

