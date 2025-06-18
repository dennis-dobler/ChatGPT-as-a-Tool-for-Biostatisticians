# Install if not already installed
install.packages(c("survival", "survminer", "dplyr"))

# Load libraries
library(survival)
library(survminer)
library(dplyr)


df <- read.csv("06_simulated_pancreatic_cancer_data.csv")

# Assuming your data is in a data frame called df with these variables:
# time, event (1=event, 0=censored), group (control/treatment), ecog (0–2)

# Convert group to factor
df$group <- factor(df$group, levels = c("control", "treatment"))

# Plot KM curves stratified by treatment within ECOG groups
ggsurv <- survfit(Surv(time, event) ~ group + strata(ecog), data = df)

# Facet plot by ECOG
ggsurvplot(ggsurv,
           data = df,
           facet.by = "ecog",
           conf.int = TRUE,
           risk.table = TRUE,
           ggtheme = theme_minimal(),
           legend.labs = c("Control", "Treatment"),
           title = "Kaplan-Meier Survival Curves by Treatment Within ECOG Groups")

cox_unadj <- coxph(Surv(time, event) ~ group, data = df)
summary(cox_unadj)

# Variables: age, sex (0/1), ecog (0–2), ca199, diabetes (0/1)
cox_adj <- coxph(Surv(time, event) ~ group + age + sex + ecog + ca199 + diabetes, data = df)
summary(cox_adj)

# Extract and format results
library(broom)

# Tidy models
unadj_tidy <- tidy(cox_unadj, exponentiate = TRUE, conf.int = TRUE) %>%
  select(term, estimate, conf.low, conf.high, p.value) %>%
  rename(HR_unadj = estimate, CI_low_unadj = conf.low, CI_high_unadj = conf.high, p_unadj = p.value)

adj_tidy <- tidy(cox_adj, exponentiate = TRUE, conf.int = TRUE) %>%
  select(term, estimate, conf.low, conf.high, p.value) %>%
  rename(HR_adj = estimate, CI_low_adj = conf.low, CI_high_adj = conf.high, p_adj = p.value)

# Merge and format
cox_results <- full_join(unadj_tidy, adj_tidy, by = "term") %>%
  mutate(
    CI_unadj = paste0(round(CI_low_unadj, 2), "–", round(CI_high_unadj, 2)),
    CI_adj = paste0(round(CI_low_adj, 2), "–", round(CI_high_adj, 2))
  ) %>%
  select(term, HR_unadj, CI_unadj, p_unadj, HR_adj, CI_adj, p_adj)

print(cox_results)
