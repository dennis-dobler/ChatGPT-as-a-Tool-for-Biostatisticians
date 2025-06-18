library(survival)
library(survminer)
library(dplyr)

# Assume sim_df is your data frame
# Variables: time, event, treatment (0/1), performance_status (0/1/2)

sim_df <- read.csv("02_simulated_pancreatic_cancer_data.csv")


# Label treatment and ECOG groups
sim_df$treatment_label <- factor(sim_df$treatment, labels = c("Standard", "New"))
sim_df$ecog_group <- factor(sim_df$performance_status, labels = c("ECOG 0", "ECOG 1", "ECOG 2"))

# One plot per ECOG group
for (ecog in levels(sim_df$ecog_group)) {
  cat("Plotting:", ecog, "\n")
  fit <- survfit(Surv(time, event) ~ treatment_label, data = filter(sim_df, ecog_group == ecog))
  g <- ggsurvplot(fit, 
                  data = filter(sim_df, ecog_group == ecog),
                  pval = TRUE,
                  title = paste("Survival by Treatment in", ecog),
                  legend.title = "Treatment",
                  xlab = "Time (months)",
                  ylab = "Survival probability")
  print(g)
}


# Unadjusted model
cox_unadj <- coxph(Surv(time, event) ~ treatment, data = sim_df)

# Adjusted model
cox_adj <- coxph(Surv(time, event) ~ treatment + age + sex + performance_status + ca19_9, data = sim_df)

# Summary of models
summary(cox_unadj)
summary(cox_adj)

library(broom)

# Extract tidy results
unadj <- tidy(cox_unadj, exponentiate = TRUE, conf.int = TRUE) %>%
  filter(term == "treatment") %>%
  select(HR_unadj = estimate, CI_low = conf.low, CI_high = conf.high, p.value)

adj <- tidy(cox_adj, exponentiate = TRUE, conf.int = TRUE) %>%
  mutate(term = recode(term, treatment = "Treatment", age = "Age", sex = "Sex",
                       performance_status = "Performance Status", ca19_9 = "CA 19-9")) %>%
  select(variable = term, HR_adj = estimate, adj_CI_low = conf.low, adj_CI_high = conf.high, adj_p = p.value)

# Merge results
table <- adj %>%
  left_join(unadj %>% mutate(variable = "Treatment"), by = "variable") %>%
  select(variable, HR_unadj, CI_low, CI_high, p.value, HR_adj, adj_CI_low, adj_CI_high, adj_p)

# Nicely formatted table
library(knitr)
kable(table, digits = 3, col.names = c("Variable", "HR unadj.", "95% CI Low", "95% CI High", "P-value",
                                       "HR adj.", "Adj. 95% CI Low", "Adj. 95% CI High", "Adj. P-value"))

