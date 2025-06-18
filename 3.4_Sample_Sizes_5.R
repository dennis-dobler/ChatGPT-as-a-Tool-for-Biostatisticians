library(survival)
library(survminer)



df <- read.csv("05_simulated_pancreatic_cancer_data.csv")


# Assuming your data is in a data.frame named `df`
# with variables: time, event, treatment, performance_status

df$treatment <- factor(df$treatment, levels = c(0, 1), labels = c("Control", "Treatment"))
df$performance_status <- as.factor(df$performance_status)

# Plot: One KM curve per ECOG group
for (ps in levels(df$performance_status)) {
  subset_df <- subset(df, performance_status == ps)
  fit <- survfit(Surv(time, event) ~ treatment, data = subset_df)
  
  ggsurvplot(fit, data = subset_df, 
             title = paste("Kaplan-Meier Curve (ECOG", ps, ")"),
             xlab = "Time (months)", ylab = "Survival Probability",
             legend.title = "Treatment Group",
             palette = "Dark2")
}


# Unadjusted Cox model
cox_unadj <- coxph(Surv(time, event) ~ treatment, data = df)

# Adjusted Cox model
cox_adj <- coxph(Surv(time, event) ~ treatment + age + sex + performance_status + ca19_9 + metastatic, data = df)

# Summary of models
summary(cox_unadj)
summary(cox_adj)


# Install if not yet available
# install.packages("broom")

library(broom)

# Extract results
tidy_unadj <- tidy(cox_unadj, exponentiate = TRUE, conf.int = TRUE)
tidy_adj <- tidy(cox_adj, exponentiate = TRUE, conf.int = TRUE)

# Merge by variable
results <- merge(tidy_unadj[, c("term", "estimate", "conf.low", "conf.high", "p.value")],
                 tidy_adj[, c("term", "estimate", "conf.low", "conf.high", "p.value")],
                 by = "term", all = TRUE, suffixes = c("_unadj", "_adj"))

# Rename columns
colnames(results) <- c("Variable", "HR unadj.", "95% CI low (unadj)", "95% CI high (unadj)", "p (unadj)",
                       "HR adj.",   "95% CI low (adj)",   "95% CI high (adj)",   "p (adj)")

print(results)
