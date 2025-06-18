# Install if not already installed
install.packages("survival")
install.packages("survminer")


df <- read.csv("08_simulated_pancreatic_cancer_data.csv")

library(survival)
library(survminer)

# KM plots per ECOG level
ecog_levels <- sort(unique(df$ecog))

for (level in ecog_levels) {
  df_subset <- subset(df, ecog == level)
  
  fit <- survfit(Surv(observed_time, event) ~ group, data = df_subset)
  
  g <- ggsurvplot(fit,
                  data = df_subset,
                  pval = TRUE,
                  title = paste("Kaplan–Meier Curve (ECOG", level, ")"),
                  xlab = "Time (months)",
                  ylab = "Survival Probability",
                  legend.title = "Group")
  
  print(g)
}

cox_unadj <- coxph(Surv(observed_time, event) ~ group, data = df)
summary(cox_unadj)

cox_adj <- coxph(Surv(observed_time, event) ~ group + age + sex + ecog + stage, data = df)
summary(cox_adj)

# Load for nice HR summary
library(broom)

# Extract tidy results
unadj_res <- tidy(cox_unadj, exponentiate = TRUE, conf.int = TRUE)
adj_res <- tidy(cox_adj, exponentiate = TRUE, conf.int = TRUE)

# Merge results
merged <- merge(unadj_res, adj_res, by = "term", all = TRUE, suffixes = c("_unadj", "_adj"))

# Clean output
merged <- merged[, c("term",
                     "estimate_unadj", "conf.low_unadj", "conf.high_unadj", "p.value_unadj",
                     "estimate_adj", "conf.low_adj", "conf.high_adj", "p.value_adj")]

# Optional: rename columns
colnames(merged) <- c("variable", "HR unadj.", "CI low", "CI high", "p-value",
                      "HR adj.", "adj. CI low", "adj. CI high", "adj. p-value")

print(merged)
