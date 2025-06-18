library(survival)
library(survminer)

# Assuming your data is in a data frame called df with the following columns:
# time, event, group (factor: "control"/"treatment"), ecog (0/1/2)

df <- read.csv("03_simulated_pancreatic_cancer_data.csv")


# KM plot stratified by ECOG and treatment
ggsurvplot_facet(
  fit = survfit(Surv(time, event) ~ group + ecog, data = df),
  data = df,
  facet.by = "ecog",
  legend.title = "Group",
  legend.labs = c("Control", "Treatment"),
  palette = c("#E69F00", "#56B4E9"),
  xlab = "Time (months)",
  ylab = "Survival probability",
  conf.int = TRUE
)


cox_unadj <- coxph(Surv(time, event) ~ group, data = df)
summary(cox_unadj)


cox_adj <- coxph(Surv(time, event) ~ group + age + sex + ecog + stage, data = df)
summary(cox_adj)


# Load library for tidy model summaries
library(broom)

# Get tidy output
tidy_unadj <- tidy(cox_unadj, exponentiate = TRUE, conf.int = TRUE)
tidy_adj <- tidy(cox_adj, exponentiate = TRUE, conf.int = TRUE)

# Merge results
results <- merge(
  tidy_unadj[, c("term", "estimate", "conf.low", "conf.high", "p.value")],
  tidy_adj[, c("term", "estimate", "conf.low", "conf.high", "p.value")],
  by = "term", suffixes = c("_unadj", "_adj")
)

# Rename columns
colnames(results) <- c(
  "Variable", "HR_unadj", "CI_low_unadj", "CI_high_unadj", "p_unadj",
  "HR_adj", "CI_low_adj", "CI_high_adj", "p_adj"
)

# Format for display
results$CI_unadj <- paste0(round(results$CI_low_unadj, 2), "–", round(results$CI_high_unadj, 2))
results$CI_adj <- paste0(round(results$CI_low_adj, 2), "–", round(results$CI_high_adj, 2))

# Final table
final_results <- results[, c("Variable", "HR_unadj", "CI_unadj", "p_unadj",
                             "HR_adj", "CI_adj", "p_adj")]

print(final_results)
