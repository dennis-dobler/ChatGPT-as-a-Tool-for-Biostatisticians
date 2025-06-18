install.packages(c("survival", "survminer", "dplyr"))


simulated_data <- read.csv("09_simulated_pancreatic_data.csv")


library(survival)
library(survminer)
library(dplyr)

# Ensure group is a factor
simulated_data$group <- factor(simulated_data$group, levels = c("control", "treatment"))
simulated_data$performance_status <- factor(simulated_data$performance_status)

# One plot per ECOG group
for (ecog in levels(simulated_data$performance_status)) {
  cat("Plotting ECOG group:", ecog, "\n")
  
  subset_data <- filter(simulated_data, performance_status == ecog)
  surv_obj <- Surv(time = subset_data$time, event = subset_data$event)
  
  ggsurvplot(
    survfit(surv_obj ~ group, data = subset_data),
    data = subset_data,
    title = paste("Kaplan-Meier Curve - ECOG", ecog),
    xlab = "Time (months)",
    ylab = "Survival Probability",
    conf.int = TRUE,
    pval = TRUE,
    legend.title = "Treatment Group",
    legend.labs = c("Control", "Treatment")
  )
}

# Unadjusted Cox model
cox_unadj <- coxph(Surv(time, event) ~ group, data = simulated_data)

# Adjusted Cox model
cox_adj <- coxph(Surv(time, event) ~ group + age + sex + performance_status + tumor_stage + log(CA19_9 + 1), 
                 data = simulated_data)

# Summarize both models
summary(cox_unadj)
summary(cox_adj)

# Function to extract HR, CI, and p-values
extract_cox <- function(model, prefix) {
  s <- summary(model)
  coefs <- s$coefficients
  confint <- s$conf.int
  data.frame(
    variable = rownames(coefs),
    HR = round(confint[, "exp(coef)"], 2),
    CI = paste0("[", round(confint[, "lower .95"], 2), ", ", round(confint[, "upper .95"], 2), "]"),
    p_value = signif(coefs[, "Pr(>|z|)"], 3),
    row.names = NULL,
    check.names = FALSE
  ) %>% rename_with(~ paste0(prefix, .), -variable)
}

# Combine results
tbl_unadj <- extract_cox(cox_unadj, "unadj_")
tbl_adj <- extract_cox(cox_adj, "adj_")

# Merge by variable
final_table <- full_join(tbl_unadj, tbl_adj, by = "variable")
final_table
