# Load necessary libraries
library(survival)
library(ggplot2)

# Load the dataset
df <- read.csv("10_final_simulated_dataset.csv")

# Convert 'group' and 'ecog' to factors
df$group <- factor(df$group, levels = c("control", "treatment"))
df$ecog <- factor(df$ecog)

# Fit Kaplan-Meier curves for each ECOG group
ggplot(df, aes(time = time, status = event, color = group)) +
  geom_step(aes(linetype = group)) +
  facet_wrap(~ecog) +
  labs(title = "Kaplan-Meier Curves by Treatment and ECOG Group",
       x = "Time (months)",
       y = "Survival Probability") +
  theme_minimal() +
  theme(legend.position = "bottom")



# Convert group to a binary numeric variable (0 = control, 1 = treatment)
df$group <- ifelse(df$group == "treatment", 1, 0)

# Unadjusted Cox model (treatment only)
cox_unadj <- coxph(Surv(time, event) ~ group, data = df)
summary(cox_unadj)

# Adjusted Cox model (age, sex, ecog, stage)
cox_adj <- coxph(Surv(time, event) ~ group + age + sex + ecog + stage, data = df)
summary(cox_adj)

# Combine results into a table for easy comparison
results <- data.frame(
  variable = c("group", "age", "sex", "ecog", "stage"),
  `HR unadj.` = c(exp(cox_unadj$coefficients["group"]), rep(NA, 4)),
  `95% CI unadj.` = c(confint(cox_unadj)["group", ], rep(NA, 4)),
  `p-value unadj.` = c(cox_unadj$coefficients["group"], rep(NA, 4)),
  `HR adj.` = c(exp(cox_adj$coefficients["group"]), exp(cox_adj$coefficients[c("age", "sex", "ecog", "stage")])),
  `95% CI adj.` = c(confint(cox_adj)["group", ], apply(confint(cox_adj)[c("age", "sex", "ecog", "stage"), ], 1, function(x) paste("[", round(exp(x[1]), 2), ",", round(exp(x[2]), 2), "]")))),
`p-value adj.` = c(cox_adj$coefficients["group"], c(cox_adj$coefficients[c("age", "sex", "ecog", "stage")], rep(NA, 4)))
)

results