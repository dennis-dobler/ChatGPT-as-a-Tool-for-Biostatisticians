temp <- tempfile()
download.file(
  "https://archive.ics.uci.edu/static/public/17/breast+cancer+wisconsin+diagnostic.zip",
  temp
)
data <- read.table(unz(temp, "wdbc.data"), sep = ",")
data$V2 <- data$V2 == "M"
unlink(temp)

# Load necessary libraries
library(MASS)

# Set up: outcome is V2 (TRUE = malignant, FALSE = benign)
data$V2 <- factor(data$V2, levels = c(FALSE, TRUE))

# List of predictor variable names (V3 to V32)
predictors <- paste0("V", 3:32)

# Start with null model
current_formula <- as.formula("V2 ~ 1")
current_model <- glm(current_formula, data = data, family = binomial)

# Track included predictors
included <- c()
remaining <- predictors
changed <- TRUE

while (changed) {
  changed <- FALSE
  
  # === Try adding one variable at a time ===
  add_candidates <- list()
  
  for (var in remaining) {
    test_formula <- as.formula(paste("V2 ~", paste(c(included, var), collapse = " + ")))
    test_model <- glm(test_formula, data = data, family = binomial)
    lrt_p <- anova(current_model, test_model, test = "LRT")[2, "Pr(>Chi)"]
    add_candidates[[var]] <- lrt_p
  }
  
  # Find best variable to add
  best_add <- names(which.min(unlist(add_candidates)))
  best_add_p <- add_candidates[[best_add]]
  
  if (!is.null(best_add) && best_add_p < 0.05) {
    included <- c(included, best_add)
    remaining <- setdiff(remaining, best_add)
    current_formula <- as.formula(paste("V2 ~", paste(included, collapse = " + ")))
    current_model <- glm(current_formula, data = data, family = binomial)
    changed <- TRUE
  }
  
  # === Try removing variables that no longer meet POUT criteria ===
  if (length(included) > 1) {
    drop_test <- drop1(current_model, test = "LRT")
    drop_p <- drop_test[-1, "Pr(>Chi)"]
    
    remove_vars <- names(drop_p[drop_p > 0.10])
    
    if (length(remove_vars) > 0) {
      included <- setdiff(included, remove_vars)
      remaining <- c(remaining, remove_vars)
      current_formula <- as.formula(paste("V2 ~", paste(included, collapse = " + ")))
      current_model <- glm(current_formula, data = data, family = binomial)
      changed <- TRUE
    }
  }
}

summary(current_model)
