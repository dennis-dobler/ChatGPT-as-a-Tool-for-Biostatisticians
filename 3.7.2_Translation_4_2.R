temp <- tempfile()
download.file(
  "https://archive.ics.uci.edu/static/public/17/breast+cancer+wisconsin+diagnostic.zip",
  temp
)
data <- read.table(unz(temp, "wdbc.data"), sep = ",")
data$V2 <- data$V2 == "M"
unlink(temp)

forward_stepwise_sp <- function(data, response, predictors, pin = 0.05, pout = 0.10) {
  selected_vars <- c()
  remaining_vars <- predictors
  model_formula <- as.formula(paste(response, "~ 1"))
  model <- glm(model_formula, data = data, family = binomial)
  
  repeat {
    # --- FORWARD STEP ---
    forward_pvals <- sapply(remaining_vars, function(var) {
      temp_formula <- as.formula(paste(response, "~", paste(c(selected_vars, var), collapse = "+")))
      temp_model <- glm(temp_formula, data = data, family = binomial)
      pval <- anova(model, temp_model, test = "LRT")$`Pr(>Chi)`[2]
      return(pval)
    })
    
    min_p <- min(forward_pvals, na.rm = TRUE)
    min_var <- remaining_vars[which.min(forward_pvals)]
    
    if (!is.na(min_p) && min_p < pin) {
      selected_vars <- c(selected_vars, min_var)
      remaining_vars <- setdiff(remaining_vars, min_var)
      model <- glm(as.formula(paste(response, "~", paste(selected_vars, collapse = "+"))),
                   data = data, family = binomial)
    } else {
      break  # No variables meet the PIN threshold
    }
    
    # --- BACKWARD STEP ---
    repeat {
      if (length(selected_vars) < 2) break  # Need at least 2 to compare
      
      pvals_back <- sapply(selected_vars, function(var) {
        reduced_vars <- setdiff(selected_vars, var)
        reduced_model <- glm(as.formula(paste(response, "~", paste(reduced_vars, collapse = "+"))),
                             data = data, family = binomial)
        pval <- anova(reduced_model, model, test = "LRT")$`Pr(>Chi)`[2]
        return(pval)
      })
      
      max_p <- max(pvals_back, na.rm = TRUE)
      if (!is.na(max_p) && max_p > pout) {
        remove_var <- selected_vars[which.max(pvals_back)]
        selected_vars <- setdiff(selected_vars, remove_var)
        model <- glm(as.formula(paste(response, "~", paste(selected_vars, collapse = "+"))),
                     data = data, family = binomial)
      } else {
        break  # No variables exceed POUT threshold
      }
    }
  }
  
  return(summary(model))
}

# Make sure outcome is a factor
data$V2 <- as.factor(data$V2)

# Define predictors (e.g., V3 to V32)
predictors <- paste0("V", 3:32)

# Run SPSS-style stepwise logistic regression
final_model <- forward_stepwise_sp(data, response = "V2", predictors = predictors)
