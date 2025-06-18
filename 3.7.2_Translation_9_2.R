temp <- tempfile()
download.file(
  "https://archive.ics.uci.edu/static/public/17/breast+cancer+wisconsin+diagnostic.zip",
  temp
)
data <- read.table(unz(temp, "wdbc.data"), sep = ",")
data$V2 <- data$V2 == "M"
unlink(temp)

stepwise_lr <- function(data, response, predictors, pin = 0.05, pout = 0.10) {
  included <- c()
  remaining <- predictors
  formula_base <- as.formula(paste(response, "~ 1"))
  current_model <- glm(formula_base, data = data, family = binomial)
  
  repeat {
    # Forward step: test all remaining predictors
    pvals <- sapply(remaining, function(var) {
      temp_model <- update(current_model, paste(". ~ . +", var))
      test <- anova(current_model, temp_model, test = "Chisq")
      test$`Pr(>Chi)`[2]  # p-value for likelihood ratio test
    })
    
    min_pval <- min(pvals)
    best_var <- remaining[which.min(pvals)]
    
    # If best variable meets PIN threshold, add it
    if (min_pval < pin) {
      included <- c(included, best_var)
      remaining <- setdiff(remaining, best_var)
      current_model <- update(current_model, paste(". ~ . +", best_var))
    } else {
      break  # no variable meets entry criterion
    }
    
    # Backward step: check if any included variable should be removed
    pvals_included <- sapply(included, function(var) {
      reduced_model <- update(current_model, paste(". ~ . -", var))
      test <- anova(reduced_model, current_model, test = "Chisq")
      test$`Pr(>Chi)`[2]  # p-value for LRT
    })
    
    max_pval <- max(pvals_included)
    worst_var <- included[which.max(pvals_included)]
    
    if (max_pval > pout) {
      included <- setdiff(included, worst_var)
      current_model <- update(current_model, paste(". ~ . -", worst_var))
    }
  }
  
  return(current_model)
}

# Prepare data
data$V2 <- as.factor(data$V2)
predictors <- paste0("V", 3:32)

# Run the custom stepwise logistic regression
model_final <- stepwise_lr(data = data, response = "V2", predictors = predictors)

# Summary of final model
summary(model_final)
