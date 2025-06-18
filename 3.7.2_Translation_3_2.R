temp <- tempfile()
download.file(
  "https://archive.ics.uci.edu/static/public/17/breast+cancer+wisconsin+diagnostic.zip",
  temp
)
data <- read.table(unz(temp, "wdbc.data"), sep = ",")
data$V2 <- data$V2 == "M"
unlink(temp)

custom_forward_stepwise <- function(data, outcome, predictors, pin = 0.05, pout = 0.10, verbose = TRUE) {
  # Initial setup
  included <- character(0)
  remaining <- predictors
  formula_base <- as.formula(paste(outcome, "~ 1"))
  model <- glm(formula_base, data = data, family = binomial)
  
  repeat {
    # Try adding each of the remaining variables and calculate p-value from LR test
    add_tests <- lapply(remaining, function(var) {
      formula_try <- as.formula(paste(outcome, "~", paste(c(included, var), collapse = "+")))
      model_try <- glm(formula_try, data = data, family = binomial)
      pval <- anova(model, model_try, test = "Chisq")$`Pr(>Chi)`[2]
      return(data.frame(variable = var, pval = pval))
    })
    
    # Combine into one data frame
    add_results <- do.call(rbind, add_tests)
    
    # Find the variable with the smallest p-value for entry
    best_candidate <- add_results[which.min(add_results$pval), ]
    
    if (verbose) {
      cat("Evaluating variable for inclusion:\n")
      print(add_results)
    }
    
    # Check if it meets the inclusion threshold
    if (best_candidate$pval < pin) {
      included <- c(included, best_candidate$variable)
      remaining <- setdiff(remaining, best_candidate$variable)
      model <- glm(as.formula(paste(outcome, "~", paste(included, collapse = "+"))), data = data, family = binomial)
      if (verbose) cat("Added:", best_candidate$variable, "\n\n")
    } else {
      if (verbose) cat("No more variables meet inclusion criteria.\n\n")
      break
    }
    
    # Check for any variables that now fail the exclusion threshold (POUT)
    if (length(included) > 1) {
      drop_tests <- lapply(included, function(var) {
        reduced <- setdiff(included, var)
        model_reduced <- glm(as.formula(paste(outcome, "~", paste(reduced, collapse = "+"))), data = data, family = binomial)
        pval <- anova(model_reduced, model, test = "Chisq")$`Pr(>Chi)`[2]
        return(data.frame(variable = var, pval = pval))
      })
      drop_results <- do.call(rbind, drop_tests)
      
      if (verbose) {
        cat("Evaluating variables for exclusion:\n")
        print(drop_results)
      }
      
      # Drop any that exceed POUT
      to_remove <- drop_results$variable[drop_results$pval > pout]
      if (length(to_remove) > 0) {
        included <- setdiff(included, to_remove)
        remaining <- union(remaining, to_remove)
        model <- glm(as.formula(paste(outcome, "~", paste(included, collapse = "+"))), data = data, family = binomial)
        if (verbose) cat("Removed:", paste(to_remove, collapse = ", "), "\n\n")
      }
    }
  }
  
  return(model)
}

# Your setup
data$V2 <- factor(data$V2, levels = c(FALSE, TRUE))
predictors <- paste0("V", 3:32)

# Run custom stepwise LR
final_model <- custom_forward_stepwise(
  data = data[, -1],  # Remove ID column
  outcome = "V2",
  predictors = predictors,
  pin = 0.05,
  pout = 0.10
)

# Check final model
summary(final_model)
