temp <- tempfile()
download.file(
  "https://archive.ics.uci.edu/static/public/17/breast+cancer+wisconsin+diagnostic.zip",
  temp
)
data <- read.table(unz(temp, "wdbc.data"), sep = ",")
data$V2 <- data$V2 == "M"
unlink(temp)

# Assign column names (from the UCI description)
colnames(data) <- c("ID", "Diagnosis", paste0("V", 3:32))
data$Diagnosis <- as.numeric(data$Diagnosis)  # TRUE (M) = 1, FALSE (B) = 0

forward_stepwise_lr <- function(data, outcome, predictors, pin = 0.05, pout = 0.10, max_iter = 20) {
  selected_vars <- c()
  remaining_vars <- predictors
  iter <- 0
  current_model <- glm(as.formula(paste(outcome, "~ 1")), data = data, family = binomial)
  
  while (length(remaining_vars) > 0 && iter < max_iter) {
    iter <- iter + 1
    pvals <- sapply(remaining_vars, function(var) {
      test_model <- glm(as.formula(paste(outcome, "~", paste(c(selected_vars, var), collapse = "+"))), 
                        data = data, family = binomial)
      lr_test <- anova(current_model, test_model, test = "LRT")
      lr_test$`Pr(>Chi)`[2]  # p-value of added variable
    })
    
    best_var <- remaining_vars[which.min(pvals)]
    best_p <- min(pvals, na.rm = TRUE)
    
    if (!is.na(best_p) && best_p < pin) {
      selected_vars <- c(selected_vars, best_var)
      remaining_vars <- setdiff(remaining_vars, best_var)
      current_model <- glm(as.formula(paste(outcome, "~", paste(selected_vars, collapse = "+"))),
                           data = data, family = binomial)
      message(sprintf("Step %d: Added %s (p = %.4f)", iter, best_var, best_p))
    } else {
      break
    }
  }
  
  # Optionally remove non-significant predictors based on POUT
  if (length(selected_vars) > 0) {
    final_model <- current_model
    repeat {
      pvals <- summary(final_model)$coefficients[-1, 4]
      worst_p <- max(pvals)
      if (worst_p > pout) {
        remove_var <- names(which.max(pvals))
        selected_vars <- setdiff(selected_vars, remove_var)
        final_model <- glm(as.formula(paste(outcome, "~", paste(selected_vars, collapse = "+"))),
                           data = data, family = binomial)
        message(sprintf("Removed %s (p = %.4f)", remove_var, worst_p))
      } else {
        break
      }
    }
    return(final_model)
  } else {
    return(current_model)
  }
}

# Assume data is already loaded and cleaned
data$Diagnosis <- as.numeric(data$Diagnosis)  # TRUE (M) = 1, FALSE (B) = 0
predictors <- paste0("V", 3:32)  # SPSS V3–V32
model <- forward_stepwise_lr(data = data, outcome = "Diagnosis", predictors = predictors)
summary(model)
