temp <- tempfile()
download.file(
  "https://archive.ics.uci.edu/static/public/17/breast+cancer+wisconsin+diagnostic.zip",
  temp
)
data <- read.table(unz(temp, "wdbc.data"), sep = ",")
data$V2 <- data$V2 == "M"
unlink(temp)

# Custom forward stepwise logistic regression using LRT
stepwise_lr <- function(data, response, predictors, pin = 0.05, pout = 0.10, verbose = TRUE) {
  selected_vars <- c()
  remaining_vars <- predictors
  current_model <- glm(as.formula(paste(response, "~ 1")), data = data, family = binomial)
  
  repeat {
    pvals <- sapply(remaining_vars, function(var) {
      test_model <- glm(
        as.formula(paste(response, "~", paste(c(selected_vars, var), collapse = "+"))),
        data = data,
        family = binomial
      )
      lr_test <- anova(current_model, test_model, test = "LRT")
      pval <- lr_test$`Pr(>Chi)`[2]
      return(pval)
    })
    
    min_pval <- min(pvals, na.rm = TRUE)
    best_var <- remaining_vars[which.min(pvals)]
    
    if (min_pval < pin) {
      if (verbose) cat("Adding:", best_var, "with p =", min_pval, "\n")
      selected_vars <- c(selected_vars, best_var)
      remaining_vars <- setdiff(remaining_vars, best_var)
      current_model <- glm(
        as.formula(paste(response, "~", paste(selected_vars, collapse = "+"))),
        data = data,
        family = binomial
      )
      
      # Check for variables to remove
      if (length(selected_vars) > 1) {
        drop_pvals <- summary(current_model)$coefficients[-1, 4]
        worst_var <- names(drop_pvals)[which.max(drop_pvals)]
        max_pval <- max(drop_pvals)
        
        if (max_pval > pout) {
          if (verbose) cat("Removing:", worst_var, "with p =", max_pval, "\n")
          selected_vars <- setdiff(selected_vars, worst_var)
          remaining_vars <- c(remaining_vars, worst_var)
          current_model <- glm(
            as.formula(paste(response, "~", paste(selected_vars, collapse = "+"))),
            data = data,
            family = binomial
          )
        }
      }
    } else {
      break
    }
  }
  
  return(current_model)
}

# Setup your data first
colnames(data) <- c("ID", "Diagnosis", paste0("V", 3:32))
data$Diagnosis <- ifelse(data$Diagnosis == "M", 1, 0)  # Binary 1/0

# Run custom stepwise regression
model <- stepwise_lr(
  data = data[, -1],  # Drop ID
  response = "Diagnosis",
  predictors = paste0("V", 3:32),
  pin = 0.05,
  pout = 0.10
)

# View final model
summary(model)
