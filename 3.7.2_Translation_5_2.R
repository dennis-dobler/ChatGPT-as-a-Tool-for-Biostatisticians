temp <- tempfile()
download.file(
  "https://archive.ics.uci.edu/static/public/17/breast+cancer+wisconsin+diagnostic.zip",
  temp
)
data <- read.table(unz(temp, "wdbc.data"), sep = ",")
data$V2 <- data$V2 == "M"
unlink(temp)

# Define response and predictor variables
response <- "V2"
predictors <- paste0("V", 3:32)
included <- c()  # Variables currently in the model
remaining <- predictors  # Variables not yet in the model

# Set PIN and max iterations
pin <- 0.05
max_iter <- 20
iter <- 0
improved <- TRUE

while (improved && iter < max_iter) {
  iter <- iter + 1
  p_values <- c()
  
  # Test adding each remaining variable one at a time
  for (var in remaining) {
    current_formula <- as.formula(
      paste(response, "~", paste(c(included, var), collapse = " + "))
    )
    base_formula <- if (length(included) == 0) {
      as.formula(paste(response, "~ 1"))
    } else {
      as.formula(paste(response, "~", paste(included, collapse = " + ")))
    }
    
    full_model <- glm(current_formula, data = data, family = binomial)
    base_model <- glm(base_formula, data = data, family = binomial)
    
    # Likelihood ratio test
    lr_test <- anova(base_model, full_model, test = "LRT")
    p_val <- lr_test$`Pr(>Chi)`[2]
    p_values <- c(p_values, p_val)
  }
  
  names(p_values) <- remaining
  
  # Find the best candidate with lowest p-value
  min_p <- min(p_values, na.rm = TRUE)
  best_var <- names(which.min(p_values))
  
  if (!is.na(min_p) && min_p < pin) {
    included <- c(included, best_var)
    remaining <- setdiff(remaining, best_var)
    cat(sprintf("Step %d: Added %s (p = %.4f)\n", iter, best_var, min_p))
  } else {
    improved <- FALSE
  }
}

# Final model
final_formula <- as.formula(paste(response, "~", paste(included, collapse = " + ")))
final_model <- glm(final_formula, data = data, family = binomial)

# Summary of the selected model
summary(final_model)
