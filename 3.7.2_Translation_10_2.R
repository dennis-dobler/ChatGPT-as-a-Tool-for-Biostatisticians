temp <- tempfile()
download.file(
  "https://archive.ics.uci.edu/static/public/17/breast+cancer+wisconsin+diagnostic.zip",
  temp
)
data <- read.table(unz(temp, "wdbc.data"), sep = ",")
data$V2 <- data$V2 == "M"
unlink(temp)

# Prepare the data
data$V2 <- as.factor(data$V2)
df <- data[, -1]  # remove ID column

# Stepwise function
forward_stepwise_lrt <- function(data, response, entry_thresh = 0.05, removal_thresh = 0.10, max_steps = 20) {
  vars <- setdiff(names(data), response)
  included <- c()
  step <- 0
  continue <- TRUE
  
  while (continue && step < max_steps) {
    step <- step + 1
    message("\n--- Step ", step, " ---")
    
    # Try adding variables
    excluded <- setdiff(vars, included)
    add_pvals <- sapply(excluded, function(var) {
      fmla <- as.formula(paste(response, "~", paste(c(included, var), collapse = "+")))
      full <- glm(fmla, data = data, family = binomial())
      if (length(included) == 0) {
        null <- glm(as.formula(paste(response, "~ 1")), data = data, family = binomial())
      } else {
        null <- glm(as.formula(paste(response, "~", paste(included, collapse = "+"))), data = data, family = binomial())
      }
      pval <- anova(null, full, test = "Chisq")[2, "Pr(>Chi)"]
      return(pval)
    })
    
    min_p <- min(add_pvals)
    best_var <- names(add_pvals)[which.min(add_pvals)]
    
    if (!is.na(min_p) && min_p < entry_thresh) {
      included <- c(included, best_var)
      message("Added: ", best_var, " (p = ", round(min_p, 4), ")")
    } else {
      message("No variable added (min p = ", round(min_p, 4), ")")
      break
    }
    
    # Try removing variables
    if (length(included) > 1) {
      drop_pvals <- sapply(included, function(var) {
        reduced <- setdiff(included, var)
        full <- glm(as.formula(paste(response, "~", paste(included, collapse = "+"))), data = data, family = binomial())
        reduced_model <- glm(as.formula(paste(response, "~", paste(reduced, collapse = "+"))), data = data, family = binomial())
        pval <- anova(reduced_model, full, test = "Chisq")[2, "Pr(>Chi)"]
        return(pval)
      })
      max_p <- max(drop_pvals)
      worst_var <- names(drop_pvals)[which.max(drop_pvals)]
      
      if (!is.na(max_p) && max_p > removal_thresh) {
        included <- setdiff(included, worst_var)
        message("Removed: ", worst_var, " (p = ", round(max_p, 4), ")")
      }
    }
  }
  
  # Final model
  final_formula <- as.formula(paste(response, "~", paste(included, collapse = "+")))
  final_model <- glm(final_formula, data = data, family = binomial())
  return(final_model)
}

# Run stepwise
model <- forward_stepwise_lrt(df, response = "V2", entry_thresh = 0.05, removal_thresh = 0.10, max_steps = 20)

# Summary
summary(model)
