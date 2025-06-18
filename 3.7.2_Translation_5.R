temp <- tempfile()
download.file(
  "https://archive.ics.uci.edu/static/public/17/breast+cancer+wisconsin+diagnostic.zip",
  temp
)
data <- read.table(unz(temp, "wdbc.data"), sep = ",")
data$V2 <- data$V2 == "M"
unlink(temp)

# Recode V2 as binary (already done in your script)
data$V2 <- data$V2 == "M"  # TRUE for malignant, FALSE for benign

# Fit null model (intercept only)
null_model <- glm(V2 ~ 1, data = data, family = binomial)

# Fit full model with predictors V3 to V32
full_model <- glm(V2 ~ ., data = data[, c("V2", paste0("V", 3:32))], family = binomial)

# Stepwise selection: forward (based on AIC by default; we'll mimic p-value criteria with direction = "forward")
# Note: R's step() uses AIC, but to mimic PIN/POUT thresholds you would need a custom approach
# For now, we'll do standard forward selection using step()
step_model <- step(null_model, 
                   scope = list(lower = null_model, upper = full_model),
                   direction = "forward",
                   trace = TRUE)

# Summary of the final model
summary(step_model)
