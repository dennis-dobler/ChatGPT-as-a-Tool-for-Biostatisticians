temp <- tempfile()
download.file(
  "https://archive.ics.uci.edu/static/public/17/breast+cancer+wisconsin+diagnostic.zip",
  temp
)
data <- read.table(unz(temp, "wdbc.data"), sep = ",")
data$V2 <- data$V2 == "M"
unlink(temp)

# Prepare data frame with proper column names (optional but useful)
colnames(data) <- c("ID", "Diagnosis", paste0("V", 3:32))
data$Diagnosis <- as.factor(data$Diagnosis)

# Fit null model (intercept only)
null_model <- glm(Diagnosis ~ 1, data = data, family = binomial)

# Fit full model with all predictors
full_model <- glm(Diagnosis ~ ., data = data[, -1], family = binomial)  # remove ID

# Stepwise logistic regression - forward selection
stepwise_model <- step(
  null_model,
  scope = list(lower = null_model, upper = full_model),
  direction = "forward",
  trace = TRUE
)

# View final model
summary(stepwise_model)
