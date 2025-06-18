temp <- tempfile()
download.file(
  "https://archive.ics.uci.edu/static/public/17/breast+cancer+wisconsin+diagnostic.zip",
  temp
)
data <- read.table(unz(temp, "wdbc.data"), sep = ",")
data$V2 <- data$V2 == "M"
unlink(temp)

# Ensure the outcome is a factor for logistic regression
data$V2 <- factor(data$V2, levels = c(FALSE, TRUE))

# Full model with all predictors
full_model <- glm(V2 ~ ., data = data[, -1], family = binomial)

# Null model (intercept only)
null_model <- glm(V2 ~ 1, data = data[, -1], family = binomial)

# Forward stepwise selection using AIC as default (we’ll emulate PIN/POUT next)
stepwise_model <- step(
  null_model,
  scope = list(lower = null_model, upper = full_model),
  direction = "forward",
  trace = 1
)

summary(stepwise_model)
