temp <- tempfile()
download.file(
  "https://archive.ics.uci.edu/static/public/17/breast+cancer+wisconsin+diagnostic.zip",
  temp
)
data <- read.table(unz(temp, "wdbc.data"), sep = ",")
data$V2 <- data$V2 == "M"
unlink(temp)

# Load necessary libraries
library(MASS)

# Set outcome variable as factor for logistic regression
data$V2 <-
  factor(data$V2, levels = c(FALSE, TRUE))  # Ensure proper binary coding

# Fit null model (intercept only)
null_model <- glm(V2 ~ 1, data = data, family = binomial)

# Fit full model with all predictors V3 to V32
full_model <-
  glm(V2 ~ ., data = data[,-1], family = binomial)  # exclude V1 (ID column)

# Perform forward stepwise selection
stepwise_model <- stepAIC(
  null_model,
  scope = list(lower = null_model, upper = full_model),
  direction = "forward",
  trace = TRUE
)
