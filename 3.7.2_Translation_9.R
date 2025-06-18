temp <- tempfile()
download.file(
  "https://archive.ics.uci.edu/static/public/17/breast+cancer+wisconsin+diagnostic.zip",
  temp
)
data <- read.table(unz(temp, "wdbc.data"), sep = ",")
data$V2 <- data$V2 == "M"
unlink(temp)

# Load required package
library(MASS)

# Set up the data: convert V2 to binary factor for logistic regression
data$V2 <- as.factor(data$V2)

# Build the null (intercept-only) model
null_model <- glm(V2 ~ 1, data = data, family = binomial)

# Build the full model with predictors V3 to V32
full_model <- glm(V2 ~ ., data = data[, 2:32], family = binomial)

# Run forward stepwise selection (similar to METHOD=FSTEP(LR) in SPSS)
stepwise_model <- stepAIC(null_model, 
                          scope = list(lower = null_model, upper = full_model), 
                          direction = "forward", 
                          trace = TRUE)
