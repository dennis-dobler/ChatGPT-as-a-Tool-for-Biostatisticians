temp <- tempfile()
download.file(
  "https://archive.ics.uci.edu/static/public/17/breast+cancer+wisconsin+diagnostic.zip",
  temp
)
data <- read.table(unz(temp, "wdbc.data"), sep = ",")
data$V2 <- data$V2 == "M"
unlink(temp)

# Convert V2 to binary (TRUE = Malignant)
data$V2 <- as.factor(data$V2)

# Fit the null model (intercept only)
null_model <- glm(V2 ~ 1, data = data, family = binomial)

# Full model with predictors V3 to V32
full_model <- glm(V2 ~ ., data = data[, -1], family = binomial)

# Perform forward stepwise selection using AIC (like SPSS FSTEP(LR))
stepwise_model <- step(null_model, 
                       scope = list(lower = null_model, upper = full_model), 
                       direction = "forward", 
                       trace = TRUE)
