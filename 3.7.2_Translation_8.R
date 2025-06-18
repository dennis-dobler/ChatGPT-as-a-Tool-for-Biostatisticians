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

library(MASS)

# Null model (intercept only)
null_model <- glm(Diagnosis ~ 1, data = data, family = binomial)

# Full model with all predictors
full_model <- glm(Diagnosis ~ ., data = data[, -1], family = binomial)  # remove ID column

# Forward stepwise selection
model_step <- stepAIC(null_model, 
                      scope = list(lower = null_model, upper = full_model),
                      direction = "forward",
                      trace = TRUE)
summary(model_step)

# install.packages("olsrr")
library(olsrr)

# Run logistic regression with forward stepwise selection
model <- glm(Diagnosis ~ ., data = data[, -1], family = binomial)
ols_step_forward_p(model, prem = 0.05, pstay = 0.10, details = TRUE)
