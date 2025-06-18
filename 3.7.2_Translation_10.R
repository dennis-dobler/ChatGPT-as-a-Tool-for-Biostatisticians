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

# Prepare the data
data$V2 <- as.factor(data$V2)  # make outcome a factor

# Define full and null models
full_model <- glm(V2 ~ ., data = data[, -1], family = binomial())  # Remove ID column V1
null_model <- glm(V2 ~ 1, data = data[, -1], family = binomial())

# Forward stepwise logistic regression using AIC
step_model <- stepAIC(null_model, 
                      scope = list(lower = null_model, upper = full_model),
                      direction = "forward",
                      steps = 20,  # corresponds to ITERATE(20)
                      trace = TRUE)

summary(step_model)
