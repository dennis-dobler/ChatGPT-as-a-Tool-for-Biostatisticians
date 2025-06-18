temp <- tempfile()
download.file(
  "https://archive.ics.uci.edu/static/public/17/breast+cancer+wisconsin+diagnostic.zip",
  temp
)
data <- read.table(unz(temp, "wdbc.data"), sep = ",")
data$V2 <- data$V2 == "M"
unlink(temp)


# Rename columns for clarity (optional but recommended)
colnames(data)[1:2] <- c("ID", "Diagnosis")  # V1 is ID, V2 is Diagnosis
# Ensure diagnosis is a factor
data$Diagnosis <- factor(data$Diagnosis, levels = c(FALSE, TRUE), labels = c("B", "M"))

# Fit initial model with no predictors (intercept-only model)
null_model <- glm(Diagnosis ~ 1, data = data, family = binomial)

# Fit full model with all predictors (V3 to V32)
full_model <- glm(Diagnosis ~ ., data = data[, -1], family = binomial)  # Exclude ID column

# Perform stepwise selection (forward selection with LR)
stepwise_model <- step(null_model, 
                       scope = list(lower = null_model, upper = full_model),
                       direction = "forward", 
                       trace = 1)  # You can set trace = 0 to suppress output

# Summary of the final model
summary(stepwise_model)
