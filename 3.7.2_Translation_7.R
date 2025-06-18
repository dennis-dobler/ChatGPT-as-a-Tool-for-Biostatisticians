temp <- tempfile()
download.file(
  "https://archive.ics.uci.edu/static/public/17/breast+cancer+wisconsin+diagnostic.zip",
  temp
)
data <- read.table(unz(temp, "wdbc.data"), sep = ",")
data$V2 <- data$V2 == "M"
unlink(temp)

# Load necessary package
library(MASS)

# Prepare the data
# Set column names (optional but helps readability)
colnames(data) <- c("ID", "Diagnosis", paste0("V", 3:32))

# Convert outcome to binary numeric: 1 = Malignant, 0 = Benign
data$Diagnosis <- ifelse(data$Diagnosis == "M", 1, 0)

# Create the full model formula
full_formula <- as.formula(paste("Diagnosis ~", paste(colnames(data)[3:32], collapse = " + ")))

# Fit the null (intercept-only) model
null_model <- glm(Diagnosis ~ 1, data = data, family = binomial)

# Fit the full model
full_model <- glm(full_formula, data = data, family = binomial)

# Stepwise selection (forward)
stepwise_model <- stepAIC(null_model, 
                          scope = list(lower = null_model, upper = full_model), 
                          direction = "forward", 
                          trace = TRUE)

# View the final model summary
summary(stepwise_model)
