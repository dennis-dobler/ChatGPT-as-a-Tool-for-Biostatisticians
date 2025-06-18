import pandas as pd
import numpy as np
import requests, zipfile, io
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Set seed for reproducibility
np.random.seed(1)

# Download and load the dataset
url = "https://archive.ics.uci.edu/static/public/17/breast+cancer+wisconsin+diagnostic.zip"
r = requests.get(url)
z = zipfile.ZipFile(io.BytesIO(r.content))
df = pd.read_csv(z.open("wdbc.data"), header=None)

# Convert target variable to binary
df[1] = df[1] == "M"

# Drop the ID column and split into train/test sets
X = df.drop(columns=[0, 1])
y = df[1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# Fit base logistic regression model (intercept only)
# Reset indices to align exog and endog
y_train_reset = y_train.reset_index(drop=True)
X_train_const = pd.DataFrame({'const': 1}, index=y_train_reset.index)

# Fit base model
base_model = sm.Logit(y_train_reset, X_train_const).fit(disp=0)

# Stepwise selection using mlxtend
lr = LogisticRegression(solver='liblinear')
sfs = SFS(lr,
          k_features='best',
          forward=True,
          floating=True,
          scoring='accuracy',
          cv=5,
          n_jobs=-1)

sfs = sfs.fit(X_train, y_train)

# Fit model with selected features
selected_features = list(sfs.k_feature_names_)
X_train_selected = X_train[selected_features]
X_test_selected = X_test[selected_features]

final_model = sm.Logit(y_train, sm.add_constant(X_train_selected)).fit(disp=0)

# Predict on test data
test_pred_prob = final_model.predict(sm.add_constant(X_test_selected))
test_pred_class = test_pred_prob > 0.5

# Accuracy
accuracy = accuracy_score(y_test, test_pred_class)
print("Accuracy:", accuracy)

# Coefficients
print("Selected features and coefficients:")
print(final_model.params)
