import pandas as pd
import numpy as np
import urllib.request
import zipfile
import tempfile
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
from statsmodels.tools import add_constant
from statsmodels.discrete.discrete_model import Logit
from mlxtend.feature_selection import SequentialFeatureSelector as SFS

# Download and extract data
url = "https://archive.ics.uci.edu/static/public/17/breast+cancer+wisconsin+diagnostic.zip"
temp = tempfile.NamedTemporaryFile(delete=False)
urllib.request.urlretrieve(url, temp.name)

with zipfile.ZipFile(temp.name, 'r') as zip_ref:
    with zip_ref.open("wdbc.data") as file:
        data = pd.read_csv(file, header=None)

# Prepare target and features
data[1] = data[1] == "M"  # Malignant as True
X = data.drop(columns=[0, 1])  # Drop ID and keep features
y = data[1]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1, stratify=y
)

# Add constant for intercept-only model
X_train_const = pd.DataFrame({'const': 1}, index=X_train.index)
X_test_const = pd.DataFrame({'const': 1}, index=X_test.index)

# Fit base model (intercept only)
base_model = Logit(y_train, X_train_const).fit(disp=0)

# Stepwise selection (forward + backward) using mlxtend
sfs = SFS(
    sm.Logit,
    k_features='best',
    forward=True,
    floating=True,
    scoring='accuracy',
    cv=0,
    verbose=0
)

sfs = sfs.fit(X_train.values, y_train.values)

selected_features = list(sfs.k_feature_idx_)
X_train_selected = X_train.iloc[:, selected_features]
X_test_selected = X_test.iloc[:, selected_features]

# Fit final model
X_train_final = add_constant(X_train_selected)
X_test_final = add_constant(X_test_selected)

final_model = Logit(y_train, X_train_final).fit()

# Summary
print(final_model.summary())

# Prediction and accuracy
test_pred_probs = final_model.predict(X_test_final)
test_pred = test_pred_probs > 0.5
accuracy = (test_pred == y_test).mean()
print("Accuracy:", accuracy)

# Coefficients
print("Selected features:", X_train_selected.columns.tolist())
