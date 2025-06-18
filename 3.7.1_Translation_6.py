import pandas as pd
import numpy as np
import tempfile
import requests
import zipfile
from io import BytesIO
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import statsmodels.api as sm
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from sklearn.linear_model import LogisticRegression

from statsmodels.stats.outliers_influence import variance_inflation_factor

def drop_high_vif(X, thresh=10.0):
    X = X.copy()
    while True:
        vif = pd.Series(
            [variance_inflation_factor(X.values, i) for i in range(X.shape[1])],
            index=X.columns
        )
        max_vif = vif.max()
        if max_vif > thresh:
            drop_col = vif.idxmax()
            print(f"Dropping '{drop_col}' with VIF: {max_vif:.2f}")
            X = X.drop(columns=[drop_col])
        else:
            break
    return X

# Download and load the data
url = "https://archive.ics.uci.edu/static/public/17/breast+cancer+wisconsin+diagnostic.zip"
response = requests.get(url)
with zipfile.ZipFile(BytesIO(response.content)) as z:
    with z.open("wdbc.data") as f:
        df = pd.read_csv(f, header=None)

# Convert diagnosis to binary
df[1] = df[1] == "M"

# Drop ID column and split
X = df.drop(columns=[0, 1])
y = df[1].astype(int)

# Remove near-constant columns (optional but safe)
X = X.loc[:, X.std() > 1e-8]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1
)

# Define custom AIC scoring function with error handling
def safe_aic_score(estimator, X, y):
    try:
        model = sm.Logit(y, sm.add_constant(X)).fit(disp=0)
        return model.aic
    except np.linalg.LinAlgError:
        return np.inf  # Penalize singular models heavily

# Logistic regression model
logreg = LogisticRegression(max_iter=10000, solver='liblinear')

# Stepwise selection
sfs = SFS(
    logreg,
    k_features="best",
    forward=True,
    floating=True,
    scoring=safe_aic_score,
    cv=0,
    n_jobs=-1
)

sfs = sfs.fit(X_train, y_train)
selected_features = list(sfs.k_feature_names_)

# Add constant and remove multicollinearity
X_train_sel = sm.add_constant(X_train[selected_features], has_constant='add')
X_train_nocol = drop_high_vif(X_train_sel)

# Make sure test data matches
X_test_sel = sm.add_constant(X_test[selected_features], has_constant='add')
X_test_nocol = X_test_sel[X_train_nocol.columns]

# Fit final model
final_model = sm.Logit(y_train, X_train_nocol).fit()
print(final_model.summary())

# Predict and evaluate
test_pred_prob = final_model.predict(X_test_nocol)
test_pred = test_pred_prob > 0.5
accuracy = accuracy_score(y_test, test_pred)
print("Accuracy:", accuracy)

# Coefficients
print("Final selected features:", final_model.params.index.tolist())

