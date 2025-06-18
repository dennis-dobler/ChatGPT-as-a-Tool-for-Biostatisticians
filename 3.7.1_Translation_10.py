import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from urllib.request import urlretrieve
from zipfile import ZipFile
import tempfile
import os
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Set seed
np.random.seed(1)

# Download and extract dataset
url = "https://archive.ics.uci.edu/static/public/17/breast+cancer+wisconsin+diagnostic.zip"
temp_file = tempfile.NamedTemporaryFile(delete=False)
urlretrieve(url, temp_file.name)

with ZipFile(temp_file.name, 'r') as zip_ref:
    zip_ref.extractall(tempfile.gettempdir())

data_path = os.path.join(tempfile.gettempdir(), "wdbc.data")
data = pd.read_csv(data_path, header=None)

# Convert diagnosis to binary
data[1] = data[1] == "M"

# Drop ID column and split data
X = data.drop(columns=[0, 1])
y = data[1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# Stepwise feature selection (AIC approximation)
def calculate_aic(model, X, y):
    llf = model.llf
    k = model.df_model + 1  # number of parameters + intercept
    return -2 * llf + 2 * k

def stepwise_selection(X, y):
    included = []
    while True:
        changed = False
        # Forward step
        excluded = list(set(X.columns) - set(included))
        new_pvals = {}
        for new_col in excluded:
            model = sm.Logit(y, sm.add_constant(pd.DataFrame(X[included + [new_col]]))).fit(disp=0)
            new_pvals[new_col] = calculate_aic(model, X[included + [new_col]], y)
        if new_pvals:
            best_new = min(new_pvals, key=new_pvals.get)
            best_new_aic = new_pvals[best_new]
            if not included:
                current_aic = float('inf')
            else:
                model_current = sm.Logit(y, sm.add_constant(pd.DataFrame(X[included]))).fit(disp=0)
                current_aic = calculate_aic(model_current, X[included], y)
            if best_new_aic < current_aic:
                included.append(best_new)
                changed = True

        # Backward step
        if len(included) > 1:
            aic_with_drops = {}
            for col in included:
                cols = list(included)
                cols.remove(col)
                model = sm.Logit(y, sm.add_constant(pd.DataFrame(X[cols]))).fit(disp=0)
                aic_with_drops[col] = calculate_aic(model, X[cols], y)
            best_drop = min(aic_with_drops, key=aic_with_drops.get)
            if aic_with_drops[best_drop] < current_aic:
                included.remove(best_drop)
                changed = True

        if not changed:
            break
    return included

selected_features = stepwise_selection(X_train, y_train)

# Fit final model
final_model = sm.Logit(y_train, sm.add_constant(X_train[selected_features])).fit()
print(final_model.summary())

# Predict on test set
test_pred_probs = final_model.predict(sm.add_constant(X_test[selected_features]))
test_pred = test_pred_probs > 0.5
accuracy = (test_pred == y_test).mean()
print(f"Test accuracy: {accuracy:.4f}")

# Coefficient names
print("Selected features:", ['const'] + selected_features)
