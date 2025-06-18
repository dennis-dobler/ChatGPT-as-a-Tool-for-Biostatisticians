import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from urllib.request import urlretrieve
import zipfile
import tempfile
import os

# Set seed
np.random.seed(1)

# Download and extract data
temp = tempfile.NamedTemporaryFile(delete=False)
urlretrieve(
    "https://archive.ics.uci.edu/static/public/17/breast+cancer+wisconsin+diagnostic.zip",
    temp.name
)

with zipfile.ZipFile(temp.name, 'r') as zip_ref:
    zip_ref.extractall(tempfile.gettempdir())

data = pd.read_csv(os.path.join(tempfile.gettempdir(), "wdbc.data"), header=None)
data[1] = data[1] == "M"  # Convert to boolean (malignant)

# Train-test split (80-20)
train, test = train_test_split(data, test_size=0.2, random_state=1)
train = train.drop(columns=[0])
test = test.drop(columns=[0])

# Define predictors and response
y_train = train[1].astype(int)
X_train = train.drop(columns=[1])
X_test = test.drop(columns=[1])
y_test = test[1].astype(int)

# AIC-based stepwise selection
def stepwise_aic(X, y, verbose=True):
    included = []
    best_aic = np.inf
    while True:
        changed = False
        excluded = list(set(X.columns) - set(included))
        candidates = []

        # Try adding features
        for col in excluded:
            model = sm.Logit(y, sm.add_constant(X[included + [col]])).fit(disp=0)
            candidates.append((model.aic, included + [col], f"add {col}"))

        # Try removing features
        for col in included:
            subset = [c for c in included if c != col]
            if subset:
                model = sm.Logit(y, sm.add_constant(X[subset])).fit(disp=0)
                candidates.append((model.aic, subset, f"drop {col}"))

        if not candidates:
            break

        # Pick the model with lowest AIC
        aic_values = [c[0] for c in candidates]
        min_idx = np.argmin(aic_values)
        best_candidate_aic, best_candidate_vars, action = candidates[min_idx]

        if best_candidate_aic + 1e-6 < best_aic:  # small tolerance for float comparison
            if verbose:
                print(f"{action:<12} -> AIC: {best_candidate_aic:.2f}")
            included = best_candidate_vars
            best_aic = best_candidate_aic
            changed = True

        if not changed:
            break

    return included

# Run stepwise AIC selection
selected_vars = stepwise_aic(X_train, y_train, verbose=True)

# Final model
X_train_final = sm.add_constant(X_train[selected_vars])
final_model = sm.Logit(y_train, X_train_final).fit()
print(final_model.summary())

# Evaluate on test data
X_test_final = sm.add_constant(X_test[selected_vars])
pred_probs = final_model.predict(X_test_final)
pred_labels = (pred_probs > 0.5).astype(int)
accuracy = (pred_labels == y_test).mean()

print(f"\nTest Accuracy: {accuracy:.4f}")
print("\nFinal model features:")
print(final_model.params.index.tolist())
