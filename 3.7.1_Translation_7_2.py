import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
import requests, zipfile, io
from itertools import combinations

# Download and read the dataset
url = "https://archive.ics.uci.edu/static/public/17/breast+cancer+wisconsin+diagnostic.zip"
r = requests.get(url)
z = zipfile.ZipFile(io.BytesIO(r.content))
data = pd.read_csv(z.open("wdbc.data"), header=None)

# Convert diagnosis column to boolean: M = True, B = False
data[1] = data[1] == "M"

# Train/test split (80/20), dropping ID column
X = data.drop(columns=[0, 1])
y = data[1]
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=1)

# Rename columns for clarity (V3 to V32)
X_train.columns = [f'V{i}' for i in range(3, 33)]
print(X_train.columns)
X_test.columns = [f'V{i}' for i in range(3, 33)]

# Combine with target
train = pd.concat([y_train.reset_index(drop=True), X_train.reset_index(drop=True)], axis=1)
train.columns = ['V2'] + list(X_train.columns)
test = pd.concat([y_test.reset_index(drop=True), X_test.reset_index(drop=True)], axis=1)
test.columns = ['V2'] + list(X_test.columns)

# Stepwise selection based on AIC
def stepwise_aic(X, y, verbose=True):
    initial_features = []
    remaining_features = list(X.columns)
    best_aic = np.inf
    current_features = []
    while True:
        aic_with_candidates = []
        # Try adding each feature
        for candidate in remaining_features:
            try_features = current_features + [candidate]
            model = sm.Logit(y, sm.add_constant(X[try_features])).fit(disp=0)
            aic_with_candidates.append((model.aic, try_features))
        # Try removing each feature
        for candidate in current_features:
            try_features = [f for f in current_features if f != candidate]
            if try_features:
                model = sm.Logit(y, sm.add_constant(X[try_features])).fit(disp=0)
                aic_with_candidates.append((model.aic, try_features))
        
        # Select the best among add/remove candidates
        aic_with_candidates.sort()
        best_new_aic, best_new_features = aic_with_candidates[0]
        
        if best_new_aic < best_aic:
            best_aic = best_new_aic
            current_features = best_new_features
            remaining_features = list(set(X.columns) - set(current_features))
            if verbose:
                print(f"Stepwise AIC improved to {best_aic:.2f}, features: {current_features}")
        else:
            break
    return current_features

# Run AIC-based stepwise selection
selected_features = stepwise_aic(train.iloc[:, 1:], train['V2'])

# Fit final model with selected features
X_train_sel = sm.add_constant(train[selected_features])
final_model = sm.Logit(train['V2'], X_train_sel).fit()
print(final_model.summary())

# Predict on test data
X_test_sel = sm.add_constant(test[selected_features])
test_pred = final_model.predict(X_test_sel)
accuracy = np.mean((test_pred > 0.5) == test['V2'])
print("Test accuracy:", accuracy)

# Output selected variable names
print("Selected variables:", final_model.params.index.tolist())
