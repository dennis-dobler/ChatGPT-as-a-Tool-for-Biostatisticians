import pandas as pd
import numpy as np
import requests, zipfile, io
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Set seed
np.random.seed(1)

# Download and load the dataset
url = "https://archive.ics.uci.edu/static/public/17/breast+cancer+wisconsin+diagnostic.zip"
r = requests.get(url)
z = zipfile.ZipFile(io.BytesIO(r.content))
df = pd.read_csv(z.open("wdbc.data"), header=None)

# Convert target variable to binary
df[1] = df[1] == "M"

# Drop ID column and split data
X = df.drop(columns=[0, 1])
y = df[1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# Reset indices
X_train = X_train.reset_index(drop=True)
y_train = y_train.reset_index(drop=True)
X_test = X_test.reset_index(drop=True)
y_test = y_test.reset_index(drop=True)

# Stepwise AIC
def calculate_aic(model):
    return model.aic

def fit_model(features):
    X_with_const = sm.add_constant(X_train[features]) if features else pd.DataFrame({'const': 1}, index=X_train.index)
    model = sm.Logit(y_train, X_with_const).fit(disp=0)
    return model

def stepwise_selection():
    remaining = set(X_train.columns)
    selected = []
    current_score, best_new_score = np.inf, np.inf
    while True:
        scores_with_candidates = []
        
        # Forward step
        for candidate in remaining:
            model = fit_model(selected + [candidate])
            aic = calculate_aic(model)
            scores_with_candidates.append((aic, candidate))
        
        scores_with_candidates.sort()
        best_new_score, best_candidate = scores_with_candidates[0]
        
        if best_new_score < current_score:
            remaining.remove(best_candidate)
            selected.append(best_candidate)
            current_score = best_new_score
        else:
            break

        # Backward step
        while len(selected) > 1:
            scores_with_removals = []
            for candidate in selected:
                temp_features = [f for f in selected if f != candidate]
                model = fit_model(temp_features)
                aic = calculate_aic(model)
                scores_with_removals.append((aic, candidate))

            scores_with_removals.sort()
            best_removal_score, worst_feature = scores_with_removals[0]

            if best_removal_score < current_score:
                selected.remove(worst_feature)
                current_score = best_removal_score
            else:
                break

    return selected

# Perform stepwise selection
selected_features = stepwise_selection()
print("Selected features:", selected_features)

# Fit final model
final_model = fit_model(selected_features)
print(final_model.summary())

# Predict and evaluate
X_test_selected = sm.add_constant(X_test[selected_features])
test_pred_prob = final_model.predict(X_test_selected)
test_pred_class = test_pred_prob > 0.5

accuracy = accuracy_score(y_test, test_pred_class)
print("Test accuracy:", accuracy)

# List coefficients
print("Model coefficients:")
print(final_model.params)
