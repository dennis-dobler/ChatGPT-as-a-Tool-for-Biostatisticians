import pandas as pd
import numpy as np
import tempfile
import zipfile
import requests
from io import BytesIO
import statsmodels.api as sm
from sklearn.model_selection import train_test_split

# Stepwise AIC selection
def stepwise_aic(X, y, 
                 initial_features=None, 
                 verbose=True):
    if initial_features is None:
        initial_features = []
    included = list(initial_features)
    best_aic = np.inf
    while True:
        changed = False
        # Forward step
        excluded = list(set(X.columns) - set(included))
        aic_with_candidates = {}
        for new_feature in excluded:
            try_features = included + [new_feature]
            model = sm.Logit(y, sm.add_constant(X[try_features])).fit(disp=False)
            aic_with_candidates[new_feature] = model.aic
        if aic_with_candidates:
            best_new_feature = min(aic_with_candidates, key=aic_with_candidates.get)
            if aic_with_candidates[best_new_feature] < best_aic:
                included.append(best_new_feature)
                best_aic = aic_with_candidates[best_new_feature]
                changed = True
                if verbose:
                    print(f'Add {best_new_feature}, AIC: {best_aic:.3f}')
        
        # Backward step
        aic_with_candidates = {}
        for feature in included:
            try_features = list(set(included) - {feature})
            if not try_features:
                continue
            model = sm.Logit(y, sm.add_constant(X[try_features])).fit(disp=False)
            aic_with_candidates[feature] = model.aic
        if aic_with_candidates:
            worst_feature = min(aic_with_candidates, key=aic_with_candidates.get)
            if aic_with_candidates[worst_feature] < best_aic:
                included.remove(worst_feature)
                best_aic = aic_with_candidates[worst_feature]
                changed = True
                if verbose:
                    print(f'Drop {worst_feature}, AIC: {best_aic:.3f}')
        if not changed:
            break
    return included

# Load data
url = "https://archive.ics.uci.edu/static/public/17/breast+cancer+wisconsin+diagnostic.zip"
response = requests.get(url)
with zipfile.ZipFile(BytesIO(response.content)) as z:
    with z.open("wdbc.data") as f:
        data = pd.read_csv(f, header=None)

# Preprocess
data[1] = data[1] == 'M'  # Convert labels to boolean (Malignant = True)
X = data.iloc[:, 2:]
y = data[1]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=1)

# Stepwise AIC
selected_features = stepwise_aic(X_train, y_train, verbose=False)

# Final model
X_train_sel = sm.add_constant(X_train[selected_features])
X_test_sel = sm.add_constant(X_test[selected_features])
model = sm.Logit(y_train, X_train_sel).fit()
print(model.summary())

# Predict and evaluate
pred_probs = model.predict(X_test_sel)
accuracy = np.mean((pred_probs > 0.5) == y_test)
print(f"Accuracy: {accuracy:.4f}")

# Selected features
print("Selected features:", selected_features)
