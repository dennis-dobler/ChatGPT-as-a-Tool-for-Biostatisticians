import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from urllib.request import urlopen
import zipfile
import io

# Download and extract the data
url = "https://archive.ics.uci.edu/static/public/17/breast+cancer+wisconsin+diagnostic.zip"
with urlopen(url) as zipresp:
    with zipfile.ZipFile(io.BytesIO(zipresp.read())) as zfile:
        with zfile.open("wdbc.data") as f:
            data = pd.read_csv(f, header=None)

# Recode diagnosis column (column 1): M = 1, B = 0
data[1] = data[1] == 'M'

# Train-test split
train_data, test_data = train_test_split(data, test_size=0.2, random_state=1)

# Define predictors and outcome
X_train = train_data.iloc[:, 2:]
y_train = train_data[1].astype(int)
X_test = test_data.iloc[:, 2:]
y_test = test_data[1].astype(int)

# Stepwise AIC-based selection
def stepwise_selection_aic(X, y, verbose=True):
    included = []
    best_aic = np.inf
    while True:
        changed = False
        # Try adding features
        excluded = list(set(X.columns) - set(included))
        add_aic = pd.Series(index=excluded, dtype=float)
        for new_col in excluded:
            try:
                model = sm.Logit(y, sm.add_constant(X[included + [new_col]])).fit(disp=0)
                add_aic[new_col] = model.aic
            except:
                continue
        if not add_aic.empty and add_aic.min() < best_aic:
            best_feature = add_aic.idxmin()
            best_aic = add_aic.min()
            included.append(best_feature)
            changed = True
            if verbose:
                print(f"Add {best_feature}, AIC: {best_aic:.2f}")

        # Try removing features
        if len(included) > 1:
            drop_aic = pd.Series(index=included, dtype=float)
            for col in included:
                trial_features = [f for f in included if f != col]
                try:
                    model = sm.Logit(y, sm.add_constant(X[trial_features])).fit(disp=0)
                    drop_aic[col] = model.aic
                except:
                    continue
            if not drop_aic.empty and drop_aic.min() < best_aic:
                worst_feature = drop_aic.idxmin()
                best_aic = drop_aic.min()
                included.remove(worst_feature)
                changed = True
                if verbose:
                    print(f"Drop {worst_feature}, AIC: {best_aic:.2f}")
        if not changed:
            break
    return included

# Run stepwise AIC selection
selected_features = stepwise_selection_aic(X_train, y_train)

# Fit final model
X_train_selected = sm.add_constant(X_train[selected_features])
final_model = sm.Logit(y_train, X_train_selected).fit()
print(final_model.summary())

# Predict on test data
X_test_selected = sm.add_constant(X_test[selected_features])
test_pred_probs = final_model.predict(X_test_selected)
test_pred = (test_pred_probs > 0.5).astype(int)

# Accuracy
accuracy = (test_pred == y_test).mean()
print("\nTest accuracy:", round(accuracy, 4))

# Final selected features
print("Selected features:", final_model.params.index.tolist())
