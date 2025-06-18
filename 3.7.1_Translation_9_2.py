import pandas as pd
import numpy as np
import urllib.request
import zipfile
import tempfile
import statsmodels.api as sm
from sklearn.model_selection import train_test_split

# --- Step 1: Load and prepare data ---
url = "https://archive.ics.uci.edu/static/public/17/breast+cancer+wisconsin+diagnostic.zip"
temp = tempfile.NamedTemporaryFile(delete=False)
urllib.request.urlretrieve(url, temp.name)

with zipfile.ZipFile(temp.name, 'r') as zip_ref:
    with zip_ref.open("wdbc.data") as file:
        data = pd.read_csv(file, header=None)

data[1] = data[1] == "M"
X = data.drop(columns=[0, 1])
y = data[1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1, stratify=y)

# --- Step 2: Stepwise Selection using AIC ---
def stepwise_selection(X, y, initial_features=[], direction='both', verbose=True):
    included = list(initial_features)
    best_aic = np.inf
    changed = True

    while changed:
        changed = False
        # Forward step
        excluded = list(set(X.columns) - set(included))
        new_aics = {}
        for new_col in excluded:
            try:
                model = sm.Logit(y, sm.add_constant(X[included + [new_col]])).fit(disp=0)
                new_aics[new_col] = model.aic
            except:
                continue
        if new_aics:
            best_new_col = min(new_aics, key=new_aics.get)
            if new_aics[best_new_col] < best_aic:
                included.append(best_new_col)
                best_aic = new_aics[best_new_col]
                changed = True
                if verbose:
                    print(f'Add {best_new_col:>10} | AIC: {best_aic:.2f}')

        # Backward step
        if direction in ['both', 'backward'] and len(included) > 1:
            aics_with_drops = {}
            for col in included:
                try:
                    cols_to_keep = list(set(included) - {col})
                    model = sm.Logit(y, sm.add_constant(X[cols_to_keep])).fit(disp=0)
                    aics_with_drops[col] = model.aic
                except:
                    continue
            if aics_with_drops:
                worst_col = min(aics_with_drops, key=aics_with_drops.get)
                if aics_with_drops[worst_col] < best_aic:
                    included.remove(worst_col)
                    best_aic = aics_with_drops[worst_col]
                    changed = True
                    if verbose:
                        print(f'Drop {worst_col:>10} | AIC: {best_aic:.2f}')
    return included

# Perform stepwise selection
selected_features = stepwise_selection(X_train, y_train)

# --- Step 3: Final model and evaluation ---
X_train_final = sm.add_constant(X_train[selected_features])
X_test_final = sm.add_constant(X_test[selected_features])

final_model = sm.Logit(y_train, X_train_final).fit()
print(final_model.summary())

# Predict and evaluate
test_pred_probs = final_model.predict(X_test_final)
test_pred = test_pred_probs > 0.5
accuracy = (test_pred == y_test).mean()
print("Test Accuracy:", accuracy)
print("Selected features:", selected_features)
