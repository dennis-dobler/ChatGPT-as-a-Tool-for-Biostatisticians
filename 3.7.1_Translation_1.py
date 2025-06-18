import pandas as pd
import numpy as np
import tempfile
import zipfile
import requests
from io import BytesIO
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from itertools import combinations

# Stepwise AIC function
def stepwise_selection(X, y, 
                       initial_features=[], 
                       threshold_in=0.01, 
                       threshold_out=0.05, 
                       verbose=True):
    included = list(initial_features)
    while True:
        changed = False
        # forward step
        excluded = list(set(X.columns) - set(included))
        new_pval = pd.Series(index=excluded, dtype=float)
        for new_column in excluded:
            model = sm.Logit(y, sm.add_constant(pd.DataFrame(X[included + [new_column]]))).fit(disp=False)
            new_pval[new_column] = model.pvalues[new_column]
        best_pval = new_pval.min() if not new_pval.empty else None
        if best_pval is not None and best_pval < threshold_in:
            best_feature = new_pval.idxmin()
            included.append(best_feature)
            changed = True
            if verbose:
                print(f'Add {best_feature} with p-value {best_pval:.6}')

        # backward step
        if included:
            model = sm.Logit(y, sm.add_constant(pd.DataFrame(X[included]))).fit(disp=False)
            pvalues = model.pvalues.iloc[1:]  # exclude intercept
            worst_pval = pvalues.max()
            if worst_pval > threshold_out:
                worst_feature = pvalues.idxmax()
                included.remove(worst_feature)
                changed = True
                if verbose:
                    print(f'Drop {worst_feature} with p-value {worst_pval:.6}')
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

# Stepwise selection
selected_features = stepwise_selection(X_train, y_train, verbose=False)

# Final model
X_train_sel = sm.add_constant(X_train[selected_features])
X_test_sel = sm.add_constant(X_test[selected_features])
model = sm.Logit(y_train, X_train_sel).fit()
print(model.summary())

# Predict and evaluate
pred_probs = model.predict(X_test_sel)
accuracy = np.mean((pred_probs > 0.5) == y_test)
print(f"Accuracy: {accuracy:.4f}")

# Coefficients
print("Selected coefficients:", model.params.index.tolist())
