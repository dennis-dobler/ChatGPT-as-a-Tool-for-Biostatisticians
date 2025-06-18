import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
import requests, zipfile, io

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

# Combine X and y for modeling
train = pd.concat([y_train.reset_index(drop=True), X_train.reset_index(drop=True)], axis=1)
train.columns = ['V2'] + [f'V{i}' for i in range(3, 33)]
test = pd.concat([y_test.reset_index(drop=True), X_test.reset_index(drop=True)], axis=1)
test.columns = ['V2'] + [f'V{i}' for i in range(3, 33)]

# Stepwise logistic regression (AIC-based)
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
        new_pvals = pd.Series(index=excluded, dtype=float)
        for new_col in excluded:
            model = sm.Logit(y, sm.add_constant(pd.DataFrame(X[included + [new_col]]))).fit(disp=0)
            new_pvals[new_col] = model.pvalues[new_col]
        best_pval = new_pvals.min()
        if best_pval < threshold_in:
            best_feature = new_pvals.idxmin()
            included.append(best_feature)
            changed = True
            if verbose:
                print(f"Add  {best_feature:<15} with p-value {best_pval:.6}")

        # backward step
        model = sm.Logit(y, sm.add_constant(pd.DataFrame(X[included]))).fit(disp=0)
        pvalues = model.pvalues.iloc[1:]  # exclude intercept
        worst_pval = pvalues.max()  # null if pvalues is empty
        if worst_pval > threshold_out:
            worst_feature = pvalues.idxmax()
            included.remove(worst_feature)
            changed = True
            if verbose:
                print(f"Drop {worst_feature:<15} with p-value {worst_pval:.6}")
        if not changed:
            break
    return included

# Run stepwise selection
selected_features = stepwise_selection(train.iloc[:, 1:], train['V2'])

# Fit final model
X_train_sel = sm.add_constant(train[selected_features])
model = sm.Logit(train['V2'], X_train_sel).fit()
print(model.summary())

# Predict on test set
X_test_sel = sm.add_constant(test[selected_features])
test_pred = model.predict(X_test_sel)
accuracy = np.mean((test_pred > 0.5) == test['V2'])
print("Test accuracy:", accuracy)

# Selected variable names
print("Selected variables:", model.params.index.tolist())
