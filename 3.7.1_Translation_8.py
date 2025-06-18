import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from urllib.request import urlopen
import zipfile
import io

# Download and load data
url = "https://archive.ics.uci.edu/static/public/17/breast+cancer+wisconsin+diagnostic.zip"
with urlopen(url) as zipresp:
    with zipfile.ZipFile(io.BytesIO(zipresp.read())) as zfile:
        with zfile.open("wdbc.data") as f:
            data = pd.read_csv(f, header=None)

# Recode diagnosis column to binary
data[1] = data[1] == 'M'

# Train-test split
train_data, test_data = train_test_split(data, test_size=0.2, random_state=1)

# Define predictors and outcome
X_train = train_data.iloc[:, 2:]
y_train = train_data[1].astype(int)
X_test = test_data.iloc[:, 2:]
y_test = test_data[1].astype(int)

# Start with intercept-only model
X_base = sm.add_constant(pd.DataFrame(index=X_train.index))
base_model = sm.Logit(y_train, X_base).fit(disp=0)

# Stepwise selection
def stepwise_selection(X, y, 
                       initial_features=[], 
                       threshold_in=0.01, 
                       threshold_out=0.05, 
                       verbose=True):
    included = list(initial_features)
    while True:
        changed = False
        # Forward step
        excluded = list(set(X.columns) - set(included))
        new_pval = pd.Series(index=excluded, dtype=float)
        for new_column in excluded:
            model = sm.Logit(y, sm.add_constant(pd.DataFrame(X[included + [new_column]]))).fit(disp=0)
            new_pval[new_column] = model.pvalues[new_column]
        best_pval = new_pval.min()
        if best_pval < threshold_in:
            best_feature = new_pval.idxmin()
            included.append(best_feature)
            changed = True
            if verbose:
                print(f"Add {best_feature} (p={best_pval:.6})")
        # Backward step
        model = sm.Logit(y, sm.add_constant(pd.DataFrame(X[included]))).fit(disp=0)
        pvalues = model.pvalues.iloc[1:]  # exclude intercept
        worst_pval = pvalues.max()
        if worst_pval > threshold_out:
            worst_feature = pvalues.idxmax()
            included.remove(worst_feature)
            changed = True
            if verbose:
                print(f"Drop {worst_feature} (p={worst_pval:.6})")
        if not changed:
            break
    return included

selected_features = stepwise_selection(X_train, y_train)

# Final model
X_train_selected = sm.add_constant(X_train[selected_features])
final_model = sm.Logit(y_train, X_train_selected).fit()
print(final_model.summary())

# Predictions
X_test_selected = sm.add_constant(X_test[selected_features])
test_pred_probs = final_model.predict(X_test_selected)
test_pred = (test_pred_probs > 0.5).astype(int)

# Accuracy
accuracy = (test_pred == y_test).mean()
print("Test accuracy:", accuracy)

# Coefficient names
print("Selected features:", final_model.params.index.tolist())
