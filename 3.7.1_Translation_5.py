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

# Download and read the data
temp = tempfile.NamedTemporaryFile(delete=False)
urlretrieve(
    "https://archive.ics.uci.edu/static/public/17/breast+cancer+wisconsin+diagnostic.zip",
    temp.name
)

with zipfile.ZipFile(temp.name, 'r') as zip_ref:
    zip_ref.extractall(tempfile.gettempdir())

data = pd.read_csv(os.path.join(tempfile.gettempdir(), "wdbc.data"), header=None)
data[1] = data[1] == "M"  # Convert to boolean: True for malignant

# Train-test split (80-20)
train, test = train_test_split(data, test_size=0.2, random_state=1)

# Remove ID column
train = train.drop(columns=[0])
test = test.drop(columns=[0])

# Base model: intercept only
y_train = train[1].astype(int)
X_train = train.drop(columns=[1])
X_test = test.drop(columns=[1])
y_test = test[1].astype(int)

# Stepwise logistic regression function
def stepwise_selection(X, y, 
                       initial_list=[], 
                       threshold_in=0.01, 
                       threshold_out=0.05, 
                       verbose=True):
    included = list(initial_list)
    while True:
        changed = False
        # forward step
        excluded = list(set(X.columns) - set(included))
        new_pval = pd.Series(index=excluded, dtype=float)
        for new_column in excluded:
            model = sm.Logit(y, sm.add_constant(pd.DataFrame(X[included + [new_column]]))).fit(disp=0)
            new_pval[new_column] = model.pvalues[new_column]
        best_pval = new_pval.min() if not new_pval.empty else None
        if best_pval is not None and best_pval < threshold_in:
            best_feature = new_pval.idxmin()
            included.append(best_feature)
            changed = True
            if verbose:
                print(f'Add  {best_feature:<30} with p-value {best_pval:.6f}')

        # backward step
        if included:
            model = sm.Logit(y, sm.add_constant(pd.DataFrame(X[included]))).fit(disp=0)
            pvalues = model.pvalues.iloc[1:]  # exclude intercept
            worst_pval = pvalues.max()
            if worst_pval > threshold_out:
                worst_feature = pvalues.idxmax()
                included.remove(worst_feature)
                changed = True
                if verbose:
                    print(f'Drop {worst_feature:<30} with p-value {worst_pval:.6f}')
        if not changed:
            break
    return included

# Run stepwise selection
selected_features = stepwise_selection(X_train, y_train, verbose=False)

# Final model
X_train_selected = sm.add_constant(X_train[selected_features])
model = sm.Logit(y_train, X_train_selected).fit()
print(model.summary())

# Prediction
X_test_selected = sm.add_constant(X_test[selected_features])
pred_probs = model.predict(X_test_selected)
pred_labels = (pred_probs > 0.5).astype(int)
accuracy = (pred_labels == y_test).mean()
print(f"\nTest Accuracy: {accuracy:.4f}")

# Print selected feature names
print("\nSelected features in final model:")
print(model.params.index.tolist())
