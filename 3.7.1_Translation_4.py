import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from urllib.request import urlretrieve
import zipfile
import os

# Set seed
np.random.seed(1)

# Download and unzip the dataset
url = "https://archive.ics.uci.edu/static/public/17/breast+cancer+wisconsin+diagnostic.zip"
temp_zip = "temp.zip"
urlretrieve(url, temp_zip)

with zipfile.ZipFile(temp_zip, 'r') as zip_ref:
    zip_ref.extractall("data")

# Load the dataset
data = pd.read_csv("data/wdbc.data", header=None)
data[1] = data[1] == "M"  # Convert diagnosis to boolean

# Split into train and test
train_data, test_data = train_test_split(data, test_size=0.2, random_state=1)

# Prepare data
X_train = train_data.iloc[:, 2:]
y_train = train_data.iloc[:, 1]
X_test = test_data.iloc[:, 2:]
y_test = test_data.iloc[:, 1]

# Add intercept
X_train_const = sm.add_constant(X_train)
X_test_const = sm.add_constant(X_test)

# Stepwise selection (AIC-based)
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
        new_pval = pd.Series(index=excluded)
        for new_column in excluded:
            model = sm.Logit(y, sm.add_constant(X[included + [new_column]])).fit(disp=0)
            new_pval[new_column] = model.pvalues[new_column]
        best_pval = new_pval.min()
        if best_pval < threshold_in:
            best_feature = new_pval.idxmin()
            included.append(best_feature)
            changed = True
            if verbose:
                print(f'Add  {best_feature:30} with p-value {best_pval:.6}')

        # backward step
        model = sm.Logit(y, sm.add_constant(X[included])).fit(disp=0)
        pvalues = model.pvalues.iloc[1:]  # exclude intercept
        worst_pval = pvalues.max()
        if worst_pval > threshold_out:
            worst_feature = pvalues.idxmax()
            included.remove(worst_feature)
            changed = True
            if verbose:
                print(f'Drop {worst_feature:30} with p-value {worst_pval:.6}')
        if not changed:
            break
    return included

selected_features = stepwise_selection(X_train, y_train, verbose=False)

# Fit final model
final_model = sm.Logit(y_train, sm.add_constant(X_train[selected_features])).fit()
print(final_model.summary())

# Predict and evaluate
pred_probs = final_model.predict(sm.add_constant(X_test[selected_features]))
pred_labels = pred_probs > 0.5
accuracy = np.mean(pred_labels == y_test)
print("Accuracy:", accuracy)

# Get feature names used
print("Selected features:", final_model.params.index.tolist())
