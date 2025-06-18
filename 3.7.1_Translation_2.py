import pandas as pd
import numpy as np
import statsmodels.api as sm
import requests, zipfile, io
from sklearn.model_selection import train_test_split
from itertools import combinations

# Set seed for reproducibility
np.random.seed(1)

# Download and extract data
url = "https://archive.ics.uci.edu/static/public/17/breast+cancer+wisconsin+diagnostic.zip"
r = requests.get(url)
z = zipfile.ZipFile(io.BytesIO(r.content))
data = pd.read_csv(z.open("wdbc.data"), header=None)

# Convert diagnosis to boolean (M = True, B = False)
data[1] = data[1] == 'M'

# Split into training and test sets (80/20)
train_df, test_df = train_test_split(data, test_size=0.2, random_state=1)

# Drop ID column
train = train_df.drop(columns=0)
test = test_df.drop(columns=0)

# Response and predictors
y_train = train[1].astype(int)
X_train_full = train.drop(columns=1)
X_test = test.drop(columns=1)
y_test = test[1].astype(int)

# Add intercept to design matrix
X_train_full_const = sm.add_constant(X_train_full)

# Stepwise AIC implementation
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
        for new_col in excluded:
            model = sm.Logit(y, sm.add_constant(pd.DataFrame(X[included + [new_col]]))).fit(disp=False)
            new_pval[new_col] = model.pvalues[new_col]
        best_pval = new_pval.min()
        if best_pval < threshold_in:
            best_feature = new_pval.idxmin()
            included.append(best_feature)
            changed = True
            if verbose:
                print(f'Add {best_feature} with p-value {best_pval:.6f}')
        
        # backward step
        model = sm.Logit(y, sm.add_constant(pd.DataFrame(X[included]))).fit(disp=False)
        pvalues = model.pvalues.iloc[1:]  # exclude intercept
        worst_pval = pvalues.max() 
        if worst_pval > threshold_out:
            worst_feature = pvalues.idxmax()
            included.remove(worst_feature)
            changed = True
            if verbose:
                print(f'Drop {worst_feature} with p-value {worst_pval:.6f}')
        
        if not changed:
            break
    return included

# Perform stepwise selection
selected_features = stepwise_selection(X_train_full, y_train, verbose=False)

# Final model
X_train_selected = sm.add_constant(X_train_full[selected_features])
final_model = sm.Logit(y_train, X_train_selected).fit()
print(final_model.summary())

# Predict on test data
X_test_selected = sm.add_constant(X_test[selected_features])
test_pred = final_model.predict(X_test_selected)

# Accuracy
accuracy = ((test_pred > 0.5) == y_test).mean()
print(f"\nTest Accuracy: {accuracy:.4f}")

# Coefficients
print("\nSelected coefficients:", final_model.params.index.tolist())
