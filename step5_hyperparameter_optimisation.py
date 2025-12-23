import pandas as pd
import numpy as np
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.ensemble import ExtraTreesRegressor
from joblib import Parallel, delayed
from itertools import product

def load_combined():
    return pd.read_csv("combined_step3.csv")

def remove_non_numeric(X):
    return X.select_dtypes(include=[np.number]).copy()

# Eq.1 and Eq.2 combined RMSE
def error_rmse(true_costs, preds, groups):

    df = pd.DataFrame({
        "Demand ID": groups,
        "True": true_costs,
        "Pred": preds
    })

    errors = []

    for demand_id, group in df.groupby("Demand ID"):

        pred_best = group.loc[group["Pred"].idxmin()]
        actual_selected_cost = pred_best["True"]
        actual_best_cost = group["True"].min()

        error = actual_best_cost - actual_selected_cost
        errors.append(error)

    return np.sqrt(np.mean(np.array(errors) ** 2))

def evaluate_hparams(X, y, groups, hparams, folds_subset):

    def evaluate_fold(train_idx, test_idx):

        model = ExtraTreesRegressor(
            n_estimators=hparams["n_estimators"],
            max_depth=hparams["max_depth"],
            min_samples_split=hparams["min_samples_split"],
            n_jobs=-1,
            random_state=0
        )

        X_train = X.iloc[train_idx]
        y_train = y.iloc[train_idx]
        X_test = X.iloc[test_idx]
        y_test = y.iloc[test_idx]
        g_test = groups.iloc[test_idx]

        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        return error_rmse(y_test.values, preds, g_test.values)

    scores = Parallel(n_jobs=-1)(
        delayed(evaluate_fold)(train, test)
        for train, test in folds_subset
    )

    return np.mean(scores)

def run_grid_search(combined):

    X = remove_non_numeric(combined.drop(columns=["Cost_USD_per_MWh"]))
    y = combined["Cost_USD_per_MWh"]
    groups = combined["Demand ID"]

    logo = LeaveOneGroupOut()
    all_folds = list(logo.split(X, y, groups))

    folds_subset = all_folds[:50]

    param_grid = {
        "n_estimators": [40, 80],
        "max_depth": [8, None],
        "min_samples_split": [2, 4]
    }

    all_combinations = list(product(
        param_grid["n_estimators"],
        param_grid["max_depth"],
        param_grid["min_samples_split"]
    ))

    best_rmse = float("inf")
    best_params = None

    for n_est, depth, split in all_combinations:

        hparams = {
            "n_estimators": n_est,
            "max_depth": depth,
            "min_samples_split": split
        }

        print("Testing:", hparams)

        rmse = evaluate_hparams(X, y, groups, hparams, folds_subset)

        print("RMSE:", rmse)
        print()

        if rmse < best_rmse:
            best_rmse = rmse
            best_params = hparams

    return best_params, best_rmse

if __name__ == "__main__":

    combined = load_combined()

    best_params, best_rmse = run_grid_search(combined)

    print("Grid Search Completed")
    print("Best Parameters:", best_params)
    print("Best RMSE:", best_rmse)