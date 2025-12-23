import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.model_selection import LeaveOneGroupOut
from joblib import Parallel, delayed
from itertools import product

def load_combined():
    return pd.read_csv("combined_step3.csv")

def remove_non_numeric(X):
    return X.select_dtypes(include=[np.number]).copy()

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

    errors = np.array(errors)
    rmse = np.sqrt(np.mean(errors ** 2))
    return rmse

def step3_train_test_score(model, combined):

    X = combined.drop(columns=["Cost_USD_per_MWh"])
    y = combined["Cost_USD_per_MWh"]
    groups = combined["Demand ID"]

    X_num = remove_non_numeric(X)

    unique_demands = groups.unique().tolist()
    np.random.shuffle(unique_demands)
    test_group = unique_demands[:20]

    train_mask = ~groups.isin(test_group)
    test_mask = groups.isin(test_group)

    X_train = X_num[train_mask]
    y_train = y[train_mask]
    X_test = X_num[test_mask]
    y_test = y[test_mask]
    g_test = groups[test_mask]

    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    rmse = error_rmse(y_test.values, preds, g_test.values)

    return rmse

def step4_logo_cv(model, combined, max_folds=50):

    X = combined.drop(columns=["Cost_USD_per_MWh"])
    y = combined["Cost_USD_per_MWh"]
    groups = combined["Demand ID"]
    X = remove_non_numeric(X)

    logo = LeaveOneGroupOut()
    all_folds = list(logo.split(X, y, groups))

    folds = all_folds[:max_folds]

    def process_fold(train_idx, test_idx):
        X_train = X.iloc[train_idx]
        y_train = y.iloc[train_idx]
        X_test = X.iloc[test_idx]
        y_test = y.iloc[test_idx]
        g_test = groups.iloc[test_idx]

        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        return error_rmse(y_test.values, preds, g_test.values)

    scores = Parallel(n_jobs=-1)(
        delayed(process_fold)(train_idx, test_idx) for train_idx, test_idx in folds
    )

    return np.mean(scores)

def step5_hyperparameter_search(model_class, param_grid, combined, max_folds=50):

    X = combined.drop(columns=["Cost_USD_per_MWh"])
    y = combined["Cost_USD_per_MWh"]
    groups = combined["Demand ID"]
    X = remove_non_numeric(X)

    logo = LeaveOneGroupOut()
    all_folds = list(logo.split(X, y, groups))
    folds = all_folds[:max_folds]

    all_params = list(product(
        param_grid["n_estimators"],
        param_grid["max_depth"],
        param_grid["min_samples_split"]
    ))

    best_rmse = float("inf")
    best_params = None

    def process_fold(model, train_idx, test_idx):
        X_train = X.iloc[train_idx]
        y_train = y.iloc[train_idx]
        X_test = X.iloc[test_idx]
        y_test = y.iloc[test_idx]
        g_test = groups.iloc[test_idx]

        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        return error_rmse(y_test.values, preds, g_test.values)

    for n_est, depth, split in all_params:

        params = {
            "n_estimators": n_est,
            "max_depth": depth,
            "min_samples_split": split,
            "n_jobs": -1,
            "random_state": 0
        }

        model = model_class(**params)

        rmse_scores = Parallel(n_jobs=-1)(
            delayed(process_fold)(model, train_idx, test_idx)
            for train_idx, test_idx in folds
        )

        rmse = np.mean(rmse_scores)
        print("Testing:", params, "RMSE:", rmse)

        if rmse < best_rmse:
            best_rmse = rmse
            best_params = params

    return best_params, best_rmse

if __name__ == "__main__":
    combined = load_combined()

    model_rf = RandomForestRegressor(n_estimators=100, n_jobs=-1, random_state=0)
    model_xt = ExtraTreesRegressor(n_estimators=100, n_jobs=-1, random_state=0)

    print("Running Step 3 for both models")
    rf_step3 = step3_train_test_score(model_rf, combined)
    xt_step3 = step3_train_test_score(model_xt, combined)

    print("RF Step 3 RMSE:", rf_step3)
    print("XT Step 3 RMSE:", xt_step3)

    print("\nRunning Step 4 LOGO CV (first 50 folds)")
    rf_step4 = step4_logo_cv(model_rf, combined, max_folds=50)
    xt_step4 = step4_logo_cv(model_xt, combined, max_folds=50)

    print("RF LOGO RMSE:", rf_step4)
    print("XT LOGO RMSE:", xt_step4)

    print("\nRunning Step 5 Grid Search for both models (first 50 folds)")

    param_grid = {
        "n_estimators": [50, 80],
        "max_depth": [8, None],
        "min_samples_split": [2, 5]
    }

    best_rf_params, best_rf_rmse = step5_hyperparameter_search(
        RandomForestRegressor, param_grid, combined, max_folds=50
    )
    best_xt_params, best_xt_rmse = step5_hyperparameter_search(
        ExtraTreesRegressor, param_grid, combined, max_folds=50
    )

    print("\nBest RF Params:", best_rf_params)
    print("Best RF RMSE:", best_rf_rmse)

    print("\nBest XT Params:", best_xt_params)
    print("Best XT RMSE:", best_xt_rmse)