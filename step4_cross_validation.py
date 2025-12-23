import pandas as pd
import numpy as np
from sklearn.model_selection import LeaveOneGroupOut, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import make_scorer

def load_combined():
    return pd.read_csv("combined_step3.csv")

def remove_non_numeric(X):
    return X.select_dtypes(include=[np.number]).copy()

def error_function(true_costs, predicted_costs, groups):

    df = pd.DataFrame({
        "Demand ID": groups,
        "True": true_costs,
        "Pred": predicted_costs
    })

    errors = []

    for demand_id, group in df.groupby("Demand ID"):
        pred_best_row = group.loc[group["Pred"].idxmin()]
        actual_selected_cost = pred_best_row["True"]
        actual_best_cost = group["True"].min()
        errors.append(actual_best_cost - actual_selected_cost)

    errors = np.array(errors)
    return np.sqrt(np.mean(errors ** 2))

def make_custom_scorer(groups_series):

    def scorer(y_true, y_pred):

        idx = y_true.index

        return -error_function(
            y_true.values,
            y_pred,
            groups_series.iloc[idx].values
        )

    return make_scorer(scorer, greater_is_better=False)

if __name__ == "__main__":

    combined = load_combined()

    X = combined.drop(columns=["Cost_USD_per_MWh"])
    y = combined["Cost_USD_per_MWh"]
    groups = combined["Demand ID"]

    X_num = remove_non_numeric(X)

    model = RandomForestRegressor(
        n_estimators=80,
        max_depth=12,
        n_jobs=-1,
        random_state=0
    )

    custom_scorer = make_custom_scorer(groups)

    logo = LeaveOneGroupOut()
    all_folds = list(logo.split(X_num, y, groups))

    selected_folds = all_folds[:50]

    print("Running Leave-One-Group-Out Cross Validation")
    print("Total folds available:", len(all_folds))
    print("Evaluating only first 50 folds to reduce runtime")
    print()

    scores = cross_val_score(
        model,
        X_num,
        y,
        cv=selected_folds,
        scoring=custom_scorer,
        n_jobs=-1
    )

    scores = -scores

    print("Number of folds evaluated:", len(scores))
    print("RMSE score per evaluated fold:", scores)
    print("Mean RMSE:", np.mean(scores))