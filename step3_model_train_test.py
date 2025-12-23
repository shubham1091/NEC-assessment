import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import random

def load_clean_data():
    demand = pd.read_csv("clean_demand.csv")
    plants = pd.read_csv("clean_plants.csv")
    costs = pd.read_csv("clean_costs.csv")
    return demand, plants, costs

def build_combined_dataset(demand, plants, costs):

    df_demand = costs[["Demand ID"]].merge(demand, on="Demand ID", how="left")
    df_plants = costs[["Plant ID"]].merge(plants, on="Plant ID", how="left")

    combined = pd.concat([df_demand, df_plants.drop(columns=["Plant ID"])], axis=1)

    combined["Cost_USD_per_MWh"] = costs["Cost_USD_per_MWh"]

    return combined

def remove_non_numeric(X):
    num_cols = X.select_dtypes(include=[np.number]).columns
    return X[num_cols].copy()

def split_train_test(combined):

    demand_ids = combined["Demand ID"].unique().tolist()
    random.shuffle(demand_ids)

    test_group = demand_ids[:20]

    train_mask = ~combined["Demand ID"].isin(test_group)
    test_mask = combined["Demand ID"].isin(test_group)

    X = combined.drop(columns=["Cost_USD_per_MWh"])
    y = combined["Cost_USD_per_MWh"]

    X_train = X[train_mask].reset_index(drop=True)
    y_train = y[train_mask].reset_index(drop=True)

    X_test = X[test_mask].reset_index(drop=True)
    y_test = y[test_mask].reset_index(drop=True)

    X_train_num = remove_non_numeric(X_train)
    X_test_num = remove_non_numeric(X_test)

    return X_train_num, X_test_num, y_train, y_test, test_group

def train_model(X_train, y_train):
    model = RandomForestRegressor(n_estimators=200, random_state=0)
    model.fit(X_train, y_train)
    return model

def compute_rmse(model, combined, test_group):

    test_df = combined[combined["Demand ID"].isin(test_group)].copy()

    X_test_num = remove_non_numeric(test_df.drop(columns=["Cost_USD_per_MWh"]))

    test_df["Predicted"] = model.predict(X_test_num)

    grouped = test_df.groupby("Demand ID")

    errors = []

    for demand_id, group in grouped:

        predicted_best_row = group.loc[group["Predicted"].idxmin()]

        actual_cost_selected = predicted_best_row["Cost_USD_per_MWh"]

        actual_best_cost = group["Cost_USD_per_MWh"].min()

        error = actual_best_cost - actual_cost_selected
        errors.append(error)

    return np.array(errors), np.sqrt(np.mean(np.array(errors) ** 2))

if __name__ == "__main__":

    print("Loading cleaned data files")
    demand, plants, costs = load_clean_data()
    print("Demand rows:", len(demand))
    print("Plant rows:", len(plants))
    print("Cost rows:", len(costs))
    print()

    print("Building combined dataset")
    combined = build_combined_dataset(demand, plants, costs)
    print("Combined dataset shape:", combined.shape)
    print("Sample rows from combined dataset")
    print(combined.head())
    print()

    combined.to_csv("combined_step3.csv", index=False)

    print("Splitting into train and test groups")
    X_train, X_test, y_train, y_test, test_group = split_train_test(combined)
    print("Training rows:", len(X_train))
    print("Testing rows:", len(X_test))
    print("Test Demand IDs:", test_group)
    print()

    print("Training Random Forest model")
    model = train_model(X_train, y_train)
    print("Model training completed")
    print()

    print("Evaluating model on test set using Eq.1 and Eq.2")
    errors, rmse = compute_rmse(model, combined, test_group)
    print("Errors for each demand in TestGroup:", errors)
    print("Mean Error:", errors.mean())
    print("RMSE according to Eq.2:", rmse)