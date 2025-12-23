import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def load_clean_data():
    demand = pd.read_csv("clean_demand.csv")
    plants = pd.read_csv("clean_plants.csv")
    costs = pd.read_csv("clean_costs.csv")
    return demand, plants, costs

def eda_demand_features(demand):
    demand_num = demand.select_dtypes(include=[np.number])

    print("Demand Feature Summary")
    print(demand_num.describe())

    print("Correlation Matrix")
    print(demand_num.corr())

    plt.figure(figsize=(12, 8))
    demand_num.hist(figsize=(12, 8))
    plt.tight_layout()
    plt.show()

def eda_cost_patterns(costs, plants):

    type_map = plants.set_index("Plant ID")["Plant Type"].to_dict()
    region_map = plants.set_index("Plant ID")["Region"].to_dict()

    costs["Plant Type"] = costs["Plant ID"].map(type_map)
    costs["Region"] = costs["Plant ID"].map(region_map)

    avg_by_type = costs.groupby("Plant Type")["Cost_USD_per_MWh"].mean()
    avg_by_region = costs.groupby("Region")["Cost_USD_per_MWh"].mean()

    print("Average Cost by Plant Type")
    print(avg_by_type)

    print("Average Cost by Region")
    print(avg_by_region)

    plt.figure(figsize=(8, 4))
    avg_by_type.plot(kind="bar")
    plt.title("Average Cost by Plant Type")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(8, 4))
    avg_by_region.plot(kind="bar")
    plt.title("Average Cost by Region")
    plt.tight_layout()
    plt.show()

def compute_baseline_rmse(costs, plants):

    costs = costs[costs["Plant ID"].isin(plants["Plant ID"])]

    best_cost = costs.groupby("Demand ID")["Cost_USD_per_MWh"].min()

    rmse_scores = {}

    grouped = costs.groupby("Plant ID")

    for plant, group in grouped:

        plant_costs = group.set_index("Demand ID")["Cost_USD_per_MWh"]

        aligned_best, aligned_plant = best_cost.align(plant_costs, join="inner")

        diff = aligned_best - aligned_plant

        rmse = np.sqrt(np.mean(diff.values ** 2))
        rmse_scores[plant] = rmse

    rmse_df = pd.DataFrame.from_dict(rmse_scores, orient="index", columns=["RMSE"])

    print("Baseline RMSE for each plant")
    print(rmse_df)

    plt.figure(figsize=(14, 6))
    rmse_df.sort_values("RMSE").plot(kind="bar")
    plt.title("RMSE for Always Selecting Each Plant")
    plt.tight_layout()
    plt.show()

    return rmse_df

if __name__ == "__main__":
    demand, plants, costs = load_clean_data()

    eda_demand_features(demand)

    eda_cost_patterns(costs, plants)

    rmse_df = compute_baseline_rmse(costs, plants)

    print("EDA completed.")
    print("Lowest RMSE plant:", rmse_df['RMSE'].idxmin(), rmse_df['RMSE'].min())
    print("Highest RMSE plant:", rmse_df['RMSE'].idxmax(), rmse_df['RMSE'].max())