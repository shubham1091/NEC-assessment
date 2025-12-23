import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def load_data():
    demand = pd.read_csv("demand.csv")
    plants = pd.read_csv("plants.csv")
    costs = pd.read_csv("generation_costs.csv")
    return demand, plants, costs

def handle_missing(demand, plants, costs):
    demand = demand.dropna()
    plants = plants.dropna()
    costs = costs.dropna()
    return demand, plants, costs

def scale_features(demand, plants):
    demand_num = demand.select_dtypes(include=[np.number])
    plants_num = plants.select_dtypes(include=[np.number])

    scaler_demand = StandardScaler()
    scaler_plants = StandardScaler()

    demand_scaled = demand.copy()
    plants_scaled = plants.copy()

    demand_scaled[demand_num.columns] = scaler_demand.fit_transform(demand_num)
    plants_scaled[plants_num.columns] = scaler_plants.fit_transform(plants_num)

    return demand_scaled, plants_scaled

def remove_poor_plants(costs, percentile=0.8):
    avg_cost = costs.groupby("Plant ID")["Cost_USD_per_MWh"].mean()
    cutoff = avg_cost.quantile(percentile)
    good_plants = avg_cost[avg_cost <= cutoff].index.tolist()
    costs_filtered = costs[costs["Plant ID"].isin(good_plants)].copy()
    return costs_filtered, good_plants

def save_clean_data(demand, plants, costs):
    demand.to_csv("clean_demand.csv", index=False)
    plants.to_csv("clean_plants.csv", index=False)
    costs.to_csv("clean_costs.csv", index=False)

if __name__ == "__main__":
    demand, plants, costs = load_data()

    demand, plants, costs = handle_missing(demand, plants, costs)

    demand, plants = scale_features(demand, plants)

    costs, good_plants = remove_poor_plants(costs)

    save_clean_data(demand, plants, costs)

    print("Data preparation completed.")
    print("Cleaned demand rows:", len(demand))
    print("Cleaned plant rows:", len(plants))
    print("Remaining plants after filtering:", len(good_plants))
    print("Clean files saved as clean_demand.csv, clean_plants.csv, clean_costs.csv.")