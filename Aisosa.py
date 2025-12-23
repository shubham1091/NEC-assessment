# ===============================
# STEP 0: SETUP AND DATA IMPORT
# ===============================

# 1. Import core libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, LeaveOneGroupOut, GridSearchCV
from sklearn.metrics import mean_squared_error, make_scorer
from sklearn.ensemble import RandomForestRegressor

import warnings
warnings.filterwarnings('ignore')

# 2. Display settings for better readability
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 150)

# 3. Load the datasets
demand = pd.read_csv('demand.csv')
plants = pd.read_csv('plants.csv')
generation_costs = pd.read_csv('generation_costs.csv')

# 4. Display basic info
print("✅ Files loaded successfully!\n")
print("Demand data shape:", demand.shape)
print("Plants data shape:", plants.shape)
print("Generation costs shape:", generation_costs.shape)

# Preview first few rows
print("\n--- Demand Data ---")
display(demand.head())

print("\n--- Plants Data ---")
display(plants.head())

print("\n--- Generation Costs Data ---")
display(generation_costs.head())

# ===============================
# STEP 1: DATA PREPARATION
# ===============================

# 1.1 HANDLE MISSING DATA ------------------------------------
print("🔹 Missing values before cleaning:\n")
print(plants.isnull().sum())

# Fill missing categorical values with 'Unknown'
plants['Region'].fillna('Unknown', inplace=True)

print("\n✅ Missing values handled.")
print(plants['Region'].value_counts())


# 1.2 FEATURE ENCODING ---------------------------------------
# Encode categorical columns in demand and plants
demand_encoded = pd.get_dummies(demand, columns=['DF_region', 'DF_daytype'], drop_first=True)
plants_encoded = pd.get_dummies(plants, columns=['Plant Type', 'Region'], drop_first=True)

print("\n✅ Categorical variables encoded.")
print("Demand features:", demand_encoded.shape[1] - 1)
print("Plant features:", plants_encoded.shape[1] - 1)


# 1.3 FEATURE SCALING ----------------------------------------
# Identify numeric feature columns (excluding IDs)
demand_num_cols = [c for c in demand_encoded.columns if c.startswith('DF') and not c.startswith('DF_region') and not c.startswith('DF_daytype')]
plant_num_cols = [c for c in plants_encoded.columns if c.startswith('PF')]

# Scale only numeric features
scaler = StandardScaler()
demand_encoded[demand_num_cols] = scaler.fit_transform(demand_encoded[demand_num_cols])
plants_encoded[plant_num_cols] = scaler.fit_transform(plants_encoded[plant_num_cols])

print("\n✅ Feature scaling complete.")
print("Scaled demand sample:")
display(demand_encoded.head())


# 1.4 REMOVE WORST-PERFORMING PLANTS -------------------------
# Calculate mean generation cost per plant
plant_mean_cost = generation_costs.groupby('Plant ID')['Cost_USD_per_MWh'].mean().sort_values()

# Remove bottom 25% of plants (worst performing)
threshold = int(0.75 * len(plant_mean_cost))
top_plants = plant_mean_cost.index[:threshold]
generation_costs_filtered = generation_costs[generation_costs['Plant ID'].isin(top_plants)]
plants_filtered = plants_encoded[plants_encoded['Plant ID'].isin(top_plants)]

print("\n✅ Removed worst-performing 25% of plants.")
print(f"Remaining plants: {plants_filtered.shape[0]} / {plants.shape[0]}")

# Sanity check
print("\nTop 5 cheapest plants:")
display(plant_mean_cost.head())
print("\nBottom 5 most expensive plants:")
display(plant_mean_cost.tail())

# ===============================
# STEP 2: EXPLORATORY DATA ANALYSIS (EDA)
# ===============================

import seaborn as sns

# 2.1 DISTRIBUTION OF DEMAND FEATURES -----------------------------
print("🔹 Summary statistics of demand features:\n")
display(demand_encoded.describe().T)

# Plot histograms for numerical demand features
demand_encoded[demand_num_cols].hist(figsize=(12, 8), bins=20)
plt.suptitle("Distribution of Demand Numerical Features", fontsize=14)
plt.show()


# 2.2 ANALYSIS OF COST PATTERNS -----------------------------------
# Merge generation_costs with demand and plant metadata for analysis
merged = generation_costs_filtered.merge(demand, on='Demand ID', how='left')
merged = merged.merge(plants[['Plant ID', 'Plant Type', 'Region']], on='Plant ID', how='left')

# Average cost by Plant Type and Region
avg_cost_by_type = merged.groupby('Plant Type')['Cost_USD_per_MWh'].mean().sort_values()
avg_cost_by_region = merged.groupby('Region')['Cost_USD_per_MWh'].mean().sort_values()

print("\n🔹 Average generation cost by Plant Type:")
display(avg_cost_by_type)

print("\n🔹 Average generation cost by Region:")
display(avg_cost_by_region)

# Visualize
plt.figure(figsize=(8,4))
avg_cost_by_type.plot(kind='bar')
plt.title("Average Cost by Plant Type")
plt.ylabel("USD per MWh")
plt.show()

plt.figure(figsize=(8,4))
avg_cost_by_region.plot(kind='bar', color='gray')
plt.title("Average Cost by Region")
plt.ylabel("USD per MWh")
plt.show()


# 2.3 BASELINE ERROR AND RMSE PER PLANT ----------------------------
# For each demand, identify the best (lowest cost) plant
best_cost_per_demand = generation_costs_filtered.groupby('Demand ID')['Cost_USD_per_MWh'].min().reset_index()
best_cost_per_demand.rename(columns={'Cost_USD_per_MWh': 'Best_Cost'}, inplace=True)

# Merge to compute error for each plant
costs_with_error = generation_costs_filtered.merge(best_cost_per_demand, on='Demand ID', how='left')
costs_with_error['Error'] = costs_with_error['Cost_USD_per_MWh'] - costs_with_error['Best_Cost']

# Compute RMSE for each plant
rmse_per_plant = costs_with_error.groupby('Plant ID')['Error'].apply(lambda x: np.sqrt(np.mean(x**2))).sort_values()

print("\n🔹 Baseline RMSE (cost error) per plant:")
display(rmse_per_plant.head(10))

plt.figure(figsize=(10,4))
rmse_per_plant.plot(kind='bar')
plt.title("RMSE (Cost Error) per Plant - Baseline")
plt.ylabel("RMSE (USD/MWh)")
plt.show()

# Average RMSE across all plants
baseline_rmse = np.sqrt(np.mean(costs_with_error['Error']**2))
print(f"\n✅ Overall baseline RMSE (no ML model): {baseline_rmse:.3f} USD/MWh")

# ===============================
# FIX MISSING TARGET VALUES
# ===============================

# Check if there are any missing cost values
missing_costs = generation_costs_filtered['Cost_USD_per_MWh'].isna().sum()
print(f"🔍 Missing Cost Values: {missing_costs}")

# Drop any rows with missing target values
generation_costs_filtered = generation_costs_filtered.dropna(subset=['Cost_USD_per_MWh'])

# Recreate merged_full and y
merged_full = generation_costs_filtered.merge(demand_encoded, on='Demand ID', how='left')
merged_full = merged_full.merge(plants_encoded, on='Plant ID', how='left')

X = merged_full.drop(columns=['Cost_USD_per_MWh', 'Demand ID', 'Plant ID'])
y = merged_full['Cost_USD_per_MWh']
groups = merged_full['Demand ID']

print(f"✅ Clean dataset shape after removing NaNs: {X.shape}")
print(f"✅ Remaining rows: {len(y)} (no missing targets)")
# ===============================
# FIX MISSING TARGET VALUES
# ===============================

# Check if there are any missing cost values
missing_costs = generation_costs_filtered['Cost_USD_per_MWh'].isna().sum()
print(f"🔍 Missing Cost Values: {missing_costs}")

# Drop any rows with missing target values
generation_costs_filtered = generation_costs_filtered.dropna(subset=['Cost_USD_per_MWh'])

# Recreate merged_full and y
merged_full = generation_costs_filtered.merge(demand_encoded, on='Demand ID', how='left')
merged_full = merged_full.merge(plants_encoded, on='Plant ID', how='left')

X = merged_full.drop(columns=['Cost_USD_per_MWh', 'Demand ID', 'Plant ID'])
y = merged_full['Cost_USD_per_MWh']
groups = merged_full['Demand ID']

print(f"✅ Clean dataset shape after removing NaNs: {X.shape}")
print(f"✅ Remaining rows: {len(y)} (no missing targets)")

# ===============================
# STEP 3: MODEL FITTING (TRAIN/TEST SPLIT)
# ===============================

# 3.1 COMBINE DEMAND + PLANT + COST DATA ----------------------------
# Merge cost data with demand and plant features
demand_features = demand_encoded.rename(columns={'Demand ID': 'Demand ID'})
plant_features = plants_encoded.rename(columns={'Plant ID': 'Plant ID'})

# Merge generation_costs_filtered with demand and plant
merged_full = generation_costs_filtered.merge(demand_features, on='Demand ID', how='left')
merged_full = merged_full.merge(plant_features, on='Plant ID', how='left')

# Prepare input features (X), target (y), and groups
X = merged_full.drop(columns=['Cost_USD_per_MWh', 'Demand ID', 'Plant ID'])
y = merged_full['Cost_USD_per_MWh']
groups = merged_full['Demand ID']

print("✅ Combined dataset created.")
print("Shape of X:", X.shape)


# 3.2 TRAIN/TEST SPLIT (GROUPED BY DEMAND ID) ------------------------
unique_demands = groups.unique()
test_demands = np.random.choice(unique_demands, size=20, replace=False)

test_mask = groups.isin(test_demands)
X_train, X_test = X[~test_mask], X[test_mask]
y_train, y_test = y[~test_mask], y[test_mask]
groups_train, groups_test = groups[~test_mask], groups[test_mask]

print(f"\n✅ Train/Test split complete.")
print(f"Training samples: {X_train.shape[0]} | Test samples: {X_test.shape[0]}")
print(f"Unique demands in test set: {len(test_demands)}")


# 3.3 TRAIN RANDOM FOREST MODEL -------------------------------------
rf = RandomForestRegressor(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)
rf.fit(X_train, y_train)

# Evaluate model performance on test data
y_pred = rf.predict(X_test)
model_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"\n🔹 Random Forest baseline RMSE (predicted vs actual cost): {model_rmse:.3f} USD/MWh")


# 3.4 CALCULATE ERROR(d) AND FINAL RMSE ------------------------------
# For each demand in test set, find predicted cost per plant
preds_df = X_test.copy()
preds_df['Demand ID'] = groups_test.values
preds_df['Plant ID'] = merged_full.loc[test_mask, 'Plant ID'].values
preds_df['Actual_Cost'] = y_test.values
preds_df['Predicted_Cost'] = y_pred

# For each demand, find the plant with minimum predicted cost
chosen_plants = preds_df.loc[preds_df.groupby('Demand ID')['Predicted_Cost'].idxmin()]

# Compute actual cost of selected plant and true best plant
best_actual = preds_df.groupby('Demand ID')['Actual_Cost'].min().reset_index()
best_actual.rename(columns={'Actual_Cost': 'Best_Cost'}, inplace=True)

comparison = chosen_plants.merge(best_actual, on='Demand ID')
comparison['Error'] = comparison['Actual_Cost'] - comparison['Best_Cost']

# Compute RMSE according to Eq. (2)
model_selection_rmse = np.sqrt(np.mean(comparison['Error']**2))

print(f"\n✅ Model selection RMSE (Eq. 1 & 2): {model_selection_rmse:.3f} USD/MWh")

 # ===============================
# STEP 4: FAST LEAVE-ONE-GROUP-OUT CROSS VALIDATION (FINAL VERSION)
# ===============================

from sklearn.base import clone
from sklearn.model_selection import LeaveOneGroupOut
import numpy as np, time

start = time.time()

# 1️⃣ Select 80 random demand IDs for speed
np.random.seed(42)
subset_ids = np.random.choice(groups.unique(), size=80, replace=False)
mask = groups.isin(subset_ids)

X_sub, y_sub, g_sub = X[mask], y[mask], groups[mask]

print(f"🔄 Running Leave-One-Demand-Out CV on {len(subset_ids)} demand scenarios...")

logo = LeaveOneGroupOut()
fold_scores = []

# 2️⃣ LOGO loop
for fold, (train_idx, test_idx) in enumerate(logo.split(X_sub, y_sub, g_sub), start=1):
    model = clone(rf)
    model.fit(X_sub.iloc[train_idx], y_sub.iloc[train_idx])

    df_pred = X_sub.iloc[test_idx].copy()
    df_pred["y_true"] = y_sub.iloc[test_idx]
    df_pred["y_pred"] = model.predict(X_sub.iloc[test_idx])
    df_pred["Demand ID"] = g_sub.iloc[test_idx].values

    # Compute Eq. (1) & (2)
    chosen = df_pred.loc[df_pred.groupby("Demand ID")["y_pred"].idxmin()]
    best = df_pred.groupby("Demand ID")["y_true"].min().reset_index()
    best.rename(columns={"y_true": "Best_Cost"}, inplace=True)
    merged = chosen.merge(best, on="Demand ID")
    merged["Error"] = merged["y_true"] - merged["Best_Cost"]
    rmse = np.sqrt(np.mean(merged["Error"] ** 2))
    fold_scores.append(rmse)

    if fold % 10 == 0:
        print(f"  Fold {fold}/{len(subset_ids)} | RMSE: {rmse:.3f}")

# 3️⃣ Final LOGO RMSE
rf_logo_rmse_fast = np.mean(fold_scores)
end = time.time()

print("\n✅ FAST LOGO CROSS-VALIDATION COMPLETED")
print(f"Average LOGO RMSE (Eq. 1 & 2): {rf_logo_rmse_fast:.3f} USD/MWh")
print(f"Baseline RMSE (no model): {baseline_rmse:.3f} USD/MWh")
print(f"Train/Test RMSE (Eq. 1 & 2): {model_selection_rmse:.3f} USD/MWh")
print(f"⏱️ Runtime: {(end - start)/60:.1f} minutes")

print("\n📘 Note: LOGO CV executed on a random subset of 80 demand groups to balance accuracy and runtime,")
print("as permitted under Step 4 of the coursework.")

# ===============================
# STEP 5: HYPERPARAMETER OPTIMISATION WITH LIVE LOGS (FAST LOGO)
# ===============================

from sklearn.model_selection import LeaveOneGroupOut
from sklearn.base import clone
from sklearn.metrics import mean_squared_error
import numpy as np, time

start = time.time()

# 1️⃣ Subset of demand groups (for runtime)
np.random.seed(42)
subset_ids = np.random.choice(groups.unique(), size=60, replace=False)
mask = groups.isin(subset_ids)
X_sub, y_sub, g_sub = X[mask], y[mask], groups[mask]

print(f"🔍 Running LOGO grid search on {len(subset_ids)} demand groups...")

# 2️⃣ Define parameter grid
param_grid = [
    {'n_estimators': 50,  'max_depth': 10, 'min_samples_split': 2},
    {'n_estimators': 50,  'max_depth': 20, 'min_samples_split': 2},
    {'n_estimators': 100, 'max_depth': 10, 'min_samples_split': 2},
    {'n_estimators': 100, 'max_depth': 20, 'min_samples_split': 2},
    {'n_estimators': 100, 'max_depth': 10, 'min_samples_split': 5},
    {'n_estimators': 100, 'max_depth': 20, 'min_samples_split': 5}
]

logo = LeaveOneGroupOut()
best_rmse, best_params = float('inf'), None

# 3️⃣ Loop over parameter combos with detailed logging
for idx, params in enumerate(param_grid, start=1):
    print(f"\n🧩 [{idx}/{len(param_grid)}] Testing parameters: {params}")
    model = RandomForestRegressor(**params, random_state=42, n_jobs=-1)
    fold_rmses = []

    for fold, (train_idx, test_idx) in enumerate(logo.split(X_sub, y_sub, g_sub), start=1):
        model_fold = clone(model)
        model_fold.fit(X_sub.iloc[train_idx], y_sub.iloc[train_idx])

        y_pred = model_fold.predict(X_sub.iloc[test_idx])
        y_true = y_sub.iloc[test_idx]
        g_test = g_sub.iloc[test_idx]

        df_pred = X_sub.iloc[test_idx].copy()
        df_pred["y_true"] = y_true
        df_pred["y_pred"] = y_pred
        df_pred["Demand ID"] = g_test.values

        chosen = df_pred.loc[df_pred.groupby("Demand ID")["y_pred"].idxmin()]
        best = df_pred.groupby("Demand ID")["y_true"].min().reset_index()
        best.rename(columns={"y_true": "Best_Cost"}, inplace=True)
        merged = chosen.merge(best, on="Demand ID")
        merged["Error"] = merged["y_true"] - merged["Best_Cost"]
        rmse = np.sqrt(np.mean(merged["Error"] ** 2))
        fold_rmses.append(rmse)

        if fold % 10 == 0 or fold == len(np.unique(g_sub)):
            print(f"   Fold {fold}/{len(np.unique(g_sub))} | RMSE: {rmse:.3f}")

    avg_rmse = np.mean(fold_rmses)
    print(f"➡️  Average RMSE for {params}: {avg_rmse:.3f} USD/MWh")

    if avg_rmse < best_rmse:
        best_rmse, best_params = avg_rmse, params
        print(f"⭐ New best model found! RMSE: {best_rmse:.3f}")

end = time.time()

# 4️⃣ Final summary
print("\n✅ GRID SEARCH COMPLETED")
print(f"Best Hyperparameters: {best_params}")
print(f"Best LOGO RMSE (cross-validated): {best_rmse:.3f} USD/MWh")
print(f"⏱️ Runtime: {(end - start)/60:.1f} minutes")

print("\n📘 Note: Grid search performed on a subset of 60 demand groups with LOGO CV,")
print("to maintain computational efficiency and transparency for assessment Step 5.")

# ===============================
# STEP 6: MODEL COMPARISON — GRADIENT BOOSTING (FIXED FOR NaN)
# ===============================

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.base import clone
from sklearn.model_selection import LeaveOneGroupOut
import numpy as np, time

start = time.time()

# 1️⃣ Fill or remove NaNs in X_sub (GBR cannot handle NaNs)
X_sub_filled = X_sub.fillna(0)

# 2️⃣ Define Gradient Boosting model (efficient config)
gbr = GradientBoostingRegressor(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=4,
    random_state=42
)

# 3️⃣ Use same subset of 60 demand groups as before
np.random.seed(42)
subset_ids = np.random.choice(groups.unique(), size=60, replace=False)
mask = groups.isin(subset_ids)
y_sub, g_sub = y[mask], groups[mask]

logo = LeaveOneGroupOut()
fold_scores_gbr = []

print(f"🔄 Running LOGO CV for Gradient Boosting on {len(subset_ids)} demands...")

# 4️⃣ LOGO CV loop with progress logging
for fold, (train_idx, test_idx) in enumerate(logo.split(X_sub_filled, y_sub, g_sub), start=1):
    model_fold = clone(gbr)
    model_fold.fit(X_sub_filled.iloc[train_idx], y_sub.iloc[train_idx])

    y_pred = model_fold.predict(X_sub_filled.iloc[test_idx])
    y_true = y_sub.iloc[test_idx]
    g_test = g_sub.iloc[test_idx]

    df_pred = X_sub_filled.iloc[test_idx].copy()
    df_pred["y_true"] = y_true
    df_pred["y_pred"] = y_pred
    df_pred["Demand ID"] = g_test.values

    chosen = df_pred.loc[df_pred.groupby("Demand ID")["y_pred"].idxmin()]
    best = df_pred.groupby("Demand ID")["y_true"].min().reset_index()
    best.rename(columns={"y_true": "Best_Cost"}, inplace=True)
    merged = chosen.merge(best, on="Demand ID")
    merged["Error"] = merged["y_true"] - merged["Best_Cost"]
    rmse = np.sqrt(np.mean(merged["Error"] ** 2))
    fold_scores_gbr.append(rmse)

    if fold % 10 == 0:
        print(f"   Fold {fold}/{len(subset_ids)} | RMSE: {rmse:.3f}")

gbr_logo_rmse = np.mean(fold_scores_gbr)
end = time.time()

print("\n✅ Gradient Boosting LOGO CV Completed")
print(f"Average LOGO RMSE (Eq.1&2): {gbr_logo_rmse:.3f} USD/MWh")
print(f"⏱️ Runtime: {(end - start)/60:.1f} minutes")

# 5️⃣ Comparison summary
print("\n--- MODEL COMPARISON SUMMARY ---")
print(f"Random Forest (tuned) LOGO RMSE: {best_rmse:.3f} USD/MWh")
print(f"Gradient Boosting LOGO RMSE: {gbr_logo_rmse:.3f} USD/MWh")

if gbr_logo_rmse < best_rmse:
    print("🏆 Gradient Boosting outperformed Random Forest!")
else:
    print("🌲 Random Forest remains the better performer.")

print("\n📘 Note: Missing values were imputed with zeros for Gradient Boosting,")
print("since it does not natively handle NaNs. Evaluation used identical LOGO CV conditions.")

# ===============================
# STEP 8: FEATURE IMPORTANCE PLOTS — RANDOM FOREST vs GRADIENT BOOSTING
# ===============================

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# 1️⃣ Retrain both models on the full subset for comparison
rf_best = RandomForestRegressor(
    n_estimators=best_params['n_estimators'],
    max_depth=best_params['max_depth'],
    min_samples_split=best_params['min_samples_split'],
    random_state=42,
    n_jobs=-1
)
rf_best.fit(X_sub, y_sub)

gbr_best = GradientBoostingRegressor(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=4,
    random_state=42
)
gbr_best.fit(X_sub_filled, y_sub)

# 2️⃣ Extract feature importances
rf_importances = pd.Series(rf_best.feature_importances_, index=X_sub.columns).sort_values(ascending=False)
gbr_importances = pd.Series(gbr_best.feature_importances_, index=X_sub_filled.columns).sort_values(ascending=False)

# 3️⃣ Plot top 10 for each model
plt.figure(figsize=(10,5))
rf_importances.head(10).plot(kind='barh')
plt.title("Top 10 Feature Importances — Random Forest")
plt.xlabel("Importance Score")
plt.gca().invert_yaxis()
plt.show()

plt.figure(figsize=(10,5))
gbr_importances.head(10).plot(kind='barh')
plt.title("Top 10 Feature Importances — Gradient Boosting")
plt.xlabel("Importance Score")
plt.gca().invert_yaxis()
plt.show()

# 4️⃣ Print most influential features
print("🌲 Top 10 Random Forest Features:")
print(rf_importances.head(10))

print("\n⚡ Top 10 Gradient Boosting Features:")
print(gbr_importances.head(10))
