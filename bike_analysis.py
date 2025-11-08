import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from scipy.stats import pearsonr
from haversine import haversine, Unit
from sklearn.linear_model import LinearRegression, ElasticNet, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# Define output folder
output_folder = r"E:\GA6"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# ----------------------------
# Data Loading & Preprocessing
# ----------------------------
df = pd.read_csv(r"E:\GA6\bike_sales_india.csv")

# Rename columns for easier reference
df.rename(columns={
    "Avg Daily Distance (km)": "AvgDistance",
    "Engine Capacity (cc)": "EngineCapacity",
    "Mileage (km/l)": "Mileage",
    "Price (INR)": "OriginalPrice",
    "Resale Price (INR)": "ResalePrice",
    "Year of Manufacture": "Year",
    "City Tier": "CityTier"
}, inplace=True)

# Compute Price Retention (ResalePrice divided by OriginalPrice)
df["PriceRetention"] = df["ResalePrice"] / df["OriginalPrice"]

# ----------------------------
# Advanced Exploratory Analytics
# ----------------------------
# Generate descriptive statistics for the entire dataset
desc_stats = df.describe()
desc_stats.to_csv(os.path.join(output_folder, "descriptive_stats.csv"))

# Correlation matrix across all numeric columns
corr_matrix = df.select_dtypes(include=[np.number]).corr()
corr_matrix.to_csv(os.path.join(output_folder, "correlation_matrix.csv"))

# Visualize distribution of numeric features
num_cols = ["OriginalPrice", "ResalePrice", "EngineCapacity", "Mileage", "AvgDistance", "PriceRetention"]
df[num_cols].hist(figsize=(15, 10), bins=20)
plt.tight_layout()
plt.savefig(os.path.join(output_folder, "numeric_distributions.png"))
plt.close()

# Correlation heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Matrix of Numeric Features")
plt.tight_layout()
plt.savefig(os.path.join(output_folder, "correlation_heatmap.png"))
plt.close()

# ----------------------------
# Compare Multiple ML Models
# ----------------------------
# Prepare features and target for model comparison
X = df[["EngineCapacity", "Mileage", "AvgDistance", "OriginalPrice"]]
y = df["ResalePrice"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features for better performance
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create dictionary of models to evaluate
models = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(alpha=1.0),
    "Lasso Regression": Lasso(alpha=0.1),
    "ElasticNet": ElasticNet(alpha=0.1, l1_ratio=0.5),
    "Decision Tree": DecisionTreeRegressor(random_state=42),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(random_state=42),
    "SVR": SVR(kernel='rbf'),
    "K-Nearest Neighbors": KNeighborsRegressor(n_neighbors=5)
}

model_results = []
for name, model in models.items():
    # Train model
    if name in ["SVR", "K-Nearest Neighbors"]:
        # These models benefit from scaled features
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
    
    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    model_results.append({
        "Model": name, 
        "MSE": mse,
        "R^2": r2
    })

# Save model comparison results
model_comparison = pd.DataFrame(model_results)
model_comparison.to_csv(os.path.join(output_folder, "model_comparison.csv"), index=False)

# ----------------------------
# Question 1: TVS motorcycle resale value factors in Gujarat (Correlation Analysis)
# ----------------------------
def solve_q1():
    """
    Q1: TVS motorcycle resale value factors in Gujarat
    Returns the coefficient with the largest absolute magnitude
    """
    # Filter for TVS in Gujarat and check if we have data
    tvs_gujarat = df[(df["Brand"] == "TVS") & (df["State"] == "Gujarat")]
    
    if tvs_gujarat.empty:
        answer = {
            "Question": "Q1: TVS in Gujarat - Price retention correlation analysis",
            "Answer": "No data available for TVS in Gujarat",
            "Solution": "The dataset does not contain any records for TVS motorcycles in Gujarat."
        }
    else:
        # Calculate actual correlations with price retention
        correlations = {}
        p_values = {}
        for feature in ["Mileage", "AvgDistance", "EngineCapacity"]:
            corr, p_val = pearsonr(tvs_gujarat[feature], tvs_gujarat["PriceRetention"])
            correlations[feature] = round(corr, 2)
            p_values[feature] = round(p_val, 3)
        
        # Format all correlation results for reference
        all_correlations = "\n".join([f"{feat}: {corr}" for feat, corr in correlations.items()])
        
        # Find feature with max absolute correlation
        max_abs_feature = max(correlations.items(), key=lambda x: abs(x[1]))
        max_abs_value = max_abs_feature[1]
        max_abs_name = max_abs_feature[0]
        
        # Map correlations to options
        answer = {
            "Question": "Q1: TVS in Gujarat - Price retention correlation analysis",
            "Answer": f"{max_abs_value}",  # Just return the maximum absolute correlation value
            "Solution": f"""Analysis of TVS motorcycles in Gujarat:
All correlations with price retention:
{all_correlations}

The absolute maximum correlation coefficient is {max_abs_value} for {max_abs_name}."""
        }
    
    return answer

# ----------------------------
# Question 2: Yamaha motorcycle resale value factors in West Bengal (Correlation Analysis)
# ----------------------------
def solve_q2():
    """
    Q2: Yamaha motorcycle resale value factors in West Bengal
    Returns the coefficient with the largest absolute magnitude
    """
    # Filter for Yamaha in West Bengal
    yamaha_wb = df[(df["Brand"] == "Yamaha") & (df["State"] == "West Bengal")]
    
    if yamaha_wb.empty:
        answer = {
            "Question": "Q2: Yamaha in West Bengal - Price retention correlation analysis",
            "Answer": "No data available for Yamaha in West Bengal",
            "Solution": "The dataset does not contain any records for Yamaha motorcycles in West Bengal."
        }
    else:
        # Calculate actual correlations with price retention
        correlations = {}
        p_values = {}
        for feature in ["Mileage", "AvgDistance", "EngineCapacity"]:
            corr, p_val = pearsonr(yamaha_wb[feature], yamaha_wb["PriceRetention"])
            correlations[feature] = round(corr, 2)
            p_values[feature] = round(p_val, 3)
        
        # Format all correlation results for reference
        all_correlations = "\n".join([f"{feat}: {corr}" for feat, corr in correlations.items()])
        
        # Find feature with max absolute correlation
        max_abs_feature = max(correlations.items(), key=lambda x: abs(x[1]))
        max_abs_value = max_abs_feature[1]
        max_abs_name = max_abs_feature[0]
        
        answer = {
            "Question": "Q2: Yamaha in West Bengal - Price retention correlation analysis",
            "Answer": f"{max_abs_value}",  # Just return the maximum absolute correlation value
            "Solution": f"""Analysis of Yamaha motorcycles in West Bengal:
All correlations with price retention:
{all_correlations}

The absolute maximum correlation coefficient is {max_abs_value} for {max_abs_name}."""
        }
    
    return answer

# ----------------------------
# Question 3: Predict resale price for a Tier 2 motorcycle (Random Forest Regressor)
# ----------------------------
def solve_q3():
    """
    Q3: Use Random Forest to predict resale price for a Tier 2 motorcycle
    """
    # Filter for Tier 2 motorcycles
    tier2_bikes = df[df["CityTier"] == "Tier 2"]
    
    if tier2_bikes.empty:
        answer = {
            "Question": "Q3: Predict resale price for a Tier 2 motorcycle",
            "Answer": "No Tier 2 motorcycle data available",
            "Solution": "The dataset does not contain any records for Tier 2 motorcycles."
        }
    else:
        # Use a specific test sample for prediction
        # If provided in the question, use that sample, otherwise use the first Tier 2 bike
        features = ["EngineCapacity", "Mileage", "AvgDistance", "OriginalPrice"]
        try:
            # Attempt to use the example values from the question if needed
            test_sample = np.array([[423.0245892, 40.51805532, 41.79131515, 264102.8387]])
            sample_description = "Example bike (EngineCapacity=423.02, Mileage=40.52, AvgDistance=41.79, OriginalPrice=264102.84)"
        except:
            # Otherwise use the first Tier 2 bike as sample
            sample_bike = tier2_bikes.iloc[0]
            test_sample = sample_bike[features].values.reshape(1, -1)
            sample_description = f"{sample_bike['Brand']} {sample_bike['Model']} (EngineCapacity={sample_bike['EngineCapacity']}, Mileage={sample_bike['Mileage']})"
        
        # Prepare data for Random Forest model
        X_tier2 = tier2_bikes[features]
        y_tier2 = tier2_bikes["ResalePrice"]
        
        # Create and train a Random Forest model
        rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
        rf_model.fit(X_tier2, y_tier2)
        
        # Make prediction
        predicted_price = rf_model.predict(test_sample)[0]
        
        # Get feature importances
        feature_importance = dict(zip(features, rf_model.feature_importances_))
        importance_str = "\n".join([f"{feat}: {imp:.4f}" for feat, imp in feature_importance.items()])
        
        answer = {
            "Question": "Q3: Predict resale price for a Tier 2 motorcycle",
            "Answer": f"{predicted_price:.0f}",  # Just the predicted price as integer
            "Solution": f"""Used Random Forest Regressor (100 trees) on {tier2_bikes.shape[0]} Tier 2 motorcycles.
Predicting for {sample_description}
Feature importances:
{importance_str}
Predicted resale price: {predicted_price:.2f}"""
        }
    
    return answer

# ----------------------------
# Question 4: Forecast 2027 resale value for Kawasaki Versys 650 in Delhi (SVR)
# ----------------------------
def solve_q4():
    """
    Q4: Use Support Vector Regression to forecast 2027 resale value for Kawasaki Versys 650 in Delhi
    """
    # Filter for Kawasaki Versys 650 in Delhi
    kv_delhi = df[(df["Brand"] == "Kawasaki") & (df["Model"] == "Versys 650") & (df["State"] == "Delhi")]
    
    if kv_delhi.empty:
        answer = {
            "Question": "Q4: Forecast 2027 resale value for Kawasaki Versys 650 in Delhi",
            "Answer": "No data available for Kawasaki Versys 650 in Delhi",
            "Solution": "The dataset does not contain any records for Kawasaki Versys 650 in Delhi."
        }
    else:
        # Scale the year data for SVR
        years = kv_delhi[["Year"]].values
        prices = kv_delhi["ResalePrice"].values
        
        # Scale the data
        year_scaler = StandardScaler()
        price_scaler = StandardScaler()
        
        years_scaled = year_scaler.fit_transform(years)
        prices_scaled = price_scaler.fit_transform(prices.reshape(-1, 1)).ravel()
        
        # Train SVR model
        svr_model = SVR(kernel='rbf', C=100, gamma=0.1)
        svr_model.fit(years_scaled, prices_scaled)
        
        # Prepare 2027 for prediction
        future_year = np.array([[2027]])
        future_year_scaled = year_scaler.transform(future_year)
        
        # Predict and inverse transform
        forecast_scaled = svr_model.predict(future_year_scaled)
        forecast_2027 = price_scaler.inverse_transform(forecast_scaled.reshape(-1, 1))[0][0]
        
        answer = {
            "Question": "Q4: Forecast 2027 resale value for Kawasaki Versys 650 in Delhi",
            "Answer": f"{forecast_2027:.0f}",  # Return forecast as integer
            "Solution": f"""Used Support Vector Regression (RBF kernel) on {kv_delhi.shape[0]} Kawasaki Versys 650 bikes in Delhi.
Data years: {kv_delhi['Year'].min()} to {kv_delhi['Year'].max()}
Applied data scaling for better SVR performance
Forecasted 2027 resale price: {forecast_2027:.2f}"""
        }
    
    return answer

# ----------------------------
# Question 5: Count anomalous mileage records for Electric bikes in Maharashtra
# ----------------------------
def solve_q5():
    """
    Q5: Use Isolation Forest to detect anomalous mileage records for Electric vehicles in Maharashtra
    """
    # Filter for Electric bikes in Maharashtra
    elec_maharashtra = df[(df["Fuel Type"] == "Electric") & (df["State"] == "Maharashtra")]
    
    if elec_maharashtra.empty:
        answer = {
            "Question": "Q5: Anomalous mileage records for Electric vehicles in Maharashtra",
            "Answer": "No data available for Electric vehicles in Maharashtra",
            "Solution": "The dataset does not contain any records for Electric vehicles in Maharashtra."
        }
    else:
        # Calculate IQR for mileage as requested
        Q1 = elec_maharashtra["Mileage"].quantile(0.25)
        Q3 = elec_maharashtra["Mileage"].quantile(0.75)
        IQR = Q3 - Q1
        
        # Define bounds with 0.5 multiplier as specified
        lower_bound = Q1 - 0.5 * IQR
        upper_bound = Q3 + 0.5 * IQR
        
        # Identify anomalies
        anomalies = elec_maharashtra[(elec_maharashtra["Mileage"] < lower_bound) | 
                                    (elec_maharashtra["Mileage"] > upper_bound)]
        
        answer = {
            "Question": "Q5: Anomalous mileage records for Electric vehicles in Maharashtra",
            "Answer": f"{len(anomalies)}",  # Just return the count as integer
            "Solution": f"""Analysis of {len(elec_maharashtra)} Electric vehicles in Maharashtra.
Used IQR method with 0.5 multiplier:
Q1={Q1:.2f}, Q3={Q3:.2f}, IQR={IQR:.2f}
Anomaly thresholds: lower={lower_bound:.2f}, upper={upper_bound:.2f}
Detected {len(anomalies)} anomalous records"""
        }
    
    return answer

# ----------------------------
# Question 6: Count anomalous mileage records for Electric bikes in Uttar Pradesh (DBScan clustering)
# ----------------------------
def solve_q6():
    """
    Q6: Identify anomalous mileage records for Electric vehicles in Uttar Pradesh
    """
    # Filter for Electric bikes in Uttar Pradesh
    elec_up = df[(df["Fuel Type"] == "Electric") & (df["State"] == "Uttar Pradesh")]
    
    if elec_up.empty:
        answer = {
            "Question": "Q6: Anomalous mileage records for Electric vehicles in Uttar Pradesh",
            "Answer": "No data available for Electric vehicles in Uttar Pradesh",
            "Solution": "The dataset does not contain any records for Electric vehicles in Uttar Pradesh."
        }
    else:
        # Calculate IQR for mileage (as in Q5)
        Q1 = elec_up["Mileage"].quantile(0.25)
        Q3 = elec_up["Mileage"].quantile(0.75)
        IQR = Q3 - Q1
        
        # Define bounds with 0.5 multiplier
        lower_bound = Q1 - 0.5 * IQR
        upper_bound = Q3 + 0.5 * IQR
        
        # Identify anomalies
        anomalies = elec_up[(elec_up["Mileage"] < lower_bound) | 
                          (elec_up["Mileage"] > upper_bound)]
        
        answer = {
            "Question": "Q6: Anomalous mileage records for Electric vehicles in Uttar Pradesh",
            "Answer": f"{len(anomalies)}",  # Just return the count as integer
            "Solution": f"""Analysis of {len(elec_up)} Electric vehicles in Uttar Pradesh.
Used IQR method with 0.5 multiplier:
Q1={Q1:.2f}, Q3={Q3:.2f}, IQR={IQR:.2f}
Anomaly thresholds: lower={lower_bound:.2f}, upper={upper_bound:.2f}
Detected {len(anomalies)} anomalous records"""
        }
    
    return answer

# ----------------------------
# Question 7: Calculate distance from Central Command Post to Silver Haven Junction
# ----------------------------
def solve_q7():
    """
    Q7: Calculate the haversine distance between two locations
    """
    central = (27.6072, -103.2503)
    silver_haven = (27.6846, -103.3366)
    
    # Calculate distance using haversine formula
    exact_distance = haversine(central, silver_haven, unit=Unit.METERS)
    
    # Options provided in the question (these would be the multiple choice options)
    options = [8200, 9400, 10600, 12000]
    
    # Find the closest option to our calculated distance
    closest_option = min(options, key=lambda x: abs(x - exact_distance))
    
    answer = {
        "Question": "Q7: Distance from Central Command Post to Silver Haven Junction",
        "Answer": f"{closest_option}",  # Return the closest option
        "Solution": f"""Used haversine formula to calculate great-circle distance:
Central Command Post: {central}
Silver Haven Junction: {silver_haven}
Exact calculated distance = {exact_distance:.2f} meters
Closest option from the choices = {closest_option} meters"""
    }
    
    return answer

# ----------------------------
# Question 8: Find closest community to Central Command Post
# ----------------------------
def solve_q8():
    """
    Q8: Find the closest community using haversine distances
    """
    central = (26.4652, -74.2424)
    communities = {
        "North Falls Sanctuary": (26.5015, -74.3304),
        "Pleasant Springs Outpost": (26.548, -74.2694),
        "Green Ridge Hamlet": (26.3444, -74.1077),
        "Blue Shores Sanctuary": (26.381, -74.1365)
    }
    
    # Calculate distances to each community
    distances = {}
    for name, coords in communities.items():
        distances[name] = haversine(central, coords, unit=Unit.METERS)
    
    # Find closest community
    closest = min(distances, key=distances.get)
    
    answer = {
        "Question": "Q8: Closest community to Central Command Post",
        "Answer": f"{closest}",  # Just return the name of the closest community
        "Solution": f"""Calculated haversine distances from Central Command Post to each community:
{', '.join([f"{name}: {dist:.0f}m" for name, dist in distances.items()])}
The closest community is {closest} at {distances[closest]:.0f} meters."""
    }
    
    return answer

# ----------------------------
# Question 9: Determine evacuation route using nearest neighbor approach
# ----------------------------
def solve_q9():
    """
    Q9: Generate evacuation route using nearest neighbor algorithm
    """
    central = (43.2752, -97.3323)
    communities = {
        "Blue Bluff Junction": (43.1811, -97.1838),
        "Eagle Glen Outpost": (43.3218, -97.2716),
        "River Bluff Station": (43.1543, -97.2825),
        "Pleasant Point Town": (43.22, -97.3679)
    }
    
    # Apply nearest neighbor algorithm
    route = ["Central Command Post"]
    current = central
    remaining = communities.copy()
    total_distance = 0
    
    while remaining:
        # Find nearest unvisited community
        next_community = min(remaining.items(), 
                           key=lambda x: haversine(current, x[1], unit=Unit.METERS))
        name, coords = next_community
        dist = haversine(current, coords, unit=Unit.METERS)
        
        # Add to route and update distance
        route.append(name)
        total_distance += dist
        
        # Update current position and remove from remaining
        current = coords
        del remaining[name]
    
    # Return to starting point
    route.append("Central Command Post")
    total_distance += haversine(current, central, unit=Unit.METERS)
    
    # Format the route as comma-separated list
    route_str = " → ".join(route)
    
    answer = {
        "Question": "Q9: Evacuation route using nearest neighbor approach",
        "Answer": f"{route_str}",  # Return the full route
        "Solution": f"""Applied nearest neighbor algorithm starting from Central Command Post.
Evaluated distances between all points using haversine formula.
Total route distance: {total_distance:.0f} meters
Final route: {route_str}"""
    }
    
    return answer

# ----------------------------
# Execute all analyses and save results
# ----------------------------
solutions = [
    solve_q1(),
    solve_q2(),
    solve_q3(),
    solve_q4(),
    solve_q5(),
    solve_q6(),
    solve_q7(),
    solve_q8(),
    solve_q9()
]

# Save the Q&A to CSV
qa_df = pd.DataFrame(solutions)
qa_df.to_csv(os.path.join(output_folder, "question_answer_solution.csv"), index=False)

# Display the results in console
if __name__ == "__main__":
    print("\n====== BIKE SALES ANALYSIS - RESULTS ======\n")
    
    for solution in solutions:
        print(f"\n{solution['Question']}")
        print("-" * len(solution['Question']))
        print(f"Answer: {solution['Answer']}")
        print(f"\nSolution Explanation:\n{solution['Solution']}")
        print("\n" + "="*50)
    
    print(f"\nAll analysis outputs saved to {output_folder}\n")