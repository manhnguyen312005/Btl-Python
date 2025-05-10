# part4.2.py: Estimating Player Transfer Values
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
import matplotlib.pyplot as plt
import re

# Function to parse Age from 'years-days' format to float
def parse_age(age_str):
    try:
        if isinstance(age_str, str) and "-" in age_str:
            years, days = map(int, age_str.split("-"))
            return years + (days / 365.0)  # Convert to decimal years
        return float(age_str)
    except (ValueError, TypeError):
        return np.nan

# Function to parse estimated_value from '€32.2M' to float
def parse_transfer_value(value):
    try:
        if isinstance(value, str):
            value = re.sub(r'[€£M]', '', value).strip()
            return float(value)
        return float(value)
    except (ValueError, TypeError):
        return np.nan

# Load data from df_combined.csv
df = pd.read_csv("df_combined.csv")

# Log available columns for debugging
print("Available columns in df_combined.csv:", df.columns.tolist())

# Parse estimated_value
df["estimated_value"] = df["estimated_value"].apply(parse_transfer_value)

# Handle missing values in estimated_value
df = df.dropna(subset=["estimated_value"])  # Drop rows with NaN in target

# Clean Age column
df["Age"] = df["Age"].apply(parse_age)
df["Age"] = df["Age"].fillna(df["Age"].median())  # Impute with median

# Define potential features
potential_numerical_features = [
    "Age", "Min", "MP", "Starts", "Gls", "Ast", "xG", "xAG",
    "YCrd", "RCrd", "PrgC", "PrgP", "PrgR", "Gls/90", "Ast/90", "xG/90", "xAG/90",
    "Cmp", "Tkl", "Int", "Blocks", "Touches", "Att 3rd", "Att Pen", "Succ%",
    "PrgDist", "CPA", "Recov", "Won%"
]
potential_categorical_features = ["Position", "Pos"]

# Check available columns
available_columns = df.columns.tolist()
numerical_features = [f for f in potential_numerical_features if f in available_columns]
categorical_features = [f for f in potential_categorical_features if f in available_columns]

# Warn about missing features
missing_features = [f for f in potential_numerical_features + potential_categorical_features if f not in available_columns]
if missing_features:
    print(f"Warning: Missing features in df_combined.csv: {missing_features}")

# Compute per-90 metrics if missing
if "Gls/90" not in available_columns and "Gls" in available_columns and "Min" in available_columns:
    df["Gls/90"] = df["Gls"] / (df["Min"] / 90.0)
    numerical_features.append("Gls/90")
if "Ast/90" not in available_columns and "Ast" in available_columns and "Min" in available_columns:
    df["Ast/90"] = df["Ast"] / (df["Min"] / 90.0)
    numerical_features.append("Ast/90")

# Ensure at least some features are available
if not numerical_features and not categorical_features:
    raise ValueError("No valid features available in df_combined.csv. Please check the input data.")

# Handle missing values in features
for col in numerical_features:
    df[col] = pd.to_numeric(df[col], errors="coerce")  # Convert to numeric, invalid to NaN
    df[col] = df[col].replace("N/a", 0).fillna(df[col].median())  # Replace N/a with 0, impute NaN with median

# Prepare features (X) and target (y)
X = df[numerical_features + categorical_features]
y = df["estimated_value"]

# Create preprocessing pipeline
transformers = [("num", StandardScaler(), numerical_features)]
if categorical_features:
    transformers.append(("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), categorical_features))

preprocessor = ColumnTransformer(transformers=transformers)

# Create model pipeline
model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("regressor", RandomForestRegressor(n_estimators=100, max_depth=10, min_samples_split=5, random_state=42))
])

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)  # Compute MSE
rmse = np.sqrt(mse)  # Compute RMSE manually

print(f"Model Performance on Test Set:")
print(f"Mean Absolute Error (MAE): {mae:.2f} million euros")
print(f"R² Score: {r2:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f} million euros")

# Cross-validation
cv_scores = cross_val_score(model, X, y, cv=5, scoring="r2")
print(f"5-Fold Cross-Validation R² Scores: {cv_scores}")
print(f"Average CV R² Score: {cv_scores.mean():.2f} ± {cv_scores.std():.2f}")

# Predict for all players
df["predicted_value"] = model.predict(X)

# Save predictions
df[["player_name", "estimated_value", "predicted_value"]].to_csv(
    "estimated_values.csv", index=False, encoding="utf-8-sig"
)

# Extract feature importance
feature_names = numerical_features
if categorical_features:
    feature_names += list(model.named_steps["preprocessor"].transformers_[1][1].get_feature_names_out())
importances = model.named_steps["regressor"].feature_importances_
feature_importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importances
}).sort_values(by="Importance", ascending=False)

# Save feature importance
feature_importance_df.to_csv("feature_importance.csv", index=False, encoding="utf-8-sig")

# Plot feature importance
plt.figure(figsize=(10, 6))
plt.barh(feature_importance_df["Feature"][:10], feature_importance_df["Importance"][:10])
plt.xlabel("Feature Importance")
plt.title("Top 10 Features Influencing Player Transfer Value")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig("feature_importance_plot.png")
plt.close()

print("Estimation complete. Results saved to 'estimated_values.csv' and 'feature_importance.csv'.")