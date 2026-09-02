import os
import pandas as pd
from sklearn.preprocessing import StandardScaler

# --------------------------------------------------
# 1. Create data directory
# --------------------------------------------------

os.makedirs("data", exist_ok=True)


# --------------------------------------------------
# 2. Create sample DataFrame
# --------------------------------------------------

data = {
    "name": [
        "Alice",
        "Bob",
        "Charlie",
        "David",
        "Eva",
        "Frank",
        "Grace",
        "Henry",
        "Ivy",
        "mack"
    ],
    "age": [25, 30, 35, 40, 28, 45, 32, 38, 27, 50],
    "salary": [
        40000,
        50000,
        60000,
        80000,
        45000,
        90000,
        55000,
        70000,
        42000,
        300000
    ],
    "experience": [1, 3, 7, 10, 2, 15, 5, 8, 2, 20],
    "purchased": [0, 1, 1, 1, 0, 1, 1, 1, 0, 1]
}

df = pd.DataFrame(data)


# --------------------------------------------------
# 3. Save raw data
# --------------------------------------------------

raw_file = "data/data.csv"

df.to_csv(raw_file, index=False)

print(f"Raw data saved to: {raw_file}")

print("\nRaw Data:")
print(df)


# --------------------------------------------------
# 4. Preprocessing
# --------------------------------------------------

# Make a copy so the original data is not changed
processed_df = df.copy()

# Drop the name column because it is not useful
# for numerical machine-learning models
processed_df = processed_df.drop(columns=["name"])


# --------------------------------------------------
# 5. Handle missing values
# --------------------------------------------------

# Fill missing numerical values with the column median
numeric_columns = ["age", "salary", "experience"]

for column in numeric_columns:
    processed_df[column] = processed_df[column].fillna(
        processed_df[column].median()
    )


# --------------------------------------------------
# 6. Scale numerical features
# --------------------------------------------------

scaler = StandardScaler()

processed_df[numeric_columns] = scaler.fit_transform(
    processed_df[numeric_columns]
)


# --------------------------------------------------
# 7. Save preprocessed data
# --------------------------------------------------

processed_file = "data/preprocessed.csv"

processed_df.to_csv(processed_file, index=False)

print(f"\nPreprocessed data saved to: {processed_file}")

print("\nPreprocessed Data:")
print(processed_df)


# --------------------------------------------------
# Done
# --------------------------------------------------

print("\nData preparation completed successfully!")