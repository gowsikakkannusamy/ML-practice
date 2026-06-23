import pandas as pd
from sklearn.preprocessing import StandardScaler

# Sample dataset
data = {
    'Age': [20, 25, 30],
    'Salary': [20000, 30000, 40000]
}

df = pd.DataFrame(data)
scaler = StandardScaler()
standardized_data = scaler.fit_transform(df)
standardized_df = pd.DataFrame(
    standardized_data,
    columns=df.columns
)

print("Original Data:")
print(df)

print("\nStandardized Data:")
print(standardized_df)