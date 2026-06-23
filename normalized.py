import pandas as pd
from sklearn.preprocessing import MinMaxScaler
data = {
    'Age': [20, 25, 30],
    'Salary': [20000, 30000, 40000]
}

df = pd.DataFrame(data)
scaler = MinMaxScaler()
normalized_data = scaler.fit_transform(df)
normalized_df = pd.DataFrame(normalized_data, columns=df.columns)

print(normalized_df)