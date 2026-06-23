import pandas as pd
data = {
    'Marks': [80, 75, 90, 85, 70, 95, 88]
}

df = pd.DataFrame(data)
print("Statistical Report:")
print(df.describe())
print("\nMean:", df['Marks'].mean())
print("Median:", df['Marks'].median())
print("Mode:", df['Marks'].mode()[0])
print("Standard Deviation:", df['Marks'].std())