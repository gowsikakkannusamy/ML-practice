import pandas as pd
from sklearn.linear_model import LinearRegression

# Load data
df = pd.read_csv('data.csv')

# Input and Output
X = df[['hours']]
y = df['score']

# Train model
model = LinearRegression()
model.fit(X, y)

# Predict score for 6 hours
prediction = model.predict([[6]])

print("Predicted Score for 6 hours:", prediction[0])