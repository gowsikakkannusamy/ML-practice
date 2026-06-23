import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Sample dataset
data = {
    'Hours': [1, 2, 3, 4, 5, 6, 7, 8],
    'Pass': [0, 0, 0, 0, 1, 1, 1, 1]
}

df = pd.DataFrame(data)

# Input and Output
X = df[['Hours']]
y = df['Pass']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create and train model
model = LogisticRegression()
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Accuracy
print("Accuracy:", accuracy_score(y_test, y_pred))

# Predict for 5 hours study
result = model.predict([[5]])
print("Pass Prediction:", result[0])