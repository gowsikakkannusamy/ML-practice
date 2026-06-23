import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Sample dataset
data = {
    'Hours': [1, 2, 3, 4, 5, 6, 7, 8],
    'Pass':  [0, 0, 0, 0, 1, 1, 1, 1]
}

df = pd.DataFrame(data)

# Features and Target
X = df[['Hours']]
y = df['Pass']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create Decision Tree model
model = DecisionTreeClassifier()

# Train model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Check accuracy
print("Accuracy:", accuracy_score(y_test, y_pred))

# Predict for 5 hours study
result = model.predict([[5]])
print("Pass Prediction:", result[0])