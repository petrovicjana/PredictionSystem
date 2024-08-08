import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import joblib
import os

# Path to the data file
data_path = os.path.join('data', "USA_Housing.csv")
model_path = os.path.join('model', "linear_regression_model.pkl")

# Load and prepare data
data = pd.read_csv(data_path)
data = data.drop(['Address'], axis=1)

X = data.drop('Price', axis=1)
y = data['Price']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Save the model
joblib.dump(model, model_path)

print("Model trained and saved at", model_path)
