import sklearn
from sklearn.linear_model import LinearRegression
import numpy as np

# Check version
print("scikit-learn version:", sklearn.__version__)

# Create some dummy data
X = np.array([[1], [2], [3], [4], [5]])
y = np.array([1, 2, 3, 4, 5])

# Train a simple linear regression model
model = LinearRegression()
model.fit(X, y)

# Make a prediction
print("Prediction for 6:", model.predict([[6]])[0])
