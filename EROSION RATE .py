
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from bayes_opt import BayesianOptimization

# Load your original dataset from a CSV file
data_path = 'erosion-dataset.csv'
column_names = ["No.", "Inner diameter (mm)", "R/D ratio", "Bending angle (°)",
                 "Bending orientation (°)", "Particle velocity (m/s)",
                 "Particle size (cm)", "Particle Mass flow rate (kg/s)",
                 "Maximum erosion rate -ERmax kg/(m2·s)"]
df = pd.read_csv(data_path, names=column_names, skiprows=1)

# Exclude non-numeric values from the dataset
df = df.apply(pd.to_numeric, errors='coerce').dropna()

# Extract features (X) and target variable (y)
X = df.drop(columns=["Maximum erosion rate -ERmax kg/(m2·s)"])
y = df["Maximum erosion rate -ERmax kg/(m2·s)"]

# Data preprocessing: Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.45, random_state=42)

# Define the objective function for Bayesian Optimization
def gpr_objective(length_scale):
    kernel = 1.0 * RBF(length_scale=length_scale)
    gpr_model = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, random_state=0)
    gpr_model.fit(X_train, y_train)
    y_pred, _ = gpr_model.predict(X_test, return_std=True)
    return -mean_squared_error(y_test, y_pred)  # Negative MSE for maximization

# Define the parameter bounds for Bayesian Optimization
param_bounds = {'length_scale': (1e-8, 1e3)}

# Perform Bayesian Optimization
bayesian_opt = BayesianOptimization(f=gpr_objective, pbounds=param_bounds, random_state=0)
bayesian_opt.maximize(init_points=10, n_iter=15)

# Get the best hyperparameters
best_length_scale = bayesian_opt.max['params']['length_scale']

# Create GPR model with the optimized kernel
best_kernel = 1.0 * RBF(length_scale=best_length_scale)
gpr_model = GaussianProcessRegressor(kernel=best_kernel, n_restarts_optimizer=10, random_state=0)

# Fit the model on the training dataset
gpr_model.fit(X_train, y_train)

# Predict on the training set
y_train_pred, _ = gpr_model.predict(X_train, return_std=True)

# Predict on the test set for the new data
y_test_pred, _ = gpr_model.predict(X_test, return_std=True)


# Plot regression for the testing data
plt.scatter(y_test, y_test_pred, label='Predicted Values', color='Blue')
plt.scatter(y_test, y_test, label='True Values', color='Yellow', alpha=0.5)  # True values in red
plt.xlabel('True Max Erosion Rate (kg/(m2·s)')
plt.ylabel('Predicted Max Erosion Rate (kg/(m2·s)')
plt.title('Regression Plot for Testing Data')
plt.legend()
plt.show()


# Calculate Mean Squared Error (MSE) and R-squared for the testing data
mse_test = mean_squared_error(y_test, y_test_pred)
r2_test = r2_score(y_test, y_test_pred)
print(f'Mean Squared Error (MSE) for Testing Data: {mse_test}')
print(f'R-squared for Testing Data: {r2_test}')



# In[ ]:




