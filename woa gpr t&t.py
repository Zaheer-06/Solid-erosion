#!/usr/bin/env python
# coding: utf-8

# In[31]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from scipy.optimize import minimize

# Load your original dataset from a CSV file
data_path = 'erosion-dataset.csv'
column_names = ["No.", "Inner diameter (mm)", "R/D ratio", "Bending angle (°)",
                 "Bending orientation (°)", "Particle velocity (m/s)",
                 "Particle size (cm)", "Particle Mass flow rate (kg/s)",
                 "Maximum erosion rate -ERmax kg/(m2·s)"]
df = pd.read_csv(data_path, names=column_names, skiprows=1)

# Handle NaN values (drop rows with NaN values for demonstration, choose your strategy)
df = df.dropna()


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
    gpr_model = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, random_state=0, optimizer='fmin_l_bfgs_b')
    gpr_model.fit(X_train, y_train)
    y_pred, _ = gpr_model.predict(X_test, return_std=True)
    return mean_squared_error(y_test, y_pred)

# Perform Whale Optimization Algorithm (WOA) with scipy.optimize.minimize
woa_bounds = [(1e-3, 1e3)]
result = minimize(gpr_objective, x0=np.random.uniform(1e-3, 1e3), bounds=woa_bounds, method='L-BFGS-B', options={'maxiter': 20})

# Get the best length scale
best_length_scale_woa = result.x[0]

# Create GPR model with the optimized kernel using WOA
best_kernel_woa = 1.0 * RBF(length_scale=best_length_scale_woa)
gpr_model_woa = GaussianProcessRegressor(kernel=best_kernel_woa, n_restarts_optimizer=10, random_state=0)
                                         # ...

# Fit the model on the training dataset
gpr_model_woa.fit(X_train, y_train)

# Predict on the training set
y_train_pred_woa, _ = gpr_model_woa.predict(X_train, return_std=True)

# Calculate regression plot for the training data
plt.scatter(y_train, y_train_pred_woa)
plt.xlabel('True Values (Training Data)')
plt.ylabel('Predictions (Training Data)')
plt.title('Regression Plot for Training Data with WOA-Optimized GPR')
plt.show()

# Calculate Mean Squared Error (MSE) and R-squared for the training data
mse_train_woa = mean_squared_error(y_train, y_train_pred_woa)
r2_train_woa = r2_score(y_train, y_train_pred_woa)
print(f'Mean Squared Error (MSE) for Training Data: {mse_train_woa}')
print(f'R-squared for Training Data: {r2_train_woa}')

# Predict on the testing set
y_test_pred_woa, _ = gpr_model_woa.predict(X_test, return_std=True)

# Calculate regression plot for the testing data
plt.scatter(y_test, y_test_pred_woa)
plt.xlabel('True Values (Testing Data)')
plt.ylabel('Predictions (Testing Data)')
plt.title('Regression Plot for Testing Data with WOA-Optimized GPR')
plt.show()

# Calculate Mean Squared Error (MSE) and R-squared for the testing data
mse_test_woa = mean_squared_error(y_test, y_test_pred_woa)
r2_test_woa = r2_score(y_test, y_test_pred_woa)
print(f'Mean Squared Error (MSE) for Testing Data: {mse_test_woa}')
print(f'R-squared for Testing Data: {r2_test_woa}')


# In[ ]:




