#!/usr/bin/env python
# coding: utf-8

# In[27]:


import numpy as np
import pandas as pd

# Set seed for reproducibility
np.random.seed(42)

# Generate synthetic data for 1000 CFD points for training
cfd_train_data = pd.DataFrame({
    'fluid_velocity': np.random.uniform(7.5, 40, 1000),
    'fluid_density': np.random.uniform(1, 50, 1000),
    'particle_size': np.random.uniform(50, 400, 1000),
    'pipe_diameter': np.random.uniform(50.8, 304.8, 1000)
})

# Generate synthetic data for 100 CFD points for testing
cfd_test_data = pd.DataFrame({
    'fluid_velocity': np.random.uniform(7.5, 40, 100),
    'fluid_density': np.random.uniform(1, 50, 100),
    'particle_size': np.random.uniform(50, 400, 100),
    'pipe_diameter': np.random.uniform(50.8, 304.8, 100)
})

# Generate synthetic data for erosion rate for both training and testing datasets
max_erosion_rate_train = np.random.uniform(0.001, 0.1, 1000)
max_erosion_rate_test = np.random.uniform(0.001, 0.1, 100)

# Combine data into a single DataFrame for training and testing
erosion_dataset_train = pd.concat([cfd_train_data, pd.Series(max_erosion_rate_train, name='max_erosion_rate')], axis=1)
erosion_dataset_test = pd.concat([cfd_test_data, pd.Series(max_erosion_rate_test, name='max_erosion_rate')], axis=1)

# Save the datasets to CSV files
erosion_dataset_train.to_csv('cfd_erosion_dataset_train.csv', index=False)
erosion_dataset_test.to_csv('cfd_erosion_dataset_test.csv', index=False)

# Load the datasets
erosion_dataset_train = pd.read_csv('cfd_erosion_dataset_train.csv')
erosion_dataset_test = pd.read_csv('cfd_erosion_dataset_test.csv')


# In[40]:


from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

# Prepare features (X) and target variable (y) for training
X_train = erosion_dataset_train.drop('max_erosion_rate', axis=1)
y_train = erosion_dataset_train['max_erosion_rate']

# Define the Gaussian Process kernel
kernel = C(1, (1e-3, 1e3)) * RBF(1, (1e-1, 1e2))

# Create the Gaussian Process Regressor
gp_model = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, random_state=42)

# Fit the model to the training data
gp_model.fit(X_train, y_train)
print("Model trained successfully.")


# In[41]:


# Prepare features (X) and target variable (y) for testing
X_test = erosion_dataset_test.drop('max_erosion_rate', axis=1)
y_test = erosion_dataset_test['max_erosion_rate']

# Make predictions on the test set
y_pred, sigma = gp_model.predict(X_test, return_std=True)
print("Predictions made successfully.")


# In[42]:


from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Evaluate the model performance
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
print(f'Mean Squared Error: {mse}')
print(f'Root Mean Squared Error: {rmse}')
# Visualize the predictions
plt.scatter(y_test, y_pred)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--k', color='red', linewidth=2)
plt.xlabel('True Max Erosion Rate')
plt.ylabel('Predicted Max Erosion Rate')
plt.title('Gaussian Process Regression - Predicted vs True')
plt.show()
print("Visualization completed.")

# Save predictions to CSV for testing dataset
predictions_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
predictions_df.to_csv('gpr_predictions_test.csv', index=False)
print("Predictions saved to 'gpr_predictions_test.csv'.")



# In[ ]:




