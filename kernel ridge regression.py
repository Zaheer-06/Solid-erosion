#!/usr/bin/env python
# coding: utf-8

# In[1]:


pip install scikit-optimize


# In[4]:


import numpy as np
import pandas as pd
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Load the previously generated datasets
erosion_dataset_train = pd.read_csv("cfd_erosion_dataset_train.csv")
erosion_dataset_test = pd.read_csv("cfd_erosion_dataset_test.csv")

# Prepare features (X) and target variable (y) for training
X_train = erosion_dataset_train.drop('max_erosion_rate', axis=1)
y_train = erosion_dataset_train['max_erosion_rate']

# Prepare features (X) and target variable (y) for testing
X_test = erosion_dataset_test.drop('max_erosion_rate', axis=1)
y_test = erosion_dataset_test['max_erosion_rate']

# Define the Kernel Ridge Regression model
kernel_ridge_model = KernelRidge(alpha=1.0, kernel='rbf', gamma=None)

# Fit the model to the training data
kernel_ridge_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = kernel_ridge_model.predict(X_test)

# Evaluate the model performance
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
print(f'Mean Squared Error: {mse}')
print(f'Root Mean Squared Error: {rmse}')

# Visualize the predictions
plt.scatter(y_test, y_pred)
plt.xlabel("True Max Erosion Rate")
plt.ylabel("Predicted Max Erosion Rate")
plt.title("Kernel Ridge Regression: True vs Predicted")
plt.show()
predictions_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
predictions_df.to_csv('krr_predictions_test.csv', index=False)
print("Predictions saved to 'krr_predictions_test.csv'.")


# In[ ]:




