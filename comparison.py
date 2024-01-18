#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt

# Load predictions from Kernel Ridge Regression
krr_predictions = pd.read_csv('krr_predictions_test.csv')

# Load predictions from Gaussian Process Regression
gpr_predictions = pd.read_csv('gpr_predictions_test.csv')

# Scatter plot for Kernel Ridge Regression
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.scatter(krr_predictions['Actual'], krr_predictions['Predicted'])
plt.xlabel('True Max Erosion Rate')
plt.ylabel('Predicted Max Erosion Rate')
plt.title('Kernel Ridge Regression: True vs Predicted')

# Scatter plot for Gaussian Process Regression
plt.subplot(1, 2, 2)
plt.scatter(gpr_predictions['Actual'], gpr_predictions['Predicted'])
plt.xlabel('True Max Erosion Rate')
plt.ylabel('Predicted Max Erosion Rate')
plt.title('Gaussian Process Regression: True vs Predicted')

plt.tight_layout()
plt.show()


# In[2]:


import pandas as pd
import matplotlib.pyplot as plt

# Load predictions from Kernel Ridge Regression
krr_predictions = pd.read_csv('krr_predictions_test.csv')

# Load predictions from Gaussian Process Regression
gpr_predictions = pd.read_csv('gpr_predictions_test.csv')

# Scatter plot for both models on the same graph
plt.figure(figsize=(10, 6))

# Kernel Ridge Regression in blue
plt.scatter(krr_predictions['Actual'], krr_predictions['Predicted'], label='KRR', color='blue', alpha=0.7)

# Gaussian Process Regression in red
plt.scatter(gpr_predictions['Actual'], gpr_predictions['Predicted'], label='GPR', color='red', alpha=0.7)

plt.plot([min(krr_predictions['Actual'].min(), gpr_predictions['Actual'].min()),
          max(krr_predictions['Actual'].max(), gpr_predictions['Actual'].max())],
         [min(krr_predictions['Actual'].min(), gpr_predictions['Actual'].min()),
          max(krr_predictions['Actual'].max(), gpr_predictions['Actual'].max())],
         '--k', color='black', linewidth=2, label='Ideal Prediction')

plt.xlabel('True Max Erosion Rate')
plt.ylabel('Predicted Max Erosion Rate')
plt.title('Comparison of KRR and GPR: True vs Predicted')
plt.legend()
plt.show()


# In[ ]:




