# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Importing the dataset
dataset = pd.read_csv('C:/Users/LM/Desktop/NUS Documents/School Notes/AD Project/AD Project Inception/dataset.csv')
dataset = dataset.replace(to_replace=['yes', 'no'], value=['1', '0'])
X = dataset.iloc[:, :-1]
y = dataset.iloc[:, -1]

# Training the Logistic Regression model on the Training set
log_reg = LogisticRegression(random_state=0)
log_reg.fit(X, y)

# Serialize the Object
pickle.dump(log_reg, open('PythonLogRegModel.pkl', 'wb'))  #serialize the object