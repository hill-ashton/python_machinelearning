#This is an evaluating classification models lab, originally performed in an online IDE for a coursera program
#For the purposes of archiving and documenting my work, I have rewritten all the code into a format which will work in this codespace
#Most of this code does not function as intended in this environment

#importing libraries
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

#load data set
data = load_breast_cancer()
X, y = data.data, data.target
labels = data.target_names
feature_names = data.feature_names

print(data.DESCR)
print(data.target_names)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

#adding noise to simulate random measurement error
#add Gaussian noise
np.random.seed(42)  #for reproducibility
noise_factor = 0.5 #adjust this to control the amount of noise
X_noisy = X_scaled + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=X.shape)

#load the original and noisy data sets into a DataFrame for comparison and visualization
df = pd.DataFrame(X_scaled, columns=feature_names)
df_noisy = pd.DataFrame(X_noisy, columns=feature_names)

#display the first few rows of the standardized original and noisy data sets for comparison
print("Original Data (First 5 rows):")
df.head()
print("\nNoisy Data (First 5 rows):")
df_noisy.head()