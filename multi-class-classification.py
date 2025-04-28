#This is a multi class classification lab, originally performed in an online IDE for a coursera program
#For the purposes of archiving, and documenting my work, I have rewritten all the code into a format which will work in this codespace
#Most of this code does not function as intended in this environment

#importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsOneClassifier
from sklearn.metrics import accuracy_score

#loading dataset
file_path = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/GkDzb7bWrtvGXdPOfk6CIg/Obesity-level-prediction-dataset.csv"
data = pd.read_csv(file_path)
data.head()

#distribution of target variable
sns.countplot(y='NObeyesdad', data=data)
plt.title('Distribution of Obesity Levels')
plt.show()

#Exercise 1 - Check for null values, and display a summary of the dataset (use .info() and .describe() methods).
#checking for null values
print(data.isnull().sum())

#dataset summary
print(data.info())
print(data.describe())

#Exercise 2 -