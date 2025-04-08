#This is a multiple linear regression lab, originally performed in an online IDE for a coursera program
#For the purposes of archiving, and documenting my work, I have rewritten all the code into a format which will work in this codespace
#Most of this code does not function as intended in this environment

#importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#%matplotlib inline

#loading dataset with pandas library
url= "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%202/data/FuelConsumptionCo2.csv"
df = pd.read_csv(url)

#verify successful load with randomly selected records
df.sample(5)

df.describe()

#drop categoricals and useless columns
df = df.drop(['MODELYEAR', 'MAKE', 'MODEL', 'VEHICLECLASS', 'TRANSMISSION', 'FUELTYPE',],axis=1)
#analyze levels of correlation
df.corr()
#drop particular categories based on correlation
df = df.drop(['CYLINDERS', 'FUELCONSUMPTION_CITY', 'FUELCONSUMPTION_HWY','FUELCONSUMPTION_COMB',],axis=1)
df.head(9)

#show scatter plots for each pair of input features
axes = pd.plotting.scatter_matrix(df, alpha=0.2)
#need to rotate axis labels so we can read them
for ax in axes.flatten():
    ax.xaxis.label.set_rotation(90)
    ax.yaxis.label.set_rotation(0)
    ax.yaxis.label.set_ha('right')

plt.tight_layout()
plt.gcf().subplots_adjust(wspace=0, hspace=0)
plt.show()

#extract the input features and labels from the data set
X = df.iloc[:,[0,1]].to_numpy()
y = df.iloc[:,[2]].to_numpy()

