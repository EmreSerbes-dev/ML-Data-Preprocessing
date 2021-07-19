import numpy as np #For data
import matplotlib.pyplot as plt #For graphics
import pandas as pd #For data
from sklearn.impute import SimpleImputer #For missing values
from sklearn.compose import ColumnTransformer #For OneHotEncode
from sklearn.preprocessing import OneHotEncoder #For OneHotEncode
from sklearn.preprocessing import LabelEncoder #For LabelEncode
from sklearn.model_selection import train_test_split #For splitting the dataset
from sklearn.preprocessing import StandardScaler #For feature scaling

#Reading dataset
dataset = pd.read_csv("Data.csv")  #dependent variable: Last column

#Creating matrix with column we chosen
x = dataset.iloc[:, :-1].values #All rows and all column without last column
y = dataset.iloc[:, -1].values  #All rows and only last column(dependent column)  # -1 is dependent column's index
z = dataset.iloc[2, :].values   #Only second rows and seconds row's all column

#Using imputer and closing nan values
imputer = SimpleImputer(missing_values = np.nan, strategy = "mean")
imputer.fit(x[:, 1:3])
x[:, 1:3] = imputer.transform(x[:, 1:3])

#Using OneHotEncoder for undependent and string variable
ct = ColumnTransformer(transformers = [('encoder', OneHotEncoder(), [0])], remainder = "passthrough") #transformer(('name of encode', class of encoder, [indexes of column])])
# NOTE: remainder: passthrough = keeps other of strings
x = np.array(ct.fit_transform(x))
# print(f"{x} \n")

#Using LabelEncoder for dependent and string variable
le = LabelEncoder()
y = le.fit_transform(y)
# print(f"{y} \n")

#Splitting the dataset
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 1) #1. parameter: Undependent variables cloumn matrix, 2. parameter:Dependent varible cloumn matrix, 3. parameter: test size, 4. parameter: random state
print(f"{x_train} \n")
print(f"{x_test} \n")
print(f"{y_train} \n")
print(f"{y_test} \n")

#Feature scaling
sc = StandardScaler()
x_train[:, 3:] = sc.fit_transform(x_train[:, 3:])
x_test[:, 3:] = sc.transform(x_test[:, 3:])
print(f"{x_train} \n")
print(f"{x_test} \n")

