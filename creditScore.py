import numpy as np
import pandas as pd

# loading our datasets
train = pd.read_csv("D:/Data/train_u6lujuX_CVtuZ9i.csv")
test = pd.read_csv("D:/Data/test_Y3wMUE5_7gLdaTN.csv")

# list of column names
print(list(train))

# data sample
print(train.head())

# types of data columns
print(train.dtypes)

# summary statistics
print(train.describe())


# Data cleaning and preprocessing

# finding missing values
train.isnull().sum()
test.isnull().sum()

# Impute missing values with mean(numerical variables)
train.fillna(train.mean(), inplace=True)
train.isnull().sum()

# for test data
test.fillna(test.mean(), inplace=True)
test.isnull().sum()


# Impute missing values with mode(categorical variables)
train.Gender.fillna(train.Gender.mode()[0], inplace=True)
train.Married.fillna(train.Married.mode()[0], inplace=True)
train.Dependents.fillna(train.Dependents.mode()[0], inplace=True)
train.Self_Employed.fillna(train.Self_Employed.mode()[0], inplace=True)
train.isnull().sum()

# for test data
test.Gender.fillna(test.Gender.mode()[0], inplace=True)
test.Dependents.fillna(test.Dependents.mode()[0], inplace=True)
test.Self_Employed.fillna(test.Self_Employed.mode()[0], inplace=True)
test.isnull().sum()

# treatment for outliers
train.Loan_Amount_Term = np.log(train.Loan_Amount_Term)

# predictive modelling
# remove Loan_ID variable which is irrelevant
train = train.drop('Loan_ID', axis=1)
test = test.drop('Loan_ID', axis=1)

# create target variable
X = train.drop('Loan_Status', 1)
Y = train.Loan_Status

# Build dummy variables for categorical variables
X = pd.get_dummies(X)
train = pd.get_dummies(train)
test = pd.get_dummies(test)

# split train test data for cross validation
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

# 1. Logistic Regression Algorithm
# Fit model
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(x_train, y_train)

# predict values for test data
predict = model.predict(x_test)

# Evaluate accuracy of model
from sklearn.metrics import accuracy_score, confusion_matrix
print(accuracy_score(y_test, predict))
matrix = confusion_matrix(y_test, predict)
print(matrix)

# 2. Decision Tree Algorithm
# fit model
from sklearn import tree
dt = tree.DecisionTreeClassifier(criterion='gini')
dt.fit(x_train, y_train)

# predict values for test data
predict_1 = dt.predict(x_test)

# Evaluate accuracy of model
print(accuracy_score(y_test, predict_1))
matrix_1 = confusion_matrix(y_test, predict_1)
print(matrix_1)


# 3. Random Forest Algorithm
# fit model
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier()
rf.fit(x_train, y_train)

# predict values for test data
predict_2 = rf.predict(x_test)

# Evaluate accuracy of model
print(accuracy_score(y_test, predict_2))
matrix_2 = confusion_matrix(y_test, predict_2)
print(matrix_2)


# 4. Support Vector Machine Algorithm
# fit model
from sklearn import svm
svm_model = svm.SVC()
svm_model.fit(x_train, y_train)

# predict values for test data
predict_3 = svm_model.predict(x_test)

# Evaluate accuracy of model
print(accuracy_score(y_test, predict_3))
matrix_3 = confusion_matrix(y_test, predict_3)
print(matrix_3)


# 5. Naive Bayes Algorithm
# fit model
from sklearn.naive_bayes import GaussianNB
nb = GaussianNB()
nb.fit(x_train, y_train)

# predict values for test data
predict_4 = nb.predict(x_test)

# Evaluate accuracy of model
print(accuracy_score(y_test, predict_4))
matrix_4 = confusion_matrix(y_test, predict_4)
print(matrix_4)

# 6. K Nearest Neighbor Algorithm
# fit model
from sklearn.neighbors import KNeighborsClassifier
kNN = KNeighborsClassifier()
kNN.fit(x_train, y_train)

# predict values for test data
predict_5 = nb.predict(x_test)

# Evaluate accuracy of model
print(accuracy_score(y_test, predict_5))
matrix_5 = confusion_matrix(y_test, predict_5)
print(matrix_5)

# predict values using test data(naive bayes)
pred_test = nb.predict(test)

# write test results in csv file
predictions = pd.DataFrame(pred_test, columns=['predictions']).to_csv('Credit_Score_Predictions.csv')
