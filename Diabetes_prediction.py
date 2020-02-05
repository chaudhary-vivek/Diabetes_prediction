# Part1: Preprocessing

# Importing required libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sklearn.impute
import sklearn.ensemble

# Importing dataset
data = pd.read_csv("pima-data.csv")

# Checking dataset for null values
print(data.isnull().values.any())

# Checking correlation between deifferent variables using correlation matrix
import seaborn as sns
import matplotlib.pyplot as plt
# Getting correlations of each features in dataset
corrmat = data.corr()
top_corr_features = corrmat.index
plt.figure(figsize=(20,20))
# Plotting heatmap of correlation values)
g=sns.heatmap(data[top_corr_features].corr(),annot=True,cmap="RdYlGn")
# Glucocose, BMI, age and num_preg are top 4 most correlated features

# Changing diabetes variable from boolean to numerical
# Cresting a dictionary for defining variables
diabetes_map = {True: 1, False: 0}
data['diabetes'] = data['diabetes'].map(diabetes_map)

# Checking if dataset is balance
diabetes_true_count = len(data.loc[data['diabetes']==True])
diabetes_false_count = len(data.loc[data['diabetes']==False])
# There are 268 true values and 500 false values. The dataset is not well balanced

# Splitting data into training set and testing set
# Impoting required library
from sklearn.model_selection import train_test_split
# Defining independent and dependent variables (Feature and target)
feature_columns = ['num_preg', 'glucose_conc', 'diastolic_bp', 'insulin', 'bmi', 'diab_pred', 'age', 'skin']
predicted_class = ['diabetes']
X = data[feature_columns].values
y = data[predicted_class].values
# Setting aside 30% of values as testing data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30, random_state = 10)

# Checking for the number of missing values in the dataset
print("total number of rows : {0}".format(len(data)))
# .format appends output of the trailing code to the string while printing
print("number of rows missing glucose_conc: {0}".format(len(data.loc[data['glucose_conc'] == 0])))
print("number of rows missing glucose_conc: {0}".format(len(data.loc[data['glucose_conc'] == 0])))
print("number of rows missing diastolic_bp: {0}".format(len(data.loc[data['diastolic_bp'] == 0])))
print("number of rows missing insulin: {0}".format(len(data.loc[data['insulin'] == 0])))
print("number of rows missing bmi: {0}".format(len(data.loc[data['bmi'] == 0])))
print("number of rows missing diab_pred: {0}".format(len(data.loc[data['diab_pred'] == 0])))
print("number of rows missing age: {0}".format(len(data.loc[data['age'] == 0])))
print("number of rows missing skin: {0}".format(len(data.loc[data['skin'] == 0])))
# There are negligible number of zero values in most variables, there are 374 zero values in insulin and 227 zero values in skin

# Replacing the zero values with the mean value of the feature
# Importing imputer from scikit learn to do the replacing
from sklearn.impute import SimpleImputer 
# missing_values arguement defines what is to be replaced, strategy defines what it is to be replaced with, and axis=0 defines axis
fill_values = SimpleImputer(missing_values=0, strategy="mean")
X_train = fill_values.fit_transform(X_train)
X_test = fill_values.fit_transform(X_test)

# Part 2: Making the model

# Importing random forest classifier from scikit learn
from sklearn.ensemble import RandomForestClassifier
# Making the model
random_forest_model = RandomForestClassifier(random_state=10)
random_forest_model.fit(X_train, y_train.ravel())

# Part 3: Evaluation and prediction

# Predicting the values with the testing data
predict_train_data = random_forest_model.predict(X_test)
# Evaluating the model
from sklearn import metrics
print("Accuracy = {0:.3f}".format(metrics.accuracy_score(y_test, predict_train_data)))
# The accuracy is 73.6%
