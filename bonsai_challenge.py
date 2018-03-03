# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
# Define functions needed for analysis
#find missing variables
def missing(dataset):
    print(dataset.apply(lambda x: sum(x.isnull().values), axis = 0))

def frequency(dataset):
        for col in dataset:
            print(dataset.groupby(col).size())

# Import necessary packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import preprocessing
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from matplotlib.colors import ListedColormap
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
import seaborn as sns


# Bring in training dataset
train_data=pd.read_csv("/Users/geoffrey.kip/Projects/customer_churn_ml/train.csv")
test_data=pd.read_csv("/Users/geoffrey.kip/Projects/customer_churn_ml/test.csv")
# Check column types
train_data.dtypes
test_data.dtypes
#check missing values
missing(train_data)
missing(test_data)

# Exploratory Data Analysis
# Out of the females that are independent, what percentage of them have a monthly charge of greater than $20?
independent_females= train_data[(train_data['Dependents']== "No") & (train_data['gender'] == "Female")]
print(float(len(independent_females[independent_females['MonthlyCharges'] > 20]))/float(len(independent_females)))

# 92 % of all independent females have a total monthly charges greater than $20

eda_dataset= train_data.set_index(['customerID'])

#check frequency distribution of all categorical variables
frequency(eda_dataset.select_dtypes(include=[np.object]))

#Print countplots for all categorical variables
for i, col in enumerate(eda_dataset.select_dtypes(include=[np.object]).columns):
    plt.figure(i)
    sns.countplot(x=col, data=eda_dataset)
    
# Print distribuitons for all continuos variables
for i, col in enumerate(eda_dataset.select_dtypes(include=[np.float64,np.int64]).columns):
    plt.figure(i)
    sns.distplot(eda_dataset[col])

# Compare label against some variables
eda_dataset['Churn']= np.where(eda_dataset["Churn"] == "Yes",1,0)
sns.barplot(x="Partner", y="Churn",hue="gender", data=eda_dataset)
sns.barplot(x="SeniorCitizen", y="Churn",hue="gender", data=eda_dataset)
sns.barplot(x="Dependents", y="Churn",hue="gender", data=eda_dataset)

#Seperate labels from other features
labels=train_data[["customerID", "Churn"]]
labels=labels.set_index(['customerID'])
labels["Churn"]= np.where(labels["Churn"] == "Yes",1,0)

train_data= train_data.drop(["Churn"],axis=1)

# Convert Senior Citizen to string for one hot encoding later
train_data['SeniorCitizen']=np.where(train_data['SeniorCitizen'] == 1,"Yes","No")
test_data['SeniorCitizen']=np.where(test_data['SeniorCitizen'] == 1,"Yes","No")


#Set customerID as the index
#train_data= train_data.set_index(['customerID'])

#Prep train data for modeling
scaler = StandardScaler()
train_quant= train_data.select_dtypes(include=[np.float64,np.int64])
train_quant  = pd.DataFrame(scaler.fit_transform(train_quant),columns=train_quant.columns)
train_categorical= train_data.select_dtypes(include=[np.object])
train_categorical= train_categorical.set_index(['customerID'])
train_categorical= pd.get_dummies(train_categorical)
train_categorical= train_categorical.reset_index(['customerID'])
train_features= pd.concat([train_categorical, train_quant], axis=1)
train_features= train_features.set_index(['customerID'])

#Prep test data for prediction later
scaler = StandardScaler()
test_quant= test_data.select_dtypes(include=[np.float64,np.int64])
test_quant  = pd.DataFrame(scaler.fit_transform(test_quant),columns=test_quant.columns)
test_categorical= test_data.select_dtypes(include=[np.object])
test_categorical= test_categorical.set_index(['customerID'])
test_categorical= pd.get_dummies(test_categorical)
test_categorical= test_categorical.reset_index(['customerID'])
test_data= pd.concat([test_categorical, test_quant], axis=1)
test_data= test_data.set_index(['customerID'])


#Split data into training and test sets
test_size = 0.30
#validation_size=0.20
seed = 7
X_train, X_test, Y_train, Y_test = train_test_split(train_features, labels, test_size=test_size, random_state=seed)
print (X_train.shape, Y_train.shape)
#print (X_validation.shape, Y_validation.shape)
print (X_test.shape, Y_test.shape)

#Evaluate different models

seed = 7
scoring = 'accuracy'
# Evaluate training accuracy
models = []
models.append(('LR', LogisticRegression()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('RANDOM FOREST', RandomForestClassifier()))
models.append(('SVM', SVC()))
models.append(('Gradient Boosting', GradientBoostingClassifier()))
models.append(('Naive Baiyes', GaussianNB()))

results = []
names = []
for name, model in models:
	kfold = model_selection.KFold(n_splits=10, random_state=seed)
	cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
	results.append(cv_results)
	names.append(name)
	msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
	print(msg)

#==============================================================================
# LR: 0.802435 (0.017573)
# KNN: 0.761616 (0.016895)
# CART: 0.724569 (0.025687)
# RANDOM FOREST: 0.770483 (0.015216)
# SVM: 0.766180 (0.016633)
# Gradient Boosting: 0.800408 (0.017902)
# 
#==============================================================================

# Compare Algorithms
fig = plt.figure(figsize=(10,10))
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()


# Fit and Evaluate with the best model

model =  GradientBoostingClassifier()
model.fit(X_train, Y_train)

feat_imp = pd.Series(model.feature_importances_, X_train.columns).sort_values(ascending=False)
fig = plt.figure(figsize=(15,10))
feat_imp.plot(kind='bar', title='Feature Importances')
plt.ylabel('Feature Importance Score')

best_features= feat_imp[feat_imp > 0]

predictions = model.predict(X_test)
predictions_probs = model.predict_proba(X_test)
preds = np.where(predictions_probs[:,1] >= 0.5 , 1, 0)
print(accuracy_score(Y_test, preds))
print(confusion_matrix(Y_test, preds))
print(classification_report(Y_test, preds))
score_test = metrics.f1_score(Y_test, preds,
                          pos_label=list(set(Y_test)), average = None)


# Predict on the given test set
churn_predictions= model.predict(test_data)
test_data= test_data.reset_index()
customer_id=pd.DataFrame(test_data['customerID'])
final_data=pd.merge(customer_id, pd.DataFrame(churn_predictions), how='left', left_index=True, right_index=True)