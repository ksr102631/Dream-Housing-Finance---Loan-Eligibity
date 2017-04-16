# -*- coding: utf-8 -*-
"""
Created on Wed Apr 12 22:47:28 2017

@author: sreeram
"""


## Data sets
# train = pd.read_csv("https://s3-ap-southeast-1.amazonaws.com/av-datahack-datacamp/train.csv")
# test = pd.read_csv("https://s3-ap-southeast-1.amazonaws.com/av-datahack-datacamp/test.csv")

import pandas as pd

train = pd.read_csv("C:/Users/sreeram/RAM_PY_FILES/train_loan.csv")

test = pd.read_csv("C:/Users/sreeram/RAM_PY_FILES/test_loan.csv")

train.head(5)

# For understanding continuos variables

train.describe()

# For understanding categorical variables

train.Property_Area.value_counts()

train.Education.value_counts()

train.Self_Employed.value_counts()

# Understanding distribution of numerical variables

# Histogram

train.ApplicantIncome.hist(bins = 50)

# Boxplot

train.boxplot(column = "ApplicantIncome")


train.LoanAmount.hist(bins = 50)

train.boxplot(column = "LoanAmount")


## Understanding distribution of categorical variables 

# Use value_counts() with train['LoanStatus'] to look at the frequency distribution

#Use crosstab with LoanStatus and CreditHistory to perform bi-variate analysis

pd.crosstab(train.Gender,train.Loan_Status,margins = True)

pd.crosstab(train.Property_Area,train.Loan_Status,margins = True)


pd.crosstab(train.Self_Employed,train.Loan_Status,margins = True)

pd.crosstab(train.Married,train.Loan_Status,margins = True) 


## See the missing values

train.Credit_History.isnull().sum()

# How many variables have missing values?

train.isnull().sum()

# Imputing missing values of LoanAmount

train.LoanAmount.fillna(train.LoanAmount.mean(),inplace = True)

## Checking any NA's after imputing

train.LoanAmount.isnull().sum()

## Impute missing values of SelfEmployed

#Similarly, to impute missing values of Categorical variables, 
#we look at the frequency table. The simplest way is to impute
# with value which has highest frequency because there is a higher probability of success.

train.Self_Employed.value_counts()

train.Self_Employed.isnull().sum()

## Fill NA's

train.Self_Employed.fillna("No",inplace =True)

train.Self_Employed.value_counts()

train.Self_Employed.isnull().sum()

## Similarly filling in Gender and  Credit_History

train.Gender.value_counts()

train.Gender.fillna("Male",inplace =True)


train.Gender.value_counts()

train.Credit_History.value_counts()

train.Credit_History.fillna(1,inplace =True)


train.Credit_History.value_counts()

train.isnull().sum()


# Treat / Tranform extreme values of LoanAmount and ApplicantIncome          

train.LoanAmount.hist(bins =50)

import numpy as np

train.LoanAmount_log = np.log(train.LoanAmount)


train.LoanAmount_log.hist(bins =50)

train.TotalIncome = train.ApplicantIncome+ train.CoapplicantIncome

train.TotalIncome.hist(bins =50)


train.TotalIncome_log = np.log(train.TotalIncome)

train.TotalIncome_log.hist(bins =50)      


## Treating categorical variables for scikit learn
      
from sklearn.preprocessing import LabelEncoder

train.Gender.value_counts()

number = LabelEncoder()

train.Gender = number.fit_transform(train.Gender.astype(str))

train.Gender.value_counts()

train['Married_new'] = number.fit_transform(train['Married'].astype(str))

train["Property_Area_new"] = number.fit_transform(train.Property_Area)

train["Property_Area_new"].value_counts()

train.Property_Area.isnull().sum()

## Model Building

## Use Logistic Regression, Decision Tree, Random Forest

import sklearn.linear_model

model = sklearn.linear_model.LogisticRegression()


train.Education.value_counts()

train["Education"] = number.fit_transform(train["Education"].astype(str))

train["Education"].value_counts()

train.head(2)     

## Building model

predictors = ["Credit_History","Education","Gender"]

# Converting predictors and outcome to numpy array

x_train = train[predictors].values

x_train.shape()

y_train = train["Loan_Status"].values

y_train

# Model Building

model = sklearn.linear_model.LogisticRegression()

model.fit(x_train,y_train) # Model is bulit


## Testing the model on Test data

## Treating categorical variables in Test Data 

test = pd.read_csv("C:/Users/sreeram/RAM_PY_FILES/test_loan.csv")

## Treating missing values

test.Credit_History.value_counts()

test.Credit_History.fillna(1,inplace =True)

test.Education.value_counts()

test.Education.isnull().sum()

test.Gender.isnull().sum()

test.Gender.fillna("Male",inplace =True)

test.isnull().sum()

## Treating categorical variables(Encoding them)

test.Gender = number.fit_transform(test.Gender.astype(str))

test.Education = number.fit_transform(test.Education.astype(str))



predictors = ["Credit_History","Education","Gender"]

x_test = test[predictors].values

#Predict Output

predicted = model.predict(x_test)

predicted

## Reverse encoding for predicted outcome

predicted = number.inverse_transform(predicted)

test["Loan_Status"] = predicted

test.to_csv("Submission.csv", columns = ["Loan_ID","Loan_Status"])


test.to_csv("Submission1.csv", columns = ["Loan_Status"])

test.to_csv("Test_new.csv")               





 

    



         


























