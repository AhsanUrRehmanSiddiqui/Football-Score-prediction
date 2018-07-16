import csv
import pandas as pd
import numpy as np
import datetime as dt
from sklearn.linear_model import LinearRegression
from sklearn import datasets
from sklearn.model_selection import train_test_split

#Taking data into dataframe and converting date into numeric value
df = pd.read_csv('data_v1.csv')
df['Date'] = pd.to_datetime(df['Date'])
df['Date']=df['Date'].map(dt.datetime.toordinal)

data = pd.DataFrame(df)
# use pd.concat to join the new columns with your original dataframe
df = pd.concat([df,pd.get_dummies(df['home'], prefix='Hteam')],axis=1)

# now drop the original 'home' column (you don't need it anymore)
df.drop(['home'],axis=1, inplace=True)
#print(pd.get_dummies(data))\

# use pd.concat to join the new columns with your original dataframe
df = pd.concat([df,pd.get_dummies(df['visitor'], prefix='Ateam')],axis=1)

# now drop the original 'home' column (you don't need it anymore)
df.drop(['visitor'],axis=1, inplace=True)
#print(pd.get_dummies(data))\

X = df
print(list(X))
X=X.drop(columns = ['hgoal'], axis=1)
X=X.drop(columns = ['vgoal'], axis=1)
print(list(X))

Y_hgoal = df['hgoal']
Y_vgoal = df['vgoal']

X_train, X_test, y_train, y_test = train_test_split(X, Y_hgoal, test_size=0.2, random_state=0)
model = LinearRegression()

model.fit(X_train, y_train)

print("Error:, " ,model.score(X_test, y_test))


print("Value of Y-intercept is:", model.intercept_)

test_result_home=model.predict(X_test)

for i in range(len(test_result_home)):
    test_result_home[i]=int(round(test_result_home[i]))

#######################################################

X_train, X_test, y_train, y_test = train_test_split(X, Y_vgoal, test_size=0.2, random_state=0)
model = LinearRegression()

model.fit(X_train, y_train)

print("Error:, " ,model.score(X_test, y_test))


print("Value of Y-intercept is:", model.intercept_)

test_result_visitor=model.predict(X_test)

for i in range(len(test_result_visitor)):
    test_result_visitor[i]=int(round(test_result_visitor[i]))

#print(test_result_home)
#print(test_result_visitor)

