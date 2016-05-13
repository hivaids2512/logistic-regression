import numpy as np
import pandas as pd
import seaborn as sb
from sklearn.cross_validation import train_test_split
from sklearn import linear_model
from sklearn import metrics

data = pd.read_csv('students.csv')
features = ['gre', 'gpa', 'rank']
target = ['admit']
X=data[features].values
y=data[target].values.ravel()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.25)
regr = linear_model.LogisticRegression()
regr.fit(X_train, y_train)

print regr.predict_proba(X_test)
print regr.score(X_test, y_test)
print('Coefficients: \n', regr.coef_)
print('Intercept: \n', regr.intercept_)

sb_plot = sb.pairplot(data, x_vars=['gre', 'gpa', 'rank'], y_vars='admit', size=10, aspect=0.7, kind='reg')
sb_plot.savefig("chart.png")


