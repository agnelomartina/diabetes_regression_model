from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

from sklearn.metrics import mean_squared_error
import pandas as pd
import matplotlib.pyplot as plt

diabetes=load_diabetes()
X=diabetes.data
Y=diabetes.target
print(X,Y)

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.3,random_state=42)
model=LinearRegression()
model.fit(X_train,Y_train)

y_pred=model.predict(X_test)
print(y_pred)
#since its a regression dataset we can do r2_score,mean absolute error,mean square error
#r2_score
r2 = r2_score(Y_test, y_pred)
print("R² Score:", r2)

#mean squared error
mse = mean_squared_error(Y_test, y_pred)
print("MSE:", mse)

#mean squared error
mse = mean_squared_error(Y_test, y_pred)
print("MSE:", mse)

#visualization plots for linear regression model
#assigning the dataframe
diabetes = load_diabetes()
X = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
Y = pd.DataFrame(diabetes.target)
bmi_df=X[['bmi']]
print("bmi",bmi_df)
print("@@@@@@@@@@@@@@@@@@@@@@")
print("target data",Y)
print("$$$$$$$$$$$$$$$$$$$$$$$$$$$")
bmi_target_df = pd.concat([bmi_df, Y], axis=1)
print(bmi_target_df.head())
#histograms
bmi_target_df.hist(bins=15, figsize=(15,10), color='skyblue')
plt.suptitle("Feature Distributions", y=1.02)
plt.show()


