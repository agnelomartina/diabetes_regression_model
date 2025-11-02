from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error
import pandas as pd
import matplotlib.pyplot as plt
import pickle

diabetes=load_diabetes()
X=diabetes.data
Y=diabetes.target


X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.3,random_state=42)
model=DecisionTreeRegressor(criterion='squared_error',max_depth=3)
model.fit(X_train,Y_train)
#prediction
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
X1 = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
Y1 = pd.DataFrame(diabetes.target,columns=['target'])

bmi_df=X1[['bmi']]

print(bmi_df,Y)
model1=LinearRegression()
model1.fit(bmi_df,Y1)

#cross validation
cv_r2_scores_lr1 = cross_val_score(model1, bmi_df, Y1, cv=5, scoring='r2')
print("cross validation for model1",cv_r2_scores_lr1)

#histograms
df = pd.concat([bmi_df, Y1], axis=1)
df.hist(bins=15, figsize=(10,5), color='skyblue')
plt.suptitle("BMI vs Target", y=1.02)
plt.show()

#boxplot
comparison_df = pd.DataFrame({
    'Actual': Y_test,
    'Predicted': y_pred
})
plt.figure(figsize=(10, 8))
comparison_df.boxplot(column=['Actual', 'Predicted'], grid=False, color=dict(boxes='green', medians='red'))
plt.title("Boxplot: Actual vs Predicted Values")
plt.ylabel("Diabetes Target")
plt.show()

with open("diabetes_linear_trained_model.pkl","wb")as file_obj:
    pickle.dump(model1,file_obj)