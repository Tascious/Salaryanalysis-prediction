from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

df= pd.read_csv("machine_learning_projects\Learning\Salary_Data.csv")
print(df.isnull().sum())
print(df.describe())
x=df["YearsExperience"]
y=df["Salary"]
print(x)
print(y)

#plotting data that is to be used,Understanding of the density and  spread of data
sns.regplot(x=x,y=y)
plt.xlabel("YearsExperience")
plt.ylabel("Salary")
plt.show()
sns.histplot(x,bins=5,kde=True)

#training of the model
model                         = LinearRegression()
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=123)
model.fit(x_train,y_train)
y_pred                         = model.predict(x_test)

print("R2 score:",r2_score(y_test,y_pred))
print("mean absolute error:",mean_absolute_error(y_test,y_pred))








