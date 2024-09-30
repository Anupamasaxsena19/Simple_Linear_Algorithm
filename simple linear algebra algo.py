import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,r2_score

#create a synthetic dataset
np.random.seed(0)
x=2*np.random.rand(100,1)
y=4+3*x+np.random.randn(100,1)

#convert to DataFrame
data=pd.DataFrame(np.hstack((x,y)),columns=('x','y'))
print(data.head())

#split the dataset
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=42)

#create and train the model
model=LinearRegression()
model.fit(x_train,y_train)

#Make predictions
y_pred=model.predict(x_test)

#evaluate the model
mse=mean_squared_error(y_test,y_pred)
r2=r2_score(y_test,y_pred)
print(f"Mean Squared Error:(mse:.2f)")
print(f"R_Squared:(r2:.2f)")

#Plotting the result
plt.scatter(x_test, y_test, color='blue', label='Actual data')
plt.plot(x_test,y_pred, color='black',linewidth=2,label='predicted line')
plt.title('Simple Linear Regression')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()
