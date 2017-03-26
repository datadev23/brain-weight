import pandas as pd
from sklearn import linear_model
import matplotlib.pyplot as plt

#read data
dataframe = pd.read_fwf('brain_body.txt')
x_values = dataframe[['Body']]
y_values = dataframe[['Brain']]
body_reg = linear_model.LinearRegression()
body_reg.fit(x_values,y_values)

#plot points on graph
plt.scatter(x_values, y_values)
plt.xlabel('Body',fontsize=14)
plt.ylabel('Brain',fontsize=14)
#plot the regression line
plt.plot(x_values,body_reg.predict(x_values))
plt.show()
