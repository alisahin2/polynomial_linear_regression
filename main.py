# -*- coding: utf-8 -*-
"""
Created on Sat Aug 22 01:55:16 2020

@author: Ali
"""

import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("polynominal_LR.csv")

y = df.car_max_speed.values.reshape(-1,1)
x = df.car_price.values.reshape(-1,1)

plt.scatter(x,y)
plt.ylabel("car_max_speed")
plt.xlabel("car_price")


#%% linear regression

from sklearn.linear_model import LinearRegression

lr = LinearRegression()

lr.fit(x,y)

#%% predict
y_head = lr.predict(x)

plt.plot(x,y_head, color = "red", label = "linear")

#%%
# polynominal regression y = b0 +b1*x + b2*x^2 + b3*x^3...+bn*x^n

from sklearn.preprocessing import PolynomialFeatures
polynomial_regression = PolynomialFeatures(degree = 4) # x^4 ye kadar olan kismi al demek

x_polynomial = polynomial_regression.fit_transform(x) # ust satirda hazirlanan islemin sonucunu gosterir(bir ust satirda ise sadece islemi gerceklestirir)

#%% fit

lr2 = LinearRegression()
lr2.fit(x_polynomial,y)

#%%

y_head2 = lr2.predict(x_polynomial)
plt.plot(x,y_head2, color = "green",label = "poly")
plt.legend()
plt.show()












