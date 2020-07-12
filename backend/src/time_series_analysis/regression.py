import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, BayesianRidge
from sklearn.svm import SVR
import string

# Loading data
data = np.genfromtxt("data/passagierzahlen.csv", delimiter=",", dtype=str)
data = np.array([int(data[i][1].translate(str.maketrans('', '', string.punctuation)))
                 for i in range(2, data.shape[0]-2)])

x = np.linspace(0, data.shape[0]-1, data.shape[0])

y = np.reshape(data/1000000, (-1, 1))
x = np.reshape(x, (-1, 1))

# Linear Regression
reg_lin = LinearRegression().fit(x, y)
pred_lin = reg_lin.predict(x)

# SVM: rbf
reg_svr = SVR(kernel='rbf', C=100, gamma=.1, epsilon=.3).fit(x, np.ravel(y))
pred_svr = reg_svr.predict(x)

# Baysian Ridge
reg_ridge = BayesianRidge(tol=1e-6, compute_score=True)
reg_ridge = reg_ridge.fit(x, np.ravel(y))
pred_ridge = reg_ridge.predict(x)


plt.figure()
plt.xlabel("Months from start (01.01.2009)")
plt.ylabel("Passengers in Mio.")
#plt.plot(x, pred_lin, c="red")
plt.plot(x, pred_svr, c="orange")
#plt.plot(x, pred_ridge, c="black")
#plt.scatter(x, y)
plt.show()