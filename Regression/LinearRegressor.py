from sklearn import linear_model
from sklearn import datasets
import matplotlib.pyplot as plt

input_data = datasets.load_iris().data
X, y = input_data[:, 2:3], input_data[:, 3:4]
training_samples = int(0.6 * len(X))

X_train, y_train = X[:training_samples], y[:training_samples]

X_test, y_test = X[training_samples:], y[training_samples:]

reg_linear = linear_model.LinearRegression()

reg_linear.fit(X_train, y_train)

y_test_pred = reg_linear.predict(X_test)

plt.scatter(X_test, y_test, color='red')
plt.plot(X_test, y_test_pred, color='black', linewidth=2)
plt.xticks(())
plt.yticks(())
plt.show()
