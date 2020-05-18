from sklearn import linear_model
from sklearn import datasets
import sklearn.metrics as sm
from sklearn.preprocessing import PolynomialFeatures

input_data = datasets.load_iris().data
X, y = input_data[:, :-1], input_data[:, -1]
# X, y = input_data[:, :-1], datasets.load_iris().target

training_samples = int(0.6 * len(X))

X_train, y_train = X[:training_samples], y[:training_samples]

X_test, y_test = X[training_samples:], y[training_samples:]

reg_linear = linear_model.LinearRegression()

reg_linear.fit(X_train, y_train)

y_test_pred = reg_linear.predict(X_test)

print("Performance of Linear regressor:")
print("Mean absolute error =",
      round(sm.mean_absolute_error(y_test, y_test_pred), 2))
print("Mean squared error =",
      round(sm.mean_squared_error(y_test, y_test_pred), 2))
print("Median absolute error =",
      round(sm.median_absolute_error(y_test, y_test_pred), 2))
print("Explain variance score =",
      round(sm.explained_variance_score(y_test, y_test_pred), 2))
print("R2 score =", round(sm.r2_score(y_test, y_test_pred), 2))

############

polynomial = PolynomialFeatures(degree=10)
X_train_transformed = polynomial.fit_transform(X_train)

poly_linear_model = linear_model.LinearRegression()

poly_linear_model.fit(X_train_transformed, y_train)

poly_datapoint = polynomial.fit_transform(X_test)

y_test_pred = poly_linear_model.predict(poly_datapoint)

print("----------------------")

print("Performance of Linear regressor:")
print("Mean absolute error =",
      round(sm.mean_absolute_error(y_test, y_test_pred), 2))
print("Mean squared error =",
      round(sm.mean_squared_error(y_test, y_test_pred), 2))
print("Median absolute error =",
      round(sm.median_absolute_error(y_test, y_test_pred), 2))
print("Explain variance score =",
      round(sm.explained_variance_score(y_test, y_test_pred), 2))
print("R2 score =", round(sm.r2_score(y_test, y_test_pred), 2))
