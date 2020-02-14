import pickle
import sklearn.preprocessing
import sklearn.linear_model
import numpy as np
from matplotlib import pyplot


# Import data from pickles
with open("Q2_data/Fx_test.pkl", "rb") as f:
    Fx_test = pickle.load(f)

with open("Q2_data/X_test.pkl", "rb") as f:
    X_test = pickle.load(f)

with open("Q2_data/X_train.pkl", "rb") as f:
    X_train = pickle.load(f)

with open("Q2_data/Y_train.pkl", "rb") as f:
    Y_train = pickle.load(f)


# Reshape X_test and Fx_test into column vectors
X_test = X_test.reshape(-1, 1)
Fx_test = Fx_test.reshape(-1, 1)

# Arrays for plotting using matplotlib
bias_squared_array = []
variance_array = []
model_complexity = range(1, 10)


# Iterate over degree from 1 to 9
for deg in range(1, 10):

    mean_of_predictions = np.zeros(Fx_test.shape)
    mean_square_predictions = np.zeros(Fx_test.shape)

    # Iterate over the models in the training set
    for model in range(20):

        # Get the part of the training data we need in this iteration
        x_data = X_train[model, :]
        y_data = Y_train[model, :]

        # Reshape the data into column vector format
        x_data = x_data.reshape(-1, 1)
        y_data = y_data.reshape(-1, 1)

        # Convert x_data into polynomial features in order to use linear regression
        poly = sklearn.preprocessing.PolynomialFeatures(deg)
        x_data = poly.fit_transform(x_data)

        # Fit a regression model to the data
        regression_model = sklearn.linear_model.LinearRegression().fit(x_data, y_data)

        # Convert the test data into a polynomial features in order to use the regression model
        poly_X_test = poly.fit_transform(X_test)

        # Make predictons on the test data using the regression model
        predictions = regression_model.predict(poly_X_test)

        # Update the running sum of square of predictions and mean of predictions
        mean_square_predictions = np.add(
            mean_square_predictions, np.multiply(predictions, predictions))
        mean_of_predictions = np.add(mean_of_predictions, predictions)

    print("Degree of the polynomial: " + str(deg))

    # Calculate bias from mean of predictions and Fx_test
    mean_of_predictions = np.multiply(mean_of_predictions, (1/20))
    bias = np.subtract(mean_of_predictions, Fx_test)
    bias_squared = np.multiply(bias, bias)
    bias_squared = np.mean(bias_squared)
    print("bias squared is ", bias_squared)
    bias_squared_array.append(bias_squared)

    # Calculate variance from mean square predictions and mean of predictions
    mean_square_predictions = np.multiply(mean_square_predictions, (1/20))
    variance = np.subtract(mean_square_predictions, np.multiply(
        mean_of_predictions, mean_of_predictions))
    variance = np.mean(variance)
    print("variance is ", variance)
    print("")
    variance_array.append(variance)

    # Create a scatter plot of the test data
    pyplot.scatter(X_test, Fx_test, label = "actual value")

    # Plot the learnt model
    pyplot.plot(X_test, mean_of_predictions, "r*", label = "predicted value")

    # Show comparision between trained model and test data
    pyplot.title("Regression for degree " + str(deg))
    pyplot.legend()
    pyplot.show()

# Plot on a graph
pyplot.plot(model_complexity, bias_squared_array, "b+-", label="Bias ^ 2")
pyplot.plot(model_complexity, variance_array, "r*-", label="Variance")
pyplot.legend()
pyplot.xlabel("Model complexity")
pyplot.title("Bias^2 and variance against model complexity")
pyplot.show()
