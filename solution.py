import numpy as np
from scipy.optimize import fmin_l_bfgs_b
import pdb


# load: open file and return the matrix of samples.
def load(name):
    # load data from file with passed name.
    data = np.loadtxt(name)
    # The matrix of samples is represented by the first n-1 columns.
    # The target variable is represented by the last column.
    X, y = data[:, :-1], data[:, -1].astype(np.int)
    return X, y


# h: predict the probability for class 1 based on the given sample
# and the given vector of weights theta.
# x and theta are row vectors.
def h(x, theta):
    # Compute value of logistic function.
    prob_1 = 1/(1 + np.exp(np.dot(theta, x)))
    # Return probability.
    return 1 - prob_1


# cost: compute the value of the cost function given feature matrix X, target variable
# values y, the weights theta and the regularization factor lambda.
def cost(theta, X, y, lambda_):
    return -1/X.shape[0] * np.sum([y[row_idx]*np.log(h(X[row_idx, :], theta)) + (1 - y[row_idx])*np.log(1 - h(X[row_idx, :], theta)) for row_idx in range(X.shape[0])])


# grad: the gradient of the cost function. Return a numpy vector that is the
# same size as the theta vector (vector of partial derivatives).
def grad(theta, X, y, lambda_):
    # Allocate array for result.
    res = np.zeros(len(theta))
    # Compute gradient values using derived forumla.
    for col_idx in range(len(theta)):
        res[col_idx] = -1/X.shape[0] * sum([(y[row_idx]  - h(X[row_idx, :], theta))*X[row_idx, col_idx] for row_idx in range(X.shape[0])])
    return res


# num_grad: compute the numeric gradient of the cost function. Return a numpy vector
# that is the same size as the vector theta. Use cost function to compute
# the numeric gradient.
def num_grad(theta, X, y, lambda_):
    # Allocate array for result.
    res = np.zeros(len(theta))
    # Define step in theta.
    dtheta = 1e-7
    # Compute gradient values using derivative definition.
    for row_idx in range(len(theta)):
        # Make a copy of the theta vector and take step in specified place.
        theta_aux = theta.copy()
        theta_aux[row_idx] = theta_aux[row_idx] + dtheta
        # Compute partial derivative using definition.
        res[row_idx] = (cost(theta_aux, X, y, lambda_) - cost(theta, X, y, lambda_)) / dtheta
    return res


# LogRegClassifier: Predict the class for the features variable vector.
# Return a list of probabilities for each class.
class LogRegClassifier(object):
    def __init__(self, th):
        self.th = th

    def __call__(self, x):
        # Append 1 to beginning of the features vector.
        x = np.hstack(([1.], x))
        # Compute probability of class 1 using the h function.
        p1 = h(x, self.th)
        # Return list of probabilities.
        return [1-p1, p1]


# LogRegLearner: build a prediction model for the training data X with classes y.
class LogRegLearner(object):

    # constructor
    def __init__(self, lambda_=0.0):
        self.lambda_ = lambda_

    # Calling the object
    def __call__(self, X, y):
        """
        Build a prediction model for learning data X with classes y.
        """
        # Horizontally stack a vector of ones with data matrix X.
        X = np.hstack((np.ones((len(X), 1)), X))

        # optimization - find values theta using the cost function and the gradient producing function.
        theta = fmin_l_bfgs_b(
            cost,
            x0=np.zeros(X.shape[1]),
            args=(X, y, self.lambda_),
            fprime=grad)[0]

        # Return the trained classifier.
        return LogRegClassifier(theta)


# test_learning: return prediction for the same samples that were used with learning.
# This is the wrong way to estimate larning success.
def test_learning(learner, X, y):
    """
    call example:
        res = test_learning(LogRegLearner(lambda_=0.0), X, y)
    """
    # Get trained classifier.
    c = learner(X,y)
    # Go over samples in X and classify them.
    # Save results to list and return it.
    results = [c(x) for x in X]
    return results


# test_cv: test the prediction success using 5-fold cross validation.
def test_cv(learner, X, y, k=5):
    pass


# CA: compute classification accuracy given a vector of real classes
# and a vector of predictions.
def CA(real, predictions):
    # Get number of correct classifications - equal to dot product
    num_correct = np.dot(real, predictions)
    return num_correct/len(real)


# AUC: measure the classification accuracy using the area under durve.
def AUC(real, predictions):
    pass

def data1():
    X = np.array([[5.0, 3.6, 1.4, 0.2],
                     [5.4, 3.9, 1.7, 0.4],
                     [4.6, 3.4, 1.4, 0.3],
                     [5.0, 3.4, 1.5, 0.2],
                     [5.6, 2.9, 3.6, 1.3],
                     [6.7, 3.1, 4.4, 1.4],
                     [5.6, 3.0, 4.5, 1.5],
                     [5.8, 2.7, 4.1, 1.0]])
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    return X, y

# If running as script...
if __name__ == "__main__":
    X, y = data1()
    learner = LogRegLearner(lambda_=0.)