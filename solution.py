import math

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import fmin_l_bfgs_b

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
    prob_1 = 1/(1 + np.exp(np.dot(theta, x)))  # Compute value of logistic function.
    return 1 - prob_1  # Return probability.


# cost: compute the value of the cost function given feature matrix X, target variable
# values y, the weights theta and the regularization factor lambda.
def cost(theta, X, y, lambda_):
    return -1/X.shape[0] * np.sum([y[row_idx]*np.log(h(X[row_idx, :], theta)) + (1 - y[row_idx])*np.log(1 - h(X[row_idx, :], theta)) for row_idx in range(X.shape[0])]) + lambda_/X.shape[0] * sum(theta)**2


# grad: the gradient of the cost function. Return a numpy vector that is the
# same size as the theta vector (vector of partial derivatives).
def grad(theta, X, y, lambda_):
    res = np.zeros(len(theta))  # Allocate array for result.
    for col_idx in range(len(theta)):  # Compute gradient values using derived formula.
        res[col_idx] = -1/X.shape[0] * sum([(y[row_idx] - h(X[row_idx, :], theta)) * X[row_idx, col_idx] for row_idx in range(X.shape[0])]) + (2*lambda_)/X.shape[0] * theta[col_idx]
    return res


# num_grad: compute the numeric gradient of the cost function. Return a numpy vector
# that is the same size as the vector theta. Use cost function to compute
# the numeric gradient.
def num_grad(theta, X, y, lambda_):
    res = np.zeros(len(theta))  # Allocate array for result.
    dtheta = 1e-7  # Define step in theta.
    for row_idx in range(len(theta)):  # Compute gradient values using derivative definition.
        theta_aux = theta.copy()  # Make a copy of the theta vector and take step in specified place.
        theta_aux[row_idx] = theta_aux[row_idx] + dtheta  # Compute partial derivative using definition. Add derivative of regularization term.
        res[row_idx] = (cost(theta_aux, X, y, 0) - cost(theta, X, y, 0)) / dtheta + (2*lambda_)/X.shape[0] * theta[row_idx]
    return res


# LogRegClassifier: Predict the class for the features variable vector.
# Return a list of probabilities for each class.
class LogRegClassifier(object):
    def __init__(self, th):
        self.th = th

    def __call__(self, x):
        x = np.hstack(([1.], x))  # Append 1 to beginning of the features vector.
        p1 = h(x, self.th)  # Compute probability of class 1 using the h function.
        return [1-p1, p1]  # Return list of probabilities.


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
    c = learner(X, y)
    # Go over samples in X and classify them.
    # Save results to list and return it.
    results = [c(x) for x in X]
    return results


# test_cv: test the prediction success using k-fold cross validation.
def test_cv(learner, X, y, k=5):
    # Shuffle the data matrix.
    X_aux = np.hstack((X, np.reshape(y, [len(y), 1])))
    # Append indices of rows to save information about the shuffling order.
    X_aux = np.hstack((np.reshape(range(X_aux.shape[0]), [X_aux.shape[0], 1]), X_aux))
    X_aux = np.random.permutation(X_aux)
    y_perm = X_aux[:, -1]       # Get reshuffled target variable values.
    X_perm = X_aux[:, 1:-1]     # Get reshuffled feature vectors matrix.
    # Get information about shuffle ordering.
    perm = X_aux[:, 0]
    num_in_group = math.ceil(X_perm.shape[0]/k)  # Compute maximum number of samples in each group.
    pred = np.empty((0, 2), dtype=float)  # Allocate starting array for prediction results matrix.
    # Go over example sets.
    for idx in range(0, X_perm.shape[0], int(num_in_group)):
        test_rows = X_perm[idx:min(idx + num_in_group, X_perm.shape[0]), :]  # Get testing rows.
        train_rows = np.vstack((X_perm[:idx, :], X_perm[min(idx + num_in_group, X_perm.shape[0]):, :]))  # Get training rows.
        train_rows_target = np.hstack((y_perm[:idx], y_perm[min(idx + num_in_group, len(y_perm)):]))  # Get test rows target variable values.
        classifier = learner(train_rows, train_rows_target)  # Train classifier.
        pred_next = np.apply_along_axis(classifier, 1, test_rows)  # Get next set of predictions.
        pred = np.vstack((pred, pred_next))  # add next set of predictions to results matrix.

    pred_aux = np.hstack((pred, np.reshape(perm, [perm.shape[0], 1])))  # Apply inverse permutation to rows of matrix of predictions.
    pred_aux = pred_aux[pred_aux[:, -1].argsort()]
    pred = pred_aux[:, :-1]
    return pred


# CA: compute classification accuracy given a vector of real classes
# and a vector of predictions.
def CA(real, predictions):
    prediction_classes = np.round(predictions)[:, 1]  # Get vector of class predictions. The value in vector is 1 if predicted class is 1 and 0 otherwise.
    num_correct = np.sum(np.equal(prediction_classes, real))  # Get number of correct predictions.
    return num_correct/len(real)  # Get proportion of correct classifications.



# AUC: measure the classification accuracy using the area under curve.
def AUC(real, predictions):
    """
    
    Iz knjige Inteligentni sistemi (Igor Kononenko, Marko Robnik Sikonja):
    "Izkaže se, da je AUC enaka verjetnosti, da bo klasifikator (ki zna napovedati verjetnosti) pravilno razločil med
     pozitivnim in negativnim primerom (tj. pozitivnemu bo pripisal večjo verjetnost, da je pozitiven)."
     
    """

    # See Inteligentni sistemi page 62. for description.

    # Get predictions for class 1.
    pos_pred = [el[1] for el in predictions]

    # roc_auc: compute ROC AUC score from given true values of y and the predictions.
    def roc_auc(y_true, y_pred):
        # roc_curve: compute roc curve (represented by row two vectors)
        def roc(y_true, y_score):
            # Get count of false positives and true positives as a function of classification threshold.
            false_pos, true_pos = threshold_f(y_true, y_score)

            # Get mask of indices of valid thresholds. Make sure to append 1 (true) for first and last treshold.
            is_optimal = np.r_[1, np.logical_or(np.diff(false_pos, 2), np.diff(true_pos, 2)), 1]
            optimal_idxs = np.where(is_optimal)[0]  # Get indices of ones (true values) in mask.

            false_pos = false_pos[optimal_idxs]  # Get number of false positives with optimal threshold indices.
            true_pos = true_pos[optimal_idxs]  # Get number of true positives with optimal threshold indices.

            # Curve starts at (0, 0) - append 0 to front.
            true_pos = np.append(0, true_pos)
            false_pos = np.append(0, false_pos)

            specificity = false_pos / false_pos[-1]  # get x-axis and y-axis values for ROC curve.
            sensitivity = true_pos / true_pos[-1]
            return specificity, sensitivity  # Return vectors representing the ROC curve

        specificity, sensitivity = roc(y_true, y_pred)         # Compute ROC curve from passed data.
        # Numerically integrate the ROC curve and return result.
        return np.trapz(sensitivity, specificity)  # Use trapezoidal integration (see data4bio notes for intuition)

    # threshold_f: calculate number of true and false positives for each binary classification threshold
    def threshold_f(y_true, y_score):
        # Convert lists to numpy arrays.
        y_score = np.array(y_score)
        y_true = np.array(y_true)
        desc_score_indices = np.argsort(y_score)[::-1]  # Get indices that would sort y_score array.
        y_score = y_score[desc_score_indices]           # Sort y_score values.
        y_true = y_true[desc_score_indices]             # sort y_true values

        # Extract indices associated with distinct values.
        unique_val_idxs = np.where(np.diff(y_score))[0]
        # Add last index to unique_val_idxs (was truncated).
        threshold_idxs = np.append(unique_val_idxs, y_true.size - 1)

        true_pos = np.cumsum(y_true)[threshold_idxs]       # Accumulate values - get true positive values for each treshold.
        false_pos = 1 + threshold_idxs - true_pos          # Get false positives for each threshold.
        return false_pos, true_pos

    # Compute and return AUC score.
    return roc_auc(real, pos_pred)


# If running as script...
if __name__ == "__main__":
    X, y = load('reg.data')

    # make a linearly spaced list of 500 lambda values on interval [0, 7]
    lam_vals = np.linspace(0, 1, 10)
    acc_cv = np.empty(len(lam_vals), dtype=float)
    acc_auc = np.empty(len(lam_vals), dtype=float)
    # Go over lambda values and save computed classification accuracy.
    for i, lam in enumerate(lam_vals):
        learner = LogRegLearner(lambda_=lam)
        pred = test_cv(learner, X, y, k=5)
        acc_cv[i] = CA(y, pred)
        acc_auc[i] = AUC(y, pred)


    # Plot the accuracies with respect to lambda.
    a = plt.plot(lam_vals, acc_auc, label='auc score')
    b = plt.plot(lam_vals, acc_cv, label="percentage correct")
    plt.legend(bbox_to_anchor=(0.866, 0.98), loc=2, borderaxespad=0.)
    plt.title("Classification Accuracy", fontsize=20)
    plt.show()
