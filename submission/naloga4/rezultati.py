from solution import load, LogRegLearner, test_cv, CA, AUC
from draw import draw_decision
import numpy as np
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")

X, y = load('reg.data')

# Prompt user for functionality choice.
while True:
    action = input('Make plot of classification accuracy with respect to lambda or visualize classification for specified lambda? (1/2): ')
    if action in {'1', '2'}:
        break
    else:
        print("Invalid input. Please try again.")

# Accuracy with respect to lambda
if action == '1':
    # Make a linearly spaced list of lambda values on specified interval of specified.
    while True:
        try:
            lam_start = float(input('Enter starting lambda value: '))
            lam_end = float(input('Enter end lambda value: '))
            num_lam = int(input('Enter numer of steps: '))
            lam_vals = np.linspace(lam_start, lam_end, num_lam)
            break
        except ValueError:
            print('Invalid input. Please try again.')

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


# Classification visualization
else:
    # Promp user for lambda value.
    while True:
        try:
            lam = float(input('Specify lambda: '))
            break
        except ValueError:
            print('Invalid input. Please try again.')

    learner = LogRegLearner(lambda_=lam)
    classifier = learner(X, y)
    draw_decision(X, y, classifier, 0, 1, lam)
