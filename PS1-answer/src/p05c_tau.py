import matplotlib.pyplot as plt
import numpy as np
import util

from p05b_lwr import LocallyWeightedLinearRegression


def main(tau_values, train_path, valid_path, test_path, pred_path):
    """Problem 5(b): Tune the bandwidth paramater tau for LWR.

    Args:
        tau_values: List of tau values to try.
        train_path: Path to CSV file containing training set.
        valid_path: Path to CSV file containing validation set.
        test_path: Path to CSV file containing test set.
        pred_path: Path to save predictions.
    """
    # Load training set
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)

    # *** START CODE HERE ***
    # Search tau_values for the best tau (lowest MSE on the validation set)
    # Fit a LWR model with the best tau value
    # Run on the test set to get the MSE value
    # Save predictions to pred_path
    # Plot data

    x_eval, y_eval = util.load_dataset(valid_path, add_intercept=True)
    MSE = np.zeros(len(tau_values))
    for i,tau in enumerate(tau_values):
        lwr = LocallyWeightedLinearRegression(tau)
        lwr.fit(x_train,y_train)
        y_eval_pred = lwr.predict(x_eval)
        MSE[i] = np.mean((y_eval-y_eval_pred)*(y_eval-y_eval_pred))
        print(f"{i}-th: MSE={MSE[i]}")
    tau_perfect = tau_values[np.argmin(MSE)]
    print(np.argmin(MSE))

    x_test, y_test = util.load_dataset(test_path, add_intercept=True)
    lwr = LocallyWeightedLinearRegression(tau_perfect)
    lwr.fit(x_train,y_train)
    y_test_pred = lwr.predict(x_test)

    plt.plot(x_train[:,-1], y_train, 'bx', label="training set")
    plt.plot(x_test[:,-1], y_test_pred, 'ro', label="test set prediction")
    plt.legend(loc="upper left")
    plt.show()
        


    # *** END CODE HERE ***
if __name__ == "__main__":
    tau_values=[3e-2, 5e-2, 1e-1, 5e-1, 1e0, 1e1]
    train_path='../data/ds5_train.csv'
    valid_path='../data/ds5_valid.csv'
    test_path='../data/ds5_test.csv'
    pred_path='../output/p05c_pred.txt'
    main(tau_values,train_path,valid_path,test_path,pred_path)