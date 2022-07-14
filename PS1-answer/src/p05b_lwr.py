import matplotlib.pyplot as plt
import numpy as np
import util

from linear_model import LinearModel

# def plot(x, y_label, y_pred, title):
#     plt.figure()
#     plt.plot(x[:,-1], y_label, 'bx', label='label')
#     plt.plot(x[:,-1], y_pred, 'ro', label='prediction')
#     plt.suptitle(title, fontsize=12)
#     plt.legend(loc='upper left')

def main(tau, train_path, eval_path):
    """Problem 5(b): Locally weighted regression (LWR)

    Args:
        tau: Bandwidth parameter for LWR.
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
    """
    # Load training set
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)

    # *** START CODE HERE ***
    # Fit a LWR model
    # Get MSE value on the validation set
    # Plot validation predictions on top of training set
    # No need to save predictions
    # Plot data

    lwr = LocallyWeightedLinearRegression(tau=0.5)
    lwr.fit(x_train,y_train)
    x_eval, y_eval = util.load_dataset(eval_path, add_intercept=True)
    y_eval_pred = lwr.predict(x_eval)

    MSE = np.mean((y_eval-y_eval_pred)*(y_eval-y_eval_pred))
    print(f"MSE={MSE}")


    # plt.plot(x_train,y_train,'bx',label="training set")
    # plt.plot(x_train,lwr.predict(x_train),'ro',label="prediction")
    # plt.legend(loc='upper left')
    # plt.show()
    

    # plt.figure()
    # plt.plot(x_eval,y_eval,'bx',label="validation set")
    # plt.plot(x_eval,y_eval_pred,'ro',label="prediction")
    # plt.legend(loc='upper left')
    # plt.show()

    plt.plot(x_train[:,-1], y_train, 'bx', label="training set")
    plt.plot(x_eval[:,-1], y_eval_pred, 'ro', label="validation set prediction")
    plt.legend(loc="upper left")
    plt.show()

    

    # *** END CODE HERE ***


class LocallyWeightedLinearRegression(LinearModel):
    """Locally Weighted Regression (LWR).

    Example usage:
        > clf = LocallyWeightedLinearRegression(tau)
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def __init__(self, tau):
        super(LocallyWeightedLinearRegression, self).__init__()
        self.tau = tau
        self.x = None
        self.y = None

    def fit(self, x, y):
        """Fit LWR by saving the training set.

        """
        # *** START CODE HERE ***
        self.x = x
        self.y = y
        # *** END CODE HERE ***

    def predict(self, x):
        """Make predictions given inputs x.

        Args:
            x: Inputs of shape (m, n).

        Returns:
            Outputs of shape (m,).
        """
        # *** START CODE HERE ***
        m,n = x.shape
        y_pred = np.zeros(m)
        for i in range(m):
            W = np.diag(np.exp(-np.linalg.norm(self.x-x[i],axis=1)**2/(2*self.tau*self.tau)))
            # print(W.shape)
        # W = w @ np.eye(w.size) 
            y_pred[i] = x[i] @ np.linalg.inv(self.x.T @ W @ self.x) @ self.x.T @ W @ self.y

        return y_pred

        # *** END CODE HERE ***

if __name__ == "__main__":
    tau=5e-1
    train_path='../data/ds5_train.csv'
    eval_path='../data/ds5_valid.csv'
    main(tau,train_path,eval_path)
