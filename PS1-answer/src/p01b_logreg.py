import numpy as np
import util
import matplotlib.pyplot as plt
from linear_model import LinearModel


def main(train_path, eval_path, pred_path):
    """Problem 1(b): Logistic regression with Newton's Method.

    Args:
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
        pred_path: Path to save predictions.
    """
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)

    # *** START CODE HERE ***

    # take a look
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.plot(x_train[y_train == 1, -2], x_train[y_train == 1, -1], 'bx', linewidth=2)
    plt.plot(x_train[y_train == 0, -2], x_train[y_train == 0, -1], 'go', linewidth=2)
    

    # train
    logreg.fit(x_train, y_train)
    y_train_pred = logreg.predict(x_train)

    # show the result for training set
    util.plot(x_train, y_train, theta=logreg.theta)
    plt.show()

    # show the result for validation set
    x_eval, y_eval = util.load_dataset(eval_path, add_intercept=True)
    util.plot(x_eval, y_eval, theta=logreg.theta)
    plt.show()

    
    # plt.savefig("../output/p01b_pred_1.pdf")

    # *** END CODE HERE ***


class LogisticRegression(LinearModel):
    """Logistic regression with Newton's Method as the solver.

    Example usage:
        > clf = LogisticRegression()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def h(self,theta,x):
        return 1/(1+np.exp(-x @ theta))

    def fit(self, x, y):
        """Run Newton's Method to minimize J(theta) for logistic regression.

        Args:
            x: Training example inputs. Shape (m, n).
            y: Training example labels. Shape (m,).
        """
        # *** START CODE HERE ***
        def H(theta):
            h_theta_x = np.reshape(self.h(theta,x),(-1,1))
            return 1/m * np.dot(x.T, (h_theta_x * (1-h_theta_x) * x))
        def fp(theta):
            return 1/m * x.T @ (self.h(theta,x)-y)

        
        m,n = x.shape
        theta = np.zeros(n)
        # print(H(theta))
        # print(fp(theta).shape)
        step = np.linalg.inv(H(theta))@fp(theta)
        while np.linalg.norm(step,1) > self.eps:
            theta = theta-step
            step = np.linalg.inv(H(theta))@fp(theta)
        self.theta = theta
        # *** END CODE HERE ***

    def predict(self, x):
        """Make a prediction given new inputs x.

        Args:
            x: Inputs of shape (m, n).

        Returns:
            Outputs of shape (m,).
        """
        # *** START CODE HERE ***
        # return self.h(self.theta,x) This is wrong!!!
        return x @ self.theta >= 0
        # *** END CODE HERE ***

if __name__ == "__main__":
    train_path='../data/ds2_train.csv'
    eval_path='../data/ds2_valid.csv'
    pred_path='../output/p01b_pred_2.txt'
    logreg = LogisticRegression()
    main(train_path,eval_path,pred_path)
