import numpy as np
import util
import matplotlib.pyplot as plt
from linear_model import LinearModel
import p01b_logreg


def main(train_path, eval_path, pred_path):
    """Problem 1(e): Gaussian discriminant analysis (GDA)

    Args:
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
        pred_path: Path to save predictions.
    """
    # Load dataset
    x_train, y_train = util.load_dataset(train_path, add_intercept=False)

    # *** START CODE HERE ***
    gda = GDA()
    gda.fit(x_train, y_train)
    # print(gda.theta)

    # logreg = p01b_logreg.LogisticRegression()
    # logreg.fit(x_train,y_train) 
    
    # print(p01b_logreg.logreg.theta)

    x_eval, y_eval = util.load_dataset(eval_path, add_intercept=False)
    util.plot2(x_eval, y_eval,theta_1=gda.theta,legend_1="GDA",theta_2=p01b_logreg.logreg.theta,legend_2="Logistic Regression")
    plt.savefig("../output/p01e_gda_ds2.pdf")

    # x_eval, y_eval = util.load_dataset(eval_path, add_intercept=False)
    util.plot(x_eval, y_eval, gda.theta)
    plt.show()
    # *** END CODE HERE ***


class GDA(LinearModel):
    """Gaussian Discriminant Analysis.

    Example usage:
        > clf = GDA()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def fit(self, x, y):
        """Fit a GDA model to training set given by x and y.

        Args:
            x: Training example inputs. Shape (m, n).
            y: Training example labels. Shape (m,).

        Returns:
            theta: GDA model parameters.
        """
        # *** START CODE HERE ***
        m,n = x.shape
        y = np.reshape(y, (-1,1))
        phi = 1/m*np.sum(y)
        mu_0 = x.T @ (1-y) / np.sum(1-y)
        mu_1 = x.T @ y / np.sum(y)
        # print(mu_0)
        # mu_i = np.tile((1-y),(m,n))*mu_0 + np.tile(y,(m,n))*mu_1

        mu_i = (1-y)@mu_0.T + y@mu_1.T
        # print(mu_i)
        Sigma = 1/m * (x-mu_i).T @ (x-mu_i)
        # print(Sigma)
        # sigma = ((x[y == 0] - mu_0.T).dot(x[y == 0] - mu_0.T).T + (x[y == 1] - mu_1.T).dot(x[y == 1] - mu_1.T).T) / m
        # print(sigma)

        Sigma_inv = np.linalg.inv(Sigma)
        theta =  Sigma_inv @ (mu_1-mu_0)
        # print(theta)
        theta_0 = np.log(phi / (1-phi)) + 1/2*mu_0.T @ Sigma_inv @ mu_0 - 1/2*mu_1.T @ Sigma_inv @ mu_1
        # print(theta_0)
        theta = theta.reshape(-1,1)
        self.theta = np.insert(theta,0,theta_0)
        # print(self.theta)
        
        # *** END CODE HERE ***

    def predict(self, x):
        """Make a prediction given new inputs x.

        Args:
            x: Inputs of shape (m, n).

        Returns:
            Outputs of shape (m,).
        """
        # *** START CODE HERE ***
        return util.add_intercept(x) @ self.theta >= 0
        # *** END CODE HERE
if __name__ == "__main__":
    train_path='../data/ds2_train.csv'
    eval_path='../data/ds2_valid.csv'
    pred_path='../output/p01b_pred_2.txt'
    main(train_path,eval_path,pred_path)