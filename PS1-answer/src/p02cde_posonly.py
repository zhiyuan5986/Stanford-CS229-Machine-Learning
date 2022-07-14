import numpy as np
import util
import matplotlib.pyplot as plt
from p01b_logreg import LogisticRegression

# Character to replace with sub-problem letter in plot_path/pred_path
WILDCARD = 'X'


def main(train_path, valid_path, test_path, pred_path):
    """Problem 2: Logistic regression for incomplete, positive-only labels.

    Run under the following conditions:
        1. on y-labels,
        2. on l-labels,
        3. on l-labels with correction factor alpha.

    Args:
        train_path: Path to CSV file containing training set.
        valid_path: Path to CSV file containing validation set.
        test_path: Path to CSV file containing test set.
        pred_path: Path to save predictions.
    """
    pred_path_c = pred_path.replace(WILDCARD, 'c')
    pred_path_d = pred_path.replace(WILDCARD, 'd')
    pred_path_e = pred_path.replace(WILDCARD, 'e')

    # *** START CODE HERE ***

    # Part (c): Train and test on true labels
    # Make sure to save outputs to pred_path_c
    x_train,t_train = util.load_dataset(train_path,label_col='t',add_intercept=True)
    logreg_c = LogisticRegression()
    logreg_c.fit(x_train,t_train)

    x_test, t_test = util.load_dataset(test_path,label_col='t',add_intercept=True)
    t_test_pred = logreg_c.predict(x_test)
    np.savetxt(pred_path, t_test_pred)
    util.plot(x_test, t_test, logreg_c.theta)
    plt.show()

    # Part (d): Train on y-labels and test on true labels
    # Make sure to save outputs to pred_path_d

    x_train,y_train = util.load_dataset(train_path,add_intercept=True)
    logreg_d = LogisticRegression()
    logreg_d.fit(x_train,y_train)

    x_test, t_test = util.load_dataset(test_path,label_col='t',add_intercept=True)
    t_test_pred = logreg_d.predict(x_test)
    np.savetxt(pred_path, t_test_pred)
    util.plot(x_test, t_test, logreg_d.theta)
    plt.show()

    # Part (e): Apply correction factor using validation set and test on true labels
    # Plot and use np.savetxt to save outputs to pred_path_e

    x_train,y_train = util.load_dataset(train_path,add_intercept=True)
    logreg_e = LogisticRegression()
    logreg_e.fit(x_train,y_train)

    x_eval, y_eval = util.load_dataset(valid_path,add_intercept=True)
    x_V_plus = x_eval[y_eval > 0]
    alpha = np.mean(logreg_e.h(logreg_e.theta,x_V_plus))
    print(alpha)
    print(logreg_e.theta)
    logreg_e.theta[0] += np.log(2/alpha-1)
    print(logreg_e.theta)

    x_test, t_test = util.load_dataset(test_path,label_col='t',add_intercept=True)
    t_test_pred = logreg_e.predict(x_test)# logreg_e.h(logreg_e.theta,x_test) / alpha > 0.5
    np.savetxt(pred_path, t_test_pred)
    util.plot(x_test, t_test, logreg_e.theta)
    plt.show()

    util.plot3(x_test, t_test,theta_1=logreg_c.theta,legend_1="problem c",theta_2=logreg_d.theta,legend_2="problem d",theta_3=logreg_e.theta,legend_3="problem e")
    plt.savefig("../output/p02cde_ds3.pdf")

    # *** END CODER HERE

if __name__ == "__main__":
    train_path='../data/ds3_train.csv'
    valid_path='../data/ds3_valid.csv'
    test_path='../data/ds3_test.csv'
    pred_path='../output/p02X_pred.txt'
    main(train_path,valid_path,test_path,pred_path)