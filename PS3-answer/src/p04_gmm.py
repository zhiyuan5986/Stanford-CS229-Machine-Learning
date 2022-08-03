import matplotlib.pyplot as plt
import numpy as np
import os

PLOT_COLORS = ['red', 'green', 'blue', 'orange']  # Colors for your plots
K = 4           # Number of Gaussians in the mixture model
NUM_TRIALS = 3  # Number of trials to run (can be adjusted for debugging)
UNLABELED = -1  # Cluster label for unlabeled data points (do not change)


def main(is_semi_supervised, trial_num):
    """Problem 3: EM for Gaussian Mixture Models (unsupervised and semi-supervised)"""
    print('Running {} EM algorithm...'
          .format('semi-supervised' if is_semi_supervised else 'unsupervised'))

    # Load dataset
    train_path = os.path.join('..', 'data', 'ds4_train.csv')
    x_all, z_all = load_gmm_dataset(train_path)

    # Split into labeled and unlabeled examples
    labeled_idxs = (z_all != UNLABELED).squeeze()
    x_tilde = x_all[labeled_idxs, :]   # Labeled examples
    z_tilde = z_all[labeled_idxs, :]   # Corresponding labels
    x = x_all[~labeled_idxs, :]        # Unlabeled examples

    # *** START CODE HERE ***
    # (1) Initialize mu and sigma by splitting the m data points uniformly at random
    # into K groups, then calculating the sample mean and covariance for each group

    m = x.shape[0]
    index = np.arange(0, m)
    np.random.shuffle(index)
    K_array = np.split(x[index,:],K,axis=0)
    # random.shuffle(K_array)
    # print(len(K_array))
    mu = [np.mean(arr,axis=0) for arr in K_array]
    # print(mu)
    # print((x-mu[0]).shape)
    sigma = [(arr-mu_j).T @ (arr-mu_j) / arr.shape[0] for arr,mu_j in zip(K_array,mu)]
    # print(sigma[0].shape)

    # (2) Initialize phi to place equal probability on each Gaussian
    # phi should be a numpy array of shape (K,)
    phi = np.ones((K,1)) / K

    # (3) Initialize the w values to place equal probability on each Gaussian
    # w should be a numpy array of shape (m, K)
    
    w = np.ones((m,K))/K
    # raise ZeroDivisionError()
    # *** END CODE HERE ***

    if is_semi_supervised:
        w = run_semi_supervised_em(x, x_tilde, z_tilde, w, phi, mu, sigma)
    else:
        w = run_em(x, w, phi, mu, sigma)

    # Plot your predictions
    z_pred = np.zeros(m)
    if w is not None:  # Just a placeholder for the starter code
        for i in range(m):
            z_pred[i] = np.argmax(w[i])

    plot_gmm_preds(x, z_pred, is_semi_supervised, plot_id=trial_num)


def run_em(x, w, phi, mu, sigma):
    """Problem 3(d): EM Algorithm (unsupervised).

    See inline comments for instructions.

    Args:
        x: Design matrix of shape (n, d).
        w: Initial weight matrix of shape (n, k).
        phi: Initial mixture prior, of shape (k,).
        mu: Initial cluster means, list of k arrays of shape (d,).
        sigma: Initial cluster covariances, list of k arrays of shape (d, d).

    Returns:
        Updated weight matrix of shape (n, d) resulting from EM algorithm.
        More specifically, w[i, j] should contain the probability of
        example x^(i) belonging to the j-th Gaussian in the mixture.
    """
    # No need to change any of these parameters
    eps = 1e-3  # Convergence threshold
    max_iter = 1000

    # Stop when the absolute change in log-likelihood is < eps
    # See below for explanation of the convergence criterion
    it = 0
    ll = prev_ll = None
    while it < max_iter and (prev_ll is None or np.abs(ll - prev_ll) >= eps):
        # pass  # Just a placeholder for the starter code
        # *** START CODE HERE

        # (1) E-step: Update your estimates in w
        for i in range(K):
            w[:,i] = np.exp(-1/2*np.sum((x-mu[i]) @ np.linalg.inv(sigma[i]) * (x-mu[i]),axis=1)) * phi[i]/ (np.linalg.det(sigma[i])**0.5)
        w /= np.sum(w,axis=1).reshape((-1,1))
        # (2) M-step: Update the model parameters phi, mu, and sigma
        phi = np.mean(w,axis=1)
        for l in range(K):
            mu[l] =  x.T @ w[:,l] / (np.sum(w[:,l]))
            sigma[l] = (w[:,l].reshape((-1,1))*(x-mu[l])).T @ (x-mu[l]) / (np.sum(w[:,l]))

        # (3) Compute the log-likelihood of the data to check for convergence.
        prev_ll = ll
        p_x_z = np.ones(w.shape)
        for i in range(K):
            p_x_z[:,i] = np.exp(-1/2*np.sum((x-mu[i]) @ np.linalg.inv(sigma[i]) * (x-mu[i]),axis=1)) * phi[i]/ (np.linalg.det(sigma[i])**0.5)
        ll = np.sum(np.log(np.sum(p_x_z,axis=1)))
        # By log-likelihood, we mean `ll = sum_x[log(sum_z[p(x|z) * p(z)])]`.
        # We define convergence by the first iteration where abs(ll - prev_ll) < eps.
        # Hint: For debugging, recall part (a). We showed that ll should be monotonically increasing.
        it += 1
        # *** END CODE HERE ***
    print(f'Number of iterations:{it}')

    return w


def run_semi_supervised_em(x, x_tilde, z_tilde, w, phi, mu, sigma):
    """Problem 3(e): Semi-Supervised EM Algorithm.

    See inline comments for instructions.

    Args:
        x: Design matrix of unlabeled examples of shape (n, d).
        x_tilde: Design matrix of labeled examples of shape (n_tilde, d).
        z_tilde: Array of labels of shape (n_tilde, 1).
        w: Initial weight matrix of shape (n, k).
        phi: Initial mixture prior, of shape (k,).
        mu: Initial cluster means, list of k arrays of shape (d,).
        sigma: Initial cluster covariances, list of k arrays of shape (d, d).

    Returns:
        Updated weight matrix of shape (n, d) resulting from semi-supervised EM algorithm.
        More specifically, w[i, j] should contain the probability of
        example x^(i) belonging to the j-th Gaussian in the mixture.
    """
    # No need to change any of these parameters
    alpha = 20.  # Weight for the labeled examples
    eps = 1e-3   # Convergence threshold
    max_iter = 1000

    # Stop when the absolute change in log-likelihood is < eps
    # See below for explanation of the convergence criterion
    it = 0
    ll = prev_ll = None
    while it < max_iter and (prev_ll is None or np.abs(ll - prev_ll) >= eps):
        # pass  # Just a placeholder for the starter code
        # *** START CODE HERE ***
        # (1) E-step: Update your estimates in w
        for i in range(K):
            w[:,i] = np.exp(-1/2*np.sum((x-mu[i]) @ np.linalg.inv(sigma[i]) * (x-mu[i]),axis=1)) * phi[i]/ (np.linalg.det(sigma[i])**0.5)
        w /= np.sum(w,axis=1).reshape((-1,1))

        # (2) M-step: Update the model parameters phi, mu, and sigma
        # print(x_tilde.shape)
        # print((z_tilde == 0).shape)
        for l in range(K):
            phi[l] = (np.sum(w[:,l]) + alpha * np.sum(z_tilde == l) ) / (w.shape[0]+alpha*z_tilde.shape[0])
            mu[l] =  (x.T @ w[:,l] + alpha*np.sum(x_tilde[z_tilde.reshape(-1,)==l],axis=0)) / (np.sum(w[:,l])+alpha*np.sum(z_tilde == l))
            sigma[l] = ((w[:,l].reshape((-1,1))*(x-mu[l])).T @ (x-mu[l]) + alpha*(x_tilde[z_tilde.reshape(-1,)==l]-mu[l]).T @ (x_tilde[z_tilde.reshape(-1,)==l]-mu[l]))/ (np.sum(w[:,l])+alpha*np.sum(z_tilde == l))

        # (3) Compute the log-likelihood of the data to check for convergence.
        prev_ll = ll
        p_x_z = np.ones(w.shape)
        p_xtilde_ztilde = 0
        for i in range(K):
            p_x_z[:,i] = np.exp(-1/2*np.sum((x-mu[i]) @ np.linalg.inv(sigma[i]) * (x-mu[i]),axis=1)) * phi[i]/ (np.linalg.det(sigma[i])**0.5)
            p_xtilde_ztilde += np.sum(-1/2*np.sum((x_tilde[z_tilde.reshape(-1,)==l]-mu[i]) @ np.linalg.inv(sigma[i]) * (x_tilde[z_tilde.reshape(-1,)==l]-mu[i]),axis=1) -np.log(np.linalg.det(sigma[i])**0.5))
        ll = np.sum(np.log(np.sum(p_x_z,axis=1))) + alpha*p_xtilde_ztilde#np.sum(np.log(np.sum(p_xtilde_ztilde,axis=1)))
        # Hint: Make sure to include alpha in your calculation of ll.
        # Hint: For debugging, recall part (a). We showed that ll should be monotonically increasing.
        it += 1
        # *** END CODE HERE ***
    print(f'Number of iterations:{it}')
    return w


# *** START CODE HERE ***
# Helper functions
# *** END CODE HERE ***


def plot_gmm_preds(x, z, with_supervision, plot_id):
    """Plot GMM predictions on a 2D dataset `x` with labels `z`.

    Write to the output directory, including `plot_id`
    in the name, and appending 'ss' if the GMM had supervision.

    NOTE: You do not need to edit this function.
    """
    plt.figure(figsize=(12, 8))
    plt.title('{} GMM Predictions'.format('Semi-supervised' if with_supervision else 'Unsupervised'))
    plt.xlabel('x_1')
    plt.ylabel('x_2')

    for x_1, x_2, z_ in zip(x[:, 0], x[:, 1], z):
        color = 'gray' if z_ < 0 else PLOT_COLORS[int(z_)]
        alpha = 0.25 if z_ < 0 else 0.75
        plt.scatter(x_1, x_2, marker='.', c=color, alpha=alpha)

    file_name = 'pred{}_{}.pdf'.format('_ss' if with_supervision else '', plot_id)
    save_path = os.path.join('../output', file_name)
    plt.savefig(save_path)


def load_gmm_dataset(csv_path):
    """Load dataset for Gaussian Mixture Model.

    Args:
         csv_path: Path to CSV file containing dataset.

    Returns:
        x: NumPy array shape (n_examples, dim)
        z: NumPy array shape (n_exampls, 1)

    NOTE: You do not need to edit this function.
    """

    # Load headers
    with open(csv_path, 'r') as csv_fh:
        headers = csv_fh.readline().strip().split(',')

    # Load features and labels
    x_cols = [i for i in range(len(headers)) if headers[i].startswith('x')]
    z_cols = [i for i in range(len(headers)) if headers[i] == 'z']

    x = np.loadtxt(csv_path, delimiter=',', skiprows=1, usecols=x_cols, dtype=float)
    z = np.loadtxt(csv_path, delimiter=',', skiprows=1, usecols=z_cols, dtype=float)

    if z.ndim == 1:
        z = np.expand_dims(z, axis=-1)

    return x, z


if __name__ == '__main__':
    np.random.seed(229)
    # Run NUM_TRIALS trials to see how different initializations
    # affect the final predictions with and without supervision
    for t in range(NUM_TRIALS):
        main(is_semi_supervised=True, trial_num=t)

        # *** START CODE HERE ***
        # Once you've implemented the semi-supervised version,
        # uncomment the following line.
        # main(is_semi_supervised=False, trial_num=t)
        # You do not need to add any other lines in this code block.
        # *** END CODE HERE ***
