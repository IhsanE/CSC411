import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_boston

BATCHES = 50

class BatchSampler(object):
    '''
    A (very) simple wrapper to randomly sample batches without replacement.

    You shouldn't need to touch this.
    '''

    def __init__(self, data, targets, batch_size):
        self.num_points = data.shape[0]
        self.features = data.shape[1]
        self.batch_size = batch_size

        self.data = data
        self.targets = targets

        self.indices = np.arange(self.num_points)

    def random_batch_indices(self, m=None):
        '''
        Get random batch indices without replacement from the dataset.

        If m is given the batch will be of size m. Otherwise will default to the class initialized value.
        '''
        if m is None:
            indices = np.random.choice(self.indices, self.batch_size, replace=False)
        else:
            indices = np.random.choice(self.indices, m, replace=False)
        return indices

    def get_batch(self, m=None):
        '''
        Get a random batch without replacement from the dataset.

        If m is given the batch will be of size m. Otherwise will default to the class initialized value.
        '''
        indices = self.random_batch_indices(m)
        X_batch = np.take(self.data, indices, 0)
        y_batch = self.targets[indices]
        return X_batch, y_batch


def load_data_and_init_params():
    '''
    Load the Boston houses dataset and randomly initialise linear regression weights.
    '''
    print('------ Loading Boston Houses Dataset ------')
    X, y = load_boston(True)
    features = X.shape[1]

    # Initialize w
    w = np.random.randn(features)

    print("Loaded...")
    print("Total data points: {0}\nFeature count: {1}".format(X.shape[0], X.shape[1]))
    print("Random parameters, w: {0}".format(w))
    print('-------------------------------------------\n\n\n')

    return X, y, w


def cosine_similarity(vec1, vec2):
    '''
    Compute the cosine similarity (cos theta) between two vectors.
    '''
    dot = np.dot(vec1, vec2)
    sum1 = np.sqrt(np.dot(vec1, vec1))
    sum2 = np.sqrt(np.dot(vec2, vec2))

    return dot / (sum1 * sum2)

def lin_reg_gradient(X, y, w):
    '''
    Compute gradient of linear regression model parameterized by w
    '''
    X_trans = np.transpose(X)
    XTX = np.matmul(X_trans, X)
    XTXw = np.matmul(XTX, w)

    XTy = np.matmul(X_trans, y)


    # ∇L(x, y, w) = 2XTXw - 2XTy = 0
    gradient = 2*XTXw - 2*XTy

    return gradient


def compute_mini_batch_gradient(X, y, m, K, w):
    batch_sampler = BatchSampler(X, y, BATCHES)

    k_sum = 0
    for i in range(K):
        X_b, y_b = batch_sampler.get_batch(m)
        batch_grad = lin_reg_gradient(X_b, y_b, w)
        k_sum += batch_grad

    return k_sum / K


def compute_true_gradient(X, y, w):
    gradient = lin_reg_gradient(X, y, w)
    return gradient


def compute_mean_square_error(y, y_computed_target):
    return np.sqrt(np.sum((y - y_computed_target)**2))


def compare_mini_batch_and_true_gradient(X, y, m, K):
    w_norm = np.random.normal(size=X.shape[1])
    mini_batch_gradient = compute_mini_batch_gradient(X, y, m, K, w_norm)
    true_gradient = compute_true_gradient(X, y, w_norm)

    cosine_metric = cosine_similarity(mini_batch_gradient, true_gradient)
    squared_distance_metric = compute_mean_square_error(mini_batch_gradient, true_gradient)
    print('cosine similarity metric: {}'.format(cosine_metric))
    print('squared distance metric: {}'.format(squared_distance_metric))


def compute_variance(x, x_prime_index):
    x_avg = np.mean(x)
    x_prime = x[x_prime_index]
    N = len(x)
    sum = 0
    for i in range(N):
        sum += (x_prime - x_avg)**2

    return sum/N


def sample_variance_plot(X, y, K):
    j = 3
    w_norm = np.random.normal(size=X.shape[1])

    m_values = range(1, 401)
    j_variances = []
    for m in m_values:
        mini_batch_gradient = compute_mini_batch_gradient(X, y, 400, K, w_norm)
        variance = compute_variance(mini_batch_gradient, j)
        j_variances.append(variance)

    m_logs = np.log(m_values)
    j_logs = np.log(j_variances)

    plt.figure(figsize=(15, 5))
    plt.plot(m_logs, j_logs)
    plt.xlabel('m')
    plt.ylabel('variance')
    plt.tight_layout()
    plt.show()


def main():
    # Load data and randomly initialise weights
    X, y, w = load_data_and_init_params()
    # Create a batch sampler to generate random batches from data
    batch_sampler = BatchSampler(X, y, BATCHES)

    # Example usage
    X_b, y_b = batch_sampler.get_batch()
    batch_grad = lin_reg_gradient(X_b, y_b, w)

    K = 500
    m = 50

    # Outputs the cosine sim error and squared dist error
    compare_mini_batch_and_true_gradient(X, y, m, K)

    # Q3.6) Plots variance j against m [1,400]
    sample_variance_plot(X, y, 500)


if __name__ == '__main__':
    main()
