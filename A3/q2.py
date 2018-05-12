import numpy as np

from sklearn.datasets import fetch_mldata
import matplotlib.pyplot as plt

np.random.seed(1847)

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

class GDOptimizer(object):
    '''
    A gradient descent optimizer with momentum
    '''

    def __init__(self, lr, beta=0.0):
        self.lr = lr
        self.beta = beta
        self.delta = 0

    def update_params(self, params, grad):
        self.delta = (-self.lr * grad) + (self.beta * self.delta)
        return params + self.delta
        # Update parameters using GD with momentum and return
        # the updated parameters

def optimize_test_function(optimizer, w_init=10.0, steps=200):
    '''
    Optimize the simple quadratic test function and return the parameter history.
    '''
    def func(x):
        return 0.01 * x * x

    def func_grad(x):
        return 0.02 * x

    w = w_init
    w_history = [w_init]

    for _ in range(steps):
        # Optimize and update the history
        w = optimizer.update_params(w,  func_grad(w))
        w_history.append(w)
    return w_history

class SVM(object):
    '''
    A Support Vector Machine
    '''

    def __init__(self, c, feature_count):
        self.c = c
        self.w = np.random.normal(0.0, 0.1, feature_count)

    def hinge_loss(self, X, y):
        '''
        Compute the hinge-loss for input data X (shape (n, m)) with target y (shape (n,)).

        Returns a length-n vector containing the hinge-loss per data point.
        '''
        c_over_n = self.c/X.shape[0]
        losses = []
        for i in range(X.shape[0]):
            hl = 1 - y[i] * (np.transpose(self.w).dot(X[i]))
            losses.append(max(hl, 0))

        return np.array(losses)

    def grad(self, X, y):
        '''
        Compute the gradient of the SVM objective for input data X (shape (n, m))
        with target y (shape (n,))

        Returns the gradient with respect to the SVM parameters (shape (m,)).
        '''
        c_over_n = self.c/X.shape[0]
        sum_vec = np.zeros(X.shape[1])
        for i in range(X.shape[0]):
            sum_vec += X[i] * y[i]

        regularized = np.insert((sum_vec[1:] * c_over_n), 0, 1, axis=0)
        return self.w - regularized

    def classify(self, X):
        '''
        Classify new input data matrix (shape (n,m)).

        Returns the predicted class labels (shape (n,))
        '''
        y = X.dot(self.w) + self.b
        return np.where(y <= 0, -1, 1)

def load_data():
    '''
    Load MNIST data (4 and 9 only) and split into train and test
    '''
    mnist = fetch_mldata('MNIST original', data_home='./data')
    label_4 = (mnist.target == 4)
    label_9 = (mnist.target == 9)

    data_4, targets_4 = mnist.data[label_4], np.ones(np.sum(label_4))
    data_9, targets_9 = mnist.data[label_9], -np.ones(np.sum(label_9))

    data = np.concatenate([data_4, data_9], 0)
    data = data / 255.0
    targets = np.concatenate([targets_4, targets_9], 0)

    permuted = np.random.permutation(data.shape[0])
    train_size = int(np.floor(data.shape[0] * 0.8))

    train_data, train_targets = data[permuted[:train_size]], targets[permuted[:train_size]]
    test_data, test_targets = data[permuted[train_size:]], targets[permuted[train_size:]]
    print("Data Loaded")
    print("Train size: {}".format(train_size))
    print("Test size: {}".format(data.shape[0] - train_size))
    print("-------------------------------")
    return train_data, train_targets, test_data, test_targets

def optimize_b(c_over_n, y):
    return -1*c_over_n * np.sum(y)

def optimize_svm(train_data, train_targets, penalty, optimizer, batchsize, iters):
    '''
    Optimize the SVM with the given hyperparameters. Return the trained SVM.
    '''
    svm = SVM(penalty, train_data.shape[1])
    batch_sampler = BatchSampler(train_data, train_targets, batchsize)

    for _ in range(iters):
        # Optimize and update the history
        estimate = 0
        for m in range(int(train_data.shape[0]/batchsize)):
            X_b, y_b = batch_sampler.get_batch()
            estimate += optimizer.update_params(svm.w,  svm.grad(X_b, y_b))
        svm.w = estimate/(train_data.shape[0]/batchsize)

    svm.b = optimize_b(penalty/train_data.shape[0], train_targets)
    return svm

def compute_avg_hinge_loss(svm, data, targets):
    losses = svm.hinge_loss(data, targets)
    return np.sum(losses)/losses.shape[0]

def compute_svm_accuracy(svm, data, targets):
    classified = svm.classify(data)
    n_diff = ((classified + targets) == 0).sum()
    return (targets.shape[0] - n_diff) / targets.shape[0]

def answer_2_1():
    GDO = GDOptimizer(1.0, beta=0.0)
    w_history = optimize_test_function(GDO)

    plt.figure(figsize=(15, 5))
    plt.plot(range(201), w_history)
    plt.xlabel('steps')
    plt.ylabel('w estimates')
    plt.tight_layout()
    plt.show()

    GDO = GDOptimizer(1.0, beta=0.9)
    w_history = optimize_test_function(GDO)

    plt.figure(figsize=(15, 5))
    plt.plot(range(201), w_history)
    plt.xlabel('steps')
    plt.ylabel('w estimates')
    plt.tight_layout()
    plt.show()

def apply_svm_to_mnist(train_data, train_targets, test_data, test_targets, beta):
    GDO = GDOptimizer(0.05, beta=beta)
    svm = optimize_svm(train_data, train_targets, 1.0, GDO, 100, 500)

    print("\nUsing Beta : %s\n" % beta)

    print("** LOSSES **\n")

    training_loss = compute_avg_hinge_loss(svm, train_data, train_targets)
    print("Training Loss: %s" % training_loss)

    test_loss = compute_avg_hinge_loss(svm, test_data, test_targets)
    print("Test Loss: %s" % test_loss)

    print("\n** ACCURACIES **\n")

    training_acc = compute_svm_accuracy(svm, train_data, train_targets)
    print("Training Accuracy: %s" % training_acc)

    test_acc = compute_svm_accuracy(svm, test_data, test_targets)
    print("Test Accuracy: %s" % test_acc)

    plt.imshow(np.reshape(svm.w[1:], (-1, 28)), cmap='gray')
    plt.show()

if __name__ == '__main__':
    # Display plots for gradient descent on f(w) = 0.01w^2
    answer_2_1()

    train_data, train_targets, test_data, test_targets = load_data()
    train_data = np.insert(train_data, 0, 1, axis=1)
    test_data = np.insert(test_data, 0, 1, axis=1)

    # Display plots/accuracies for svm on training/tests
    apply_svm_to_mnist(train_data, train_targets, test_data, test_targets, 0.0)
    apply_svm_to_mnist(train_data, train_targets, test_data, test_targets, 0.1)
