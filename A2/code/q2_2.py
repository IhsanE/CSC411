'''
Question 2.2 Skeleton Code

Here you should implement and evaluate the Conditional Gaussian classifier.
'''

import data
import numpy as np
# Import pyplot - plt.imshow is useful!
import matplotlib.pyplot as plt

D = 64
def compute_mean_mles(train_data, train_labels):
    '''
    Compute the mean estimate for each digit class

    Should return a numpy array of size (10,64)
    The ith row will correspond to the mean estimate for digit class i
    '''
    means = []
    for i in range(0, 10):
        i_digits = data.get_digits_by_label(train_data, train_labels, i)
        mean_1d = np.sum(i_digits, axis=0)/i_digits.shape[0]
        means.append(mean_1d)
    return np.array(means)

def compute_sigma_mles(train_data, train_labels):
    '''
    Compute the covariance estimate for each digit class

    Should return a three dimensional numpy array of shape (10, 64, 64)
    consisting of a covariance matrix for each digit class
    '''
    cov = []
    for i in range(0, 10):
        i_digits = data.get_digits_by_label(train_data, train_labels, i)
        mean_1d = np.sum(i_digits, axis=0)/i_digits.shape[0]
        cov_i = 0
        for x in i_digits:
            x_minus_mew = x - mean_1d
            x_minus_mew = x_minus_mew.reshape(-1, 64)
            cov_i += np.matmul(np.transpose(x_minus_mew), x_minus_mew)

        cov_i /= i_digits.shape[0]
        cov.append(cov_i)
    cov += 0.01 * np.identity(64)
    return np.array(cov)

def plot_cov_diagonal(covariances):
    cov = []
    for i in range(10):
        cov_diag = np.log(np.diag(covariances[i]))
        cov.append((cov_diag).reshape(-1, 8))

    all_concat = np.concatenate(np.array(cov), 1)
    plt.imshow(all_concat, cmap='gray')
    plt.show()

def generative_likelihood(digits, means, covariances):
    '''
    Compute the generative log-likelihood:
        log p(x|y,mu,Sigma)

    Should return an n x 10 numpy array
    '''
    log_2pi = np.log((2*np.pi)**(-D/2))
    likelihoods = []
    for x in digits:
        class_likelihoods = []
        for k in range(10):
            cov_k = covariances[k]
            mean_k = means[k]
            log_cov = np.log(np.linalg.det(cov_k)**(-1/2))
            x_minus_mew = x - mean_k
            x_minus_mew_trans = np.transpose(x_minus_mew)
            logged_exponent = (-1/2) * np.matmul(x_minus_mew_trans, np.linalg.inv(cov_k))
            logged_exponent = np.matmul(logged_exponent, x_minus_mew)
            class_likelihood = log_2pi + log_cov + logged_exponent
            class_likelihoods.append(class_likelihood)
        likelihoods.append(class_likelihoods)
    return np.array(likelihoods)

def conditional_likelihood(digits, means, covariances):
    '''
    Compute the conditional likelihood:

        log p(y|x, mu, Sigma)

    This should be a numpy array of shape (n, 10)
    Where n is the number of datapoints and 10 corresponds to each digit class
    '''

    p_x_y_mu_sigma = generative_likelihood(digits, means, covariances)

    return p_x_y_mu_sigma + np.log(1/10)

def avg_conditional_likelihood(digits, labels, means, covariances):
    '''
    Compute the average conditional likelihood over the true class labels

        AVG( log p(y_i|x_i, mu, Sigma) )

    i.e. the average log likelihood that the model assigns to the correct class label
    '''
    cond_likelihood = conditional_likelihood(digits, means, covariances)
    avg = 0
    for i, datum in enumerate(cond_likelihood):
        avg += datum[int(labels[i])]

    return avg/cond_likelihood.shape[0]

def classify_data(digits, means, covariances):
    '''
    Classify new points by taking the most likely posterior class
    '''
    cond_likelihood = conditional_likelihood(digits, means, covariances)
    likelihoods = np.exp(cond_likelihood)
    most_likely_classes = []
    for datum in likelihoods:
        most_likely_class = np.argmax(datum)
        most_likely_classes.append(most_likely_class)

    return most_likely_classes

def compute_likelihood_accuracy(predicted_labels, labels):
    count = 0
    for i in range(labels.shape[0]):
        if (predicted_labels[i] == labels[i]):
            count += 1

    return count/labels.shape[0]

def question_2_2_3(train_data, train_labels, test_data, test_labels, means, covariances):
    predicted_training_labels = classify_data(train_data, means, covariances)
    training_accuracy = compute_likelihood_accuracy(predicted_training_labels, train_labels)
    print ("Training Accuracy: {}".format(training_accuracy))

    predicted_test_labels = classify_data(test_data, means, covariances)
    test_accuracy = compute_likelihood_accuracy(predicted_test_labels, test_labels)
    print ("Test Accuracy: {}".format(test_accuracy))


def main():
    train_data, train_labels, test_data, test_labels = data.load_all_data('data')

    # Fit the model
    means = compute_mean_mles(train_data, train_labels)
    covariances = compute_sigma_mles(train_data, train_labels)

    # Plot the covariance diagonals
    print("========================================================")
    print("Plotting log of diagonal elements of covariance matrices")
    print("========================================================\n")
    plot_cov_diagonal(covariances)


    # Compute the average train and test log likelihoods
    print("==========================================================")
    print("Computing average conditional likelihood (training & test)")
    print("==========================================================\n")
    avg_train_log_likelihood = avg_conditional_likelihood(train_data, train_labels, means, covariances)
    avg_test_log_likelihood = avg_conditional_likelihood(test_data, test_labels, means, covariances)

    print("Average Training log-likelihood: {}\nAverage Test log-likelihood: {}".format(
        avg_train_log_likelihood,
        avg_test_log_likelihood
    ))

    print("=========================================================================")
    print("Computing most likely posterior class, printing accuracy (training, test)")
    print("=========================================================================\n")

    question_2_2_3(train_data, train_labels, test_data, test_labels, means, covariances)


    # Evaluation

if __name__ == '__main__':
    main()
