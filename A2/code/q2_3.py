'''
Question 2.3 Skeleton Code

Here you should implement and evaluate the Naive Bayes classifier.
'''

import data
import numpy as np
# Import pyplot - plt.imshow is useful!
import matplotlib.pyplot as plt


BETA_A = 2
BETA_B = 2

def binarize_data(pixel_values):
    '''
    Binarize the data by thresholding around 0.5
    '''
    return np.where(pixel_values > 0.5, 1.0, 0.0)

def compute_parameters(train_data, train_labels):
    '''
    Compute the eta MAP estimate/MLE with augmented data

    You should return a numpy array of shape (10, 64)
    where the ith row corresponds to the ith digit class.
    '''
    classes = []
    for i in range(0, 10):
        i_digits = data.get_digits_by_label(train_data, train_labels, i)
        class_sum = np.sum(i_digits, axis=0)
        Nc = class_sum
        N = i_digits.shape[0]
        class_probability = (Nc + BETA_A - 1)/(N + BETA_A + BETA_B - 2)
        classes.append(class_probability)
    return np.array(classes)


def plot_images(class_images):
    '''
    Plot each of the images corresponding to each class side by side in grayscale
    '''
    reshaped_class_images = []
    for i in range(10):
        img_i = class_images[i]
        img_i = np.reshape(img_i, (-1, 8))
        reshaped_class_images.append(img_i)
    plt.imshow(np.concatenate(np.array(reshaped_class_images), 1), cmap='gray')
    plt.show()

def generate_new_data(eta):
    '''
    Sample a new data point from your generative distribution p(x|y,theta) for
    each value of y in the range 0...10

    Plot these values
    '''

    samples = []
    for k in eta:
        sample_k = np.random.binomial(1, p=k)
        samples.append(sample_k)

    plot_images(np.array(samples))

def generative_likelihood(bin_digits, eta):
    '''
    Compute the generative log-likelihood:
        log p(x|y, eta)

    Should return an n x 10 numpy array
    '''
    gen_likelihood = []
    for b in bin_digits:
        b_prob = []
        for i in range(0,10):
            nk = eta[i]
            prob_k = 0
            for j in range(0, 64):
                nkj_bj = nk[j]**(b[j])
                one_nkj_nj = (1 - nk[j])**(1 - b[j])
                prob_k += np.log(nkj_bj) + np.log(one_nkj_nj)
            b_prob.append(prob_k)
        gen_likelihood.append(np.array(b_prob))

    return np.array(gen_likelihood)

def conditional_likelihood(bin_digits, eta):
    '''
    Compute the conditional likelihood:

        log p(y|x, eta)

    This should be a numpy array of shape (n, 10)
    Where n is the number of datapoints and 10 corresponds to each digit class
    '''
    prob_yk = 1/10
    gen_likelihood = generative_likelihood(bin_digits, eta)
    return gen_likelihood + np.log(prob_yk)

def avg_conditional_likelihood(bin_digits, labels, eta):
    '''
    Compute the average conditional likelihood over the true class labels

        AVG( log p(y_i|x_i, eta) )

    i.e. the average log likelihood that the model assigns to the correct class label
    '''
    cond_likelihood = conditional_likelihood(bin_digits, eta)
    avg = 0
    for i, datum in enumerate(cond_likelihood):
        avg += datum[int(labels[i])]

    return avg/cond_likelihood.shape[0]

def classify_data(bin_digits, eta):
    '''
    Classify new points by taking the most likely posterior class
    '''
    cond_likelihood = conditional_likelihood(bin_digits, eta)
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

def question_2_3_6(train_data, train_labels, test_data, test_labels):
    train_eta = compute_parameters(train_data, train_labels)
    predicted_training_labels = classify_data(train_data, train_eta)
    training_accuracy = compute_likelihood_accuracy(predicted_training_labels, train_labels)
    print ("Training Accuracy: {}".format(training_accuracy))

    test_eta = compute_parameters(test_data, test_labels)
    predicted_test_labels = classify_data(test_data, test_eta)
    test_accuracy = compute_likelihood_accuracy(predicted_test_labels, test_labels)
    print ("Test Accuracy: {}".format(test_accuracy))

def main():
    train_data, train_labels, test_data, test_labels = data.load_all_data('data')
    train_data, test_data = binarize_data(train_data), binarize_data(test_data)

    # Fit the model
    print("=================")
    print("Computing the ETA")
    print("=================\n")
    eta = compute_parameters(train_data, train_labels)

    # Evaluation
    print("============")
    print("Plotting ETA")
    print("============\n")

    plot_images(eta)

    generate_new_data(eta)

    # Compute the average train and test log likelihoods
    print("==========================================================")
    print("Computing average conditional likelihood (training & test)")
    print("==========================================================\n")

    avg_train_log_likelihood = avg_conditional_likelihood(train_data, train_labels, eta)
    avg_test_log_likelihood = avg_conditional_likelihood(test_data, test_labels, eta)

    print("Average Training log-likelihood: {}\nAverage Test log-likelihood: {}".format(
        avg_train_log_likelihood,
        avg_test_log_likelihood
    ))

    print("=========================================================================")
    print("Computing most likely posterior class, printing accuracy (training, test)")
    print("=========================================================================\n")

    question_2_3_6(train_data, train_labels, test_data, test_labels)


if __name__ == '__main__':
    main()
