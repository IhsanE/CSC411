'''
Question 2.1 Skeleton Code

Here you should implement and evaluate the k-NN classifier.
'''

import data
import numpy as np
# Import pyplot - plt.imshow is useful!
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold

class KNearestNeighbor(object):
    '''
    K Nearest Neighbor classifier
    '''

    def __init__(self, train_data, train_labels):
        self.train_data = train_data
        self.train_norm = (self.train_data**2).sum(axis=1).reshape(-1,1)
        self.train_labels = train_labels

    def l2_distance(self, test_point):
        '''
        Compute L2 distance between test point and each training point

        Input: test_point is a 1d numpy array
        Output: dist is a numpy array containing the distances between the test point and each training point
        '''
        # Process test point shape
        test_point = np.squeeze(test_point)
        if test_point.ndim == 1:
            test_point = test_point.reshape(1, -1)
        assert test_point.shape[1] == self.train_data.shape[1]

        # Compute squared distance
        train_norm = (self.train_data**2).sum(axis=1).reshape(-1,1)
        test_norm = (test_point**2).sum(axis=1).reshape(1,-1)
        dist = self.train_norm + test_norm - 2*self.train_data.dot(test_point.transpose())
        return np.squeeze(dist)

    def query_knn(self, test_point, k):
        '''
        Query a single test point using the k-NN algorithm

        You should return the digit label provided by the algorithm
        '''
        distances = self.l2_distance(test_point)

        # Compute minimum k distances, record the related index
        min_dist_indices = np.argpartition(distances, k)[:k]
        closest_labels = self.train_labels[min_dist_indices]
        closest_labels = np.array([int(i) for i in closest_labels])

        # Compute counts for each class label
        bincount_labels = np.bincount(closest_labels)

        # All counts are unique, return the most frequent label
        if (k == 1 or len(bincount_labels) == len(set(bincount_labels))):
            return np.bincount(closest_labels).argmax()
        else:
            return self.query_knn(test_point, k - 1)

def cross_validation(knn, test_data, test_labels, k_range=np.arange(1,16)):
    kf = KFold(n_splits = 10)
    X = knn.train_data
    y = knn.train_labels
    k_fold_avg = []
    for k in k_range:
        k_accuracy_avg = 0
        for train_idx, test_idx in kf.split(X):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            k_knn = KNearestNeighbor(X_train, y_train)

            eval_data = compute_knn_data(X_test, k, k_knn)
            accuracy = classification_accuracy(k_knn, k, eval_data, y_test)
            k_accuracy_avg += accuracy
        k_accuracy_avg /= 10
        k_fold_avg.append(k_accuracy_avg)
        print("K : {}\nAVERAGE ACCURACY : {}".format(k, k_accuracy_avg))

    optimal_k = len(k_fold_avg) - np.array(k_fold_avg[::-1]).argmax()

    return optimal_k


def classification_accuracy(knn, k, eval_data, eval_labels):
    '''
    Evaluate the classification accuracy of knn on the given 'eval_data'
    using the labels
    '''
    return accuracy_score(eval_labels, eval_data)

def compute_knn_data(data, k, knn):
    eval_data = []
    for datum in data:
        datum_label = knn.query_knn(datum, k)
        eval_data.append(datum_label)
    return eval_data

def answer_2_1_helper(knn, test_data, test_labels, k):
    k_1_eval_data = compute_knn_data(test_data, k, knn)

    k_1_accuracy = classification_accuracy(knn, k, k_1_eval_data, test_labels)
    print("[TEST]\nK : {}\nACCURACY : {}".format(k, k_1_accuracy))

    k_1_train_data = compute_knn_data(knn.train_data, k, knn)

    k_1_train_accuracy = classification_accuracy(knn, k, k_1_train_data, knn.train_labels)
    print("[TRAINING]\nK : {}\nACCURACY : {}".format(k, k_1_train_accuracy))

def answer_2_1(knn, test_data, test_labels):
    k = 1
    answer_2_1_helper(knn, test_data, test_labels, k)

    k = 15
    answer_2_1_helper(knn, test_data, test_labels, k)



def main():
    train_data, train_labels, test_data, test_labels = data.load_all_data('data')
    knn = KNearestNeighbor(train_data, train_labels)

    # Compute result for section 2.1.1
    answer_2_1(knn, test_data, test_labels)

    # Perform K-Fold for section 2.1.3
    opt_k = cross_validation(knn, test_data, test_labels)
    print("Optimal K from cross-validation: {}".format(opt_k))

    knn = KNearestNeighbor(train_data, train_labels)
    answer_2_1_helper(knn, test_data, test_labels, opt_k)

if __name__ == '__main__':
    main()
