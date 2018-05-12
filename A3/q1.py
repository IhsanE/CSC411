'''
Question 1 Skeleton Code


'''

import sklearn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import BernoulliNB
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.neural_network import MLPClassifier as MLP
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2


def load_data():
    # import and filter data
    newsgroups_train = fetch_20newsgroups(subset='train',remove=('headers', 'footers', 'quotes'))
    newsgroups_test = fetch_20newsgroups(subset='test',remove=('headers', 'footers', 'quotes'))

    return newsgroups_train, newsgroups_test

def bow_features(train_data, test_data):
    # Bag-of-words representation
    bow_vectorize = CountVectorizer()
    bow_train = bow_vectorize.fit_transform(train_data.data) #bag-of-word features for training data
    bow_test = bow_vectorize.transform(test_data.data)
    feature_names = bow_vectorize.get_feature_names() #converts feature index to the word it represents.
    shape = bow_train.shape
    print('{} train data points.'.format(shape[0]))
    print('{} feature dimension.'.format(shape[1]))
    print('Most common word in training set is "{}"'.format(feature_names[bow_train.sum(axis=0).argmax()]))
    return bow_train, bow_test, feature_names

def tf_idf_features(train_data, test_data):
    # Bag-of-words representation
    tf_idf_vectorize = TfidfVectorizer()
    tf_idf_train = tf_idf_vectorize.fit_transform(train_data.data) #bag-of-word features for training data
    feature_names = tf_idf_vectorize.get_feature_names() #converts feature index to the word it represents.
    tf_idf_test = tf_idf_vectorize.transform(test_data.data)
    return tf_idf_train, tf_idf_test, feature_names

def bnb_baseline(bow_train, train_labels, bow_test, test_labels):
    # training the baseline model
    binary_train = (bow_train>0).astype(int)
    binary_test = (bow_test>0).astype(int)

    model = BernoulliNB()
    model.fit(binary_train, train_labels)

    #evaluate the baseline model
    train_pred = model.predict(binary_train)
    print('BernoulliNB baseline train accuracy = {}'.format((train_pred == train_labels).mean()))

    test_pred = model.predict(binary_test)
    print('BernoulliNB baseline test accuracy = {}'.format((test_pred == test_labels).mean()))

    return model

def cross_validation(model, hyperparam, X, y, test_data, test_labels, hyper_range):
    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import KFold
    kf = KFold(n_splits = 10)
    k_fold_avg = []
    for k in hyper_range:
        k_accuracy_avg = 0
        for train_idx, test_idx in kf.split(X):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            setattr(model, hyperparam, k)
            model.fit(X_train, y_train)
            test_pred = model.predict(X_test)

            k_accuracy_avg += accuracy_score(test_pred, y_test)
        k_accuracy_avg /= 10
        k_fold_avg.append(k_accuracy_avg)
        print("{} : {}\nAVERAGE ACCURACY : {}".format(hyperparam, k, k_accuracy_avg))

    optimal_k = len(k_fold_avg) - np.array(k_fold_avg[::-1]).argmax()
    print("optimal {} : {}".format(hyperparam, optimal_k))

    return optimal_k

def knn(train_data, train_labels, test_data, test_labels):
    model = KNN(n_neighbors=9) # cross validated to get this
    model.fit(train_data, train_labels)

    #evaluate the baseline model
    train_pred = model.predict(train_data)
    print('KNN train accuracy = {}'.format((train_pred == train_labels).mean()))

    test_pred = model.predict(test_data)
    print('KNN test accuracy = {}'.format((test_pred == test_labels).mean()))

    return model

def random_forest(train_data, train_labels, test_data, test_labels):
    model = RandomForestClassifier(n_estimators=150)
    model.fit(train_data, train_labels)

    #evaluate the baseline model
    train_pred = model.predict(train_data)
    print('Random Forest train accuracy = {}'.format((train_pred == train_labels).mean()))

    test_pred = model.predict(test_data)
    print('Random Forest test accuracy = {}'.format((test_pred == test_labels).mean()))

    return model

def confusion_matrix(y_true, y_pred, labels):
    fig, ax = plt.subplots(figsize=(30,10))
    fig.patch.set_visible(False)
    ax.axis('off')
    ax.axis('tight')

    matrix = [[0 for y in range(20)] for x in range(20)]
    for label, pred in zip(y_true, y_pred):
        matrix[label][pred] += 1

    main_table = ax.table(cellText=matrix, colLabels=list(range(20)), rowLabels=list(range(20)), loc='center')
    main_table.auto_set_font_size(False)
    main_table.set_fontsize(11)

    legend = ax.table(cellText=[list(range(20))], loc='upper center', colLabels=labels)

    legend.auto_set_font_size(False)
    legend.set_fontsize(6)

    fig.tight_layout()
    plt.show()

def svm(train_data, train_labels, test_data, test_labels, class_names):
    model = SGDClassifier(n_iter=45)
    # model = LinearSVC(random_state=0)
    model.fit(train_data, train_labels)

    #evaluate the baseline model
    train_pred = model.predict(train_data)
    print('SVM train accuracy = {}'.format((train_pred == train_labels).mean()))

    test_pred = model.predict(test_data)
    print('SVM test accuracy = {}'.format((test_pred == test_labels).mean()))

    confusion_matrix(test_labels, test_pred, class_names)
    return model

if __name__ == '__main__':
    train_data, test_data = load_data()
    train_bow, test_bow, _ = bow_features(train_data, test_data)
    train_tfidf, test_tfidf, feature_names = tf_idf_features(train_data, test_data)

    print ('Running Bernoulli Naive Bayes Model\n')
    bnb_model = bnb_baseline(train_bow, train_data.target, test_bow, test_data.target)


    KBest_tfidf = SelectKBest(chi2, k=65000)

    train_tfidf = KBest_tfidf.fit_transform(train_tfidf, train_data.target.reshape((-1, 1)))
    test_tfidf = KBest_tfidf.transform(test_tfidf)

    print ('\nRunning Random Forest Model\n')
    random_forest_model = random_forest(train_tfidf, train_data.target, test_tfidf, test_data.target)

    KBest_tfidf = SelectKBest(chi2, k=40000)

    train_tfidf = KBest_tfidf.fit_transform(train_tfidf, train_data.target.reshape((-1, 1)))
    test_tfidf = KBest_tfidf.transform(test_tfidf)


    print ('\nRunning SVM Model\n')
    svm_model = svm(train_tfidf, train_data.target, test_tfidf, test_data.target, train_data.target_names)

    KBest_tfidf = SelectKBest(chi2, k=200)

    train_tfidf = KBest_tfidf.fit_transform(train_tfidf, train_data.target.reshape((-1, 1)))
    test_tfidf = KBest_tfidf.transform(test_tfidf)

    print ('\nRunning KNN Model\n')
    knn_model = knn(train_tfidf, train_data.target, test_tfidf, test_data.target)
