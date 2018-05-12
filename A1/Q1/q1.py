from sklearn import datasets
import matplotlib.pyplot as plt
import numpy as np

def load_data():
    boston = datasets.load_boston()
    X = boston.data
    y = boston.target
    features = boston.feature_names
    return X,y,features


def compute_2D_mask(inner_length, size, probability_distribution):
    return np.random.choice(
        [True, False],
        size,
        p=probability_distribution
    )


def compute_training_testing_data(X, y, num_features):
    N = num_features + 1
    data_sets = {}
    random_mask = compute_2D_mask(N, len(X), [0.8, 0.2])

    # Modify mask to have dimensionality equal to feature space
    X_mask = np.array([[i]*N for i in random_mask])
    y_mask = np.array(random_mask)

    # Apply masks to multi-dimensional input data
    X_training_masked = np.ma.masked_array(X, X_mask)
    X_test_masked = np.ma.masked_array(X, ~X_mask)

    # Apply masks to 1-D target data, which removes elements under mask
    y_training = y[y_mask]
    y_test = y[~y_mask]

    # Clean mask from input data
    data_sets['X_training'] = np.array(list(filter(lambda x: np.ma.is_masked(x), X_training_masked)))
    data_sets['X_test'] = np.array(list(filter(lambda x: np.ma.is_masked(x), X_test_masked)))

    data_sets['y_training'] = y_training
    data_sets['y_test'] = y_test

    return data_sets


# Create a table with feature as column title, and weights as data
def feature_weight_table(w, features):
    fig, ax = plt.subplots(figsize=(20,5))
    fig.patch.set_visible(False)
    ax.axis('off')
    ax.axis('tight')


    ax.table(cellText=[w], colLabels=['BIAS'] + features.tolist(), loc='center')

    fig.tight_layout()

    plt.show()


def visualize(X, y, features, title=None):
    plt.figure(figsize=(20, 5))

    # The first 'feature' is the bias, skip over this
    feature_count = X.shape[1] - 1

    # i: index
    for i in range(0, feature_count):
        ax = plt.subplot(3, 5, i + 1)
        ax.scatter([training_vector[i+1] for training_vector in X], y)
        ax.set_xlabel(features[i])
        ax.set_ylabel('target')
        if title:
            ax.set_title(title)

    plt.tight_layout()
    plt.show()


def fit_regression(X,Y):
    # Remember to use np.linalg.solve instead of inverting!

    X_trans = np.transpose(X)
    XTX = np.matmul(X_trans, X)
    I = np.identity(len(XTX))

    XTY = np.matmul(X_trans, Y)

    # w*((XTX)^-1) = (XTy)
    # w* = ((XTX)^-1) x (XTy)
    w_opt = np.linalg.solve(XTX, XTY)

    return w_opt


def compute_linear_regression(X, w):
    return np.matmul(X, w)


def compute_mean_square_error(y, y_computed_target):
    return ((y - y_computed_target)**2).mean()


def compute_mean_absolute_error(y, y_computed_target):
    return (abs(y - y_computed_target)).mean()


def compute_huber_error(y, y_computed_target):
    # Huber constant for c
    c = 1.345
    sum = 0
    i = 0
    for y_i, y_target_i in tuple(zip(y, y_computed_target)):
        i += 1
        residual = y_i - y_target_i
        if (abs(residual) <= c):
            sum += residual**2
        else:
            sum += c * ((2 * abs(residual)) - c)
    return sum/i


def get_mean_square_error(X, y, w):
    y_computed_target = compute_linear_regression(X, w)
    return compute_mean_square_error(y, y_computed_target)


def main():
    # Load the data
    X, y, features = load_data()

    # Add bias term of 1
    X = np.insert(X, 0, 1, axis=1)

    print("Features: {}".format(features))

    # Visualize the features and targets of training data
    visualize(X, y, features, '[training] features vs y-values')

    # Split data into train and test
    testing_training_data = compute_training_testing_data(X, y, len(features))
    X_training = testing_training_data['X_training']
    y_training = testing_training_data['y_training']

    # Fit regression model
    w = fit_regression(X_training, y_training)
    targets = compute_linear_regression(X_training, w)

    # Visualize the features against target values
    visualize(X_training, y_training, features, '[training] features vs targets')

    # Visualize the feature/weight table
    feature_weight_table(w, features)

    # Compute fitted values, MSE, etc.
    X_test = testing_training_data['X_test']
    y_test = testing_training_data['y_test']

    y_computed_target = compute_linear_regression(X_test, w)

    mse = compute_mean_square_error(y_test, y_computed_target)
    mae = compute_mean_absolute_error(y_test, y_computed_target)
    huber_error = compute_huber_error(y_test, y_computed_target)

    visualize(X_test, y_computed_target, features, '[test] features vs targets')

    print("Mean Square Error: {}".format(mse))
    print("Mean Absolute Error: {}".format(mae))
    print("Huber Error: {}".format(huber_error))


if __name__ == "__main__":
    main()
