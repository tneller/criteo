"""Modified from https://kaggle2.blob.core.windows.net/forum-message-attachments/53646/1539/fast_solution.py
Details at https://www.kaggle.com/c/criteo-display-ad-challenge/discussion/10322
Changes:
- uses 1000000-entry data subset.
- n-fold cross-validation (CV)
- refactored for modularity and CV
"""


from datetime import datetime
from math import exp, log, sqrt, floor
import pandas as pd

# parameters #################################################################

train = 'train_small.txt'  # path to training file
# test = 'test.csv'  # path to testing file

D = 2 ** 20  # number of weights use for learning
alpha = .1  # learning rate for sgd optimization


def get_categorical_data():
    # Read data file
    print('Reading data... ', end='')
    with open(train) as file:
        rows = [line.rstrip('\n').split('\t') for line in file]
    print('({}, {})'.format(len(rows), len(rows[0])))
    y = [1 if row[0] is '1' else 0 for row in rows]

    # Convert to categorical features
    print('Converting all features to categorical... ', end='')
    cat_features = [row[1:] for row in rows]
    X = [get_x(X_cat_row, D) for X_cat_row in cat_features]
    print('done.')

    return X, y


# function definitions #######################################################

# A. Bounded logloss
# INPUT:
#     pred: our prediction
#     read: real answer
# OUTPUT
#     logarithmic loss of p given y
def logloss(pred, real):
    pred = max(min(pred, 1. - 10e-12), 10e-12)
    return -log(pred) if real == 1. else -log(1. - pred)


# B. Apply hash trick of the original csv row
# for simplicity, we treat both integer and categorical features as categorical
# INPUT:
#     csv_row: a csv dictionary, ex: {'Lable': '1', 'I1': '357', 'I2': '', ...}
#     num_hash_indices: the max index that we can hash to
# OUTPUT:
#     x: a list of indices that its value is 1
def get_x(csv_row, num_hash_indices):
    x = [0]  # 0 is the index of the bias term
    x.extend([int(value + str(idx), 16) % num_hash_indices for idx, value in enumerate(csv_row)])  # WARNING: weak hash
    # TODO - murmurhash is recommended by many in the Kaggle community
    # see murmurhash in http://scikit-learn.org/stable/developers/utilities.html
    return x  # x contains indices of bias term plus features that have a value of 1


# C. Get probability estimation on x
# INPUT:
#     x: features
#     w: weights
# OUTPUT:
#     probability of p(y = 1 | x; w)
def get_p(x, w):
    feature_weight_sum = 0.
    for i in x:  # compute feature weight sum
        feature_weight_sum += w[i] * 1.  # w[i] * x[i], but if i in x we got x[i] = 1.
    return 1. / (1. + exp(-max(min(feature_weight_sum, 20.), -20.)))  # bounded sigmoid


# D. Update given model
# INPUT:
#     w: weights
#     n: a counter that counts the number of times we encounter a feature
#        this is used for adaptive learning rate
#     x: feature
#     p: prediction of our model
#     y: answer
# OUTPUT:
#     w: updated model
#     n: updated count
def update_w(w, n, x, p, y):
    for i in x:
        # alpha / (sqrt(n) + 1) is the adaptive learning rate heuristic
        # (p - y) * x[i] is the current gradient
        # note that in our case, if i in x then x[i] = 1
        w[i] -= (p - y) * alpha / (sqrt(n[i]) + 1.)
        n[i] += 1.

    return w, n


def logistic_regression(X, y):
    # training and testing #######################################################

    # initialize our model
    w = [0.] * D  # weights
    n = [0.] * D  # number of times we've encountered a feature

    # start training a logistic regression model using on pass sgd
    loss = 0.
    size = len(X)
    for t in range(size):
        # main training procedure

        # step 1, get the hashed features
        x = X[t]
        target = y[t]

        # step 2, get prediction
        p = get_p(x, w)

        # for progress validation, useless for learning our model
        loss += logloss(p, target)
        if t % 100000 == 0 and t > 1:
            print('%s\tencountered: %d\tcurrent logloss: %f' % (datetime.now(), t, loss/t))

        # step 3, update model with answer
        w, n = update_w(w, n, x, p, target)

    print('%s\tencountered: %d\tcurrent logloss: %f' % (datetime.now(), size, loss / size))

    # return model
    return w, n


# # testing (build kaggle's submission file)
# with open('submission1234.csv', 'w') as submission:
#     submission.write('Id,Predicted\n')
#     for t, row in enumerate(DictReader(open(test))):
#         Id = row['Id']
#         del row['Id']
#         x = get_x(row, D)
#         p = get_p(x, w)
#         submission.write('%s,%f\n' % (Id, p))


def cross_validate(folds, X, y):
    size = len(X)
    fold_breaks = [int(floor(i * size / folds)) for i in range(folds)]
    fold_breaks.append(size)
    log_losses = []
    for i in range(folds):
        print('============================ Fold {}'.format(i + 1))
        # divide into training and validation data, keeping blocks continuous because of sequential nature of data
        X_train = X[0:fold_breaks[i]] + X[fold_breaks[i + 1]:size]
        y_train = y[0:fold_breaks[i]] + y[fold_breaks[i + 1]:size]
        X_val = X[fold_breaks[i]:fold_breaks[i + 1]]
        y_val = y[fold_breaks[i]:fold_breaks[i + 1]]

        # compute logistic regression model
        w, n = logistic_regression(X_train, y_train)

        # compute prodictions for validation set based on regression model and compute mean log-loss
        log_loss_sum = 0
        for j in range(len(X_val)):
            log_loss_sum += logloss(get_p(X_val[j], w), y_val[j])
        log_loss = log_loss_sum / len(X_val)
        log_losses.append(log_loss)
    print("Fold \tLog-Loss")
    for i in range(folds):
        print('{} \t{}'.format(i + 1, log_losses[i]))
    print('Log-Loss Statistics:')
    lls = pd.DataFrame(log_losses)
    print(lls.describe())


def main():
    print('Converting all data to categorical using hashing trick.')
    X, y = get_categorical_data()
    print('Performing 10-fold cross validation:')
    folds = 10
    cross_validate(folds, X, y)


main()
