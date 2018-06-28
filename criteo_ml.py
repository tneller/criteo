"""A modular framework for experimenting with different learning ideas with the Criteo Display Advertising Kaggle
Challenge.
"""
import pandas as pd
import xgboost as xgb
from sklearn.feature_extraction import FeatureHasher
import copy
import numpy as np
# from scipy import sparse

# Parameters:
train_filename = 'train_small.txt'  # path to Criteo training file
fields_filename = 'fields.txt'  # path to fields of Criteo data
verbose = False
num_threads = 20
pd.set_option('display.max_columns', 10)
pd.set_option('display.width', 80)
# more on modifying pandas output:
# https://stackoverflow.com/questions/11707586/python-pandas-how-to-widen-output-display-to-see-more-columns

# Read Criteo field names
if verbose:
    print('Reading field names from {}...'.format(fields_filename))
with open(fields_filename) as file:
    field_names = file.readline().rstrip('\n').split('\t')
# print(field_names)
target_name = field_names[0]
predictor_names = field_names[1:]
categorical_predictor_names = [field_name for field_name in predictor_names if field_name.startswith('C')]
noncategorical_predictor_names = [field_name for field_name in predictor_names if not field_name.startswith('C')]


def file_to_dataframe(filename):
    # Create and return pandas dataframe, reading directly from Criteo data
    if verbose:
        print('Reading data in {} as pandas dataframe...'.format(train_filename))
    df = pd.read_table(filename, names=field_names, header=None)
    if verbose:
        print(df.shape)
        print(df.describe())
        print(df.dtypes)
    return df


def xgb_counts_only():
    print('Using Criteo count predictors only, ignoring categoricals:')
    df = file_to_dataframe(train_filename)
    y = df[[target_name]]
    x = df[noncategorical_predictor_names]
    # DataFrame.dtypes for data must be int, float or bool, so a lazy, minimalist use of XGBoost might only treat
    # numeric count data.

    num_folds = 10
    early_stop_rounds = 5
    max_rounds = 5000
    # params documentation: https://xgboost.readthedocs.io/en/latest/python/python_api.html
    params = {'objective': 'binary:logistic', 'silent': 1, 'eval_metric': 'logloss', 'nthread': num_threads}
    xg_train = xgb.DMatrix(x, label=y)
    print("{}-fold cross validation with logloss metric, early stopping after {} non-decreasing logloss iterations."
          .format(num_folds, early_stop_rounds))
    cv = xgb.cv(params, xg_train, max_rounds, nfold=num_folds, early_stopping_rounds=early_stop_rounds, verbose_eval=1)
    # Note: cv is a pandas DataFrame with each row representing a round's logloss results. I only print the last.
    # The test-logloss-mean is the main measure of interest for our comparison.
    print(cv[-1:])


def xgb_categorical_hashing(hash_size):
    print('Using Criteo count predictors and {}-hashed categorical features:'.format(hash_size))
    df = file_to_dataframe(train_filename)
    y = df[[target_name]]
    x_noncat = df[noncategorical_predictor_names]
    x_cat = df[categorical_predictor_names]
    # DataFrame.dtypes for data must be int, float or bool, so one common approach to categorical data is to
    # prepend the field name to the category string and hash it to an index (e.g. 0 - 999,999), so that each
    # categorical results in a one-hot hashed encoding. Collisions may occur, but with enough indices, the trick
    # works well in practice.

    # This code is based on the "hashing trick" used in a number of CTR prediction competitions and discussed here:
    # https://blog.myyellowroad.com/using-categorical-data-in-machine-learning-with-python-from-dummy-variables-to-deep-category-66041f734512
    x_cat_hash = copy.copy(x_cat)
    for i in range(x_cat_hash.shape[1]):
        x_cat_hash.iloc[:, i] = x_cat_hash.columns[i] + ':' + x_cat_hash.iloc[:, i].astype('str')
    h = FeatureHasher(n_features=hash_size, input_type="string")
    x_cat_hash = pd.SparseDataFrame(h.transform(x_cat_hash.values))
    x = x_noncat.to_sparse(fill_value=None).join(x_cat_hash)
    # x = sparse.csr_matrix(x.to_coo())
    num_folds = 10
    early_stop_rounds = 5
    max_rounds = 5000
    # params documentation: https://xgboost.readthedocs.io/en/latest/python/python_api.html
    params = {'objective': 'binary:logistic', 'silent': 1, 'eval_metric': 'logloss', 'nthread': num_threads}
    xg_train = xgb.DMatrix(x, label=y, )
    print("{}-fold cross validation with logloss metric, early stopping after {} non-decreasing logloss iterations."
          .format(num_folds, early_stop_rounds))
    cv = xgb.cv(params, xg_train, max_rounds, nfold=num_folds, early_stopping_rounds=early_stop_rounds, verbose_eval=1)
    # Note: cv is a pandas DataFrame with each row representing a round's logloss results. I only print the last.
    # The test-logloss-mean is the main measure of interest for our comparison.
    print(cv[-1:])


def one_hot_most_freq_categories():
    print('One-hot-encoding most frequent categorical values... ', end='')
    df = file_to_dataframe(train_filename)
    y = df[[target_name]]
    num_rows = y.shape[0]
    x_noncat = df[noncategorical_predictor_names]
    x_cat = df[categorical_predictor_names]
    # Read most frequent categorical values
    with open('1000_most_frequent_categories.txt') as file:
        freq_cats = [line.rstrip('\n') for line in file]
        cat_set = set(freq_cats)
        x_onehot = pd.DataFrame(0, index=np.arange(num_rows), columns=freq_cats)
        for r in np.arange(num_rows):
            for c in categorical_predictor_names:
                cat = '{}:{}'.format(c, x_cat)
                if cat in freq_cats:
                    x_onehot.loc[r][cat] = 1
    x = x_noncat.join(x_onehot, how='right')
    print(x.describe())
    print(x.head())
    print('done.')
    return y, x

