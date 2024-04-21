from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pandas as pd
import pickle
import time


# optimized to run on gpu
# @jit(target_backend='cuda')

# data set for model ===================================================================================================
# data_set = pd.read_csv('datasets/dataset-preprocessed.csv')
# data_set = pd.read_csv('datasets/dimensionality_reduction/random_forest_feature_importances/dataset-syn-flag.csv')
# data_set = pd.read_csv('datasets/dimensionality_reduction/low_variance_filter/dataset-red.csv')
# data_set = pd.read_csv('datasets/dimensionality_reduction/low_variance_filter/dataset-orange.csv')
# data_set = pd.read_csv('datasets/dimensionality_reduction/low_variance_filter/dataset-yellow.csv')
#
# features = data_set.iloc[:, :-1].values
# labels = pd.factorize(data_set.iloc[:, -1].values)[0]
#
# features_train, features_test, labels_train, labels_test = train_test_split(
#     features, labels, test_size=0.2, random_state=1234
# )

# print(len(features_train), len(features_test))

# model = RandomForest(n_trees=20)
# model.fit(features_train, labels_train)


# Hyperparameters =====================================================================================================
# # Number of trees in random forest
# n_estimators = [int(x) for x in np.linspace(start=100, stop=2000, num=10)]
# # Number of features to consider at every split
# max_features = ['log2', 'sqrt']
# # Maximum number of levels in tree
# max_depth = [int(x) for x in np.linspace(10, 110, num=11)]
# max_depth.append(None)
# # Minimum number of samples required to split a node
# min_samples_split = [2, 5, 10]
# # Minimum number of samples required at each leaf node
# min_samples_leaf = [1, 2, 4]
# # Method of selecting samples for training each tree
# bootstrap = [True, False]
# # Create the random grid
# random_grid = {
#     'n_estimators': n_estimators,
#     # 'max_features': max_features,
#     'max_depth': max_depth,
#     'min_samples_split': min_samples_split,
#     'min_samples_leaf': min_samples_leaf,
#     'bootstrap': bootstrap
# }

# model training =======================================================================================================
# skit_model = RandomForestRegressor(n_estimators=2000, min_samples_split=5, min_samples_leaf=4, max_features='sqrt',
#                                    max_depth=None, bootstrap=False)
# skit_model = RandomForestRegressor(n_estimators=100, min_samples_split=2, min_samples_leaf=4,
#                                    max_depth=90, bootstrap=True)
# skit_model = RandomForestRegressor()
# skit_model = RandomForestClassifier()
# rf_random = RandomizedSearchCV(estimator=skit_model, param_distributions=random_grid,
#                                n_iter=100, cv=3, verbose=2,
#                                random_state=42, n_jobs=-1)

# skit_model.fit(features_train, labels_train)
# rf_random.fit(features_train, labels_train)
# print(rf_random.best_params_)
# {'n_estimators': 2000, 'min_samples_split': 5, 'min_samples_leaf': 4, 'max_features': 'sqrt', 'max_depth': None,
# 'bootstrap': False}
# {'n_estimators': 100, 'min_samples_split': 2, 'min_samples_leaf': 4, 'max_depth': 90, 'bootstrap': True}

# saving model ========================================================================================================
# # Save the model
# with open('model/model.pkl', 'wb') as f:
#     pickle.dump(skit_model, f)
#
#
# # Load the model
# with open('model/model.pkl', 'rb') as f:
#     model_from_pickle = pickle.load(f)

# model testing ========================================================================================================

# predictions = model_from_pickle.predict(features_test)
# predictions = skit_model.predict(features_test)

# efficiency metrics ==================================================================================================

# loss_function = metrics.log_loss(labels_test, predictions)
# f1_score = metrics.f1_score(labels_test, predictions)
# accuracy_rate = metrics.accuracy_score(labels_test, predictions)
# precision = metrics.precision_score(labels_test, predictions)
# recall = metrics.recall_score(labels_test, predictions)
# roc_auc = metrics.roc_auc_score(labels_test, predictions)
#
# print(loss_function)
#
# print(f1_score)
# print(accuracy_rate)
# print(precision)
# print(recall)
# print(roc_auc)


# random forest feature importance =================================================================================
# features = data_set.columns
# importances = skit_model.feature_importances_
# indices = np.argsort(importances)[-9:]  # top 10 features
# plt.title('Feature Importances')
# plt.barh(range(len(indices)), importances[indices], color='b', align='center')
# plt.yticks(range(len(indices)), [features[i] for i in indices])
# plt.xlabel('Relative Importance')
# plt.show()
#
# print([features[i] for i in indices])

def print_results(skit_model_rf, skit_model_tree, skit_model_knn, features_test, labels_test):
    for model in [skit_model_rf, skit_model_tree, skit_model_knn]:
        start = time.time()
        predictions = model.predict(features_test)
        stop = time.time()

        loss_function = metrics.log_loss(labels_test, predictions)
        f1_score = metrics.f1_score(labels_test, predictions)
        accuracy_rate = metrics.accuracy_score(labels_test, predictions)
        precision = metrics.precision_score(labels_test, predictions)
        recall = metrics.recall_score(labels_test, predictions)
        roc_auc = metrics.roc_auc_score(labels_test, predictions)

        print("train_model_preprocessed_knn")
        print(f"predict_time = {stop - start}")
        print(f"loss_function = {loss_function}")
        print(f"f1_score = {f1_score}")
        print(f"accuracy_rate = {accuracy_rate}")
        print(f"precision = {precision}")
        print(f"recall = {recall}")
        print(f"roc_auc = {roc_auc}")
        print()


def train_models(data_set):
    features = data_set.iloc[:, :-1].values
    labels = pd.factorize(data_set.iloc[:, -1].values)[0]

    features_train, features_test, labels_train, labels_test = train_test_split(
        features, labels, test_size=0.2, random_state=1234
    )

    skit_model_rf = RandomForestClassifier()
    skit_model_tree = DecisionTreeClassifier()
    skit_model_knn = KNeighborsClassifier()

    skit_model_rf.fit(features_train, labels_train)
    skit_model_tree.fit(features_train, labels_train)
    skit_model_knn.fit(features_train, labels_train)

    return skit_model_rf, skit_model_tree, skit_model_knn, features_test, labels_test


def train_model_preprocessed():
    data_set = pd.read_csv('datasets/dataset-preprocessed.csv')

    skit_model_rf, skit_model_tree, skit_model_knn, features_test, labels_test = train_models(data_set)

    with open('model/model_preprocessed_rf.pkl', 'wb') as f:
        pickle.dump(skit_model_rf, f)
    with open('model/model_preprocessed_tree.pkl', 'wb') as f:
        pickle.dump(skit_model_tree, f)
    with open('model/model_preprocessed_knn.pkl', 'wb') as f:
        pickle.dump(skit_model_knn, f)

    # with open('model/model_preprocessed.pkl', 'rb') as f:
    #     model_from_pickle = pickle.load(f)

    # predictions = model_from_pickle.predict(features_test)

    print_results(
        skit_model_rf,
        skit_model_tree,
        skit_model_knn,
        features_test,
        labels_test
    )


def train_model_syn_flag():
    data_set = pd.read_csv('datasets/dimensionality_reduction/random_forest_feature_importances/dataset-syn-flag.csv')

    skit_model_rf, skit_model_tree, skit_model_knn, features_test, labels_test = train_models(data_set)

    with open('model/model_syn_flag_rf.pkl', 'wb') as f:
        pickle.dump(skit_model_rf, f)
    with open('model/model_syn_flag_tree.pkl', 'wb') as f:
        pickle.dump(skit_model_tree, f)
    with open('model/model_syn_flag_knn.pkl', 'wb') as f:
        pickle.dump(skit_model_knn, f)

    # with open('model/model_syn_flag.pkl', 'rb') as f:
    #     model_from_pickle = pickle.load(f)
    #
    # predictions = model_from_pickle.predict(features_test)

    print_results(
        skit_model_rf,
        skit_model_tree,
        skit_model_knn,
        features_test,
        labels_test
    )


def train_model_red():
    data_set = pd.read_csv('datasets/dimensionality_reduction/low_variance_filter/dataset-red.csv')

    skit_model_rf, skit_model_tree, skit_model_knn, features_test, labels_test = train_models(data_set)

    with open('model/model_red_rf.pkl', 'wb') as f:
        pickle.dump(skit_model_rf, f)
    with open('model/model_red_tree.pkl', 'wb') as f:
        pickle.dump(skit_model_tree, f)
    with open('model/model_red_knn.pkl', 'wb') as f:
        pickle.dump(skit_model_knn, f)

    # with open('model/model_red.pkl', 'rb') as f:
    #     model_from_pickle = pickle.load(f)
    #
    # predictions = model_from_pickle.predict(features_test)

    print_results(
        skit_model_rf,
        skit_model_tree,
        skit_model_knn,
        features_test,
        labels_test
    )


def train_model_orange():
    data_set = pd.read_csv('datasets/dimensionality_reduction/low_variance_filter/dataset-orange.csv')

    skit_model_rf, skit_model_tree, skit_model_knn, features_test, labels_test = train_models(data_set)

    with open('model/model_orange_rf.pkl', 'wb') as f:
        pickle.dump(skit_model_rf, f)
    with open('model/model_orange_tree.pkl', 'wb') as f:
        pickle.dump(skit_model_tree, f)
    with open('model/model_orange_knn.pkl', 'wb') as f:
        pickle.dump(skit_model_knn, f)

    # with open('model/model_orange.pkl', 'rb') as f:
    #     model_from_pickle = pickle.load(f)
    #
    # predictions = model_from_pickle.predict(features_test)

    print_results(
        skit_model_rf,
        skit_model_tree,
        skit_model_knn,
        features_test,
        labels_test
    )


def train_model_yellow():
    data_set = pd.read_csv('datasets/dimensionality_reduction/low_variance_filter/dataset-yellow.csv')

    skit_model_rf, skit_model_tree, skit_model_knn, features_test, labels_test = train_models(data_set)

    with open('model/model_yellow_rf.pkl', 'wb') as f:
        pickle.dump(skit_model_rf, f)
    with open('model/model_yellow_tree.pkl', 'wb') as f:
        pickle.dump(skit_model_tree, f)
    with open('model/model_yellow_knn.pkl', 'wb') as f:
        pickle.dump(skit_model_knn, f)

    # with open('model/model_yellow.pkl', 'rb') as f:
    #     model_from_pickle = pickle.load(f)
    #
    # predictions = model_from_pickle.predict(features_test)

    print_results(
        skit_model_rf,
        skit_model_tree,
        skit_model_knn,
        features_test,
        labels_test
    )


def train_model_correlation_orange():
    data_set = pd.read_csv(
        'datasets/dimensionality_reduction/high_correlation_filter/data-set-correlation-orange-1.csv'
    )

    skit_model_rf, skit_model_tree, skit_model_knn, features_test, labels_test = train_models(data_set)

    with open('model/model_correlation_orange_rf.pkl', 'wb') as f:
        pickle.dump(skit_model_rf, f)
    with open('model/model_correlation_orange_tree.pkl', 'wb') as f:
        pickle.dump(skit_model_tree, f)
    with open('model/model_correlation_orange_knn.pkl', 'wb') as f:
        pickle.dump(skit_model_knn, f)

    # with open('model/model_correlation_orange.pkl', 'rb') as f:
    #     model_from_pickle = pickle.load(f)
    #
    # predictions = model_from_pickle.predict(features_test)

    print_results(
        skit_model_rf,
        skit_model_tree,
        skit_model_knn,
        features_test,
        labels_test
    )


def train_model_correlation_orange_backward():
    data_set = pd.read_csv(
        'datasets/dimensionality_reduction/backward_feature_elimination/data-set-correlation-orange-1-backward.csv')

    skit_model_rf, skit_model_tree, skit_model_knn, features_test, labels_test = train_models(data_set)

    with open('model/model_correlation_orange_backward_rf.pkl', 'wb') as f:
        pickle.dump(skit_model_rf, f)
    with open('model/model_correlation_orange_backward_tree.pkl', 'wb') as f:
        pickle.dump(skit_model_tree, f)
    with open('model/model_correlation_orange_backward_knn.pkl', 'wb') as f:
        pickle.dump(skit_model_knn, f)

    # with open('model/model_correlation_orange_backward.pkl', 'rb') as f:
    #     model_from_pickle = pickle.load(f)
    #
    # predictions = model_from_pickle.predict(features_test)

    print_results(
        skit_model_rf,
        skit_model_tree,
        skit_model_knn,
        features_test,
        labels_test
    )


train_model_preprocessed()
train_model_syn_flag()
train_model_red()
train_model_orange()
train_model_yellow()
train_model_correlation_orange()
train_model_correlation_orange_backward()
