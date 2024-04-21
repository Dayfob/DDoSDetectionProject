import numpy as np
from numba import jit, cuda
from collections import Counter


class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, *, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def is_leaf_node(self):
        return self.value is not None


def _split(features_column, split_thresh):
    left_indexes = np.argwhere(features_column <= split_thresh).flatten()
    right_indexes = np.argwhere(features_column > split_thresh).flatten()
    return left_indexes, right_indexes


def _entropy(labels):
    # hist = np.bincount(labels)
    label_options, hist = np.unique(labels, return_counts=True)
    # ps = hist / len(labels)
    ps = np.divide(hist, len(labels))
    return -np.sum([p * np.log(p) for p in ps if p > 0])


def _most_common_label(labels):
    counter = Counter(labels)
    value = counter.most_common(1)[0][0]
    return value


def _information_gain(labels, features_column, threshold):
    # parent entropy
    parent_entropy = _entropy(labels)

    # create children
    left_indexes, right_indexes = _split(features_column, threshold)

    if len(left_indexes) == 0 or len(right_indexes) == 0:
        return 0

    # calculate the weighted avg. entropy of children
    n = len(labels)
    n_l, n_r = len(left_indexes), len(right_indexes)
    e_l, e_r = _entropy(labels[left_indexes]), _entropy(labels[right_indexes])
    child_entropy = (n_l / n) * e_l + (n_r / n) * e_r

    # calculate the IG
    information_gain = parent_entropy - child_entropy
    return information_gain


def _best_split(features, labels, feat_indexes):
    best_gain = -1
    split_index, split_threshold = None, None

    for features_index in feat_indexes:
        features_column = features[:, features_index]
        thresholds = np.unique(features_column)

        for threshold in thresholds:
            # calculate the information gain
            gain = _information_gain(labels, features_column, threshold)

            if gain > best_gain:
                best_gain = gain
                split_index = features_index
                split_threshold = threshold

    return split_index, split_threshold


class DecisionTree:
    def __init__(self, min_samples_split=2, max_depth=100, n_features=None):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.n_features = n_features
        self.root = None

    def fit(self, features, labels):
        self.n_features = features.shape[1] if not self.n_features else min(features.shape[1], self.n_features)
        self.root = self._grow_tree(features, labels)

    def _grow_tree(self, features, labels, depth=0):
        n_samples, n_features = features.shape
        n_labels = len(np.unique(labels))

        # check the stopping criteria
        if depth >= self.max_depth or n_labels == 1 or n_samples < self.min_samples_split:
            leaf_value = _most_common_label(labels)
            return Node(value=leaf_value)

        feat_indexes = np.random.choice(n_features, self.n_features, replace=False)

        # find the best split
        best_feature, best_thresh = _best_split(features, labels, feat_indexes)

        # create child nodes
        left_indexes, right_indexes = _split(features[:, best_feature], best_thresh)
        left = self._grow_tree(features[left_indexes, :], labels[left_indexes], depth + 1)
        right = self._grow_tree(features[right_indexes, :], labels[right_indexes], depth + 1)
        return Node(best_feature, best_thresh, left, right)

    def predict(self, features):
        return np.array([self._traverse_tree(feature, self.root) for feature in features])

    def _traverse_tree(self, features, node):
        if node.is_leaf_node():
            return node.value

        if features[node.feature] <= node.threshold:
            return self._traverse_tree(features, node.left)
        return self._traverse_tree(features, node.right)
