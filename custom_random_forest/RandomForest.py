from custom_random_forest.DecisionTree import DecisionTree
import numpy as np
from collections import Counter


def _bootstrap_samples(features, labels):
    n_samples = features.shape[0]
    indexes = np.random.choice(n_samples, n_samples, replace=True)
    return features[indexes], labels[indexes]


def _most_common_label(labels):
    counter = Counter(labels)
    most_common = counter.most_common(1)[0][0]
    return most_common


class RandomForest:
    def __init__(self, n_trees=10, max_depth=10, min_samples_split=2, n_features=None):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.n_features = n_features
        self.trees = []

    def fit(self, features, labels):
        self.trees = []
        for _ in range(self.n_trees):
            tree = DecisionTree(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                n_features=self.n_features)
            features_sample, labels_sample = _bootstrap_samples(features, labels)
            tree.fit(features_sample, labels_sample)
            self.trees.append(tree)

    def predict(self, features):
        predictions = np.array([tree.predict(features) for tree in self.trees])
        tree_predictions = np.swapaxes(predictions, 0, 1)
        predictions = np.array([_most_common_label(pred) for pred in tree_predictions])
        return predictions
