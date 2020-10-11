import pickle

import matplotlib.pyplot as plt

from decision_trees.build_decision import BuildDecisionTree, Node
from decision_trees.create_data_q2 import CreateData


class PrunedTreeBuilding:
    def __init__(self):
        self.build = BuildDecisionTree()

    def build_depth_pruned_tree_id3(self, features, labels, depth, prune_depth=None):
        """
        Uses the above methods to recursively build a decision tree. Uses the id3 algorithm to identify
        best possible splitting
        :param features: The train features
        :param labels: The train labels
        :param depth: The depth of the tree (Used for termination)
        :param prune_depth: The depth at which the tree should be pruned. Anything after this depth
        is solely measured on frequencies.
        :return: root of the decision tree and the depth of the tree.
        """
        if prune_depth is not None:
            if depth == prune_depth:
                return Node(1, leaf=True) if labels.count(1) > labels.count(0) else Node(0, leaf=True)
        dataset_entropy = self.build.get_entropy(labels)
        max_ig_feature = self.build.get_max_information_gain_feature(features, labels, dataset_entropy)
        split_features, split_labels = self.build.split_data(features, labels, max_ig_feature)
        if len(split_features[0]) == 0:
            root = Node(split_labels[1][0], leaf=True)
        elif len(split_features[1]) == 0:
            root = Node(split_labels[0][0], leaf=True)
        else:
            root = Node(max_ig_feature)
            depth += 1
            root.left = self.build_depth_pruned_tree_id3(split_features[0], split_labels[0], depth, prune_depth)
            root.right = self.build_depth_pruned_tree_id3(split_features[1], split_labels[1], depth, prune_depth)

        return root

    def build_sample_size_pruned_tree(self, features, labels, sample_threshold=None):
        """
        Uses the above methods to recursively build a decision tree. Uses the id3 algorithm to identify
        best possible splitting
        :param features: The train features
        :param labels: The train labels
        :param sample_threshold:
        :return: root of the decision tree
        """
        dataset_entropy = self.build.get_entropy(labels)
        max_ig_feature = self.build.get_max_information_gain_feature(features, labels, dataset_entropy)
        split_features, split_labels = self.build.split_data(features, labels, max_ig_feature)

        if len(split_features[0]) <= sample_threshold:
            root = Node(1, leaf=True) if split_labels[0].count(1) > split_labels[0].count(0) else Node(0, leaf=True)
        elif len(split_features[1]) <= sample_threshold:
            root = Node(1, leaf=True) if split_labels[1].count(1) > split_labels[1].count(0) else Node(0, leaf=True)
        else:
            root = Node(max_ig_feature)
            root.left = self.build_sample_size_pruned_tree(split_features[0], split_labels[0], sample_threshold)
            root.right = self.build_sample_size_pruned_tree(split_features[1], split_labels[1], sample_threshold)

        return root


def run_q1():
    builder = BuildDecisionTree()
    m_list = list(range(10, 1001, 10))
    test_m = 500
    error_list = []
    for m in m_list:
        new_f, new_l = CreateData(m).create_data_set()
        r = builder.build_tree_id3(new_f, new_l, 0)
        f_list = []
        l_list = []
        for i in range(100):
            new_f, new_l = CreateData(test_m).create_data_set()
            f_list.append(new_f)
            l_list.append(new_l)
        typical_error = builder.get_typical_error(r, f_list, l_list) / test_m
        error_list.append(typical_error)
    plt.plot(m_list, error_list)
    plt.xlabel("Number of data points used for training")
    plt.ylabel("err")
    plt.show()


def run_q2():
    builder = BuildDecisionTree()
    m_list = [1000, 10000, 100000, 1000000]
    percentage_irrelevants = []
    for m in m_list:
        irrelevant = 0
        new_f, new_l = CreateData(m).create_data_set()
        r = builder.build_tree_id3(new_f, new_l, 0)
        tree_node_indices = builder.get_tree_nodes_indices(r, [])
        for i in range(15, 21):
            irrelevant += tree_node_indices.count(i)
        percentage_irrelevants.append((irrelevant / len(tree_node_indices)) * 100)

    print(percentage_irrelevants)


def create_and_pickle_data():
    train_features, train_labels = CreateData(8000).create_data_set()
    test_features, test_labels = CreateData(2000).create_data_set()
    with open("stored_data", "wb") as std:
        pickle.dump((train_features, train_labels, test_features, test_labels), std)


def load_pickled_data():
    with open("stored_data", "rb") as std:
        return pickle.load(std)


def run_q3a():
    train_features, train_labels, test_features, test_labels = load_pickled_data()
    pruned_builder = PrunedTreeBuilding()
    depth_list = list(range(0, 20))
    train_errors = []
    test_errors = []
    # Iterate for each depth
    for i in depth_list:
        # For each depth, compute the decision tree and find the train and test errors
        r = pruned_builder.build_depth_pruned_tree_id3(train_features, train_labels, 0, i)
        train_errors.append(pruned_builder.build.get_error(train_features, train_labels, r) / len(train_features))
        test_errors.append(pruned_builder.build.get_error(test_features, test_labels, r) / len(test_features))
    plt.plot(depth_list, train_errors, label="Train")
    plt.plot(depth_list, test_errors, label="Test")
    plt.xlabel("Depth of the tree")
    plt.ylabel("Error")
    plt.legend()
    plt.show()


def run_q3b():
    train_features, train_labels, test_features, test_labels = load_pickled_data()
    pruned_builder = PrunedTreeBuilding()
    train_errors = []
    test_errors = []
    sample_range = list(range(1000, 0, -10))

    # Iterate for each sample threshold
    for i in sample_range:
        # For each sample_threshold, compute the decision tree and find the train and test errors
        r = pruned_builder.build_sample_size_pruned_tree(train_features, train_labels, i)
        train_errors.append(pruned_builder.build.get_error(train_features, train_labels, r) / len(train_features))
        test_errors.append(pruned_builder.build.get_error(test_features, test_labels, r) / len(test_features))
    plt.plot(sorted(sample_range), train_errors, label="Train")
    plt.plot(sorted(sample_range), test_errors, label="Test")
    plt.xlabel("Sample threshold of the tree")
    plt.ylabel("Error")
    plt.legend()
    plt.show()


def run_q4():
    """As we concluded that a good threshold depth is 9 in the previous question, we use that to prune
    our tree and then compute the frequency of spurious variables
    """
    pruned_builder = PrunedTreeBuilding()
    builder = BuildDecisionTree()
    m_list = [1000, 10000, 100000, 1000000]
    percentage_irrelevants = []
    for m in m_list:
        irrelevant = 0
        new_f, new_l = CreateData(m).create_data_set()
        r = pruned_builder.build_depth_pruned_tree_id3(new_f, new_l, 0, 9)
        tree_node_indices = builder.get_tree_nodes_indices(r, [])
        for i in range(15, 21):
            irrelevant += tree_node_indices.count(i)
        percentage_irrelevants.append((irrelevant / len(tree_node_indices)) * 100)

    print(percentage_irrelevants)


def run_q5():
    """As we concluded that a good sample threshold is 780 in the previous question, we use that to prune
    our tree and then compute the frequency of spurious variables
    """
    pruned_builder = PrunedTreeBuilding()
    builder = BuildDecisionTree()
    m_list = [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000, 100000, 1000000]
    percentage_irrelevants = []
    for m in m_list:
        irrelevant = 0
        new_f, new_l = CreateData(m).create_data_set()
        r = pruned_builder.build_sample_size_pruned_tree(new_f, new_l, 780)
        tree_node_indices = builder.get_tree_nodes_indices(r, [])
        if len(tree_node_indices) == 0:
            percentage_irrelevants.append(-1)
            continue
        for i in range(15, 21):
            irrelevant += tree_node_indices.count(i)
        percentage_irrelevants.append((irrelevant / len(tree_node_indices)) * 100)

    print(percentage_irrelevants)


if __name__ == '__main__':
    run_q5()
