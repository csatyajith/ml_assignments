from math import log

import matplotlib.pyplot as plt

from decision_trees.create_data import CreateData


class Node:
    def __init__(self, value, leaf=False):
        self.left = None
        self.right = None
        self.value = value
        self.leaf = leaf


class BuildDecisionTree:
    def __init__(self):
        pass

    @staticmethod
    def get_entropy(labels):
        """
        For the given list of labels, we compute the total positives, and
        total negatives in the list and then computethe entropy of the label
        set using the below formula.
        :param labels: The list of labels for which entropy needs to be computed
        :return: The computed entropy
        """
        p = labels.count(1)
        n = labels.count(0)
        if p == 0 or n == 0:
            return 0
        return (p / (p + n)) * (log((p + n) / p)) + (n / (p + n)) * (log((p + n) / n))

    @staticmethod
    def split_data(features, labels, feature_index):
        """
        This function splits the features and labels into two distinct parts based on the
        value of the feature at the feature_index location.
        :param features: The list of features
        :param labels: The labels associated with the features
        :param feature_index: The index in the list of features using which we should
        partition the data set
        :return: features partitioned into two and their corresponding labels partitioned into two.
        """
        split_features = ([], [])
        split_labels = ([], [])
        for i, l in enumerate(labels):
            if features[i][feature_index] == 0:
                split_features[0].append(features[i])
                split_labels[0].append(labels[i])
            else:
                split_features[1].append(features[i])
                split_labels[1].append(labels[i])
        return split_features, split_labels

    def get_max_information_gain_feature(self, features, labels, dataset_entropy):
        """
        Finds the feature with the maximum information gain among the available features
        :param features: The features among which we need to find the max_ig
        :param labels: The labels associated with those features
        :param dataset_entropy: The entropy of the entire dataset
        :return: The index of the feature that provides the maximum information gain
        """
        feature_indices = range(len(features[0]))
        max_ig = -1
        max_ig_index = None
        for f_i in feature_indices:
            # We split the features and labels according to each feature index and measure the entropy
            split_features, split_labels = self.split_data(features, labels, f_i)
            avg_info_entropy = 0
            total_p, total_n = (labels.count(1), labels.count(0))
            for s_f, s_l in zip(split_features, split_labels):
                split_p = s_l.count(1)
                split_n = s_l.count(0)
                info_entropy = self.get_entropy(s_l)
                avg_info_entropy += ((split_p + split_n) / (total_p + total_n)) * info_entropy

            # The information gain is the difference between the entropy of the dataset and entropy of the feature
            ig = dataset_entropy - avg_info_entropy
            if ig > max_ig:
                max_ig = ig
                max_ig_index = f_i
        return max_ig_index

    @staticmethod
    def get_gini_impurity(labels):
        """
        Computes the gini impurity for a given set of labels
        :param labels: The labels for which GINI impurity needs to be computed
        :return: The gini impurity for those labels
        """
        p = labels.count(1)
        n = labels.count(0)
        if p == 0 or n == 0:
            return 0
        return 1 - ((p / (p + n)) ** 2) + ((n / (p + n)) ** 2)

    def get_best_gini_impurity_feature(self, features, labels):
        """
        Of all the features in the given data, computes the feature with the least average gini
        impurity and returns its index
        :param features: Features of the data
        :param labels: Labels corresponding to the features.
        :return: An index of the best feature to choose.
        """
        feature_indices = range(len(features[0]))
        min_gini_impurity = 100000000
        min_gini_impurity_index = None
        for f_i in feature_indices:
            split_features, split_labels = self.split_data(features, labels, f_i)
            weighted_gini_impurity = 0
            total_p, total_n = (labels.count(1), labels.count(0))
            for s_f, s_l in zip(split_features, split_labels):
                split_p = s_l.count(1)
                split_n = s_l.count(0)
                gini_impurity = self.get_gini_impurity(s_l)
                weighted_gini_impurity += ((split_p + split_n) / (total_p + total_n)) * gini_impurity

            if weighted_gini_impurity < min_gini_impurity:
                min_gini_impurity = weighted_gini_impurity
                min_gini_impurity_index = f_i
        return min_gini_impurity_index

    def build_tree_gini_impurity(self, features, labels, depth):
        """
        Uses the above methods to recursively build a decision tree. Uses gini impurity to identify
        best possible splitting
        :param features: The train features
        :param labels: The train labels
        :param depth: The depth of the tree (Used for termination)
        :return: root of the decision tree
        """
        max_ig_feature = self.get_best_gini_impurity_feature(features, labels)
        split_features, split_labels = self.split_data(features, labels, max_ig_feature)
        if len(split_features[0]) == 0:
            root = Node(split_labels[1][0], leaf=True)
        elif len(split_features[1]) == 0:
            root = Node(split_labels[0][0], leaf=True)
        else:
            root = Node(max_ig_feature)
            depth += 1
            root.left = self.build_tree_id3(split_features[0], split_labels[0], depth)
            root.right = self.build_tree_id3(split_features[1], split_labels[1], depth)

        return root

    def build_tree_id3(self, features, labels, depth):
        """
        Uses the above methods to recursively build a decision tree. Uses the id3 algorithm to identify
        best possible splitting
        :param features: The train features
        :param labels: The train labels
        :param depth: The depth of the tree (Used for termination)
        :return: root of the decision tree
        """
        dataset_entropy = self.get_entropy(labels)
        max_ig_feature = self.get_max_information_gain_feature(features, labels, dataset_entropy)
        split_features, split_labels = self.split_data(features, labels, max_ig_feature)
        if len(split_features[0]) == 0:
            root = Node(split_labels[1][0], leaf=True)
        elif len(split_features[1]) == 0:
            root = Node(split_labels[0][0], leaf=True)
        else:
            root = Node(max_ig_feature)
            depth += 1
            root.left = self.build_tree_id3(split_features[0], split_labels[0], depth)
            root.right = self.build_tree_id3(split_features[1], split_labels[1], depth)

        return root

    @staticmethod
    def get_error(features, labels, tree_root):
        """
        Predicts the labels for features and compares them with the actual labels to return an error.
        :param features: The features for which predictions need to be made.
        :param labels: The actual labels associated with those features.
        :param tree_root: The root node of the decision tree to be used.
        :return: Total number of mis-matches between the predictions and the actual data.
        """
        predictions = []
        errors = 0
        for i, f in enumerate(features):
            pos = tree_root
            while pos.leaf is False:
                if f[pos.value] == 0:
                    pos = pos.left
                else:
                    pos = pos.right
            predictions.append(pos.value)
            if pos.value != labels[i]:
                errors += 1
        return errors

    def print_tree_in_order(self, root):
        """
        Printing the tree in-order for recreation
        :param root: The root node of the decision tree
        :return: None
        """
        if root:
            if root.leaf:
                print("Leaf:{}".format(root.value)),
            else:
                print(root.value)
            self.print_tree_in_order(root.left)
            self.print_tree_in_order(root.right)

    def get_tree_nodes_indices(self, root, result):
        if root:
            self.get_tree_nodes_indices(root.left, result)
            if root.leaf is False:
                result.append(root.value)
            self.get_tree_nodes_indices(root.right, result)
            return result

    def get_typical_error(self, root, features_list, labels_list):
        error = 0
        for f1, l1 in zip(features_list, labels_list):
            error += self.get_error(f1, l1, root)
        average_error = error / len(features_list)
        return average_error


def run_q3():
    builder = BuildDecisionTree()
    new_f, new_l = CreateData(4, 30).create_data_set()
    r = builder.build_tree_id3(new_f, new_l, 0)
    training_err = builder.get_error(new_f, new_l, r)
    for f, l in zip(new_f, new_l):
        print(f, l)

    print("The tree is:")
    builder.print_tree_in_order(r)
    print("\n Train error:", training_err)


def run_q4():
    builder = BuildDecisionTree()
    new_f, new_l = CreateData(4, 30).create_data_set()
    r = builder.build_tree_id3(new_f, new_l, 0)
    test_m = 100
    f_list = []
    l_list = []
    for i in range(1000):
        new_f, new_l = CreateData(4, test_m).create_data_set()
        f_list.append(new_f)
        l_list.append(new_l)
    print("The typical error is:", builder.get_typical_error(r, f_list, l_list)/test_m)


def run_q5():
    builder = BuildDecisionTree()
    m_list = list(range(10, 1001, 10))
    test_m = 500
    error_list = []
    for m in m_list:
        new_f, new_l = CreateData(10, m).create_data_set()
        r = builder.build_tree_id3(new_f, new_l, 0)
        training_err = builder.get_error(new_f, new_l, r) / m
        f_list = []
        l_list = []
        for i in range(100):
            new_f, new_l = CreateData(10, test_m).create_data_set()
            f_list.append(new_f)
            l_list.append(new_l)
        typical_error = builder.get_typical_error(r, f_list, l_list) / test_m
        error_list.append(abs(typical_error - training_err))
    plt.plot(m_list, error_list)
    plt.xlabel("Number of data points used for training")
    plt.ylabel("|err_train-err|")
    plt.show()


def run_q6():
    builder = BuildDecisionTree()
    m_list = list(range(10, 1001, 10))
    test_m = 500
    error_list = []
    for m in m_list:
        new_f, new_l = CreateData(10, m).create_data_set()
        r = builder.build_tree_gini_impurity(new_f, new_l, 0)
        training_err = builder.get_error(new_f, new_l, r) / m
        f_list = []
        l_list = []
        for i in range(100):
            new_f, new_l = CreateData(10, test_m).create_data_set()
            f_list.append(new_f)
            l_list.append(new_l)
        typical_error = builder.get_typical_error(r, f_list, l_list) / test_m
        error_list.append(abs(typical_error - training_err))
    plt.plot(m_list, error_list)
    plt.xlabel("Number of data points used for training")
    plt.ylabel("|err_train-err|")
    plt.show()


if __name__ == '__main__':
    run_q6()
