from math import log

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
        p = labels.count(1)
        n = labels.count(0)
        if p == 0 or n == 0:
            return 0
        return (p / (p + n)) * (log((p + n) / p)) + (n / (p + n)) * (log((p + n) / n))

    @staticmethod
    def split_data(features, labels, feature_index):
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

    def get_max_information_gain_feature(self, features, labels, dataset_entropy, feature_indices=None):
        if feature_indices is None:
            feature_indices = range(len(features[0]))
        max_ig = -1
        max_ig_index = None
        for f_i in feature_indices:
            split_features, split_labels = self.split_data(features, labels, f_i)
            avg_info_entropy = 0
            total_p, total_n = (labels.count(1), labels.count(0))
            for s_f, s_l in zip(split_features, split_labels):
                split_p = s_l.count(1)
                split_n = s_l.count(0)
                info_entropy = self.get_entropy(s_l)
                avg_info_entropy += ((split_p + split_n) / (total_p + total_n)) * info_entropy
            ig = dataset_entropy - avg_info_entropy
            if ig > max_ig:
                max_ig = ig
                max_ig_index = f_i
        return max_ig_index

    def build_tree_id3(self, features, labels, parents):
        if len(parents) == len(features[0]) - 1:
            if labels.count(0) > labels.count(1):
                return Node(0, leaf=True)
            return Node(1, leaf=True)
        if labels.count(1) == len(labels):
            return Node(1, leaf=True)
        if labels.count(0) == len(labels):
            return Node(0, leaf=True)
        dataset_entropy = self.get_entropy(labels)
        max_ig_feature = self.get_max_information_gain_feature(features, labels, dataset_entropy)
        split_features, split_labels = self.split_data(features, labels, max_ig_feature)
        root = Node(max_ig_feature)
        parents.append(root)
        root.left = self.build_tree_id3(split_features[0], split_labels[0], parents)
        root.right = self.build_tree_id3(split_features[1], split_labels[1], parents)
        return root

    def printInorder(self, root):
        if root:
            if root.leaf:
                print("Leaf:{}".format(root.value)),
            else:
                print(root.value)
            self.printInorder(root.left)
            self.printInorder(root.right)


if __name__ == '__main__':
    builder = BuildDecisionTree()
    new_f, new_l = CreateData(3, 5).create_data_set()
    r = builder.build_tree_id3(new_f, new_l, [])
    print(new_f)
    print(new_l)
    builder.printInorder(r)
