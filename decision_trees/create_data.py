import random


class CreateData:
    def __init__(self, k, m):
        """
        Initialize the Data creation class
        :param k: Number of features
        :param m: Number of data points
        """
        self.k = k
        self.m = m

    @staticmethod
    def generate_features(k):
        """
        Generates k features based on the given distribution in the assignment
        :param k: number of features to be generated
        :return: A feature list with k features
        """
        x1 = 1 if random.random() <= 0.5 else 0
        x = [x1]
        for i in range(k - 1):
            rand = random.random()
            if rand < 0.25:
                x.append(1 - x[i - 1])
            else:
                x.append(x[i - 1])
        return x

    @staticmethod
    def create_label(features):
        """
        Using the features, this generates the labels for those features using the weighted average
        technique described in the assignment
        :param features: The features for which a label needs to be computed.
        :return: A label for the input features
        """
        weights_denom = 0
        # As we need to create consider weights from 2 to k, we iterate from 2 to len(features) + 1
        for i in range(2, len(features) + 1):
            weights_denom += 0.9 ** i
        weights = []
        for i in range(2, len(features) + 1):
            weights.append((0.9 ** i) / weights_denom)

        # We then compute the weighted average of the features using the weights computed above.
        weighted_av = 0
        for i in range(len(features)):
            if i == 0:
                continue
            weighted_av += weights[i - 1] * features[i]

        # We then use the weighted average computed above to create a label for the given set of features.
        return features[0] if weighted_av >= 0.5 else 1 - features[0]

    def create_data_set(self):
        features_list = []
        labels_list = []
        for i in range(self.m):
            f = self.generate_features(self.k)
            features_list.append(f)
            labels_list.append(self.create_label(f))
        return features_list, labels_list


if __name__ == '__main__':
    print(CreateData(5, 5).create_data_set())
