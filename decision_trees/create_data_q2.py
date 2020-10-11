import random


class CreateData:
    def __init__(self, m):
        """
        Initialize the Data creation class
        :param m: Number of data points
        """
        self.m = m

    @staticmethod
    def generate_features():
        """
        Generates 21 features based on the given distribution in the assignment
        :return: A feature list with 21 features
        """
        x0 = 1 if random.random() <= 0.5 else 0
        x = [x0]
        for i in range(1, 15):
            rand = random.random()
            if rand < 0.25:
                x.append(1 - x[i - 1])
            else:
                x.append(x[i - 1])
        for i in range(15, 21):
            x.append(1 if random.random() <= 0.5 else 0)
        return x

    @staticmethod
    def create_label(features):
        """
        Using the features, this generates the labels for those features using the majority
        technique described in the assignment
        :param features: The features for which a label needs to be computed.
        :return: A label for the input features
        """
        if features[0] == 0:
            return 1 if features[1:8].count(1) > features[1:8].count(0) else 0
        return 1 if features[8:15].count(1) > features[8:15].count(0) else 0

    def create_data_set(self):
        features_list = []
        labels_list = []
        for i in range(self.m):
            f = self.generate_features()
            features_list.append(f)
            labels_list.append(self.create_label(f))
        return features_list, labels_list


if __name__ == '__main__':
    print(CreateData(5).create_data_set())
