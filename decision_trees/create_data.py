import random


class CreateData:
    def __init__(self, k, m):
        '''
        Initialize the Data creation class
        :param k: Number of features
        :param m: Number of data points
        '''
        self.k = k
        self.m = m

    @staticmethod
    def generate_features(k):
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
        weights_denom = 0
        for i in range(len(features)):
            weights_denom += 0.9 ** i
        weights = []
        for i in range(len(features)):
            weights.append((0.9 ** i) / weights_denom)

        weighted_av = 0
        for i in range(1, len(features)):
            weighted_av += weights[i] * features[i]

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
