from sklearn.base import BaseEstimator
import utils
import numpy as np
import scipy.stats


class GaussianNaiveBayes(BaseEstimator):

    def __init__(self, classes): #library requirement is to explicity put parameters to be copied during cross-validaion process
        self.classes = classes

    def get_attr_count(self, X):
        return len(X[0])

    def get_class_probs(self, X, y):
        data = utils.merge_attrs(X, y)
        class_index = utils.get_class_index(data)
        result = dict()
        for record in data:
            class_key = record[class_index]
            if class_key in result:
                result[class_key] += 1
            else:
                result[class_key] = 1
        for key in result:
            result[key] = result[key] / len(data)
        return result

    def get_attr_measures(self, X, y):
        data = utils.merge_attrs(X, y)
        first_attr_index = 0
        last_attr_index = utils.get_class_index(data) - 1
        result = [[] for _ in range(self.attr_count)]
        for record in data:
            for attr_index in range(first_attr_index, last_attr_index + 1):
                result[attr_index].append(record[attr_index])
        for i in range(len(result)):
            attr_values = result[i]
            mean = np.mean(attr_values)
            std = np.std(attr_values)
            result[i] = [mean, std]
        return result

    def get_attr_by_class_measures(self, X, y):
        data = utils.merge_attrs(X, y)
        class_index = utils.get_class_index(data)
        result = dict()
        for record in data:
            class_key = record[class_index]
            if class_key in result:
                result[class_key].append(record)
            else:
                result[class_key] = [record]
        for key in result:
            class_X, class_y = utils.horizontal_split(result[key])
            result[key] = self.get_attr_measures(class_X, class_y)

        return result

    #override
    def fit(self, X, y):
        self.attr_count = self.get_attr_count(X)
        self.class_probs = self.get_class_probs(X, y)
        self.attr_measures = self.get_attr_measures(X, y)
        self.attr_by_class_measures = self.get_attr_by_class_measures(X, y)

        # print(f"get_attr_count: {self.attr_count}")
        # print(f"class_probs: {self.class_probs}")
        # print(f"classes: {self.classes}")
        # print(f"get_attr_probs(data): {self.attr_probs}")
        # print(f"get_attr_by_class_probs(data): {self.attr_by_class_probs}")
        return self

    def classify_many(self, X):
        y = []
        for x in X:
            y.append(self.classify(x))
        return y

    def classify(self, x):
        class_x_probs = dict()
        for clazz in self.get_params(False)['classes']:
            class_x_probs[clazz] = self.get_class_x_prob(x, clazz)
        return max(class_x_probs, key=class_x_probs.get)

    def get_class_x_prob(self, x, clazz):
        return self.class_probs[clazz] * self.get_x_class_prob(x, clazz) / self.get_x_prob(x)
        # return get_x_class_prob(x, clazz) also works...

    def get_x_class_prob(self, x, clazz):
        measures = self.attr_by_class_measures[clazz]
        prob = self.gaussian_prob(measures[0][0], measures[0][1], x[0])
        for i in range(1, self.attr_count):
            prob_next = self.gaussian_prob(measures[i][0], measures[i][1], x[i])
            prob *= prob_next
        return prob

    def get_x_prob(self, x):
        prob = self.gaussian_prob(self.attr_measures[0][0], self.attr_measures[0][1], x[0])
        for i in range(1, self.attr_count):
            prob *= self.gaussian_prob(self.attr_measures[i][0], self.attr_measures[i][1], x[i])
        return prob

    def gaussian_prob(self, mean, std, x):  # Gaussian Probability Density Function,
        # use trick with 'f' to work on integers and get density between 0 and 1 which seems probability
        # but in fact it's not needed...
        f = 5
        x = round(x, f) * f
        mean = round(mean, f) * f
        std = round(std, f) * f
        prob = scipy.stats.norm.pdf(x, mean, std)
        # if prob > 1:
        #     print(f"WARN in gaussian_prob: {prob}")
        #     print(f"when x: {x}, mean: {mean}, std: {std}")
        # print(prob)
        return prob

    # override
    def predict(self, X):
        return self.classify_many(X)
