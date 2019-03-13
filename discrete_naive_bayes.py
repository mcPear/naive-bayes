import utils
import numpy as np
from bin import Bin
from abstract_naive_bayes import AbstractNaiveBayes


class DiscreteNaiveBayes(AbstractNaiveBayes):

    def __init__(self, classes,
                 bins_count):  # library requirement is to explicity put parameters to be copied during cross-validation process
        self.classes = classes
        self.bins_count = bins_count

    def get_attr_probs(self, X, attr_bins):

        # fill counters
        for i in range(self.attr_count):
            for j in range(len(X)):
                for l in range(len(attr_bins[i])):
                    min = attr_bins[i][l].min
                    max = attr_bins[i][l].max
                    val = X[j][i]
                    if min <= val < max:
                        attr_bins[i][l].counter += 1

        # fix all bins counters with +1 if zero
        for i in range(self.attr_count):
            for l in range(len(attr_bins[i])):
                bin = attr_bins[i][l]
                if bin.counter == 0:
                    bin.counter += 1

        # fill probs
        size = len(X)
        for i in range(self.attr_count):
            for j in range(len(X)):
                for l in range(len(attr_bins[i])):
                    bin = attr_bins[i][l]
                    bin.prob = bin.counter / size

        return attr_bins

    def get_attr_by_class_probs(self, X, y):
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
            result[key] = self.get_attr_probs(class_X, self.get_attr_bins(X))

        return result

    def get_attr_bins(self, X):
        attr_bins = []
        k = self.get_params()['bins_count']
        mins = np.min(X, 0)
        maxs = np.max(X, 0)
        for i in range(self.attr_count):
            bins = []
            min = mins[i]
            max = maxs[i]
            width = (max - min) / k
            while min < max:
                bins.append(Bin(min, min + width))
                min += width
            attr_bins.append(bins)
        return attr_bins

    # override
    def fit(self, X, y):

        self.attr_count = self.get_attr_count(X)
        self.class_probs = self.get_class_probs(X, y)
        self.attr_probs = self.get_attr_probs(X, self.get_attr_bins(X))
        self.attr_by_class_probs = self.get_attr_by_class_probs(X, y)

        return self

    def get_x_class_prob(self, x, clazz):
        probs = self.attr_by_class_probs[clazz]
        prob = self.prob(x[0], probs[0])
        for i in range(1, self.attr_count):
            prob_next = self.prob(x[i], probs[i])
            prob *= prob_next
        return prob

    def get_x_prob(self, x):
        prob = self.prob(x[0], self.attr_probs[0])
        for i in range(1, self.attr_count):
            prob *= self.prob(x[i], self.attr_probs[i])
        return prob

    def prob(self, x, bins):
        for bin in bins:
            if bin.min <= x < bin.max:
                return bin.prob
        return 0.0001  # fixed minimum prob for values out of discretization bins range
