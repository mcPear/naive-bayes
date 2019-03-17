import utils
import numpy as np
from bin import Bin
from abstract_naive_bayes import AbstractNaiveBayes
from verbose_exception import VerboseException


class FrequencyDiscreteNaiveBayes(AbstractNaiveBayes):

    def __init__(self, classes, attr_ranges,
                 frequency):  # library requirement is to explicity put parameters to be copied during cross-validation process
        self.classes = classes
        self.attr_ranges = attr_ranges
        self.frequency = frequency

    def get_attr_probs(self, X):

        attr_ranges = self.get_params()['attr_ranges']
        attr_bins = []
        freq = self.get_params()['frequency']
        for i in range(self.attr_count):
            Xsorted = np.array(X, dtype='f8')
            Xsorted.view(('f8,' * self.attr_count)[:-1]).sort(order=[f'f{i}'], axis=0)
            # print(Xsorted)
            # mam przesortowane po kolumnie atrybutu, teraz trzeba przejść po tym i budować attr_bins
            bins = []
            last_val = None
            min, max = attr_ranges[i]
            for j in range(len(Xsorted)):
                curr_val = Xsorted[j][i]
                if not bins or ():  # nie ma koszyka
                    bins.append(Bin(min, None))
                    bins[-1].counter += 1
                elif bins[-1].counter < freq or (
                        last_val == curr_val):  # tu po or jest zabezpieczenie przed przenoszeniem tych samych do innego koszyka
                    bins[-1].counter += 1
                else:  # trzeba zrobić kolejny koszyk
                    bins.append(Bin(last_val, None))
                    bins[-1].counter += 1
                bins[-1].max = curr_val
                last_val = curr_val
            bins[len(bins) - 1].max = max
            attr_bins.append(bins)
        # print(attr_bins)

        size = len(X)

        # increment all bins counters if zero
        for i in range(self.attr_count):
            for l in range(len(attr_bins[i])):
                bin = attr_bins[i][l]
                if bin.counter == 0:
                    bin.counter += 1
                    size += 1

        # fill probs
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
            result[key] = self.get_attr_probs(class_X)

        return result

    # override
    def fit(self, X, y):

        self.attr_count = self.get_attr_count(X)
        self.class_probs = self.get_class_probs(X, y)
        self.attr_probs = self.get_attr_probs(X)
        self.attr_by_class_probs = self.get_attr_by_class_probs(X, y)

        # print("attr_bins: " + self.attr_probs.__str__())  # coś za duże te prob i countery tutaj

        return self

    def get_x_class_prob(self, x, clazz):
        probs = self.attr_by_class_probs[clazz]
        # print(f"x class PROBS: {probs.__str__()}")
        prob = self.prob(x[0], probs[0])
        for i in range(1, self.attr_count):
            prob_next = self.prob(x[i], probs[i])
            # print(f"prob next: {prob_next}")
            prob *= prob_next
        return prob

    def get_x_prob(self, x):
        prob = self.prob(x[0], self.attr_probs[0])
        for i in range(1, self.attr_count):
            prob *= self.prob(x[i], self.attr_probs[i])
        return prob

    def prob(self, x, bins):
        for bin in bins:
            if bin.min <= x <= bin.max:
                return bin.prob
        raise VerboseException(f"Value {x} is out of bins range! Learn me on full range of data.")
        # return 0.0001  # fixed minimum prob for values out of discretization bins range
