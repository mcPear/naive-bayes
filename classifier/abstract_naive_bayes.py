from sklearn.base import BaseEstimator
from util import utils
import copy


class AbstractNaiveBayes(BaseEstimator):

    def get_attr_count(self, X):
        return len(X[0])

    def get_class_probs(self, X, y):
        data = utils.merge_attrs(X, y)
        class_index = utils.get_class_index(data)
        result = self.get_empty_classes_dict(0)

        for record in data:
            class_key = record[class_index]
            result[class_key] += 1

        for key in result:
            result[key] = result[key] / len(data)
        return result

    def get_empty_classes_dict(self, empty_elem):
        classes = self.get_params()['classes']
        result = dict()
        for class_name in classes:
            result[class_name] = copy.deepcopy(empty_elem)
        return result

    def classify_many(self, X):
        y = []
        for x in X:
            y.append(self.classify(x))
        return y

    def classify(self, x):
        class_x_probs = dict()
        for clazz in self.get_params(False)['classes']:
            class_x_probs[clazz] = self.get_class_x_prob(x, clazz)
        # print(class_x_probs)
        return max(class_x_probs, key=class_x_probs.get)

    def get_class_x_prob(self, x, clazz):
        # print(f"{self.class_probs[clazz]} | {self.get_x_class_prob(x, clazz)} | {self.get_x_prob(x)}")
        return self.class_probs[clazz] * self.get_x_class_prob(x, clazz) / self.get_x_prob(x)
        # return self.get_x_class_prob(x, clazz) # also works...

    # override
    def predict(self, X):
        return self.classify_many(X)
