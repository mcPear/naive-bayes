import csv, random
import utils
from sklearn.datasets import load_iris
import numpy as np


def load_diabetes_data():
    return load('data/diabetes_data')


def load_glass_data():
    return load('data/glass_data')


def load_wine_data():
    return load('data/wine_data')


def load_iris_data():
    iris = load_iris()
    X, y = [[elem for elem in record] for record in iris.data], np.asarray(iris.target)
    # print(X)
    return utils.merge_attrs(X, y)


def load(file):
    iterator = csv.reader(open(file, "r"))
    data = list(iterator)
    class_index = utils.get_class_index(data)
    for i in range(len(data)):
        for j in range(len(data[i])):
            val = data[i][j]
            data[i][j] = float(val) if j != class_index else val
    return data


def random_split(data, percent):
    random.shuffle(data)
    half = len(data) * percent // 100
    train_data = data[:half]
    test_data = data[half:]
    return [train_data, test_data]
