import csv, random
import utils


def load_iris_data():
    return load('iris_data')


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
