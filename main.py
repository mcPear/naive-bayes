import csv, random
import numpy as np
import scipy.stats


def load(file):
    iterator = csv.reader(open(file, "r"))
    data = list(iterator)
    class_index = get_class_index(data)
    for i in range(len(data)):
        for j in range(len(data[i])):
            val = data[i][j]
            data[i][j] = float(val) if j != class_index else val
    print(data)
    return data


def split(data):
    random.shuffle(data)
    half = len(data) / 2
    train_data = data[:half]
    test_data = data[half:]
    return [train_data, test_data]


def attr_count(data):
    return len(data[0]) - 1


def get_class_index(data):
    return len(data[0]) - 1


def class_probs(data):
    class_index = get_class_index(data)
    result = dict()
    for record in data:
        class_key = record[class_index]
        if class_key in result:
            result[class_key] += 1
        else:
            result[class_key] = 1
    for key in result:
        print(result[key])
        result[key] = result[key] / len(data)
    return result


def attr_measures(data):
    first_attr_index = 0
    last_attr_index = get_class_index(data) - 1
    result = [[] for _ in range(attr_count(data))]
    for record in data:
        for attr_index in range(first_attr_index, last_attr_index + 1):
            result[attr_index].append(record[attr_index])
    for i in range(len(result)):
        attr_values = result[i]
        mean = np.mean(attr_values)
        variance = np.var(attr_values)
        result[i] = [mean, variance]
    return result


def attr_by_class_measures(data):
    class_index = get_class_index(data)
    result = dict()
    for record in data:
        class_key = record[class_index]
        if class_key in result:
            result[class_key].append(record[:class_index])
        else:
            result[class_key] = [record[:class_index]]
    return result


def gaussian_prob(mean, var, x):  # Gaussian Probability Density Function
    return scipy.stats.norm(mean, var).pdf(x)  # not sure about params


data = load("iris_data")
print(attr_count(data))
print(class_probs(data))
print(attr_measures(data))
print(attr_by_class_measures(data))
