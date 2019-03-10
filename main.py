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
    half = len(data) // 2
    train_data = data[:half]
    test_data = data[half:]
    return [train_data, test_data]


def get_attr_count(data):
    return len(data[0]) - 1


def get_class_index(data):
    return len(data[0]) - 1


def get_class_probs(data):
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


def get_classes(class_probs):
    return class_probs.keys()


def get_attr_measures(data):
    first_attr_index = 0
    last_attr_index = get_class_index(data) - 1
    result = [[] for _ in range(get_attr_count(data))]
    for record in data:
        for attr_index in range(first_attr_index, last_attr_index + 1):
            result[attr_index].append(record[attr_index])
    for i in range(len(result)):
        attr_values = result[i]
        mean = np.mean(attr_values)
        std = np.std(attr_values)
        result[i] = [mean, std]
    return result


def get_attr_by_class_measures(data):
    class_index = get_class_index(data)
    result = dict()
    for record in data:
        class_key = record[class_index]
        if class_key in result:
            result[class_key].append(record)
        else:
            result[class_key] = [record]
    for key in result:
        result[key] = get_attr_measures(result[key])

    return result


iris_data = load("iris_data")
train_data, test_data = split(iris_data)
print(f"train_data len: {len(train_data)}")
print(f"test_data len: {len(test_data)}")
attr_count = get_attr_count(train_data)
print(f"get_attr_count: {attr_count}")
class_probs = get_class_probs(train_data)
print(f"class_probs: {class_probs}")
classes = get_classes(class_probs)
print(f"classes: {classes}")
attr_measures = get_attr_measures(train_data)
print(f"get_attr_measures(data): {attr_measures}")
attr_by_class_measures = get_attr_by_class_measures(train_data)
print(f"get_attr_by_class_measures(data): {attr_by_class_measures}")
print("---CLASSIFICATION BELOW---")
tester = test_data[0][:get_class_index(test_data)]
print(f"tester attrs: {tester}")
classTester = test_data[0][get_class_index(test_data)]
print(f"tester class: {classTester}")


def classify_many(data):
    counter = 0
    for test_x in data:
        test_x_attrs = test_x[:get_class_index(data)]
        test_x_class = test_x[get_class_index(data)]
        found_class = classify(test_x_attrs)
        if test_x_class == found_class:
            counter += 1
    return counter / len(data)


def classify(x):
    class_x_probs = dict()
    for clazz in classes:
        class_x_probs[clazz] = get_class_x_prob(x, clazz)
    return max(class_x_probs, key=class_x_probs.get)


def get_class_x_prob(x, clazz):
    return class_probs[clazz] * get_x_class_prob(x, clazz) / get_x_prob(x)
    # return get_x_class_prob(x, clazz)


def get_x_class_prob(x, clazz):
    measures = attr_by_class_measures[clazz]
    prob = gaussian_prob(measures[0][0], measures[0][1], x[0])
    for i in range(1, attr_count):
        prob_next = gaussian_prob(measures[i][0], measures[i][1], x[i])
        prob *= prob_next
    return prob


def get_x_prob(x):
    prob = gaussian_prob(attr_measures[0][0], attr_measures[0][1], x[0])
    for i in range(1, attr_count):
        prob *= gaussian_prob(attr_measures[i][0], attr_measures[i][1], x[i])
    return prob


def gaussian_prob(mean, std, x):  # Gaussian Probability Density Function,
    # use trick to work on integers and get density between 0 and 1 which seems probability
    # but i don't really get it...
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


print(classify(tester))
print(classify_many(test_data))
