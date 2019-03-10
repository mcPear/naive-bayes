import copy


# assumption that class is always the last attribute
def get_class_index(data):
    return len(data[0]) - 1


def horizontal_split(data):
    size = len(data[0])
    attrs = [record[:size - 1] for record in data]
    classes_nested = [record[size - 1:] for record in data]  # can flat it here instead of extra line below
    classes = [item for sublist in classes_nested for item in sublist]
    return [attrs, classes]


def merge_attrs(X, y):
    result = []
    for i in range(len(X)):
        result.append(copy.deepcopy(X[i]))
        result[i].append(y[i])
    return result
