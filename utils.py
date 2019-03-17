import copy


# assumption that class is always the last attribute
def get_class_index(data):
    return len(data[0]) - 1


def unique_classes(classes):
    return list(set(classes))


def attr_ranges(attrs):
    ranges = []
    for c in range(len(attrs[0])):
        ranges.append([999999, -999999])
    for r in range(len(attrs)):
        for c in range(len(attrs[r])):
            if ranges[c][0] > attrs[r][c]:
                ranges[c][0] = attrs[r][c]
            if ranges[c][1] < attrs[r][c]:
                ranges[c][1] = attrs[r][c]
    return ranges


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
