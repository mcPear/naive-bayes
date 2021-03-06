from classifier.gaussian_naive_bayes import GaussianNaiveBayes
from classifier.width_discrete_naive_bayes import WidthDiscreteNaiveBayes
from classifier.frequency_discrete_naive_bayes import FrequencyDiscreteNaiveBayes
from classifier.entropy_discrete_naive_bayes import EntropyDiscreteNaiveBayes
from util import validation, utils, data_providers as dp
import random


def score(data, estimator_const, est_param, k=10, cross_val=True, stratified=True):
    attrs, classes = utils.horizontal_split(data)
    attr_ranges = utils.attr_ranges(attrs)
    unique_classes = utils.unique_classes(
        classes)  # it's important to recognize classes from both training and test set(whole data)
    estimator = estimator_const(unique_classes, est_param, attr_ranges)
    scoring = ['f1_macro', 'accuracy', 'precision_macro', 'recall_macro']
    score = validation.k_fold(data, estimator, scoring, k, stratified) \
        if cross_val else validation.single_split(data, estimator, scoring)
    print(("mean " if cross_val else "") + f"score: {score}")
    return score


def discretization_comparsion():
    data_set = dp.load_wine_data()
    random.shuffle(data_set)
    score(data_set, EntropyDiscreteNaiveBayes, utils.get_entropy_intervals(data_set))
    score(data_set, WidthDiscreteNaiveBayes, 5)
    score(data_set, FrequencyDiscreteNaiveBayes, 21)
    score(data_set, GaussianNaiveBayes, None)


def on_many_datasets_comparsion():
    for data_set in [dp.load_wine_data(), dp.load_glass_data(), dp.load_iris_data(), dp.load_diabetes_data()]:
        random.shuffle(data_set)
        score(data_set, EntropyDiscreteNaiveBayes, utils.get_entropy_intervals(data_set))


def find_discretization_param(data_set):
    max_score = 0
    for i in range(1000):
        score1 = score(data_set, FrequencyDiscreteNaiveBayes, i + 1)
        print(i)
        if max_score < score1:
            max_score = score1
            print(f"{max_score} {i + 1}")


def different_data_splits_comparsion():
    result = []
    data_set = dp.load_glass_data()
    data_set_shuff = dp.load_glass_data()
    random.shuffle(data_set_shuff)
    for k in [2, 3, 5, 10]:
        sns = score(data_set, EntropyDiscreteNaiveBayes, utils.get_entropy_intervals(data_set), k=k, stratified=False)
        ss = score(data_set, EntropyDiscreteNaiveBayes, utils.get_entropy_intervals(data_set), k=k)
        sns_shuff = score(data_set_shuff, EntropyDiscreteNaiveBayes, utils.get_entropy_intervals(data_set), k=k,
                          stratified=False)
        ss_shuff = score(data_set_shuff, EntropyDiscreteNaiveBayes, utils.get_entropy_intervals(data_set), k=k)
        result.append([ss, sns, ss_shuff, sns_shuff])
    # todo single split
    for si in range(len(result[0])):
        for ki in range(len(result)):
            print(result[ki][si], end=' ')
        print()

# find_discretization_param(dp.load_wine_data())
# discretization_comparsion()
# on_many_datasets_comparsion()
different_data_splits_comparsion()

# testowanie metod dyskretyzacji (tu można dorzucić gaussa i wybrać zbiór)
# badanie klasyfikatora na 3 różnych zbiorach
# badanie różnych podziałów danych - foldy, stratified, single split

# params
# wine, bins_count - 5
# diabetes, bins_count - 7
# iris, bins_count - 8
# glass, bins_count - 6
# wine, frequency - 21
# diabetes, frequency - 431
# iris, frequency - 3
# glass, frequency - 62
