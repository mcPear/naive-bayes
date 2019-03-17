from gaussian_naive_bayes import GaussianNaiveBayes
from width_discrete_naive_bayes import WidthDiscreteNaiveBayes
from frequency_discrete_naive_bayes import FrequencyDiscreteNaiveBayes
import validation
import data_providers as dp
import utils


def score(data, name, estimator_const, est_param, cross_val=True, stratified=True):
    attrs, classes = utils.horizontal_split(data)
    attr_ranges = utils.attr_ranges(attrs)
    unique_classes = utils.unique_classes(
        classes)  # it's important to recognize classes from both training and test sets
    estimator = estimator_const(unique_classes, attr_ranges, est_param)
    scoring = 'f1_macro'
    score = validation.ten_fold(data, estimator, scoring, stratified) \
        if cross_val else validation.single_split(data, estimator, scoring)
    # print(f"{name} - " + ("mean " if cross_val else "") + f"score: {score}")
    return score


max_score = 0
for i in range(1000):
    score1 = score(dp.load_glass_data(), 'glass', WidthDiscreteNaiveBayes, i + 1, stratified=True)
    if max_score < score1:
        max_score = score1
        print(f"{max_score} {i}")

# todo
# - 3rd discretization
# various measures - Confusion matrix, Accuracy, Precision, Recall and Fscore
# implement measures and understand F-score error at glasses

# params
# wine, width - 4
# diabetes, width - 6
# iris, width - 7
# glass, width - 5 #fixme warns
