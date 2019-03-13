from gaussian_naive_bayes import GaussianNaiveBayes
from discrete_naive_bayes import DiscreteNaiveBayes
import validation
import data_providers as dp
import utils


def ten_folds_score(data, name, stratified=True):
    classes = list(
        set(utils.horizontal_split(data)[1]))  # it's important to recognize classes from both training and test sets
    mean_score = validation.ten_fold(data, GaussianNaiveBayes(classes), 'f1_macro', stratified)
    print(f"{name} - mean score: {mean_score}")


def single_split_score(data, name):
    classes = list(
        set(utils.horizontal_split(data)[1]))  # it's important to recognize classes from both training and test sets
    score = validation.ten_fold(data, GaussianNaiveBayes(classes), 'f1_macro')
    print(f"{name} - score: {score}")

def single_split_score_doscrete(data, name):
    classes = list(
        set(utils.horizontal_split(data)[
                1]))  # it's important to recognize classes from both training and test sets
    score = validation.ten_fold(data, DiscreteNaiveBayes(classes, 5), 'f1_macro')
    print(f"{name} - score: {score}")

def ten_folds_score_discrete(data, name, stratified=True):
    classes = list(
        set(utils.horizontal_split(data)[
                1]))  # it's important to recognize classes from both training and test sets
    mean_score = validation.ten_fold(data, DiscreteNaiveBayes(classes, 10), 'f1_macro', stratified)
    print(f"{name} - mean score: {mean_score}")

ten_folds_score_discrete(dp.load_iris_data(), 'iris')
ten_folds_score(dp.load_iris_data(), 'iris')
# ten_folds_score(dp.load_iris_data(), 'iris', False)
# single_split_score(dp.load_iris_data(), 'iris')
# ten_folds_score(dp.load_diabetes_data(), 'diabetes')
# ten_folds_score(dp.load_glass_data(), 'glass')  # fixme problematic
# ten_folds_score(dp.load_wine_data(), 'wine')

# todo
# - Equal width discretization
# - Equal frequency discretization
# - be aware od 0 probs, increase all by one occurence

# - include all classes in get_class_probs
# various measures - Confusion matrix, Accuracy, Precision, Recall and Fscore
# implement measures and understand F-score error at glasses
