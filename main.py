from gaussian_naive_bayes import GaussianNaiveBayes
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


# fix
ten_folds_score(dp.load_iris_data(), 'iris')
ten_folds_score(dp.load_iris_data(), 'iris', False)
single_split_score(dp.load_iris_data(), 'iris')
# ten_folds_score(dp.load_diabetes_data(), 'diabetes')
# ten_folds_score(dp.load_glass_data(), 'glass')  # fixme problematic
# ten_folds_score(dp.load_wine_data(), 'wine')

# todo
# - discretization(various types), be aware od 0 probs, increase by one occurence
# various measures - Confusion matrix, Accuracy, Precision, Recall and Fscore
# implement measures and understand F-score error at glasses
