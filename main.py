from gaussian_naive_bayes import GaussianNaiveBayes
import validation
import data_providers as dp


def ten_folds_score(data, name):
    mean_score = validation.ten_fold_stratified(data, GaussianNaiveBayes(), 'f1_macro')
    print(f"{name} - mean score: {mean_score}")

# fix
# ten_folds_score(dp.load_iris_data(), 'iris')
# ten_folds_score(dp.load_diabetes_data(), 'diabetes')
ten_folds_score(dp.load_glass_data(), 'glass') #fixme problematic
# ten_folds_score(dp.load_wine_data(), 'wine')

# todo
# - discretization(various types), be aware od 0 probs, increase by one occurence
# raw cross validation and simple split
# various measures - Confusion matrix, Accuracy, Precision, Recall and Fscore
