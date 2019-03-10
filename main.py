from gaussian_naive_bayes import GaussianNaiveBayes
import validation, data_providers
import utils

data = data_providers.load_iris_data()
attrs, classes = utils.horizontal_split(data)
mean_score = validation.ten_fold_stratified(attrs, classes, GaussianNaiveBayes(), 'f1_macro')
print(f"Mean score: {mean_score}")
