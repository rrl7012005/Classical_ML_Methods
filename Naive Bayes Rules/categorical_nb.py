import pandas as pd
import numpy as np

data = pd.read_csv('mushrooms.csv')

y = data['class']
X = data.drop(columns=['class']) #Drop this column from the data

unique_categories = {col: X[col].unique() for col in X.columns} #dictionary of unique categories for each column

category_counts = {col: len(unique_categories[col]) for col in X.columns} #number of categories in each column

categories = list(unique_categories.keys())

for col in X.columns:
    for category_no, category in enumerate(unique_categories[col]):
        X.loc[X[col] == category, col] = category_no

X_coded = X.astype(int)
y_coded = y.apply(lambda x: 1 if x == 'p' else 0)

X_np = np.array(X_coded)
y_np = np.array(y_coded)

np.random.seed(68)

permutation = np.random.permutation(X_np.shape[ 0 ])
X_np = X_np[ permutation, : ]
y_np = y_np[ permutation ]

np.random.seed(None)

n_data = len(X_np)
r_train = 0.8
r_test = 1 - r_train

X_train = X_np[:int(r_train * n_data), :]
y_train = y_np[:int(r_train * n_data)]

X_test = X_np[int(r_train * n_data):,:]
y_test = y_np[int(r_train * n_data):]

n_train = len(X_train)
n_test = len(X_test)


#Compute priors

priors = np.array([np.sum(y_train == i)/ n_train for i in range(2)])
priors = priors[::-1]
log_priors = np.log(priors)


distribution_1 = {} #Distribution of features in each class
distribution_2 = {}


#Compute likelihoods

alpha = 0 #Laplace smoothing constant

for col_num, column in enumerate(X_train.T):
    feature_data_1 = X_train[y_train == 1, col_num] #feature data given class 1
    feature_data_2 = X_train[y_train == 0, col_num] #feature data given class 2 (y = 0)

    col_name = categories[col_num]
    categories_ftc = unique_categories[col_name]
    no_categoris_ftc = category_counts[col_name] #ftc- for this column

    feature_1_distribution = np.array([(np.count_nonzero(feature_data_1 == i) + alpha) for i in range(no_categoris_ftc)])/(len(feature_data_1) + alpha * no_categoris_ftc)
    feature_2_distribution = np.array([(np.count_nonzero(feature_data_2 == i) + alpha) for i in range(no_categoris_ftc)])/(len(feature_data_2) + alpha * no_categoris_ftc)

    distribution_1[col_name] = feature_1_distribution
    distribution_2[col_name] = feature_2_distribution


log_likelihoods_1 = np.zeros(X_test.shape[0])
log_likelihoods_2 = np.zeros(X_test.shape[0])

for j in range(X_test.shape[1]):
    col_name = categories[j]
    log_likelihoods_1 += np.log(distribution_1[col_name][X_test[:, j]])
    log_likelihoods_2 += np.log(distribution_2[col_name][X_test[:, j]])

ll_overall = np.vstack((log_likelihoods_1, log_likelihoods_2)).T

log_posteriors = log_priors + ll_overall
posteriors = np.exp(log_posteriors)
posteriors /= posteriors.sum(axis=1, keepdims=True)
predictions = 1 - np.argmax(log_posteriors, axis=1)
accuracy = np.mean(predictions == y_test) * 100

fn = np.sum((predictions == 1) & (y_test == 0))
tn = np.sum((predictions == 1) & (y_test == 1))
tp = np.sum((predictions == 0) & (y_test == 0))
fp = np.sum((predictions == 0) & (y_test == 1))

confusion_matrix = np.array([[tn, fp], [fn, tp]])

for i in range(n_test):
    print("PROBABILITIES ARE {} PREDICTION IS {}. THE OUTPUT IS {}. ITS {}".format(posteriors[i], predictions[i], y_test[i], predictions[i] == y_test[i]))

print(100 * confusion_matrix / len(predictions))
print(accuracy)