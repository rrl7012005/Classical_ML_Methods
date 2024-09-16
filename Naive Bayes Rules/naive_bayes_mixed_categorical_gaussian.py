import pandas as pd
import numpy as np
from scipy.stats import multivariate_normal

data = pd.read_csv('Stars.csv')

y_raw = data['Spectral Class'].copy()
X_raw = data.drop(columns='Spectral Class')

#First 4 columns are continuous values, the last 3 discrete

unique_categories = {col: X_raw[col].unique() for col in X_raw.columns[-3:]}
categories = list(unique_categories.keys())
category_counts = {col: len(unique_categories[col]) for col in categories}
output_classes = y_raw.unique()
output_encodings = {}

#Encode categories into numerical values

for col in categories:
    for category_no, category in enumerate(unique_categories[col]):
        X_raw.loc[X_raw[col] == category, col] = category_no

for category_no, category in enumerate(output_classes):
    y_raw.loc[y_raw == category] = category_no
    output_encodings[category_no] = category

X_continuous = X_raw.iloc[:, :4]
X_categorical = X_raw.iloc[:, -3:].astype(int)

X_encoded = pd.concat([X_continuous, X_categorical], axis=1)
y_encoded = y_raw.astype(int)

X_np = np.array(X_encoded)
y_np = np.array(y_encoded)

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


X_means = np.concatenate([np.mean(X_train[:,:4], axis=0), np.zeros(X_train[:,4:].shape[1])], 0)
X_vars = np.concatenate([np.var(X_train[:,:4], axis=0), np.ones(X_train[:,4:].shape[1])], 0)

X_train = (X_train - X_means)/ np.sqrt(X_vars)

X_test = X_np[int(r_train * n_data):,:]
y_test = y_np[int(r_train * n_data):]

X_test = (X_test - X_means) / np.sqrt(X_vars)

n_train = len(X_train)
n_test = len(X_test)


#Compute priors

n_classes = len(output_classes)
prior_alpha = 0 #Smoothing

priors = np.array([np.sum(y_train == i) + prior_alpha for i in range(n_classes)]) / (n_train + prior_alpha * n_classes)
log_priors = np.log(priors)

means = np.array([np.mean(X_train[y_train == i, :4], axis=0) for i in range(n_classes)])
covariances = np.array([np.cov(X_train[y_train == i, :4].T, bias=True) for i in range(n_classes)]) #QDA

# variances = np.array([np.var(X_train[y_train == i, :4], axis=0) for i in range(n_classes)])
# covariances = np.array([np.diag(variances[i]) for i in range(variances.shape[0])]) #Gaussian NB (Assumes features are independent)

def eval_continuous_log_likelihood(means, covariances, X_test):
    log_likelihood = np.zeros((X_test.shape[0], means.shape[0]))

    for i in range(means.shape[0]):
        log_likelihood[:, i] = -(X_test.shape[1]/2) * np.log(2 * np.pi) - np.log(np.linalg.det(covariances[i]))/2 \
            - 0.5 * np.diagonal(np.dot((X_test - means[i]), np.dot(np.linalg.inv(covariances[i]), (X_test - means[i]).T)))

    return log_likelihood


#Create probability distributions

alpha = 0
# feature_distributions = {categories[feature_num]: {category_no: [np.sum(X_train[(X_train[:, 4 + feature_num] == 0) \
#     & (y_train == i), 4 + feature_num]) + alpha for i in range(n_classes)]/(X_train[X_train[:,4+feature_num] == category_no, 4 +feature_num] \
#     + alpha * n_classes) for category_no in unique_categories[categories[feature_num]]} for feature_num in range(X_train[:,4:].shape[1])}

feature_distributions = {}

for feature_num in range(X_train[:, 4:].shape[1]):
    feature_name = categories[feature_num]
    feature_distributions[feature_name] = {}
    
    for category_no, category in enumerate(unique_categories[feature_name]):
        class_counts = np.array([np.sum((X_train[:, 4 + feature_num] == category_no) & (y_train == i)) + alpha for i in range(n_classes)])
        
        total_count = np.sum(X_train[:, 4 + feature_num] == category_no) + alpha * n_classes
        
        feature_distributions[feature_name][category_no] = np.log(class_counts / total_count)


def eval_discrete_log_likelihood(X_test, feature_distributions, categories, n_classes):
    ll = np.zeros((X_test.shape[0], n_classes))
    for i in range(X_test.shape[0]):
        ll_sum = np.zeros(n_classes)
        for feature_num, column in enumerate(X_test.T):
            ll_sum += feature_distributions[categories[feature_num]][column[i]]
        ll[i] = ll_sum
    
    return ll



cont_ll = eval_continuous_log_likelihood(means, covariances, X_test[:,:4])
discrete_ll = eval_discrete_log_likelihood(X_test[:,4:], feature_distributions, categories, n_classes)

total_ll = cont_ll + discrete_ll

log_posteriors = log_priors + total_ll
posteriors = np.exp(log_posteriors)
posteriors /= posteriors.sum(axis=1, keepdims=True)
predictions = np.argmax(log_posteriors, axis=1)
accuracy = np.mean(predictions == y_test) * 100

for i in range(n_test):
    print("PROBABILITIES ARE {} PREDICTION IS {}. THE OUTPUT IS {}. ITS {}".format(posteriors[i], predictions[i], y_test[i], predictions[i] == y_test[i]))

print(accuracy)