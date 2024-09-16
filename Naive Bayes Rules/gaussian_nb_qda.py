import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

#Generate and separate data

np.random.seed(25)

class_1_inputs = np.random.multivariate_normal([-1, 5], np.eye(2), 75)
class_2_inputs = np.random.multivariate_normal([2, 0], np.eye(2), 75)
class_3_inputs = np.random.multivariate_normal([-3, -4], np.eye(2), 75)


np.random.seed(None)

X = np.concatenate([class_1_inputs, class_2_inputs, class_3_inputs], 0)
y = np.concatenate([np.ones(class_1_inputs.shape[0]), 2 * np.ones(class_2_inputs.shape[0]), 3 * np.ones((class_3_inputs.shape[0]))], 0)

np.random.seed(43)

permutation = np.random.permutation(X.shape[ 0 ])
X = X[ permutation, : ]
y = y[ permutation ]

n_data = len(X)

r_train = 0.8
r_test = 1 - r_train

X_train = X[:int(0.8*n_data),:]
y_train = y[:int(0.8*n_data)]

X_test = X[int(0.8*n_data):,:]
y_test = y[int(0.8*n_data):]

n_train = len(X_train)
n_test = len(X_test)

np.random.seed(None)

#Plot training data

x_min, x_max = X_train[ :, 0 ].min() - .5, X_train[ :, 0 ].max() + .5
y_min, y_max = X_train[ :, 1 ].min() - .5, X_train[ :, 1 ].max() + .5
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
plt.figure()
plt.xlim(xx.min(None), xx.max(None))
plt.ylim(yy.min(None), yy.max(None))
ax = plt.gca()
ax.plot(X_train[y_train == 1, 0], X_train[y_train == 1, 1], 'ro', label = 'Class 1')
ax.plot(X_train[y_train == 2, 0], X_train[y_train == 2, 1], 'bo', label = 'Class 2')
ax.plot(X_train[y_train == 3, 0], X_train[y_train == 3, 1], 'ko', label = 'Class 3')
plt.xlabel('X1')
plt.ylabel('X2')
plt.title('Plot training data')
plt.legend(loc = 'upper left', scatterpoints = 1, numpoints = 1)
plt.show()

#Priors for each class

counts = np.array([np.count_nonzero(y_train == i+1) for i in range(3)])
priors = counts / n_train
log_priors = np.log(priors)

#Gaussian parameters for each class

means = np.array([np.mean(X_train[y_train == i + 1], axis=0) for i in range(3)])
covariances = np.array([np.cov(X_train[y_train == i + 1].T, bias=True) for i in range(3)]) #Quadratic Discriminant Analysis (accounts for feature dependencies)

# variances = np.array([np.var(X_train[y_train == i + 1], axis=0) for i in range(3)])
# covariances = np.array([np.diag(variances[i]) for i in range(variances.shape[0])]) #Gaussian NB (Assumes features are independent)s

#Testing

def eval_likelihood(means, covariances, X_test):
    likelihood = np.zeros((X_test.shape[0], means.shape[0]))

    for i in range(means.shape[0]):
        rv = multivariate_normal(mean=means[i], cov=covariances[i])
        likelihood[:, i] = rv.pdf(X_test)

    return likelihood

likelihoods = eval_likelihood(means, covariances, X_test)
posteriors = priors * likelihoods
posteriors /= posteriors.sum(axis=1, keepdims=True)
predictions = np.argmax(posteriors, axis=1) + 1
prediction_probabiltiies = np.max(posteriors, axis=1)


for i in range(n_test):
    print("PROBABILITIES ARE {} PREDICTION IS {}. THE OUTPUT IS {}. ITS {}".format(posteriors[i], predictions[i], y_test[i], predictions[i] == y_test[i]))

accuracy = np.mean(predictions == y_test) * 100

print("THE ACCURACY IS ", accuracy)


def generate_decision_boundary_data(mux, sigmax, muy, sigmay, priorx, priory, X):
    x_min, x_max = X[ :, 0 ].min() - .5, X[ :, 0 ].max() + .5
    x = np.linspace(x_min, x_max, 300)
    A = np.linalg.inv(sigmax) - np.linalg.inv(sigmay)
    b = 2 * (np.dot(np.linalg.inv(sigmay), muy) - np.dot(np.linalg.inv(sigmax), mux))
    c = np.dot(mux.T, np.dot(np.linalg.inv(sigmax), mux)) - np.dot(muy.T, np.dot(np.linalg.inv(sigmay), muy)) \
        - np.log(np.linalg.det(sigmay)/np.linalg.det(sigmax)) - 2 * np.log(priorx/priory)

    y1 = -((A[0][1] + A[1][0]) * x + b[1])/(2*A[1][1]) + np.sqrt((((A[0][1] + A[1][0]) * x + b[1])/(2*A[1][1])) ** 2 - b[0] * x/A[1][1] \
        - c / A[1][1] - A[0][0] * (x ** 2) / A[1][1])
    y2 = -((A[0][1] + A[1][0]) * x + b[1])/(2*A[1][1]) - np.sqrt((((A[0][1] + A[1][0]) * x + b[1])/(2*A[1][1])) ** 2 - b[0] * x/A[1][1] \
        - c / A[1][1] - A[0][0] * (x ** 2) / A[1][1])

    return x, y1, y2

#Overlay training and testing data

x_min, x_max = X[ :, 0 ].min() - .5, X[ :, 0 ].max() + .5
y_min, y_max = X[ :, 1 ].min() - .5, X[ :, 1 ].max() + .5
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
plt.figure()
plt.xlim(xx.min(None), xx.max(None))
plt.ylim(yy.min(None), yy.max(None))
ax = plt.gca()
ax.plot(X_train[y_train == 1, 0], X_train[y_train == 1, 1], 'ro', label = 'Class 1')
ax.plot(X_test[y_test == 1, 0], X_test[y_test == 1, 1], 'rD', label = 'Class 1')
ax.plot(X_train[y_train == 2, 0], X_train[y_train == 2, 1], 'bo', label = 'Class 2')
ax.plot(X_test[y_test == 2, 0], X_test[y_test == 2, 1], 'bD', label = 'Class 2')
ax.plot(X_train[y_train == 3, 0], X_train[y_train == 3, 1], 'ko', label = 'Class 3')
ax.plot(X_test[y_test == 3, 0], X_test[y_test == 3, 1], 'kD', label = 'Class 3')
x, y1, y2 = generate_decision_boundary_data(means[0], covariances[0], means[1], covariances[1], priors[0], priors[1], X)
ax.plot(x, y1, label='1-2 boundary', color = 'blue')
ax.plot(x, y2, label='1-2 boundary', color = 'blue')
x, y1, y2 = generate_decision_boundary_data(means[1], covariances[1], means[2], covariances[2], priors[1], priors[2], X)
ax.plot(x, y1, label='2-3 boundary', color = 'red')
ax.plot(x, y2, label='2-3 boundary', color = 'red')
x, y1, y2 = generate_decision_boundary_data(means[2], covariances[2], means[0], covariances[0], priors[2], priors[0], X)
ax.plot(x, y1, label='1-3 boundary', color = 'black')
ax.plot(x, y2, label='1-3 boundary', color = 'black')
plt.legend()
plt.xlabel('X1')
plt.ylabel('X2')
plt.title('Plot training and testing data')
plt.legend(loc = 'upper left', scatterpoints = 1, numpoints = 1)
plt.show()