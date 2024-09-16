import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class ClassificationPreprocessing():
    def __init__(self):
        self.data = None
        self.train_input = None
        self.train_output = None
        self.val_input = None
        self.val_output = None
        self.test_input = None
        self.test_output = None
        self.n_samples = None
        self.n_train_samples = None
        self.n_val_samples = None
        self.n_test_samples = None
        self.dimension = None
        self.train_tilde = None
        self.val_tilde = None
        self.test_tilde = None

    def load_data(self, filename, seed=None):
        """Seed is for random shuffling"""

        np.random.seed(seed)

        data = pd.read_csv(filename)
        print(data.head())

        self.data = np.array(data)

        np.random.shuffle(self.data)

        np.random.seed(None)

        return self.data
    
    def load_txt_data(self, xfilename, yfilename, seed=None):
        """Seed is for random shuffling"""

        np.random.seed(seed)

        X = np.loadtxt(xfilename)
        Y = np.loadtxt(yfilename)
        Y = Y.reshape(Y.shape[0], 1)

        self.data = np.concatenate([X, Y], 1)

        np.random.shuffle(self.data)

        np.random.seed(None)

        return self.data
    
    def generate_normal_data(self, dimension, n_points, mean1, sd1, mean0, sd0, seed=16):
        #dimension, number of data, distribution (mean and sd), seed
        np.random.seed(seed)

        class_1_inputs = np.random.multivariate_normal(mean1, sd1 * np.eye(dimension), n_points)
        class_0_inputs = np.random.multivariate_normal(mean0, sd0 * np.eye(dimension), n_points)

        ones = np.ones([n_points, 1])
        zeros = np.zeros([n_points, 1])

        class_1_inputs = np.concatenate([class_1_inputs, ones], 1)
        class_0_inputs = np.concatenate([class_0_inputs, zeros], 1)

        data = np.concatenate([class_1_inputs, class_0_inputs], 0)

        np.random.shuffle(data)

        self.data = data

        np.random.seed(None)

        return self.data


    def separate_data(self, col_start, col_end, r_train, r_val, r_test, normalize=False):

        """Assuming the output is the -1 entry"""

        if self.data is None:
            raise ValueError("There is no data")
        
        if abs(r_train + r_val + r_test - 1) > 1e-5:
            raise ValueError("Ratios do not add to 1")


        inp = self.data[:, col_start:col_end]
        out = np.expand_dims(self.data[:, -1], 1)

        self.n_samples = len(inp)

        self.train_input = inp[:int(r_train * self.n_samples), :]
        self.train_output = out[:int(r_train * self.n_samples), :]

        self.val_input = inp[int(r_train * self.n_samples):int((r_train + r_val) * self.n_samples), :]
        self.val_output = out[int(r_train * self.n_samples):int((r_train + r_val) * self.n_samples), :]

        self.test_input = inp[int((r_train + r_val) * self.n_samples):, :]
        self.test_output = out[int((r_train + r_val) * self.n_samples):, :]

        self.n_train_samples = len(self.train_input)
        self.n_val_samples = len(self.val_input)
        self.n_test_samples = len(self.test_input)
        self.dimension = len(self.train_input[0])

        if normalize:
            mean = np.mean(self.train_input, axis=0)
            variance = np.var(self.train_input, axis=0)
            std_dev = np.sqrt(variance)

            self.train_input = (self.train_input - mean) / std_dev

            mean = np.mean(self.val_input, axis=0)
            variance = np.var(self.val_input, axis=0)
            std_dev = np.sqrt(variance)

            self.val_input = (self.val_input - mean) / std_dev

            mean = np.mean(self.test_input, axis=0)
            variance = np.var(self.test_input, axis=0)
            std_dev = np.sqrt(variance)

            self.test_input = (self.test_input - mean) / std_dev

        return self.n_train_samples, self.n_val_samples, self.n_test_samples, self.train_input, \
        self.train_output, self.val_input, self.val_output, self.test_input, self.test_output

    
    def plot_data(self, input_data=None, showing=True):

        if self.dimension != 2:
            raise ValueError("Only supporting 2 dimensional graphs")
        

        if np.all(input_data == None):
            input_data = self.train_input
            
        x = input_data[:,0] 
        y = input_data[:,1]

        x_min, x_max = x.min() - 0.5, x.max() + 0.5
        y_min, y_max = y.min() - 0.5, y.max() + 0.5
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))

        plt.figure()
        plt.xlim(xx.min(None), xx.max(None))
        plt.ylim(yy.min(None), yy.max(None)) 

        ax = plt.gca()
        ax.plot(x[self.train_output[:, 0] == 0], y[self.train_output[:, 0] == 0], 'ro', label = 'Class 0')
        ax.plot(x[self.train_output[:, 0] == 1], y[self.train_output[:, 0] == 1], 'bo', label = 'Class 1')
        plt.xlabel('X1')
        plt.ylabel('X2')
        plt.title('Plot data')
        plt.legend(loc = 'upper left', scatterpoints = 1, numpoints = 1)

        if showing:
            plt.show()
        return xx, yy  

        
class LogisticClassificationTraining():
    def __init__(self, model, method='logistic'):
        self.model = model
        self.method = method

    def sigmoid(self, x):  #For high x, 36 is the limit for both sigmoid and ligmoid, sig -> 1, lig -> 0 (make sure not on denom)
        return 1/(1 + np.exp(-x))
    
    def log_sigmoid(self, x): #For low x, -709 is the limit which is allowed. sigmoid goes to 0, lig to -inf (make sure not on denom)

        result = np.zeros_like(x)

        result[x < -709] = x[x < -709]
        result[x > 36] = -np.log(1+np.exp(-36))
        mask = (x >= -709) & (x <= 36)
        result[mask] = -np.log(1+np.exp(-x[mask]))

        return result
    
    def basisfunction(self, X, type='gaussianrbf', otherargs = 0.1, index = None):
        if type == 'linear' or index == 0:
            X = np.concatenate([np.ones([X.shape[0], 1]), X], 1)
            return X
        elif type == 'gaussianrbf':
            l = otherargs
            Z = self.model.train_input
            X2 = np.sum(X**2, 1)
            Z2 = np.sum(Z**2, 1)
            ones_Z = np.ones(Z.shape[ 0 ])
            ones_X = np.ones(X.shape[ 0 ])
            r2 = np.outer(X2, ones_Z) - 2 * np.dot(X, Z.T) + np.outer(ones_X, Z2)
            r2 = np.exp(-0.5 / l**2 * r2)
            r2 = np.concatenate([np.ones([r2.shape[0], 1]), r2], 1)
            return r2
        else:
            raise NotImplementedError("TBC")
    
    def compute_avg_ll(self, X, y, w):
        s1 = self.log_sigmoid(np.dot(X, w))
        s2 = self.log_sigmoid(-np.dot(X, w))
        return np.mean(y * s1 + (1 - y) * s2)
        
    def gradient_ascent_logistic(self, lr, epochs, no_batches=None, batch_size=None):

        ll_train = np.zeros(epochs)
        ll_val = np.zeros(epochs)
        self.model.train_tilde = self.basisfunction(self.model.train_input)
        self.model.val_tilde = self.basisfunction(self.model.val_input)
        weights = np.random.randn(self.model.train_tilde.shape[1], 1)

        for i in range(epochs):
            sigmoid_value = self.sigmoid(np.dot(self.model.train_tilde, weights))
            sigmoid_value = sigmoid_value.reshape(sigmoid_value.shape[0], 1)
            grad = np.dot(self.model.train_tilde.T, (self.model.train_output - sigmoid_value))

            weights += lr * grad

            ll_train[i] = self.compute_avg_ll(self.model.train_tilde, self.model.train_output, weights)
            ll_val[i] = self.compute_avg_ll(self.model.val_tilde, self.model.val_output, weights) #average validation log likelihood

            print("EPOCH {} RESULTS: THE GRADIENT IS {} AND THE PARAMETERS ARE {} AND THE TRAIN LIKELIHOOD IS {}, VAL LIKELIHOOD {}"\
                  .format(i+1, grad, weights, ll_train[i], ll_val[i]))
            
        return ll_train, ll_val, weights
    
    def plot_ll(self, ll, type):

        epochs = len(ll)
        x = np.arange(1, epochs + 1)

        plt.figure()
        ax = plt.gca()
        plt.xlim(0, epochs + 2)
        plt.ylim(min(ll) - 0.1, max(ll) + 0.1)
        ax.plot(x, ll, 'r-')
        plt.xlabel('Steps')
        plt.ylabel('Average {} log-likelihood'.format(type))
        plt.title('Plot Average {} Log-likelihood Curve'.format(type))
        plt.show()

    def predict(self, weights):

        self.model.test_tilde = self.basisfunction(self.model.test_input)
        prediction = self.sigmoid(np.dot(self.model.test_tilde, weights))
        prediction = prediction.reshape(prediction.shape[0])

        test_ll = self.compute_avg_ll(self.model.test_tilde, self.model.test_output, weights)

        tp, fp, tn, fn = 0, 0, 0, 0

        for i in range(len(prediction)):
            if prediction[i] >= 0.5:
                output = 1
            else:
                output = 0
            
            if output == 0:
                if output == self.model.test_output[i]:
                    tn += 1
                else:
                    fn += 1
            elif output == 1:
                if output == self.model.test_output[i]:
                    tp += 1
                else:
                    fp += 1
    
        confusion_matrix = 100 * np.array([[tn, fp], [fn, tp]]) / len(prediction)

        accuracy = confusion_matrix[1][1] + confusion_matrix[0][0]

        return prediction, confusion_matrix, test_ll, accuracy
    
    def plot_predictive_distribution(self, data, w, map_inputs = lambda x : x):

        # if self.model.dimension != 2:
        #     raise ValueError("Plotting only for 2D inputs")
        
        xx, yy = self.model.plot_data(input_data = data, showing=False)
        ax = plt.gca()
        X_tilde = map_inputs(np.concatenate((xx.ravel().reshape((-1, 1)), yy.ravel().reshape((-1, 1))), 1))
        Z = self.sigmoid(np.dot(X_tilde, w))
        Z = Z.reshape(xx.shape)
        cs2 = ax.contour(xx, yy, Z, cmap = 'RdBu', linewidths = 2)
        plt.clabel(cs2, fmt = '%2.1f', colors = 'k', fontsize = 14)
        plt.show()