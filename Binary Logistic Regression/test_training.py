from logistic_regression import *

def test_log_sigmoid():
    model = LogisticClassificationTraining([])
    z = np.array([-900, -442, 68, -191, 3, 1.2, -887, -709, 36])
    y = model.log_sigmoid(z)
    assert len(y) == len(z)
    assert y[0] == -900
    assert y[2] != 0
    assert y[6] == -887

def test_basis_function():
    model = LogisticClassificationTraining([])
    X = np.random.randn(3, 3)
    Y = model.basisfunction(X, type='linear')

    assert np.all(Y[:, 0] == 1)

def test_compute_avg_ll():
    model = LogisticClassificationTraining([])
    X = np.random.randn(3, 3)
    y = np.random.randn(3, 1)
    w = np.random.randn(3, 1)

    ll = model.compute_avg_ll(X, y, w)
    
    assert ll != None

def test_plot_ll():
    model = LogisticClassificationTraining([])
    ll = np.random.randn(13)

    model.plot_ll(ll, type='Validation')

# def test_plot_predictive_distribution():
#     X = np.random.randn(3, 3)
#     y = np.random.randn(3, 1)
#     w = np.random.randn(3, 1)

#     model = LogisticClassificationTraining(ClassificationPreprocessing())
#     model.plot_predictive_distribution(X, y, w)