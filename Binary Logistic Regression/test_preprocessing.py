from logistic_regression import *

global classification1, classification2
classification1 = ClassificationPreprocessing()
classification2 = ClassificationPreprocessing()

def test_load_data():
    filename = 'mushroom_cleaned.csv'
    data = classification1.load_data(filename)

    assert data.dtype == np.float64

def test_separate_data():
    r_train = 0.8
    r_val = 0.1
    r_test = 0.1
    normalize = True

    n_tr, n_v, n_t, tr_i, tr_o, v_i, v_o, t_i, t_o = classification1.separate_data(1, 4, r_train, r_val, r_test, normalize)

    assert tr_i.shape[0] == tr_o.shape[0]
    assert t_i.shape[0] == t_o.shape[0]
    assert v_i.shape[0] == v_o.shape[0]

    assert t_o.shape[1] == v_o.shape[1]
    assert t_o.shape[1] == tr_o.shape[1]

    assert n_tr == tr_i.shape[0]
    assert n_v == v_i.shape[0]
    assert n_t == t_i.shape[0]


    if normalize:
        assert abs(np.min(tr_i) - np.max(tr_i)) <= 10

def test_generate_data():
    n_points = 50
    data = classification2.generate_normal_data(2, n_points, [1.0, 4.0], 1.0, [3.0, 2.0], 1.0)

    assert np.all((data[:, -1] == 0) | (data[:, -1] == 1))

    assert 2 * n_points == len(data)


def test_plot():
    classification2.separate_data(0, -1, 0.8, 0.1, 0.1, normalize=False)
    classification2.plot_data()