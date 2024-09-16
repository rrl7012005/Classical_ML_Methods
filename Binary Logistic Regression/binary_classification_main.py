from logistic_regression import *

def main():
    dataset = ClassificationPreprocessing()

    dataset.load_data('mushroom_cleaned.csv', seed=20)
    # data = dataset.generate_normal_data(2, 8800, [-1, 3], 1.0, [-2, 3], 1.0, seed=20)
    # dataset.load_txt_data('X.txt', 'Y.txt')

    r_train = 0.8
    r_val = 0.1
    r_test = 0.1
    col_start = 0
    col_end = -1
    
    _, _, _, training_input, _, _, _, _, _ = dataset.separate_data(col_start, col_end, r_train, r_val, r_test, normalize=True)
    # dataset.plot_data()

    model = LogisticClassificationTraining(dataset)

    lr = 0.00001
    epochs = 100

    ll_train, ll_val, weights = model.gradient_ascent_logistic(lr, epochs)

    model.plot_ll(ll_train, 'Training')
    model.plot_ll(ll_val, 'Validatiion')

    predictions, confusion_matrix, ll_test, accuracy = model.predict(weights)

    print("THE CONFUSION MATRIX IS ", confusion_matrix)
    print("THE TESTING LIKELIHOOD IS", ll_test)
    print("THE MODEL ACCURACY IS", accuracy)

    #model.plot_predictive_distribution(training_input, weights, lambda x : model.basisfunction(x))


main()