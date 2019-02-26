import numpy as np
from random import shuffle
import pandas as pd
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV, KFold
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.regularizers import l1
from keras.optimizers import Adamax, Adam
from nn_lib import Preprocessor
import tensorflow as tf
from illustrate import illustrate_results_ROI
from keras.utils.np_utils import to_categorical
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score
from numpy.random import randint
from confusion_matrix import precision
from imblearn.over_sampling import SMOTE, ADASYN

def create_model(activation='relu', dropout=0, input_shape=3, output_shape=4, num_layers1=1, num_layers2=1, num_layers3=1,
                 num_neurons_1=1000, num_neurons_2 = 500, num_neurons_3 = 200, learning_rate=0.01):

    opt = Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999)

    model = Sequential()

    first = True
    for i in range(num_layers1):
        if first:
            model.add(Dense(num_neurons_1, input_dim=input_shape, activation=activation))
            first = False
        else:
            model.add(Dense(num_neurons_1, activation=activation))
        if dropout != 0:
            model.add(Dropout(dropout))

    for i in range(num_layers2):
        if first:
            model.add(Dense(num_neurons_2, input_dim=input_shape, activation=activation))
            first = False
        else:
            model.add(Dense(num_neurons_2, activation=activation))
        if dropout != 0:
            model.add(Dropout(dropout))

    for i in range(num_layers3):
        if first:
            model.add(Dense(num_neurons_3, input_dim=input_shape, activation=activation))
            first = False
        else:
            model.add(Dense(num_neurons_3, activation=activation))
        if dropout != 0:
            model.add(Dropout(dropout))

    model.add(Dense(output_shape, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

    return model



def main():
    dataset = np.loadtxt("ROI_dataset.dat")
    #######################################################################
    #                       ** START OF YOUR CODE **
    #######################################################################
    shuffle(dataset)
    x = dataset[:, :3]
    y = dataset[:, 3:]

    proc = Preprocessor(x)
    proc.apply(x)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
    base_model = Sequential()

    x_resampled, y_resampled = oversample(x_train, y_train)


    # x_resampled, y_resampled = SMOTE().fit_resample(x_train, y_train)

    # class_weight = {0: 9.,
    #                 1: 11.,
    #                 2: 100.,
    #                 3: 1.}

    base_model.add(Dense(units=64, activation='relu', input_dim=3))
    base_model.add(Dense(units=4, activation='softmax'))

    # optimizer = Adamax(lr=0.006, beta_1=0.9, beta_2=0.999, epsilon=1.0)
    #
    base_model.compile(
        loss='categorical_crossentropy',
        optimizer="adam",
        metrics=['accuracy']
    )

    batch_size = 100
    epochs = 30

    base_model.fit(x_resampled, y_resampled, batch_size=batch_size, epochs=epochs, verbose=1)

    base_pred = base_model.predict(x_test, batch_size=batch_size)

    evaluate_architecture(y_test, base_pred)

    #  *********** Random search  ***********

    model = KerasClassifier(build_fn=create_model, epochs=epochs, batch_size=batch_size, verbose=1)

    grid = parameter_tuning(model, x, y)

    grid_result = grid.fit(x_resampled, y_resampled)
    best_params = grid_result.best_params_
    print(best_params)

    cv_results_df = pd.DataFrame(grid_result.cv_results_)
    cv_results_df.to_csv('gridsearch.csv')
    print(cv_results_df)


    # *************** BEST MODEL TRAINING ******************
    best_model = create_model(best_params['activation'], best_params['dropout'], 3, 4, best_params['num_layers1'], best_params['num_layers3'], best_params['num_layers3'],
                              best_params['num_neurons_1'], best_params['num_neurons_2'], best_params['num_neurons_3'], best_params['learning_rate'])

    best_model.fit(x_resampled, y_resampled, epochs=epochs, batch_size=batch_size)

    softmax_pred = best_model.predict(x_test, batch_size=batch_size)

    evaluate_architecture(y_test, softmax_pred)

    #######################################################################
    #                       ** END OF YOUR CODE **
    #######################################################################
    # illustrate_results_ROI(network, prep)

def evaluate_architecture(y_test, softmax_pred):

    pred = np.argmax(softmax_pred, axis=1)

    y_test_argmax = np.argmax(y_test, axis=1)
    cm = confusion_matrix(y_test_argmax, pred)
    print("before normalisation")
    print(cm)

    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    print("after normalisation")
    print(cm)

    f1 = f1_score(y_test_argmax, pred, average='weighted')
    recall = recall_score(y_test_argmax, pred, average='weighted')
    prec = precision_score(y_test_argmax, pred, average='weighted')
    precision_weighted_sklearn = precision_score(y_test_argmax, pred, average='weighted')
    print("precision: " + str(prec))

    precision_weighted = [ precision(cm, i) for i in range (4)]
    print("precision weighted: " + str(precision_weighted))
    print("precision weighted sklearn: " + str(precision_weighted_sklearn))
    print("f1: " + str(f1))
    print("recall: " + str(recall))

    # score = model.evaluate(x_test, y_test, batch_size=batch_size)

    # print(score)
    # print("test set accuracy: " + "{:.6f}".format(score[1] * 100) + "%")

def predict_hidden(dataset):
    shuffle(dataset)
    x = dataset[:, :3]
    y = dataset[:, 3:]

    proc = Preprocessor(x)
    proc.apply(x)

    # INSERT BEST model here


def parameter_tuning(model, x_train, y_train):
    # perform random search for a wide range of parameters

    # learning algorithm parameters
    lr = [1e-2, 1e-3, 1e-4]

    # activation
    activation = ['relu', 'sigmoid', 'tanh']

    # numbers of layers
    num_layers1 = [1, 2, 3]
    num_layers2 = [1, 2, 3]
    num_layers3 = [1, 2, 3]

    # neurons in each layer
    num_neurons_1 = [16, 32, 64]
    num_neurons_2 = [16, 32, 64]
    num_neurons_3 = [16, 32, 64]

    # dropout and regularisation
    dropout = [0, 0.1, 0.2, 0.3, 0.4, 0.5]

    # dictionary summary
    param_grid = dict(
        num_layers1=num_layers1, num_layers2=num_layers2, num_layers3=num_layers3,
        num_neurons_1=num_neurons_1, num_neurons_2=num_neurons_2, num_neurons_3=num_neurons_3,
        activation=activation, learning_rate=lr, dropout=dropout,
        input_shape=[x_train.shape[1]], output_shape=[y_train.shape[1]],
    )

    grid = RandomizedSearchCV(estimator=model, cv=KFold(3), param_distributions=param_grid,
                              verbose=20, n_iter=10, n_jobs=3)

    return grid

def oversample(x_data, y_data):
    length = len(x_data)
    classification = np.argmax(y_data, axis=1)
    class_count = np.bincount(classification)
    max_class_count = class_count[np.argmax(class_count)]

    for c in range(len(class_count)):

        indices = [ i for i in range(length) if classification[i] == c  ]

        x_class_data, y_class_data = x_data[indices], y_data[indices]
        diff = max_class_count - class_count[c]
        class_data_len = len(x_class_data)
        for i in range(diff):
            rand = randint(0, class_data_len)
            x_data = np.concatenate([x_data, [x_class_data[rand]]])
            y_data = np.concatenate([y_data, [y_class_data[rand]]])

    return x_data, y_data



if __name__ == "__main__":
    main()
