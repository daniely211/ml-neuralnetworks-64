import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from random import shuffle
# from sklearn import preprocessing
from nn_lib import Preprocessor
# from sklearn.multioutput import MultiOutputRegressor
from sklearn.neural_network import MLPRegressor


def main():
    dataset = np.loadtxt("FM_dataset.dat")

    #######################################################################
    #                       ** START OF YOUR CODE **
    #######################################################################
    shuffle(dataset)
    p = Preprocessor(dataset)
    dataset = p.apply(dataset)

    x = dataset[:, :3]
    y = dataset[:, 3:]

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    model = MLPRegressor(

        learning_rate = 'adaptive',
        random_state = 42
    )

    param_grid = dict(
        hidden_layer_sizes = [(32, 16, 8), (16, 8)],
        activation = ['relu', 'tanh', 'logistic'],
        solver = ['adam', 'sgd'],
        learning_rate_init = np.arange(0.001, 0.01, 0.001)
    )

    grid = GridSearchCV(
        estimator = model,
        param_grid = param_grid,
        cv = 3,
        scoring = 'r2'
    )

    grid_result = grid.fit(x_train, y_train)

    print(grid_result.best_params_)

    print('test result: ', grid.score(x_test, y_test))

    #######################################################################
    #                       ** END OF YOUR CODE **
    #######################################################################

if __name__ == "__main__":
    main()
