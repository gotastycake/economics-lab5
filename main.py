import math
import numpy as np
np.set_printoptions(suppress=True, precision=2)

albreht_set = [
    # IN OUT FILE INQ FP SLOC RawFP EFFORT
    [25, 150, 60, 75, 17.50, 130.000, 17.50, 102.4],
    [193, 98, 36, 70, 19.02, 318.000, 19.02, 105.2],
    [70, 27, 12, 0, 4.28, 20.000, 5.35, 11.1],
    [40, 60, 12, 20, 7.59, 54.000, 6.60, 21.1],
    [10, 69, 9, 1, 4.31, 62.000, 4.78, 28.8],
    [13, 19, 23, 0, 2.83, 28.000, 3.7733, 10],
    [34, 14, 5, 0, 2.05, 35.000, 2.5625, 8],
    [17, 17, 5, 15, 2.89, 30.000, 2.6273, 4.9],
    [45, 64, 16, 14, 6.80, 48.000, 7.1579, 12.9],
    [40, 60, 15, 20, 7.94, 93.000, 6.9043, 19],
    [41, 27, 5, 29, 5.12, 57.000, 4.6545, 10.8],
    [33, 17, 5, 8, 2.24, 22.000, 2.9859, 2.9],
    [28, 41, 11, 16, 4.17, 24.000, 4.9059, 7.5],
    [43, 40, 35, 20, 6.82, 42.000, 8.0235, 12],
    [7, 12, 8, 13, 2.09, 40.000, 5.5091, 4.1],
    [28, 38, 9, 24, 5.12, 96.000, 4.8762, 15.8],
    [42, 57, 5, 12, 6.06, 40.000, 5.5091, 18.3],
    [27, 20, 6, 24, 4.00, 52.000, 3.6364, 8.9],
    [48, 66, 50, 13, 1.235, 94.000, 10.7391, 38.1],
    [69, 112, 39, 21, 1.572, 110.000, 13.10, 61.2],
    [25, 28, 22, 4, 5.00, 15.000, 4.7619, 3.6],
    [61, 68, 11, 0, 6.94, 24.000, 6.94, 11.8],
    [15, 15, 3, 6, 1.99, 3.000, 1.8952, 0.5],
    [12, 15, 15, 0, 2.60, 29.000, 2.3768, 6.1],
]


def divide_set_on_train_test(_set, divide):
    np.random.shuffle(_set)
    return _set[:divide], _set[divide:]


def divide_set_on_x_y(_set):
    return np.array(_set)[::, :-1:], np.array(_set)[::, -1::],


def get_length(elem1, elem2):
    if len(elem1) != len(elem2):
        print('Array sizes are not equal', elem1, elem2)
        return

    length = 0
    elements = len(elem1)
    for i in range(elements):
        t = (elem1[i] - elem2[i]) ** 2
        length += t
    length /= elements

    return math.sqrt(length)


def calc_effort(elem1, elem2, neighbour_effort):
    if len(elem1) != len(elem2):
        print('Array sizes are not equal', elem1, elem2)
        return

    _effort = 0
    elements = len(elem1)
    for i in range(elements):
        if (elem2[i]) == 0:
            continue

        _effort += elem1[i] / elem2[i]
    _effort /= elements
    _effort *= neighbour_effort
    return _effort


if __name__ == '__main__':
    print('Albreht dataset size:', len(albreht_set))
    train, test = divide_set_on_train_test(albreht_set, 20)
    print('Train set size: {}, test set size: {}'.format(len(train), len(test)))

    train_x, train_y = divide_set_on_x_y(train)
    test_x, test_y = divide_set_on_x_y(test)

    test_distances = np.empty((len(test), len(train)))
    for i, test_i in enumerate(test_x):
        for j, train_i in enumerate(train_x):
            test_distances[i][j] = get_length(test_i, train_i)

    min_neighbours_indexes = list(map(lambda x: np.argmin(x), test_distances))
    for i, test_i in enumerate(test_x):
        effort = calc_effort(test_i, train_x[min_neighbours_indexes[i]], train_y[i])
        p_e, t_e = float(effort[0]), float(test_y[i][0])
        abs_error = abs(t_e - p_e)
        rel_error = abs_error / t_e * 100
        print('Calculated effort: {:.2f}, actual effort: {:.2f}, absolute error: {:.2f}, relative error: {:.2f}%'.format(p_e, t_e, abs_error, rel_error))

    print('Done.')
