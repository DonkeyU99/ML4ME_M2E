import numpy as np


def gradient_filter(type):
    if type == "0":
        filter = [[0, 0, 0], [0, 1, 0], [0, 0, 0]]

    if type == "x":
        filter = [[1 / 4, 0, -1 / 4], [2 / 4, 0, -2 / 4], [1 / 4, 0, -1 / 4]]

    if type == "y":
        filter = [[1 / 4, 2 / 4, 1 / 4], [0, 0, 0], [-1 / 4, -2 / 4, -1 / 4]]

    if type == "xy":
        filter = [[1 / 2, 0, -1 / 2], [0, 0, 0], [-1 / 2, 0, 1 / 2]]

    if type == "xx":
        filter = [
            [1 / 8, 2 / 8, 1 / 8],
            [-2 / 8, -4 / 8, -2 / 8],
            [1 / 8, 2 / 8, 1 / 8],
        ]

    if type == "yy":
        filter = [
            [1 / 8, -2 / 8, 1 / 8],
            [2 / 8, -4 / 8, 2 / 8],
            [1 / 8, -2 / 8, 1 / 8],
        ]
    return np.array(filter)


def omega(input):
    if input == "0":
        q = 0
    elif input == "x" or input == "y":
        q = 1
    elif input == "xx" or input == "yy" or input == "xy":
        q = 2
    return 50 / 2**q
