"""
Provides testing data sets.

SHAPES will always be:
inputs X: [n_batch, n_timesteps, n_states]
outputs y: [n_batch, n_timesteps, n_states]

"""
import numpy as np
from matplotlib import pyplot as plt

# TODO: add Lorenz and other test cases (with potential limits to the number of states etc.)


def gen_sine(n=10, omega=np.pi):
    # generates a sequence of sines and cosines with a given frequency. sampling is 10 points per period
    t = np.linspace(start=0, stop=n / 50 * omega, num=n, endpoint=True)
    return np.sin(omega * t)


def gen_cos(n=10, omega=np.pi):
    # generates a sequence of sines and cosines with a given frequency. sampling is 10 points per period
    t = np.linspace(start=0, stop=n / 50 * omega, num=n, endpoint=True)
    return np.cos(omega * t)


def split_sequence(signal, n_batch, n_time_in, n_time_out):
    # expects [n_timesteps, n_states] sequence

    # convert into inputs (function at last n_time_in points) and outputs (function at next n_time_out points)
    x, y = [], []
    for i in range(n_batch):
        idx_in_end = i + n_time_in
        idx_out_end = idx_in_end + n_time_out
        x.append(signal[i:idx_in_end, :])  # last n_timestep_in points
        y.append(signal[idx_in_end:idx_out_end, :])  # next n_timestep_out points
    x, y = np.array(x), np.array(y)

    return x, y


def train_test_split(x, y):
    # train-test split 80% sample random points from the sequence and return inputs and outputs
    n = x.shape[0]

    ratio = 0.8

    split_idx = np.max([1, int(n * ratio)])

    shuffle_idx = np.random.choice(n, size=n, replace=False)
    train_idx, test_idx = shuffle_idx[:int(n * ratio)], shuffle_idx[int(n * ratio):]

    # split data according to train/test split and return.
    return x[train_idx], x[test_idx], y[train_idx], y[test_idx]


def sine_pred(n_batch, n_time_in, n_time_out, n_states):
    # predict a sine signal. Single- and multi-step prediction supported


    # we will create different signal frequencies for the different states
    signal, omega = [], np.pi
    for _ in range(n_states):
        signal.append(gen_sine(n=n_batch + (n_time_in + n_time_out) + 1, omega=omega))
        omega += 0.314

    # 2D array of shape [n_timesteps, n_states]
    signal = np.array(signal).transpose()

    # split into inputs and outputs
    x, y = split_sequence(signal, n_batch, n_time_in, n_time_out)

    # train-test split
    X_train, X_test, y_train, y_test = train_test_split(x, y)

    return X_train, X_test, y_train, y_test


"""
generate some training and testing data from the sine function. 

CASE 1: scalar inputs, scalar outputs
"""


def scalar_to_scalar(name, n_batch: int = 50):

    # make sure to have at least 1 testing sample
    n_batch = np.max([n_batch, 2])

    if name == 'sine_pred':
        # single-step predict a sine signal
        X_train, X_test, y_train, y_test = sine_pred(n_batch=n_batch, n_states=1, n_time_in=1, n_time_out=1)

    print(f'shape of inputs (training): {X_train.shape}, shape of outputs (training): {y_test.shape}')
    return X_train, X_test, y_train, y_test


"""
CASE 2: Vector to Vector
"""


def vector_to_vector(name, n_batch: int = 50, n_states: int = 2):

    # make sure to have at least 1 testing sample
    n_batch = np.max([n_batch, 2])

    if name == 'sine_pred':
        # single-step predict a vector of sine signals
        X_train, X_test, y_train, y_test = sine_pred(n_batch=n_batch, n_states=n_states, n_time_in=1, n_time_out=1)

    print(f'shape of inputs (training): {X_train.shape}, shape of outputs (training): {y_test.shape}')
    return X_train, X_test, y_train, y_test


"""
CASE 3: Sequence to sequence
"""


def sequence_to_sequence(name, n_batch: int = 50, n_states: int = 2, n_time: int = 3):

    # make sure to have at least 1 testing sample
    n_batch = np.max([n_batch, 2])

    if name == 'sine_pred':
        # multi-step predict a vector of sine signals
        X_train, X_test, y_train, y_test = sine_pred(n_batch=n_batch, n_states=n_states,
                                                     n_time_in=n_time, n_time_out=n_time)

    print(f'shape of inputs (training): {X_train.shape}, shape of outputs (training): {y_test.shape}')
    return X_train, X_test, y_train, y_test


def x_to_x(name, n_batch: int = 50, n_states_in: int = 2, n_states_out: int = 2, n_time_int: int = 1, n_time_out: int =
1):

    # make sure to have at least 1 testing sample
    n_batch = np.max([n_batch, 2])

    # full flexibility in creating input and output shapes
    n_states = np.max([n_states_in, n_states_out])

    if name == 'sine_pred':
        # single-step predict a vector of sine signals
        X_train, X_test, y_train, y_test = sine_pred(n_batch=n_batch, n_states=n_states, n_time_in=1, n_time_out=1)

    # cut data if input and output vector length is not the same
    X_train, X_test = X_train[:, :, :n_states_in], X_test[:, :, :n_states_in]
    y_train, y_test = y_train[:, :, :n_states_out], y_test[:, :, :n_states_out]

    print(f'shape of inputs (training): {X_train.shape}, shape of outputs (training): {y_test.shape}')

    return X_train, X_test, y_train, y_test


if __name__ == '__main__':

    # case 1

    # n_time = X_train.shape[1]
    # plt.figure()
    # for i in range(3):
    #     plt.plot(np.arange(start=1, stop=n_time), X_train[i,:,0], label='input')
    #
    # plt.legend()
    # plt.show()

    X_train, X_test, y_train, y_test = scalar_to_scalar(name='sine_pred', n_batch=50)

    X_train, X_test, y_train, y_test = vector_to_vector(name='sine_pred', n_batch=1, n_states=3)

    X_train, X_test, y_train, y_test = sequence_to_sequence(name='sine_pred', n_batch=50, n_states=4, n_time=15)

    X_train, X_test, y_train, y_test = x_to_x(name='sine_pred', n_batch=100, n_states_in=4, n_states_out=3,
                                              n_time_int=10, n_time_out=2)
