import numpy as np
import random
from abc import ABC, abstractmethod
import os
import sys
from typing import Union
import copy

from .layers import Layer, InputLayer, ReservoirLayer, ReadoutLayer
from .optimizers import Optimizer, assign_optimizer
from .metrics import assign_metric
from .utils_networks import gen_init_states, set_spec_rad, is_zero_col_and_row, remove_node, get_num_nodes
from .metrics import assign_metric, available_metrics


def sample_random_nodes(total_nodes: int, fraction: float):
    # select a subset of randomly chosen nodes
    return np.random.choice(total_nodes, size=int(total_nodes * fraction), replace=False)


from matplotlib import pyplot as plt


# the Model class will be the super-class (abstract base class, ABC) for non-autonomous, hybrid, and autonomous RC
# models

# #### <model> class methods
# - model.compile(optimizer, metrics, ...)
# - model.fit(X, y)
# - model.predict(X)
# - model.predict_ar()
# - model.evaluate(X, y)
# - model.save(path)


class CustomModel(ABC):

    def __init__(self):

        # the relevant layers
        self.input_layer: InputLayer
        self.reservoir_layer: ReservoirLayer
        self.readout_layer: ReadoutLayer

        self.metrics = []
        self.optimizer: Optimizer = assign_optimizer('ridge')  # the method for finding the readout weights. Default is
        # Ridge Regression

    def add(self, layer: Layer):
        # add some layer(s) to the model. Will not do more at this point in time

        # check for layer type
        if type(layer) == InputLayer:
            self.input_layer = layer
        elif issubclass(type(layer), ReservoirLayer):
            self.reservoir_layer = layer
        elif type(layer) == ReadoutLayer:
            self.readout_layer = layer

    def _set_readout_nodes(self, nodes: Union[list, np.ndarray] = None):
        # fix the nodes that will be linked to the output
        if nodes is None:
            nodes = sample_random_nodes(total_nodes=self.reservoir_layer.nodes,
                                        fraction=self.readout_layer.fraction_out)

        self.readout_layer.readout_nodes = nodes

    def _connect_input_to_reservoir(self, nodes: Union[list, np.ndarray] = None):
        # wire input layer with reservoir layer. Creates a random matrix of shape nodes x n_states, i.e. number of
        # reservoir nodes x state dimension of input
        input_size = self.input_layer.n_states
        n = self.reservoir_layer.nodes

        # generate random input connection matrix [nodes, n_states]
        self.input_layer.weights = np.random.randn(input_size, n)

        # select the read-in connections
        if (nodes is None) and (self.reservoir_layer.fraction_input < 1.0):
            input_nodes = sample_random_nodes(total_nodes=n, fraction=self.reservoir_layer.fraction_input)
        else:
            input_nodes = nodes

        mask = np.zeros_like(self.input_layer.weights)
        mask[:, input_nodes] = 1
        self.input_layer.weights = self.input_layer.weights * mask

    def _set_optimizer(self, optimizer: str):
        # maps strings of optimizer names to the actual Optimizer object
        self.solver = assign_optimizer(optimizer)

    def _set_metrics(self, metrics: Union[list, str]):
        # assigns names of metrics to a list of strings in the model instance
        if type(metrics) == str:  # only single metric given
            self.metrics = [metrics]

        self.metrics_fun = []  # the actual callable metric functions
        for metric in self.metrics:
            self.metrics_fun.append(assign_metric(metric))

    def _set_init_states(self, init_states=None, method=None):

        if init_states is not None:
            if init_states.shape[0] != self.reservoir_layer.nodes:
                raise (ValueError('initial states not matching the number of reservoir nodes!'))
            self.reservoir_layer.set_initial_state(r_init=init_states)
        elif (init_states is None) and (method is not None):
            init_states = gen_init_states(self.reservoir_layer.nodes, method)
            self.reservoir_layer.set_initial_state(r_init=init_states)
        else:
            raise (ValueError('provide either an array of initial states or a method for sampling those'))

    def compile(self, optimizer: str = 'ridge', metrics: list = ['mse']):

        # set the metrics (like in TensorFlow)
        self._set_metrics(metrics)

        # set the optimizer that will find the readout weights
        self._set_optimizer(optimizer)

        # 1. check consistency of layers, data shapes etc.
        # TODO: do we have input, reservoir and readout layer?
        # TODO: are all shapes correct on input and output side?

        # 2. Sample the input connections: create W_in read-in weight matrix
        self._connect_input_to_reservoir()  # check for dependency injection here!

        # 3. Select readout nodes according to the fraction specified by the user in the readout layer
        self._set_readout_nodes()

        # 4. set reservoir initialization

    def compute_reservoir_state(self, X: np.ndarray, seed=None) -> np.ndarray:
        # expects an input of shape [n_batch, n_timesteps, n_states]
        # returns the reservoir states of shape [(n_batch * n_timesteps), N]

        # (except for .predict)! Don't let the prediction on a sample depend on which
        # sample you computed before.

        # get data shapes
        n_samples = X.shape[0]
        n_time = X.shape[1]
        n_states = X.shape[2]

        # extract layer properties for easier syntax below:
        N = self.reservoir_layer.nodes
        g = self.reservoir_layer.activation_fun
        alpha = self.reservoir_layer.leakage_rate
        A = self.reservoir_layer.weights
        W_in = self.input_layer.weights
        R0 = self.reservoir_layer.initial_res_states

        # now loop over all samples, compute reservoir states for each sample, and re-initialize reservoir (in case
        # one_shot parameter = False

        R_all = []  # collects reservoir states [n_sample, [n_nodes, n_time]]
        for sample in range(n_samples):  # loop over training samples, the batch
            # print(f'{sample}/{n_samples} ...')

            R = np.zeros([N, n_time + 1])
            # if one_shot and sample>0:   # re-use last reservoir state from previous sample
            #     R0 = R_all[-1][:,-1]

            R[:, 0] = R0

            for t, x in enumerate(X[sample, :, :]):  # go through time steps (X[1]) for current training sample
                R[:, t + 1] = (1 - alpha) * R[:, t] + alpha * g(np.dot(A, R[:, t].T) + np.dot(W_in.T, x))

            R_all.append(R[:, 1:])  # get rid of initial reservoir state

        # concatenate all reservoir states
        R_all = np.hstack(R_all).T

        return R_all  # all reservoir states: [n_nodes, (n_batch * n_time)]

    def fit(self, X: np.ndarray, y: np.ndarray, one_shot: bool = False, n_init: int = 1, store_states: bool = False):
        # val_data: dict = None):

        # expects data in particular format that is reasonable for univariate/multivariate time series data
        # - X input data of shape [n_batch, n_time_in, n_states_in]
        # - y target data of shape [n_batch, n_time_out, n_states_out]
        # - n_init: number of times that initial reservoir states are sampled.
        # - store_states returns the full time trace of reservoir states (memory-heavy!)

        n_batch = X.shape[0]
        n_time = X.shape[1]
        n_states_out = y.shape[-1]

        # one_shot = True will *not* re-initialize the reservoir from sample to sample. Introduces a dependency on the
        # sequence by which the samples are given

        # train the RC to the given data. Will also need to check for consistency of everything
        # returns some values that describe how well the training went
        # if it's an evolutionary RC network, this will be the growth stage

        # loop across many different initial conditions for the reservoir states R(t=0) --> n_init
        n_R0, n_weights, n_scores, n_res_states = [], [], [], []
        for i in range(n_init):
            print(f'initialization {i}/{n_init}: computing reservoir states')

            # set the initial reservoir state (should involve some randomness if n_init > 1)
            # TODO: call a pre-defined initializer according to some input (0, random normal, uniform)
            self._set_init_states(method=self.reservoir_layer.init_res_sampling)

            # feed training data through RC and obtain reservoir states
            reservoir_states = self.compute_reservoir_state(X)  # complete n states

            # set up the linear regression problem Ax=b, A=R, b=y, x=W_out
            # mask reservoir states that are not selected as output nodes
            # TODO: make this more efficient, do not create full mask array
            mask = np.zeros_like(reservoir_states)
            mask[:, self.readout_layer.readout_nodes] = 1
            A = reservoir_states * mask

            # reshape targets y [n_batch, n_time, n_out] to [(n_batch * n_time), n_out],
            # i.e. stack all targets into long vector along time dimension
            b = y.reshape(n_batch * n_time, n_states_out)

            # 2. solve regression problem and update readout matrix
            # R * W_out = y --> [n_nodes, (n_batch * n_time)].T * W_out = [(n_batch * n_time), n_out]
            self.readout_layer.weights = self.optimizer.solve(A=A, b=b)

            # 3. compute score on training set (required to find the best initial reservoir state)
            # TODO: replace the default L1 with the first metric chosen by the user
            score = np.mean(np.abs(y - self.predict(X=X)))

            # store intermediate results for the n_init loop
            n_R0.append(self.reservoir_layer.initial_res_states)
            n_weights.append(self.readout_layer.weights)
            n_scores.append(score)

            if store_states:
                n_res_states.append(reservoir_states)

        # select the best prediction, i.e. the best initial condition
        if n_init > 1:
            idx_optimal = np.argmin(n_scores)
        else:
            idx_optimal = 0
        self.reservoir_layer.set_initial_state(n_R0[idx_optimal])
        self.readout_layer.weights = n_weights[idx_optimal]

        # built a history object to store detailed information about the training process
        history = dict()
        history['init_res_states'] = n_R0
        history['readout_weights'] = n_weights
        history['train_scores'] = n_scores

        if store_states:
            history['res_states'] = n_res_states
        # uncertain if we should store the reservoir states by default, will be a large memory consumption

        return history


    def fit_evolve(self, X: np.ndarray, y: np.ndarray):
        # build an evolving reservoir computer: performance-dependent node addition and removal

        history = None
        return history


    def fit_prune(self, X: np.ndarray, y: np.ndarray, loss_metric='mse', max_perf_drop=0.1):
        # build a reservoir computer by performance-informed pruning the initial reservoir network down to better
        # performance OR a tolerated performance reduction.

        # expects data in particular format that is reasonable for univariate/multivariate time series data
        # - X input data of shape [n_batch, n_time_in, n_states_in]
        # - y target data of shape [n_batch, n_time_out, n_states_out]
        # - loss_metric: the metric based on which the performance-informed pruning is performed. Must be a member of
        # the existing metrics of respy
        # - max_perf_drop: percentage of by how much the performance may drop before we stop pruning. default: 0.1 (10%)

        # data set shape
        n_batch = X.shape[0]
        n_time = X.shape[1]
        n_states_out = y.shape[-1]

        loss_fun = assign_metric(loss_metric)  # get a callable loss function for performance-informed node removal

        self._set_init_states(method=self.reservoir_layer.init_res_sampling)

        # get size of original reservoir
        num_nodes = self.reservoir_layer.weights.shape[0]

        # 0. obtain performance of the initial reservoir without having pruned a single node
        # TODO: all of the following is a copy-paste from model.fit(), hence we should remove all the duplicates and
        #  simplify the code

        def train_reservoir(X, y):
            reservoir_states = self.compute_reservoir_state(X)

            mask = np.zeros_like(reservoir_states)
            mask[:, self.readout_layer.readout_nodes] = 1
            A = reservoir_states * mask

            self.readout_layer.weights = self.optimizer.solve(A=A, b=y.reshape(n_batch * n_time, n_states_out))

        # compute score of full network on training set
        train_reservoir(X, y)
        init_score = loss_fun(y, self.predict(X=X))

        # 1. define a stopping criterion which will stop removing nodes.
        def keep_pruning(init_score, current_score, max_perf_drop):
            if current_score < (init_score * (1.0 + max_perf_drop)):  # we accept a performance drop of 30%
                return True
            else:
                print('pruning stopping criterium reached.')
                return False



        # 2. loop: identify the node which causes the least performance drop, remove it from the network, and obtain
        # the loss score of the new network (train, predict, evaluate)
        i = 0
        current_score = init_score
        current_num_nodes = get_num_nodes(self.reservoir_layer.weights)
        score_per_node = []
        history = dict()
        history['pruned_nodes'] = [-1]
        history['pruned_nodes_scores'] = [init_score]
        history['num_nodes'] = [current_num_nodes]
        while keep_pruning(init_score, current_score, max_perf_drop) and (i<num_nodes):
            print(f'pruning iteration {i}')

            score_per_node.append([])
            max_loss = init_score

            # we have to reset the temporarily deleted nodes after the following loop
            orig_weights = copy.deepcopy(self.reservoir_layer.weights)
            orig_initial_res_states = copy.deepcopy(self.reservoir_layer.initial_res_states)

            for del_idx in range(num_nodes):

                # TODO: we could parallelize this loop!
                if not is_zero_col_and_row(self.reservoir_layer.weights, del_idx):

                    # 2.a: temporarily  remove node i from adjacency matrix and from reservoir initial state matrix
                    # TODO: is there a better way to mask certain columns and rows?
                    self.reservoir_layer.weights = remove_node(self.reservoir_layer.weights, del_idx)
                    self.reservoir_layer.initial_res_states = remove_node(self.reservoir_layer.initial_res_states,
                                                                          del_idx)

                    # 2.b: (TODO Manish) remove potentially isolated and  non-input/output-connected communities)

                    # 2.c: re-scale to desired spectral radius. TODO: we have issues, sometimes the radius gets down
                    #  to zero!
                    # self.reservoir_layer.weights = set_spec_rad(self.reservoir_layer.weights,
                    #                                             spec_radius=self.reservoir_layer.spec_rad)

                    # 2.d: train, predict, evaluate, save score
                    train_reservoir(X, y)
                    score_per_node[i].append(loss_fun(y, self.predict(X=X)))

                    print(
                        f'pruning node {del_idx} / {current_num_nodes}: loss = {score_per_node[i][-1]:.5f}, original loss = {init_score:.5f}')



                    max_loss = np.max([max_loss, score_per_node[i][-1]])

                else:  # this node has already been removed. should not be considered further
                    print(f'node {del_idx} is not existent in the adjacency matrix')
                    score_per_node[i].append(None)

                # 2.e: reset original row and col values in adjacency matrix and initial reservoir states
                # TODO: somehow we arrive at zero weights here, don't know where they get lost!!!
                self.reservoir_layer.weights[:, del_idx] = orig_weights[:, del_idx]
                self.reservoir_layer.weights[del_idx, :] = orig_weights[del_idx, :]
                self.reservoir_layer.initial_res_states[del_idx] = orig_initial_res_states[del_idx]



            # now find the node which affects the loss the least, i.e. obtains the minimal loss (could even be
            # smaller than the one of the original model. First replace all None values by a loss that is larger than
            # the rest
            max_loss = max_loss + 1
            score_per_node[i] = [max_loss if x is None else x for x in score_per_node[i]]
            idx_del_node = np.argmin(score_per_node[i])
            current_score = score_per_node[i][idx_del_node]
            rel_score = (current_score-init_score)/init_score*100

            # if that score is still ok, we will remove nodes and keep going. otherwise stop here
            if keep_pruning(init_score, current_score, max_perf_drop):
                # delete this node permanently from adjacency matrix and initial reservoir states
                self.reservoir_layer.weights = remove_node(self.reservoir_layer.weights, idx_del_node)
                self.reservoir_layer.initial_res_states = remove_node(self.reservoir_layer.initial_res_states,
                                                                      idx_del_node)

                current_num_nodes = get_num_nodes(self.reservoir_layer.weights)

                print(f'removing node {idx_del_node}: new loss = {current_score:.5f}, original loss = {init_score:.5f}'
                      f' ({rel_score:+.2f} %); {current_num_nodes} nodes remain')

                  # 3. store some of the pruning history to the history object returned to the user
                history['pruned_nodes'].append(idx_del_node)
                history['pruned_nodes_scores'].append(score_per_node[i][idx_del_node])
                history['num_nodes'].append(current_num_nodes)

            i += 1

        return history

    # @abstractmethod
    def predict(self, X: np.ndarray, one_shot: bool = False) -> np.ndarray:
        # makes prediction for given input (single-step prediction)
        # expects inputs of shape [n_batch, n_timestep, n_states]
        # returns predictions in shape of [n_batch, n_timestep, n_states]

        # one_shot = True will *not* re-initialize the reservoir from sample to sample. Introduces a dependency on the
        # sequence by which the samples are given

        # TODO Merten: return some random number that have the correct shape

        # TODO: external function that is going to check the dimensionality
        # and raise an error if shape is not correct
        n_batch, n_timesteps, n_states = X.shape[0], X.shape[1], X.shape[2]

        # iterate over batch to obtain predictions
        reservoir_states = self.compute_reservoir_state(X)

        # make predictions y = R * W_out, W_out has a shape of [n_out, N]
        y_pred = np.dot(reservoir_states, self.readout_layer.weights)

        # reshape predictions into 3D [n_batch, n_time_out, n_state_out]
        n_time_out = int(y_pred.shape[0] / n_batch)
        n_states_out = y_pred.shape[-1]
        y_pred = y_pred.reshape(n_batch, n_time_out, n_states_out)

        return y_pred

    # @abstractmethod
    def evaluate(self, X: np.ndarray, y: np.ndarray,
                 metrics: Union[str, list, None] = None) -> tuple:
        # evaluate metrics on predictions made for input data
          # expects: X of shape [n_batch, n_timesteps, n_states]
        # expects: y of shape [n_batch, n_timesteps_out, n_states_out]
        # depends on self.metrics = metrics from .compile()
        # returns float, if multiple metrics, then in given order (TODO: implement this)

        if metrics is None:  # user did not specify metric, take the one(s) given to .compile()
            metrics = self.metrics
        if type(metrics) is str:  # make sure that we are working with lists of strings
            metrics = [metrics]

        # self.metrics_available = ['mse', 'mae']
        #
        # eval_metrics = self.metrics + metrics  # combine from .compile and user specified
        # eval_metrics = list(set(eval_metrics))  # removes potential duplicates

        # get metric function handle from the list of metrics specified as str
        metric_funs = [assign_metric(m) for m in metrics]

        # make predictions
        y_pred = self.predict(X)

        # get metric values
        metric_values = []
        for _metric_fun in metric_funs:
            metric_values.append(float(_metric_fun(y, y_pred)))

        return metric_values

    # @abstractmethod
    def get_params(self, deep=True):
        # needed for scikit-learn compatibility
        return {
            'input_layer': self.input_layer,
            'reservoir_layer': self.reservoir_layer,
            'readout_layer': self.readout_layer
        }

    # @abstractmethod
    def save(self, path: str):
        # store the model to disk
        pass

    def plot(self, path: str):
        # print the model to some figure file
        pass


class AutoRC(CustomModel):

    def __int__(self):
        pass

    def predict_ar(self, X: np.ndarray, n_steps: int = 10):
        # auto-regressive prediction -> time series forecasting
        pass


class HybridRC(CustomModel):
    def __int__(self):
        pass


class RC(CustomModel):  # the non-auto version
    def __int__(self):
        pass

    def get_params(self, deep=True):
        # needed for scikit-learn compatibility 
        return {}


if __name__ == '__main__':
    print('hello')
