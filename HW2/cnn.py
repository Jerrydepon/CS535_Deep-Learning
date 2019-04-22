"""
Chih-Hsiang, Wang
"""

from __future__ import division
from __future__ import print_function

try:
    import _pickle as pickle
except:
    import pickle  # What pickle does is that it “serialises” the object first before writing it to file.

import numpy as np
import matplotlib.pyplot as plt
from sympy import *
import time
import random
import sys


# from scipy.special import xlogy


# This is a class for a LinearTransform layer which takes an input
# weight matrix W and computes W x as the forward step
class LinearTransform(object):

    def __init__(self, W, B):
        self.w = W
        self.b = B

    def forward(self, x):
        return x @ self.w + self.b

    @staticmethod
    def backward(x):
        return x


# This is a class for a ReLU layer max(x,0)
class ReLU(object):

    @staticmethod
    def forward(x):
        x[x < 0] = 0
        return x

    @staticmethod
    def backward(x):
        x[x > 0] = 1
        x[x < 0] = 0
        x[x == 0] = random.uniform(0, 1)
        return x


# This is a class for a sigmoid layer followed by a cross entropy layer, the reason
# this is put into a single layer is because it has a simple gradient form
class SigmoidCrossEntropy(object):

    @staticmethod
    def forward(x, y):
        for idx, xi in enumerate(x):
            if xi >= 0:
                x[idx] = 1 / (1 + np.exp(-xi))
            else:
                x[idx] = np.exp(xi) / (1 + np.exp(xi))
        z3 = x
        z3[z3 <= 1e-9] += 1e-9  # to avoid log0
        z3[z3 >= 1 - 1e-9] -= 1e-9  # to avoid log0
        # entropy = -(np.multiply(y, np.log(z3)) + np.multiply((1 - y), np.log(1 - z3)))
        entropy = -(y * np.log(z3) + (1-y) * np.log(1-z3))
        # entropy = -(xlogy(y, z3) + xlogy(1 - y, 1 - z3))
        avg_loss = entropy.sum() / y.shape[0]
        return avg_loss, z3

    @staticmethod
    def backward(z3, y):
        return z3 - y  # ( (z3 - y) / (z3 * (1 - z3)) ) * z3 * (1 - z3)


# This is a class for the Multilayer perceptron
class MLP(object):

    def __init__(self, input_x, hidden, weight1, bias1, weight2, bias2, size_batch, l_rate, m, l2, d):
        self.x = input_x
        self.h = hidden
        self.w1 = weight1
        self.b1 = np.tile(bias1, (size_batch, 1)).astype('float64')
        self.w2 = weight2
        self.b2 = np.tile(bias2, (size_batch, 1)).astype('float64')

        self.l_rate = l_rate
        self.m = m
        self.l2 = l2
        self.d = d

        self.z1 = None
        self.z2 = None
        self.z3 = None
        self.act = None
        self.xb = None
        self.yb = None

    def forward(self, xb, yb):
        self.xb = xb
        self.yb = yb
        self.z1 = LinearTransform(self.w1, self.b1).forward(xb)
        self.act = ReLU().forward(self.z1)
        self.z2 = LinearTransform(self.w2, self.b2).forward(self.act)

        entropy_loss, self.z3 = SigmoidCrossEntropy().forward(self.z2, yb)

        return entropy_loss, self.z3

    def backward(self):
        dL_z2 = SigmoidCrossEntropy().backward(self.z3, self.yb)
        dL_z1 = np.multiply(dL_z2 @ self.w2.T, ReLU().backward(self.z1))
        dL_w2 = LinearTransform(self.w2, self.b2).backward(self.act).T @ dL_z2
        dL_w1 = LinearTransform(self.w1, self.b1).backward(self.xb).T @ dL_z1
        dL_b2 = dL_z2
        dL_b1 = dL_z1

        # SGD with momentum: D_0 = 0, D_t+1 = uD_t - alpha*G_t, W_t+1 = W_t + D_t+1 & L2 regularization
        self.d[0] = self.m * self.d[0] - self.l_rate * dL_w2 + 2 * self.l2 * self.w2
        self.d[1] = self.m * self.d[1] - self.l_rate * dL_w1 + 2 * self.l2 * self.w1
        self.d[2] = self.m * self.d[2] - self.l_rate * dL_b2
        self.d[3] = self.m * self.d[3] - self.l_rate * dL_b1
        self.w2 += self.d[0]
        self.w1 += self.d[1]
        self.b2 += self.d[2]
        self.b1 += self.d[3]

    @staticmethod
    def evaluate(x, y):
        x[x >= 0.5] = 1
        x[x < 0.5] = 0
        return np.sum(x == y)


if __name__ == '__main__':

    # ================== Data Read ==================
    if sys.version_info[0] < 3:
        data = pickle.load(open('cifar_2class_py2.p', 'rb'))
    else:
        data = pickle.load(open('cifar_2class_py2.p', 'rb'), encoding='bytes')

    train_x = data[b'train_data']  # 10000 x 3072(32x32x3)
    train_y = data[b'train_labels']  # 10000 x 1, label 0 is an airplane, label 1 is a ship
    test_x = data[b'test_data']  # 2000 x 3072
    test_y = data[b'test_labels']  # 2000 x 1

    # ================== Variable ==================
    num_examples, input_dims = train_x.shape
    num_examples_test = test_x.shape[0]
    num_epochs = 50
    batch_size = 50
    learning_rate = 1e-3
    hidden_units = 50
    momentum = 0.6
    l2_penalty = 1e-5
    accuracy_list = []

    # choose the plot type
    plot_idx = 0  # 0: run once, 1: batch size, 2: learning rate, 3: hidden units (x-label variant)
    plot_title = ["test accuracy for normal one run",
                  "test accuracy with different number of batch size",
                  "test accuracy with different learning rate",
                  "test accuracy with different number of hidden units"]
    plot_x = ["None", "# of batch size", "learning rate", "# of hidden units"]

    normal = [0]
    batch_size_list = [5, 10, 50, 200, 400, 1000]
    learning_rate_list = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1]
    hidden_units_list = [10, 50, 100, 150, 200, 250]
    var_list = [normal, batch_size_list, learning_rate_list, hidden_units_list]

    plt.figure()
    plt.title(plot_title[plot_idx])
    plt.xlabel(plot_x[plot_idx])
    plt.ylabel("test accuracy (%)")

    # ================== Initialization ==================
    # initialize weight & bias
    weight_1 = 0.01 * np.random.randn(input_dims, hidden_units)
    bias_1 = 0.01 * np.random.randn(1, hidden_units)
    weight_2 = 0.01 * np.random.randn(hidden_units, 1)
    bias_2 = 0.01 * np.random.randn(1, 1)

    # normalization
    train_x = (train_x - train_x.mean()) / train_x.std()
    test_x = (test_x - test_x.mean()) / test_x.std()

    # ================== Main Process ==================
    print("\nStart:", time.asctime(time.localtime(time.time())))
    for i, v in enumerate(var_list[plot_idx]):  # 0: run once, 1: batch size, 2: learning rate, 3: hidden units
        if plot_idx != 0:
            print("{}: [{}] -- {}".format(i, plot_x[plot_idx], v))
        inertia_d = [0, 0, 0, 0]
        test_accuracy = 0
        if plot_idx == 1:
            batch_size = v
        elif plot_idx == 2:
            learning_rate = v
        elif plot_idx == 3:
            hidden_units = v

        # create new class mlp for each iteration
        mlp = MLP(input_dims, hidden_units, weight_1, bias_1, weight_2, bias_2, batch_size,
                  learning_rate, momentum, l2_penalty, inertia_d)

        for epoch in range(num_epochs):
            train_loss = 0.0
            test_loss = 0.0
            num_train_correct = 0
            num_test_correct = 0

            # training
            for j, b in enumerate(range(0, num_examples, batch_size)):
                x_batch = train_x[b: b + batch_size].astype('float128')
                y_batch = train_y[b: b + batch_size]

                loss, output = mlp.forward(x_batch, y_batch)
                mlp.backward()

                num_train_correct += int(mlp.evaluate(output, y_batch))
                train_loss += loss

                print('\r[Train: Epoch {}, mini-batch {}]  Avg.Loss = {:.3f}'.format(epoch, j, loss), end='')
                sys.stdout.flush()

            # testing
            for j, b in enumerate(range(0, num_examples_test, batch_size)):
                x_batch = test_x[b: b + batch_size].astype('float128')
                y_batch = test_y[b: b + batch_size]

                loss, output = mlp.forward(x_batch, y_batch)

                num_test_correct += int(mlp.evaluate(output, y_batch))
                test_loss += loss

                print('\r[Test: Epoch {}, mini-batch {}]  Avg.Loss = {:.3f}'.format(epoch, j, loss), end='')
                sys.stdout.flush()

            test_accuracy = 100 * num_test_correct / num_examples_test

            print("\r<Epoch {}> train_loss: {:.3f}  train_accuracy: {:.2f}".format(
                epoch, train_loss / (num_examples / batch_size), 100 * num_train_correct / num_examples))
            print("\r<Epoch {}> test_loss: {:.3f}  test_accuracy: {:.2f}".format(
                epoch, test_loss / (num_examples_test / batch_size), test_accuracy))
            sys.stdout.flush()

        accuracy_list.append(test_accuracy)
        del mlp
        print("\n")

    if plot_idx != 0:
        plt.plot(var_list[plot_idx], accuracy_list)
        plt.xscale('log')
        plt.show()
    print("Finished: ", time.asctime(time.localtime(time.time())))
