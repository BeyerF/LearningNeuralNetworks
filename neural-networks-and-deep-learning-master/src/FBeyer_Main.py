# -*- coding: utf-8 -*-
"""
Created on Wed May 23 12:03:12 2018

@author: beyer
"""

import mnist_loader
import network



# Initalize data sets
training_data = False
if not training_data:
    training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
else:
    print('Training data found.')





#-----------------------------------------------
# Initalize hyperparameters:
epochs = 30
batch_size = 10
learning_rate = 3.0

# Initalize network parameters:
num_inputs = 784
num_outputs = 10
num_neurons_layer_one = 20
num_neurons_layer_two = 10

net = network.Network([num_inputs, num_neurons_layer_one, num_neurons_layer_two, num_outputs])
net.SGD(training_data, epochs, batch_size, learning_rate, test_data=test_data)