#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 15 11:39:52 2022

@author: gitkatrin, input from: mrdbourke
"""
# Introduction to Regression with Neural Networks in Tensorflow

# There are many definitions for regession problem but in our case, we're goint to simplify it:
#   predicitiong a numerical variable based in sime ither combination of variables, even shorter ...
#   predicition a number.

import tensorflow as tf
print(tf.__version__)

# Creating data to view and fit -----------------------------------------------
print('\033[1m' + 'Creating data to view and fit' + '\033[0m')
import matplotlib.pyplot as plt
import numpy as np

# Create features
X = np.array([-7.0, -4.0, -1.0, 2.0, 5.0, 8.0, 11.0, 14.0])

# Create labels
y = np.array([3.0, 6.0, 9.0, 12.0, 15.0, 18.0, 21.0, 24.0])

# Visualize it
#plt.scatter(X,y);

print(y == X+10) # X: input features, Y: output

# Input and Output shapes -----------------------------------------------------
print()
print('\033[1m' + 'Input and Output shapes' + '\033[0m')
# Create a demo tensor for our housting price prediciton problem
house_info = tf.constant(["bedroom", "bathroom", "garage"])
house_price = tf.constant([939700])

print(house_info, house_price)

print(X[0], y[0])
print(X[1], y[1])

input_shape = X[0].shape
output_shape = y[0].shape
print(input_shape, output_shape)

# Turn Numpy arrays into tensors
# X = tf.constant(X)
# y = tf.constant(y)
X = tf.cast(tf.constant(X), dtype=tf.float32)
y = tf.cast(tf.constant(y), dtype=tf.float32)


print(X,y)
input_shape = X[0].shape
output_shape = y[0].shape
print(input_shape, output_shape)


# Steps with modeling in tensorflow ===========================================
print()
print('\033[1m' + 'Steps with modeling in tensorflow' + '\033[0m')
# 1. Creating a model - define the input and output layers, as well as the hiddenlayers of a deep learning model.
# 2. Compiling a model - define the loss function (in other words, the function which tells our model how wrong it is) and
#    the optimizer (tells our model how to improve the patterns its learning) and evaluation metrics
#    (what we can use to interpret the performance of our model).
# 3. Fitting a model - letting the model try to find patterns between X &y (features and labels)



# Set random seed
tf.random.set_seed(42)

# 1. Create a model using the Sequential API
# List Version:
# model = tf.keras.Sequential([
#     tf.keras.layers.Dense(1)
#     ])
# Add Version:
# model = tf.keras.Sequential()
# model.add(tf.keras.layers.Dense(1))
# model.add(tf.keras.layers.Dense(1))
    
# 2. Compile the model
# model.compile(loss=tf.keras.losses.mae, # mae is short for mean absolute error
#              optimizer=tf.keras.optimizers.SGD(), # SGD is short for stichastic gradient decent
#              metrics=["mae"])

# 3. Fit the model
# model.fit(tf.expand_dims(X, axis=-1), y, epochs=5)

# print(X,y)

# # Try and make a prediction usion our model
# y_pred = model.predict([17.0])
# print(y_pred)

# print(y_pred+11)

# Improving a model ===========================================================
print()
print('\033[1m' + 'Improving a model' + '\033[0m')
# We can improve our model, by altering the steps we took to create a model
#   1. Creating a model: here we might add more layers,
#       increase the number of hidden units (all called neurons) within each of the hidden layers,
#       change the activation function of each layer.
#   2. Compinling a model: here we might change the optimization function or
#       perhaps the learning rate of the optimization function.
#   3. Fitting a model: here we might fit a model for more epochs (leave it training for longer)
#       or on more data (give the model more examples to learn from).

# Let's rebuild our model

# # 1. Create the model
# model = tf.keras.Sequential([
#     tf.keras.layers.Dense(50, activation=None),
#     tf.keras.layers.Dense(1)
#     ])

# # 2. Compile the model
# model.compile(loss=tf.keras.losses.mae, # mae is short for mean absolute error
#              optimizer=tf.keras.optimizers.Adam(learning_rate=0.01), # SGD is short for stichastic gradient decent
#              metrics=["mae"])

# # 3. Fit the model
# model.fit(tf.expand_dims(X, axis=-1), y, epochs=100)

# # Let's see if our model's prediction has improved
# print(model.predict([17.0]))


# Evaluating a model ==========================================================
print()
print('\033[1m' + 'Evaluating a model' + '\033[0m')
# In practice, a typical workflow you'll go through when building neural networls is:
# Build a model -> fit it -> evaluate it ->  tweak a model -> fit it -> evaluate it -> tweak a model -> ...

# It's a good idea to visualize:
#   - The data - what data are we working with? What does it look like?
#   - The model itself: what does our model look like?
#   - The training of a model - how does a model perform while it learns?
#   - The predicitions of the model - how do the predictions of a model line up against the ground truth (the orginal labels)?

# Make a bigger dataset
X = tf.range(-100,100,4)
print(X)

# Make labels for the dataset
y = X + 10
print(y)

# Visualize the data
import matplotlib.pyplot as plt
# plt.scatter(X,y)

# The 3 sets
# Training set - the model learns from this data, which is typically 70-80% of the total data you have available.
# Validation set - the model gets tuned on this data, which is typically 10-15% of the data available.
# Test set - the model gets evaluated on this data to test what is has learned, this set is typically 10-15% of the total available.

# Chech the lenght of how many samples we have
print(len(X)) #50

# Split the data into train and test sets
X_train = X[:40] # first 40 are training samples (80% of the data) 
y_train = y[:40]

X_test = X[40:] # last 10 are ttraining samples (20% of the data)
y_test = y[40:]

print(len(X_train), len(X_test), len(y_train), len(y_test))

# # Visualizing the data
# plt.figure(figsize=(10,7))
# #Plot training data in blue
# plt.scatter(X_train, y_train, c="b", label="Training data")
# # Plot test data green
# plt.scatter(X_test, y_test, c="g", label="Testing data")
# # Show a legend
# plt.legend();

# Let's have a look at how to build a neural networ for our data

# 1. Create the model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(1)
    ])

# 2. Compile the model
model.compile(loss=tf.keras.losses.mae, # mae is short for mean absolute error
             optimizer=tf.keras.optimizers.Adam(learning_rate=0.01), # SGD is short for stichastic gradient decent
             metrics=["mae"])

# 3. Fit the model
#model.fit(X_train, y_train, epochs=100)

# Visualizing the model
#model.summary() # Error

# Let's create a model which builds automatically by defining the input_chape argument in the fist layer
tf.random.set_seed(42)
# 1. Create the same model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(10, input_shape=[1], name='input_layer'),
    tf.keras.layers.Dense(1, name='output_layer')
    ], name='model_X')

# 2. Compile the model
model.compile(loss=tf.keras.losses.mae, # mae is short for mean absolute error
             optimizer=tf.keras.optimizers.SGD(), # SGD is short for stichastic gradient decent
             metrics=["mae"])
model.summary()

# Total params - total number of parameters in the model.
# Trainable params - these are the parameters (patterns) the model can update as it trains.
# Non-trainable params - these parameters aren't updated during training
#   (this is typical when you bring in already learn patterns or parameters from each other models during transfer learning)

# 3. Let's fit our model to the training data
model.fit(X_train, y_train, epochs=100, verbose=0)

from tensorflow.keras.utils import plot_model

plot_model(model=model, show_shapes=True)
#plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

# Visualizing our model's predections -----------------------------------------
print()
print('\033[1m' + "Visualizing our model's predection" + '\033[0m')

# To visualize predictions, it's a good idea to plot them against the gound truth labels.
# Often you'll see this in the form of y_test or y_true versus y_pred (ground truth versus your model's predictions).

# Make some predictions
y_preds = model.predict(X_test)
print(y_preds)
print(y_test)

# Let's create a plotting function
# def plot_predictions(train_data = X_train,
#                      train_labels = y_train,
#                      test_data = X_test,
#                      test_labels = y_test,
#                      predictions = y_pred):
    
#     # Plots training data, test data and compares predictions to ground truth labels
#     plt.figure(figsize=(10,7))
#     # Plot training data in blue
#     plt.scatter(train_data, train_labels, c="b", label="Training data")   
#     # Plot testing data in green
#     plt.scatter(test_data, test_labels, c="g", label="Testing data")
#     # Plot model's predictions in red
#     plt.scatter(test_data, y_pred, c="r", label="Predictions")
#     # Show the legend
#     plt.legend();
    
def plot_predictions(train_data=X_train, 
                     train_labels=y_train, 
                     test_data=X_test, 
                     test_labels=y_test, 
                     predictions=y_preds):

    #Plots training data, test data and compares predictions.
    plt.figure(figsize=(10, 7))
    # Plot training data in blue
    plt.scatter(train_data, train_labels, c="b", label="Training data")
    # Plot test data in green
    plt.scatter(test_data, test_labels, c="g", label="Testing data")
    # Plot the predictions in red (predictions were made on the test data)
    plt.scatter(test_data, predictions, c="r", label="Predictions")
    # Show the legend
    plt.legend();

plot_predictions(train_data=X_train,
                  train_labels=y_train,
                  test_data=X_test,
                  test_labels=y_test,
                  predictions=y_preds)

# Evaluating our model's predictions with regression evaluation metrics -------
print()
print('\033[1m' + "Evaluating our model's predictions with regression evaluation metrics" + '\033[0m')
# Depending on the problem you're working on, there will be different
# evaluation metrics to evaluate your model's performance.
# Since we're working on a regression, two of the main metrics:
#   1. MAE - mean absolute error, "on average, how wrong is each of my model's prediction
#   2. MSE - mean square error, "square the average errors"

# Evaluate the model on the test
model.evaluate(X_test, y_test)

# Calculate the mean absoulute error
# print(tf.keras.losses.MAE(y_test, y_pred))
mae = tf.metrics.mean_absolute_error(y_true=y_test,
                                     y_pred=tf.squeeze(y_preds))
print(mae)

# Calculate the mean square error
mse = tf.metrics.mean_squared_error(y_true=y_test,
                                   y_pred=tf.squeeze(y_preds))
print(mse)

# Made some functions to reuse MAE and MSE
def mae(y_true, y_pred):
    return tf.metrics.mean_absolute_error(y_true=y_true,
                                          y_pred=tf.squeeze(y_pred))
def mse(y_true, y_pred):
    return tf.metrics.mean_squared_error(y_true=y_true,
                                         y_pred=tf.squeeze(y_pred))

# Running experiments to improve our model ====================================
print()
print('\033[1m' + "Running experiments to improve our model" + '\033[0m')
# Build a mode -> fit it -> evaluate it -> tweak it -> fit it -> evaluate it -> tweak it -> ...

# 1. Get more data - get more examples for your model to train on (more opportunities
#   to learn patterns or relationships between features and labels)
# 2. Make your model larger (usning more complex model) - this might come in the form of more layers
#   or more hidde units in each layer.
# 3. Train for longer - give your model more of a chance to find patterns in the data.

# Let's do 3 modelling experiments:
#   1. Model 1: same as the original model, 1 layer, trained for 100 epochs
#   2. Model 2: 2 layers, trained for 100 epochs
#   3. Model 3: 2 layers, trained for 500 epochs

# Build model_1 ---------------------------------------------------------------
print()
print('\033[1m' + "Build model_1" + '\033[0m')
# Set random seed
tf.random.set_seed(42)
print(X_train, y_train)
# 1. Create the model
model_1 = tf.keras.Sequential([
    tf.keras.layers.Dense(1)
    ])

# 2. Compile the model
model_1.compile(loss=tf.keras.losses.mae,
                optimizer=tf.keras.optimizers.SGD(),
                metrics=["mae"])

# 3. Fit the model
model_1.fit(tf.expand_dims(X_train, axis =-1), y_train, epochs=100, verbose=0)


# print(X_train)
# print(tf.expand_dims(X_train, axis =-1))

# Make and plot predictions for model_1
y_preds_1 = model_1.predict(X_test)
print(y_preds_1)
print(y_test)
plot_predictions(predictions=y_preds_1)


# Calculate model_1 evaluation metrics
mae_1 = mae(y_test, y_preds_1) # 18.745327
mse_1 = mse(y_test, y_preds_1) # 353.5734
print("Model_1: ", mae_1, mse_1)

# Build model_2 ---------------------------------------------------------------
print()
print('\033[1m' + "Build model_2" + '\033[0m')
tf.random.set_seed(42)

# 1. Create the model
model_2 = tf.keras.Sequential([
    tf.keras.layers.Dense(10),
    tf.keras.layers.Dense(1)
    ])

# 2. Compile the model
model_2.compile(loss=tf.keras.losses.mae,
                optimizer=tf.keras.optimizers.SGD(),
                metrics=["mse"])

# 3. Fit the model
model_2.fit(tf.expand_dims(X_train, axis=-1), y_train, epochs=100, verbose=0)

# Make and plot prediction for model_2
y_preds_2 = model_2.predict((X_test))
plot_predictions(predictions=y_preds_2)
print(y_preds_2)
# Calculate model_2 evaluation metrics
mae_2 = mae(y_test, y_preds_2) # 3.1969643
mse_2 = mse(y_test, y_preds_2) # 13.0703
print("Model_2: ", mae_2, mse_2)

# Build model_3 ---------------------------------------------------------------
print()
print('\033[1m' + "Build model_2" + '\033[0m')
tf.random.set_seed(42)

# 1. Create the model
model_3 = tf.keras.Sequential([
    tf.keras.layers.Dense(10),
    tf.keras.layers.Dense(1)
    ])

# 2. Compile the model
model_3.compile(loss=tf.keras.losses.mae,
                optimizer=tf.keras.optimizers.SGD(),
                metrics=["mae"])

# 3. Fit the model
model_3.fit(tf.expand_dims(X_train, axis=-1), y_train, epochs=500, verbose=0)

# Make and plot some predictions
y_preds_3 = model_3.predict(X_test)
plot_predictions(predictions=y_preds_3)

# Calculate model_3 evaluation metrics
mae_3 = mae(y_test, y_preds_3) # 68.71362
mse_3 = mse(y_test, y_preds_3) # 4808.0283
print("Model_3: ", mae_3, mse_3)

# Note: You want to start with small experimets (small models) and make sure they work
#       and inrease their scale when necessary.

# Comparing the results of our experiments ------------------------------------
print()
print('\033[1m' + "Comparing the results of our experiments" + '\033[0m')
# Let's compare our model's results using pandas DataFrame
import pandas as pd

model_results = [["model_1", mae_1.numpy(), mse_1.numpy()],
                 ["model_2", mae_2.numpy(), mse_2.numpy()],
                 ["model_3", mae_3.numpy(), mse_3.numpy()]]

all_results = pd.DataFrame(model_results, columns=["model", "mae", "mse"])
print(all_results) # Looks like model_2 performed the best

# Note: One of your main goals should be to minimize the time between your experiments.
#       The more experiments you do, the more things you'll figure out which dont't work
#       and in turn, get closer to figuring out what does work.

# Tracking your experiments ---------------------------------------------------
# One really good habit in machine learning modelling is to track the results of your experiments.
# And when doing so, it can be tedious if yo're running lots of experiments.
# Luckily, there are tools to help us!
# 1. TensorBoard: a component of TensorFlow library to help track modelling experiments
# 2. Weights & Bias: a tool for tracking all of kinds of machine learning experiments (plugs straight into TensorBoard)

# Saving our models ===========================================================
print()
print('\033[1m' + "Saving our models" + '\033[0m')
# Saving our models allows us to use them outside of Spyder (or wherever they were trained)
# such as in a web application or a mobile app
# There are two main formats we can save our model's too:
#   1. The SavedModel format
#   2. The HDF5 format

# Save model using the SavedModel format
model_2.save("best_model_SavedModel_format")

# Save model using the HDF5 format
model_2.save("best_model_HDF5_format.h5")

# Loading in a saved model ====================================================
print()
print('\033[1m' + "Loading in a saved model" + '\033[0m')

# Load in the SavedModel format model
loaded_SavedModel_format = tf.keras.models.load_model("./best_model_SavedModel_format")
loaded_SavedModel_format.summary()
model_2.summary()

print('\033[1m' + "SavedModel format" + '\033[0m')
# Compare model_2 predictions with SavedModel format model predicitions
model_2_preds = model_2.predict(X_test)
loaded_SavedModel_format_preds = loaded_SavedModel_format.predict(X_test)
print(model_2_preds == loaded_SavedModel_format_preds)
print(model_2_preds, loaded_SavedModel_format_preds)

# Compare the MAE of model_2 preds and loaded_SavedModel_preds
print(mae(y_true=y_test, y_pred=model_2_preds) == mae(y_true=y_test, y_pred=loaded_SavedModel_format_preds))

print('\033[1m' + ".h5 format" + '\033[0m')
# Load in a model using the .h5 format
loaded_h5_model = tf.keras.models.load_model("./best_model_HDF5_format.h5")
loaded_h5_model.summary()
model_2.summary()

# Check to see if loaded .h5 model predictions match model_2
model_2_preds = model_2.predict(X_test)
loaded_h5_model_preds = loaded_h5_model.predict(X_test)
print(model_2_preds == loaded_h5_model_preds)

# A larger example ============================================================
#50




