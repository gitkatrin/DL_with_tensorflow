#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 23 12:01:04 2022

@author: bhtcsuperuser
"""
# Introduction to neural network classification with Tensorflow
# In this script we're going to learn how to write neural networks for classification
# A classification problem is where you try to classify something as one thing or another
# A few types of classification problems:
#   - Binary classification
#   - Multiclass classification
#   - Multilabel classification

# Creating data to view and fit ===============================================
print('\033[1m' + "Creating data to view and fit" + '\033[0m')

from sklearn.datasets import make_circles

# Make 1000 examples
n_samples = 1000

# Create circles
X, y = make_circles(n_samples,
             noise=0.03,
             random_state=42)

# Check out features and labels
print(X)
print(y[:10])

# Our dara is a little hard to understand right now... let's visualize it!
import pandas as pd
circles = pd.DataFrame({"X0":X[:,0], "X1":X[:,1], "label":y})
print(circles)

print(circles["label"].value_counts())

# Visualize with a plot
import matplotlib.pyplot as plt
plt.scatter(X[:,0], X[:,1], c=y, cmap=plt.cm.RdYlGn)

# Input and Output shapes =====================================================
print()
print('\033[1m' + "Input and Output shapes" + '\033[0m')

print(X.shape, y.shape)

# How many samples we're working with
print(len(X), len(y))

# View the firsst example of features and labels
print(X[0], y[0])

# Steps in modelling ==========================================================
print()
print('\033[1m' + "Steps in modelling" + '\033[0m')

# The steps in modelling with TensorFlow are typically:
#   1. Create or import a model
#   2. Compile the model
#   3. Fit the model
#   4. Evaluate the model
#   5. Tweak
#   6. Evaluate ...

# Model_1 ---------------------------------------------------------------------
print('\033[1m' + "Model_1" + '\033[0m')
import tensorflow as tf
# Set the random seed
tf.random.set_seed(42)

# 1. Create the model using the Sequential API
model_1 = tf.keras.Sequential([
    tf.keras.layers.Dense(1)
    ])

# 2. Compile the model
model_1.compile(loss=tf.keras.losses.BinaryCrossentropy(),
                optimizer=tf.keras.optimizers.SGD(),
                metrics=["accuracy"])

# 3. Fit the model
#model_1.fit(X, y, epochs=5)

# Let's try and improve our model by training for longer
model_1.fit(X, y, epochs=200, verbose=0)
model_1.evaluate(X, y)

# Since we're working on a binary classification problem and our model is getting around ~50% accuracy,
# it's performing as if it's guessing.
# So let's step things up a notch and add an extra layer.

# Model_2 ---------------------------------------------------------------------
print('\033[1m' + "Model_2" + '\033[0m')
# Set the random seed
tf.random.set_seed(42)

# 1. Create the model
model_2 = tf.keras.Sequential([
    tf.keras.layers.Dense(1),
    tf.keras.layers.Dense(1)
    ])

# 2. Compile the model
model_2.compile(loss=tf.keras.losses.BinaryCrossentropy(),
                optimizer=tf.keras.optimizers.SGD(),
                metrics=["accuracy"])

# 3. Fit the model
model_2.fit(X, y, epochs=100, verbose=0)

# 4. Evaluate the model
model_2.evaluate(X, y)

# Improving our model =========================================================
print()
print('\033[1m' + "Improving our model" + '\033[0m')

# Let's look into our bag of tricks to see how we can improve our model.
print('\033[1m' + "Model_3" + '\033[0m')
# Set the random seed
tf.random.set_seed(42)

# 1. Create the model (this time 3 layers)
model_3 = tf.keras.Sequential([
    tf.keras.layers.Dense(100), # add 100 Dense neurons
    tf.keras.layers.Dense(10), # add another layer with 10 neurons
    tf.keras.layers.Dense(1)])

print(model_2)

# 2. Compile the model
model_3.compile(loss=tf.keras.losses.BinaryCrossentropy(),
                optimizer=tf.keras.optimizers.Adam(),
                metrics=["accuracy"])

# 3. Fit the model
model_3.fit(X, y, epochs=100, verbose=0)

# 4. Evaluate the model
model_3.evaluate(X, y)

#print(model_3.predict(X))

# To visulize our model's predictions, let's create a function 'plot_decision_boundary()',
# this function will:
#   - Take in a trained model, features (X) and labels (y)
#   - Create a meshgrid of the different X values
#   - Make predictions across the meshgrid
#   - Plot the predictions as well as a line between zones (where each uniques class falls)

import numpy as np

def plot_decision_boundary(model, X, y):
    """
    Plots the decision boundary created by a model predicting on X.
    This function was inspired by two resources:
        1. https://cs231n.github.io/neural-networks-case-study/
        2. https://github.com/madewithml/basics/blob/master/notebooks/09_Multilayer_Perceptrons/09_TF_Multilayer_Perceptrons.ipynb
    """
    # Define the axis boundaries of the plot and create a meshgrid
    x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                         np.linspace(y_min, y_max, 100))
    
    # Create X valus (we're goning to make prediction on these)
    x_in = np.c_[xx.ravel(), yy.ravel()] # stack 2D arrays together
    
    # Make predictions
    y_pred = model.predict(x_in)
    
    # Check for multi-class
    if len(y_pred[0]) > 1:
        print("doing multiclass classification")
        # We have to reshape our prediction to get them ready for plotting
        y_pred = np.argmax(y_pred, axis=1).reshape(xx.shape)
    else:
        print("doing binary classification")
        y_pred = np.round(y_pred).reshape(xx.shape)
        
    # Plot the decision boundary
    plt.contourf(xx, yy, y_pred, cmap=plt.cm.RdYlBu, alpha=0.7)
    plt.scatter(X[:, 0], X[:,1], c=y, s=40, cmap=plt.cm.RdYlBu)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    
# Check out the predictions our model is making
plot_decision_boundary(model=model_3,
                       X=X,
                       y=y)

# Lets see if our model can be used for a regression problem.
tf.random.set_seed(42)

# Create some regression data
X_regression = tf.range(0, 1000, 5)
y_regression = tf.range(100, 1100, 5) # y = X + 100

print(X_regression, y_regression)

# Split our regression data into training and test sets
X_reg_train = X_regression[:150]
X_reg_test = X_regression[150:]
y_reg_train = y_regression[:150]
y_reg_test = y_regression[150:]

# Fit our model to the regression data
#model_3.fit(tf.expand_dims(X_reg_train, axis=-1), y_reg_train, epochs=100)

# We compiled our model for a binary classification problem.
# But we're now working on regression problem,
# let's change the mode to suit our data.

print('\033[1m' + "Model_4" + '\033[0m') --------------------------------------
# Set the random seed
tf.random.set_seed(42)

# 1. Create the model (this time 3 layers)
model_4 = tf.keras.Sequential([
    tf.keras.layers.Dense(100), # add 100 Dense neurons
    tf.keras.layers.Dense(10), # add another layer with 10 neurons
    tf.keras.layers.Dense(1)])


# 2. Compile the model
model_4.compile(loss=tf.keras.losses.mae,
                optimizer=tf.keras.optimizers.Adam(),
                metrics=["mae"])

# 3. Fit the model
model_4.fit(tf.expand_dims(X_reg_train, axis=-1), y_reg_train, epochs=100, verbose=0)

# Make predictions with our trained model
y_reg_preds = model_4.predict(X_reg_test)

# Plot the model's prediction against our regression data
plt.figure(figsize =(10,7))
plt.scatter(X_reg_train, y_reg_train, c="b", label="Training data")
plt.scatter(X_reg_test, y_reg_test, c="g", label="Test")
plt.scatter(X_reg_test, y_reg_preds, c="r", label="Prediction")
plt.legend();

# The missing piece: Non-linearity ============================================
print()
print('\033[1m' + "The missing piece: Non-linearity" + '\033[0m')

# Part 2