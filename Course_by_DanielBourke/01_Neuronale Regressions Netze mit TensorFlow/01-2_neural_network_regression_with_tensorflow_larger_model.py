#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 22 13:08:49 2022

@author: gitkatrin, input from: mrdbourke
"""
# A larger Model ==============================================================
print('\033[1m' + "A larger Model" + '\033[0m')
# Import required libraries
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt

# Read in the insurace datasset 
insurance = pd.read_csv("https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/insurance.csv")
print(insurance)

#print(insurance["smoker"], insurance ["age"])

# Let's try one-hot encode our Dataframe so it's all numbers
insurance_one_hot = pd.get_dummies(insurance)
print(insurance_one_hot.head())

# Create X & y values (features and labels) -----------------------------------
print()
print('\033[1m' + "Create X & y values (features and labels)" + '\033[0m')

X = insurance_one_hot.drop("charges", axis=1)
y = insurance_one_hot["charges"]

# View X and y
print("X: ", X.head())
print("Y: ", y.head())

# Create training and test sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("Länge von X, X_train, X_test: ", len(X), len(X_train), len(X_test))

# Build a neurnal network (sort of like model_2 above)-------------------------
print()
print('\033[1m' + "Model_1" + '\033[0m')
tf.random.set_seed(42)

# 1. Create a model
insurance_model = tf.keras.Sequential([
    tf.keras.layers.Dense(10),
    tf.keras.layers.Dense(1)
    ])

# 2. Compile the model
insurance_model.compile(loss=tf.keras.losses.mae,
                        optimizer=tf.keras.optimizers.SGD(),
                        metrics=["mae"])

# 3. Fit the model
insurance_model.fit(X_train, y_train, epochs=100, verbose=0)

# Check the results of the insurance model on the test data
insurance_model.evaluate(X_test, y_test)

print(y_train.median(), y_train.mean())

# Right now it looks like our model isn't performing too well.. let's try and improve it!
# To (try) improve our model, we'll run 2 experiments:
#   1. Add an extra layer with more hidden units and use the Adam optimizer
#   2. Same as above but train for longer (200 epochs)

# Model_2 ---------------------------------------------------------------------
print()
print('\033[1m' + "Model_2" + '\033[0m')
# Set random seed
tf.random.set_seed(42)

# 1. Create the model
insurance_model_2 = tf.keras.Sequential([
    tf.keras.layers.Dense(100),
    tf.keras.layers.Dense(10),
    tf.keras.layers.Dense(1)
    ])

# 2. Compile the model
insurance_model_2.compile(loss=tf.keras.losses.mae,
                          optimizer=tf.keras.optimizers.Adam(), # Change optimizer
                          metrics=["mae"])

# 3. Fit the model
insurance_model_2.fit(X_train, y_train, epochs=100, verbose=0)

# Evaluate the larger model
insurance_model_2.evaluate(X_test, y_test)

# Model_3 ---------------------------------------------------------------------
print()
print('\033[1m' + "Model_3" + '\033[0m')
# Set random seed
tf.random.set_seed(42)

# 1. Create the model
insurance_model_3 = tf.keras.Sequential([
    tf.keras.layers.Dense(100),
    tf.keras.layers.Dense(10),
    tf.keras.layers.Dense(1)
    ])

# 2. Compile the model
insurance_model_3.compile(loss=tf.keras.losses.mae,
                          optimizer=tf.keras.optimizers.Adam(), # Change optimizer
                          metrics=["mae"])

# 3. Fit the model
history = insurance_model_3.fit(X_train, y_train, epochs=200, verbose=0)

# Evaluate the larger model
insurance_model_3.evaluate(X_test, y_test)

# Plot history (also known as a loss curve or a training curve)
pd.DataFrame(history.history).plot()
plt.ylabel("loss")
plt.xlabel("epochs")

# How long should you train for?
# It depends on the problem you're working on. However, Tensorflow has a solution: EarlyStopping Callback,
# which is a TensorFlow component you can add to your model to stop training once it stops improving a certain metric.

# Preprocessing data (normalisation and standardization) ======================
print()
print('\033[1m' + "Preprocessing data (normalisation and standardization)" + '\033[0m')

print(X)
