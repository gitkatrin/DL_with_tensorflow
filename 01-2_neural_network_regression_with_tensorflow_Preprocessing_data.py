#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 22 16:31:45 2022

@author: bhtcsuperuser
"""
# Import required libraries
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt

# Read in the insurace datasset 
insurance = pd.read_csv("https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/insurance.csv")
print(insurance)

# insurance_one_hot = pd.get_dummies(insurance)
# print(insurance_one_hot)

# Preprocessing data (normalisation and standardization) ======================
print()
print('\033[1m' + "Preprocessing data (normalisation and standardization)" + '\033[0m')
# In terms of scaling values, neural networks tend to prefer normalization.
# If you're not sure on which to use, you could try bouth and see which performs better.

# print(X)
# X["age"].plot(kind="hist")
# X["bmi"].plot(kind="hist")
# print(X["children"].value_counts())

# To prepare our data, we can borrow a few classes from Scikit-Learn
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.model_selection import train_test_split

#X["age"].plot(kind="hist")

# Create a colum transformer
ct = make_column_transformer(
    (MinMaxScaler(), ["age", "bmi", "children"]), # Turn all values in these columns between 0 and 1
    (OneHotEncoder(handle_unknown="ignore"), ["sex", "smoker", "region"])
    )

# Create X & y
X = insurance.drop("charges", axis=1)
y = insurance["charges"]

# Build our train and test sets 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("Länge von X, X_train, X_test: ", len(X), len(X_train), len(X_test))

# Fit column transformer to out training data
ct.fit(X_train)

# transform training and test data with normalisation (MinMaxScaler) anf OneHotEncoder
X_train_normal = ct.transform(X_train)
X_test_normal = ct.transform(X_test)

# What does our data look like?
print(X_train.loc[0])
print(X_train_normal[0])
print(X_train.shape, X_train_normal.shape)

# Build a neural network model to fit on our normalized data ==================
print()
print('\033[1m' + "Build a neural network model to fit on our normalized data" + '\033[0m')
# Build a neural network model to fit on our normalized data
tf.random.set_seed(42)

# 1. Create the model
insurance_model_4 = tf.keras.Sequential([
    tf.keras.layers.Dense(100),
    tf.keras.layers.Dense(10),
    tf.keras.layers.Dense(1)                                        
])

# 2. Compile the model
insurance_model_4.compile(loss=tf.keras.losses.mae,
                          optimizer=tf.keras.optimizers.SGD(),
                          metrics=["mae"])

# 3. Fit the model
insurance_model_4.fit(X_train_normal, y_train, epochs=100)

# 55 on google colab