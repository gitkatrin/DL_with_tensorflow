# -*- coding: utf-8 -*-
"""
Created on Tue Feb 15 11:39:52 2022

@author: gitkatrin, input from: mrdbourke
"""

# 1. Introduction to tensors
# 2. Getting Information for tensors
# 3. Manipulating tensors
# 4. Tensors & Numpy
# 5. Using @tf.function
# 6. Using GPUs with TF (or TPU)

# 1. Introduction to Tensors ==================================================
print('\033[1m' + '1. Introduction to Tensors' + '\033[0m')

# Import TensorFlow
import tensorflow as tf
print(tf.__version__)


# Create tensors with tf.constant()
scalar = tf.constant(7)
print(scalar)

# Check the number of dimensions of a tensor (ndim = number pf dimensions)
print(scalar.ndim)

# Create a vector
vector = tf.constant([10, 10])
print(vector)

# Check the dimention of our vector
print(vector.ndim)

# Create a matrix (has more than 1 dim)
matrix = tf.constant([[10, 7],
                      [7, 10]])
print(matrix)
print(matrix.ndim)

# Create another matrix
another_matrix = tf.constant([[10., 7.],
                              [3., 2.],
                              [8., 9.]], dtype=tf.float16) # specify the data type with dtype parameter
print(another_matrix)

# Check the dimentions of our matrix
print(another_matrix.ndim)

# Create a Tensor
tensor = tf.constant([[[1, 2, 3],
                       [4, 5, 6]],
                      [[7, 8, 9],
                       [10, 11, 12]],
                      [[13, 14, 15],
                       [16, 17, 18]]])
print(tensor)
print(tensor.ndim)

# Definitions:
# - Scalar: a single number
# - Vector: number with direction (e.g. wind speed and direction)
# - Matrix: a 2-dimensional array of numbers
# - Tensor: an n-dimensional array of numbers (when n can be any number, a 0-dimensional tensor is a scalar, a 1-dimensional tensor is a vector)

# Creating Tensors with tf.constant() and tf.Variable() -----------------------
print()
print('\033[1m' + 'Creating Tensors with tf.constant() and tf.Variable()' + '\033[0m')

# Create the same tensor with tf.Variable() as above
changeable_tensor = tf.Variable([10, 7])
unchangeable_tensor = tf.constant([10, 7])
print(changeable_tensor, unchangeable_tensor)

# Change one of the elements in the changeable tensor
#changeable_tensor[0] = 7
#print(changeable_tensor)
changeable_tensor[0].assign(7)
print(changeable_tensor)

# Change the unchangable tensor
# unchangeable_tensor[0] = 7
# print(unchangeable_tensor)
# unchangeable_tensor[0].assign(7)
# print(unchangeable_tensor)

# Note: Rarely in practice will you need to decide whether to use tf.constant or tf.Variable to create tensors,
# as TensorFlow does this for you. However, if in doubt, use tf.constant and change it later if you needed.

# Creating random tensors -----------------------------------------------------
print()
print('\033[1m' + 'Creating random tensors' + '\033[0m')

# Create two ransom (but the same) tensors
random_1 = tf.random.Generator.from_seed(7) # set seed for reproductibility
random_1 = random_1.normal(shape=(3,2))
print(random_1)

random_2 = tf.random.Generator.from_seed(7)
random_2 = random_2.normal(shape=(3,2))
print(random_2)

# Are they equal?
print(random_1 == random_2)

# Shuffle a tensor (valuable for when you shuffle your data so the inherent order doesn't effect learning)
not_shuffled = tf.constant([[10, 7],
                            [3, 4],
                            [2, 5]])
print(not_shuffled.ndim)
print(not_shuffled)

# Shuffle the non-shuffled tensor
shuffled = tf.random.shuffle(not_shuffled)
print(shuffled)

# Shuffle the non-shuffled tensor
tf.random.set_seed(42) # global-level seed
shuffled = tf.random.shuffle(not_shuffled, seed=42) # operation-level seed
print(shuffled)


# "If both the global and the operation seed are set: Both seeds are used in conjunction
# to determine the ransom sequence."
# If the shuffled tensors should be in the same order, we've got to use the global level random seed
# as well as the operation level random seed.

# Other ways to make tensors --------------------------------------------------
print()
print('\033[1m' + 'Weitere Möglichkeiten um Tensoren zu erstellen' + '\033[0m')

# Create a tensor of all ones
print(tf.ones([10, 7]))

# Create a tensor of all zeros
print(tf.zeros(shape=(3, 4)))

# The main difference between NumPy arrays and TensorFlow tensors is that tensors
# can be run on a GPU (much faster for numerical computing)

# Turn NumPy arrays into tensors
import numpy as np
numpy_A = np.arange(1, 25, dtype=np.int32) # crreate a NumPy array between 1 and 25
print(numpy_A) 

# X = tf.constant(some_matrix) # capital for matrix or tensor
# y = tf.constant(vector) # non-capital for vector

A = tf.constant(numpy_A, shape=(2, 3, 4))
print(A)
B = tf.constant(numpy_A)
print(B)

# 2. Getting Information from tensors =========================================
print()
print()
print('\033[1m' + '2. Getting Information from tensors' + '\033[0m')

# Shape, Rank, Axis or dimension, Size

# Create a rank 4 tensor (4 dimensions)
rank_4_tensor = tf.zeros([2,3,4,5])
print(rank_4_tensor)

print(rank_4_tensor[0])
print(rank_4_tensor.shape, rank_4_tensor.ndim, tf.size(rank_4_tensor))
print(2*3*4*5)

# Get various attributes of our tensors
print("Datatype of every element:", rank_4_tensor.dtype)
print("Number of dimensions (rank):", rank_4_tensor.ndim)
print("Shape of tensor:", rank_4_tensor.shape)
print("Elements along the 0 axis:", rank_4_tensor.shape[0])
print("Elements along the last axis:", rank_4_tensor.shape[-1])
print("Total number of elements in our tensor:", tf.size(rank_4_tensor))
print("Total number of elements in our tensor:", tf.size(rank_4_tensor).numpy())

# Indexing and expanding tensors ----------------------------------------------
print()
print('\033[1m' + 'Indexing and expanding tensors' + '\033[0m')
# Tensors can be indexed just like Python lists.

# Get the first 2 elements of each dimensions
some_list = [1,2,3,4]
print(some_list[:2])

print(rank_4_tensor[:2,:2,:2,:2])

# Get the first element from each dimension from each index except for the final one
print(some_list[:1])

print(rank_4_tensor.shape)
print(rank_4_tensor[:1,:1,:1,:])

# Create a rank 2 tensor (2 dimensions)
rank_2_tensor = tf.constant([[10,7],
                             [3,4]])
print(rank_2_tensor.shape, rank_2_tensor.ndim)

# Get the öast item of each of row of our rank 2 tensor
print(some_list[-1])

print(rank_2_tensor[:, -1])

# Add in extra dimension to our raink 2 tensor
rank_3_tensor = rank_2_tensor[..., tf.newaxis] # ... = every axis bevor the last one
print(rank_3_tensor)

# Alternative to tf.axis
print(tf.expand_dims(rank_2_tensor, axis =-1)) # -1 means expand the final axis

# Expand the 0-axis
print(tf.expand_dims(rank_2_tensor, axis =0)) 

# 3. Manipulating tensors =====================================================
print()
print()
print('\033[1m' + '3. Manipulating tensors' + '\033[0m')

# Basic operations +, -, *, /

# You can add values to a tensor unsing the addition operator
tensor = tf.constant([[10,7],[3,4]])
print(tensor+10) # Addition
print(tensor) # Original tensor is unchanged
print(tensor*10) # Multiplication
print(tensor-10) # Substraction
print(tensor/10)# Division

# TensorFow built-in functions
print(tf.math.add(tensor, 10)) # Addition
print(tf.multiply(tensor, 10)) # Multiplication
print(tf.math.subtract(tensor, 10)) # Substraction
print(tf.math.divide(tensor, 10)) # Division

# Matrix multiplications ------------------------------------------------------
print()
print('\033[1m' + 'Matrix multiplication 1' + '\033[0m')

# In machine learning, matrix multiplication is one of the most common tensor operations.
# Scalar - "dot product"

# Matrix multiplication in TensorFlow
print(tensor)
print(tf.linalg.matmul(tensor, tensor)) # Scalar

print(tensor*tensor) # element-wise

# Matrix multifplication with Python operator "@"
print(tensor @ tensor)

print(tensor.shape)

# Create a (3, 2) tensor
X = tf.constant([[1,2],
                 [3,4],
                 [5,6]])
# Create a (3, 2) tensor
Y = tf.constant([[7,8],
                 [9,10],
                 [11,12]])
print("X und Y: ", X, Y)

# matrix multifly tensors of same shape
#print(X @ Y) -> ERROR

# There are two rules our tensors (or metrices) need to fulfil if we're going to matrix mlutiply them:
    # 1. The inner dimensions must match
    # 2. The resulting matrix has the shap of the outer dimensions
    
print()
print('\033[1m' + 'Matrix multiplication 2' + '\033[0m')    

# Change the chape of Y
print(tf.reshape(Y, shape=(2,3)))

# Try to multiply X by reshaped Y
print(X.shape, tf.reshape(Y, shape=(2,3)).shape)
print(X @ tf.reshape(Y, shape=(2,3)))

print(tf.matmul(X, tf.reshape(Y, shape=(2,3))))

# Try to change the shape pf X instead of Y
print(tf.matmul(tf.reshape(X, shape=(2,3)), Y))

print(tf.reshape(X, shape=(2,3)).shape, Y.shape)

# The same with transpose
print(X, tf.transpose(X), tf.reshape(X, shape=(2,3)))

# transpose flippes the axis, reshape shuffles the tensor around

# Try matrix multiplication with transpose rather than reshape
print(tf.matmul(tf.transpose(X), Y))

print()
print('\033[1m' + 'Matrix multiplication 3: The dot product' + '\033[0m')

# Matrix multiplication is also referred to as the dot product.
# You can perform matrix multiplication using:
#   - tf.matmul()
#   - tf.tensordot()

# Perform the dot product on X and Y (requires X or Y to be transposed)
print(tf.tensordot(tf.transpose(X), Y, axes=1))

# Perform matrix multiplication between X and Y (transposed)
print(tf.matmul(X, tf.transpose(Y)))

# Perform matrix multiplication betreen X and Y (reshaped)
print(tf.matmul(X, tf.reshape(Y, shape=(2,3))))

# Check values of Y, reshape Y and transposed Y
print('\033[1m', "Normal Y: ", Y, "\n")
print("Y reshaped to (2,3): ", tf.reshape(Y, (2,3)), "\n")

print("Y transponsed: ", tf.transpose(Y), '\033[0m')

# Generally, when performing matrix multiplication on two tensors and one of the axis
# doesn't line up, you will transponse (rather than reshape) one of the tensors to get
# satisify the matrix multipliation rules.

# Changing the datatype of a tensor -------------------------------------------
print()
print('\033[1m' + 'Changing the datatype of a tensor' + '\033[0m')

# Create a new tensor with default datatype (float32)
B = tf.constant([1.7,7.4])
print(B.dtype) #float32

C = tf.constant([7,10])
print(C.dtype) #int32

# Change from float32 to float 16 (reduced precision)
D = tf.cast(B, dtype=tf.float16)
print(D)

# Change from int32 to float32
E = tf.cast(C, dtype=tf.float16)
print(E)

E_float16 = tf.cast(E, dtype=tf.float16)
print(E_float16)

# Aggregation tensors ---------------------------------------------------------
print()
print('\033[1m' + 'Aggregation tensors' + '\033[0m')

# Aggregation tensors = condensing them from multiple values down to a smaller amount of values

# Get the absolut values
D = tf.constant([-7,-10])
print(D)
print(tf.abs(D))

# Forms of aggregation:
#   - Get the minimum
#   - Get the maximum
#   - Get the mean of a tensor
#   - Get the sum of a tensor

# Creating a random tensor with values between 0 and 100 of size 50
E = tf.constant(np.random.randint(0,100,size=50))
print(E, tf.size(E), E.shape, E.ndim)

# Find the minimum
print(tf.reduce_min(E))

# Find the maximum
print(tf.reduce_max(E))

# Find the mean
print(tf.reduce_mean(E))

# Find the sum
print(tf.reduce_sum(E))
print(tf.__version__)

# Find the variance
print(tf.math.reduce_variance(tf.cast(E, dtype=tf.float32)))
# To find the variance of our tensor, we need access to tensorflow_probability
import tensorflow_probability as tfp
print(tfp.stats.variance(E))

# Find the standard deviation
print(tf.math.reduce_std(tf.cast(E, dtype=tf.float32)))

# Find the positional maximum and minimum -------------------------------------
print()
print('\033[1m' + 'Find the positional maximum and minimum' + '\033[0m')

# Create a new tensor for findung positional minimum and maximum
tf.random.set_seed(42)
F = tf. random.uniform(shape=[50])
print(F)

# Find the positional maximum
print(tf.argmax(F))

# Index on our largest value position
print(F[tf.argmax(F)])

# Find the max value of of
print(tf.reduce_max(F))

print(F[tf.argmax(F)] == tf.reduce_max(F))

# Find the positional minimum
print(tf.argmin(F))

# Find the minimum using the positional minimum index
print(F[tf.argmin(F)])

# Squeezing a tensor (removing all single dimensions) -------------------------
print()
print('\033[1m' + 'Squeezing a tensor (removing all single dimensions)' + '\033[0m')

# Creare a tensor to get started
tf.random.set_seed(42)
G = tf.constant(tf.random.uniform(shape=[50]), shape=(1,1,1,1,50))
print(G)

print(G.shape)

G_squeezed = tf.squeeze(G)
print(G_squeezed, G_squeezed.shape)

# One-hot encoding tensors (Form of numerical encoding) -----------------------
print()
print('\033[1m' + 'One-hot encoding tensors (Form of numerical encoding)' + '\033[0m')

# Create a list of indices
some_list = [0,1,2,3] # could be red, green, blue, purple

# One-hot encode our list of indices
print(tf.one_hot(some_list, depth=4))

# Specify custom values for one-hot encoding
print(tf.one_hot(some_list, depth=4, on_value="hello", off_value="bye"))

# Squaring, log, square root --------------------------------------------------
print()
print('\033[1m' + 'quaring, log, square root' + '\033[0m')

# Create a new tensor
H = tf.range(1,10)
print(H)

# Square it
print(tf.square(H))

# Find the Square root
print(tf.sqrt(tf.cast(H, dtype=tf.float32)))

# Find the log
print(tf.math.log(tf.cast(H, dtype=tf.float32)))

# 4. Tensors & Numpy ==========================================================
# TensorFlow interacts beautifully with NumPy arrays.
print()
print('\033[1m' + '4. Tensors & Numpy' + '\033[0m')

# Create a tensor directly form a NumPy array
J = tf.constant(np.array([3., 7., 10.]))
print(J)

# Convert our tensor back to a NumPy array
print(np.array(J), type(np.array(J)), np.array(J).dtype)

# Convert tensor J to a NumPy array
print(J.numpy(), type(J.numpy()), J.numpy().dtype)

# The default types of each are slightly different
numpy_J = tf.constant(np.array([3.,7.,10.]))
tensor_J = tf.constant([3.,7.,10.])

# Check the datatypes
print(numpy_J.dtype, tensor_J.dtype) # <dtype: 'float64'> <dtype: 'float32'>

# 5. Using @tf.function =======================================================
print()
print('\033[1m' + '5. Using @tf.function' + '\033[0m')

# Decorators modify a function in one way or another

# Create a simple function
def function(x, y):
    return print("1: ", x ** 2 + y)

x = tf.constant(np.arange(0,10))
y = tf.constant(np.arange(10,20))
function(x,y)

# Create the same function and decorate it with tf.function
@tf.function
def tf_function(x, y):
    return print("2: ", x ** 2 + y)

tf_function(x,y)

# See: no differenceh, but
# Much of the difference happens behind the scenes. One of the main ones being potential code speed-ups where possible.

# 6. Using GPUs with TF (or TPU) ==============================================
print()
print('\033[1m' + '6. Using GPUs with TF (or TPU)' + '\033[0m')

print(tf.config.list_physical_devices('GPU'))

!nvidia-smi
