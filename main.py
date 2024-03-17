import tensorflow as tf
import numpy as np

print(tf.__version__)
scalar = tf.constant(7)
scalar
print(scalar)
print(scalar.ndim)

# Create a vector
vector = tf.constant([10,10])
print(vector)

# check dimension of vector
print(vector.ndim)

# Create a matrix (has more than 1 dimension)
matrix = tf.constant([[10,7],[7,10]])
print(matrix)
print(matrix.ndim)

# Create another matrix
another_matrix = tf.constant([[10., 7.],
                              [3., 2.],
                              [8.,9.]], dtype=tf.float16)


print(another_matrix)
print(another_matrix.ndim)

# Let's create a tensor
tensor = tf.constant([[[1,2,3],
                       [4,5,6]],
                       [[7,8,9],
                        [10,11,12]],
                        [[13,14,15],
                         [16,17,18]]])

print(tensor)
print(tensor.ndim)

# Create the same tensor with tf.Variable()
changeable_tensor = tf.Variable([10,7])
unchangeable_tensor = tf.constant([10,7])
print(changeable_tensor, unchangeable_tensor)

# Chane changeable_tensor
changeable_tensor[0].assign(7)
print(changeable_tensor)

# create two random (but the same) tensors
random_1 = tf.random.Generator.from_seed(42)
random_1 = random_1.normal(shape=(3,2))

random_2 = tf.random.Generator.from_seed(42)
random_2 = random_2.normal(shape=(3,2))

print(random_1, random_2, random_1==random_2)

not_shuffled = tf.constant([[10,7],
                            [3,4],
                            [2,5]])

tf.random.set_seed(42)
tf.random.shuffle(not_shuffled, seed=42)

#Create a tensor of all ones
ones = tf.ones([10,7])
print(ones)

zeros = tf.zeros(shape=(3,4))
print(zeros)

numpy_A = np.arange(1,25, dtype=np.int32)
print(numpy_A)
A=tf.constant(numpy_A, shape=(2,3,4))
print(A)

#Create a rank 4 tensor
rank_4_tensor = tf.zeros(shape=[2,3,4,5])
print(rank_4_tensor[0])

print("Datatype of every element:", rank_4_tensor.dtype)
print("Number of dimensions (rank):", rank_4_tensor.ndim)
print("Shape of tensor:", rank_4_tensor.shape)
print("Elements along the 0 axis:", rank_4_tensor.shape[0])
print("Elements along the last axis:", rank_4_tensor.shape[-1])
print("Total number of elements in our tensor:", tf.size(rank_4_tensor))
print("Total number of elements in our tensor:", tf.size(rank_4_tensor).numpy())
print(rank_4_tensor[:2,:2,:2,:2])
print(rank_4_tensor[:1,:1,:,:1])

rank_2_tensor = tf.constant([[10,7],
                             [7,10]]) 
print(rank_2_tensor.shape, rank_2_tensor.ndim)
print(rank_2_tensor[:, -1])
rank_3_tensor = rank_2_tensor[..., tf.newaxis]
print(rank_3_tensor)
rrr=tf.expand_dims(rank_2_tensor, axis=-1)
print(rrr)
xpand0=tf.expand_dims(rank_2_tensor, axis=0)
print(xpand0)

tensor = tf.constant([[10,7],[3,4]])
addtensor=tensor+10
print(addtensor)

multensor=tensor*10
print(multensor)

subtensor=tensor-10
print(subtensor)
print(tf.multiply(tensor, 10))
print(tf.matmul(tensor, tensor))
X=tf.constant([[1,2],
               [3,4],
               [5,6]])

Y=tf.constant([[7,8],
               [9,10],
               [11,12]])

print(tf.matmul(X, tf.reshape(Y, shape=(2,3))))
print(tf.matmul(tf.reshape(X, shape=(2,3)), Y))
print(tf.matmul(tf.transpose(X), Y))
print(tf.matmul(tf.transpose(Y), X))
print(tf.tensordot(tf.transpose(X), Y, axes=1))
print(tf.matmul(X, tf.transpose(Y)))
print(tf.matmul(X, tf.reshape(Y, shape=(2,3))))