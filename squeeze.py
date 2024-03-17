import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
tf.random.set_seed(42)
G=tf.constant(tf.random.uniform(shape=[50]), shape=(1,1,1,1,50))
print(G)
G_squeezed = tf.squeeze(G)
print(G_squeezed, G_squeezed.shape)

some_list=[0,1,2,3]
print(tf.one_hot(some_list, depth=4))
print(tf.one_hot(some_list, depth=4, on_value="yo i love deep learning", off_value="I also like to dance"))
H=tf.range(1,10)
print(tf.square(H))
print(tf.sqrt(tf.cast(H, dtype=tf.float32)))
print(tf.math.log(tf.cast(H, dtype=tf.float32)))
J=tf.constant(np.array([3.,7.,10.]))
print(J)
print("checkk")
print(np.array(J))
print(type(np.array(J)))
print(J.numpy)
J=tf.constant([3.])
print(J.numpy()[0])
numpy_J=tf.constant(np.array([3.,7.,10]))
tensor_J=tf.constant([3.,7.,10.])
print(numpy_J.dtype)
print(tensor_J.dtype)