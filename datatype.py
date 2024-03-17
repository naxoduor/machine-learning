import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np

B=tf.constant([1.7, 7.4])
print(B.dtype)

C=tf.constant([7,10])
print(C.dtype)

D=tf.cast(B, dtype=tf.float16)
print(D.dtype)

E=tf.cast(C, dtype=tf.float32)
print(E.dtype)

D=tf.constant([-7, -10])
print(tf.abs(D))

E=tf.constant(np.random.randint(0,100, size=50))
print(E)
print(tf.size(E), E.shape, E.ndim)
print(tf.reduce_min(E))
print(tf.reduce_max(E))
print(tf.reduce_mean(E))
print(tf.reduce_sum(E))
print(tfp.stats.variance(E))
print(tf.math.reduce_std(tf.cast(E, dtype=tf.float32)))
print(tf.math.reduce_variance(tf.cast(E, dtype=tf.float32)))

tf.random.set_seed(42)
F=tf.random.uniform(shape=[50])
print(F)
print(tf.argmax(F))
print(F[tf.argmax(F)])
print(tf.reduce_max(F))
print(F[tf.argmax(F)] == tf.reduce_max(F))
print(tf.argmin(F))
print(F[tf.argmin(F)])