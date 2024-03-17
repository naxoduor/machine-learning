import tensorflow as tf
import numpy as np
# import matplotlib.pyplot as plt
import plotext as plt

print(tf.__version__) 

X=np.array([-7.0, -4.0, -1.0, 2.0, 5.0, 8.0, 11.0, 14.0])

y=np.array([3.0, 6.0, 9.0, 12.0, 15.0, 18.0, 21.0, 24.0])

# plt.scatter(X,y)
# plt.show()

house_info=tf.constant(["bedroom", "bathroom", "garage"])
house_price=tf.constant([939700])
print(house_info)
print(house_price)

X=tf.cast(tf.constant(X),dtype=tf.float32)
y=tf.cast(tf.constant(y),dtype=tf.float32)
print(X)
print(y)

tf.random.set_seed(42)
model=tf.keras.Sequential([
    tf.keras.layers.Input(shape=(1,)),
    tf.keras.layers.Dense(1)
])

model.compile(loss=tf.keras.losses.mae,
              optimizer=tf.keras.optimizers.SGD(),
              metrics=["mae"])

model.fit(X, y, epochs=5)
y_pred=model.predict([17.0])
print(y_pred)

tf.random.set_seed(42)
model=tf.keras.Sequential([
    tf.keras.layers.Input(shape=(1,)),
    tf.keras.layers.Dense(1)
])

model.compile(loss=tf.keras.losses.mae,
              optimizer=tf.keras.optimizers.SGD(),
              metrics=["mae"])

model.fit(X, y, epochs=100)
y_pred=model.predict([17.0])
print(y_pred)

model=tf.keras.Sequential([
    tf.keras.layers.Input(shape=(1,)),
    tf.keras.layers.Dense(100, activation="relu"),
    tf.keras.layers.Dense(1)
])

model.compile(loss="mae",
              optimizer=tf.keras.optimizers.Adam(lr=0.01),
              metrics=["mae"])

model.fit(X,y, epochs=100)

y_pred=model.predict([17.0])
print(y_pred)