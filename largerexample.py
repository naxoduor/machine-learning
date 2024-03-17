import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt



insurance=pd.read_csv("https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/insurance.csv")
print(insurance)

insurance_one_hot=pd.get_dummies(insurance)
print(insurance_one_hot)
print(insurance_one_hot.head())


X = insurance_one_hot.drop("charges", axis=1)
y= insurance_one_hot["charges"]

print(X.head())
print(y.head())

X_train, X_test, y_train, y_test=train_test_split(X, y, test_size=0.2, random_state=42)

print(len(X), len(X_train), len(X_test))

tf.random.set_seed(42)

insurance_model=tf.keras.Sequential([
    tf.keras.layers.Dense(10),
    tf.keras.layers.Dense(1)
])

insurance_model.compile(loss=tf.keras.losses.mae,
                        optimizer=tf.keras.optimizers.SGD(),
                        metrics=["mae"])

insurance_model.fit(np.asarray(X_train).astype(np.float32), np.asarray(y_train).astype(np.float32), epochs=100)

print(insurance_model.evaluate(np.asarray(X_test).astype(np.float32), np.asarray(y_test).astype(np.float32)))

tf.random.set_seed(42)

insurance_model2=tf.keras.Sequential([
    tf.keras.layers.Dense(100),
    tf.keras.layers.Dense(10),
    tf.keras.layers.Dense(1)
])

insurance_model2.compile(loss=tf.keras.losses.mae,
                         optimizer=tf.keras.optimizers.Adam(),
                         metrics=["mae"])

insurance_model2.fit(np.asarray(X_train).astype(np.float32), np.asarray(y_train).astype(np.float32), epochs=100, verbose=1)

insurance_model2.evaluate(np.asarray(X_test).astype(np.float32), np.asarray(y_test).astype(np.float32))

insurance_model_3=tf.keras.Sequential([
    tf.keras.layers.Dense(100),
    tf.keras.layers.Dense(10),
    tf.keras.layers.Dense(1)
])

insurance_model_3.compile(loss=tf.keras.losses.mae,
                          optimizer=tf.keras.optimizers.Adam(),
                          metrics=["mae"])

def create_array(data):
    return np.asarray(data).astype(np.float32)


history=insurance_model_3.fit(create_array(X_train), create_array(y_train), epochs=200)

insurance_model_3.evaluate(create_array(X_test), create_array(y_test))

insurance_model.evaluate(create_array(X_test), create_array(y_test))


print(history.history)

numpy_A = np.arange(0,200, dtype=np.int32)


plt.figure(figsize=(10,7))

plt.scatter( numpy_A, history.history["loss"], c="pink", label="loss")

plt.legend()

plt.show()




pd.DataFrame(history.history).plot()
plt.ylabel("loss")
plt.xlabel("epochs")