import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from keras.utils import plot_model
import pandas as pd

X=tf.range(-100, 100, 4)
y=X+10

X_train=X[:40]
y_train=y[:40]

X_test=X[40:]
y_test=y[40:]

print(len(X_train), len(X_test), len(y_train), len(y_test))

# plt.figure(figsize=(10,7))

# plt.scatter(X, y)

# plt.scatter(X_train, y_train, c="b", label="Training data")

# plt.scatter(X_test, y_test, c="g", label="Testing data")

# plt.legend()

# plt.show()

model=tf.keras.Sequential([
    tf.keras.layers.Dense(10, input_shape=[1], name="input_layer"),
    tf.keras.layers.Dense(1, input_shape=[1], name="output_layer"),

], name="model_1")

model.compile(loss=tf.keras.losses.mae,
              optimizer=tf.keras.optimizers.SGD(),
              metrics=["mae"])

print(model.summary())

plot_model(model=model, show_shapes=True)

model.fit(X_train, y_train, epochs=100, verbose=0)

y_pred=model.predict(X_test)
print(y_pred)
print(y_test)

def plot_predictions(train_data=X_train, train_labels=y_train,
                     test_data=X_test, test_labels=y_test,
                     predictions=y_pred):
    plt.figure(figsize=(10,7))

    plt.scatter(train_data, train_labels, c="b", label="Training data")

    plt.scatter(test_data, test_labels, c="g", label="Testing data")

    plt.scatter(test_data, predictions, c="r", label="Predictions")

    plt.legend()

    plt.show()

plot_predictions(train_data=X_train, 
                 train_labels=y_train,
                 test_data=X_test,
                 test_labels=y_test,
                 predictions=y_pred)

print("Evaluate")
print(model.evaluate(X_test, y_test))

mae=tf.metrics.mean_absolute_error(y_true=y_test, y_pred=tf.constant(y_pred))
print(mae)
print(y_test)
print(tf.constant(y_pred))
mae=tf.metrics.mean_absolute_error(y_true=y_test, y_pred=tf.squeeze(y_pred))
print(mae)
mse=tf.metrics.mean_squared_error(y_true=y_test, y_pred=tf.squeeze(y_pred))
print(mse)

def mae(y_true, y_pred):
    return tf.metrics.mean_absolute_error(y_true=y_true, y_pred=tf.squeeze(y_pred))

def mse(y_true, y_pred):
    return tf.metrics.mean_squared_error(y_true=y_true, y_pred=tf.squeeze(y_pred))


tf.random.set_seed(42)

model_1=tf.keras.Sequential([
    tf.keras.layers.Dense(1, input_shape=[1])
])

model_1.compile(loss=tf.keras.losses.mae,
                optimizer=tf.keras.optimizers.SGD(),
                metrics=["mae"])

model_1.fit(X_train, y_train, epochs=100)

y_preds_1=model_1.predict(X_test)
plot_predictions(predictions=y_preds_1)

mae_1=mae(y_test, y_preds_1)
mse_1=mse(y_test, y_preds_1)
print(mae_1, mse_1)

tf.random.set_seed(42)


model_2=tf.keras.Sequential([
    tf.keras.layers.Dense(100, input_shape=[1]),
    tf.keras.layers.Dense(1)
])

model_2.compile(loss=tf.keras.losses.mae,
                optimizer=tf.keras.optimizers.SGD(),
                metrics=["mse"])

model_2.fit(X_train, y_train, epochs=100)

y_preds_2 = model_2.predict(X_test)
plot_predictions(predictions=y_preds_2)
# print(y_preds_2)

mae_2 = mae(y_test, y_preds_2)
mse_2 = mse(y_test, y_preds_2)

print(mae_2, mse_2)

tf.random.set_seed(42)

model_3= tf.keras.Sequential([
    tf.keras.layers.Dense(100, input_shape=[1]),
    tf.keras.layers.Dense(1)
])

model_3.compile(loss=tf.keras.losses.mae,
                optimizer=tf.keras.optimizers.SGD(),
                metrics=["mae"])

model_3.fit(X_train, y_train, epochs=500)

y_pred_3 = model_3.predict(X_test)
plot_predictions(predictions=y_pred_3)


mae_3=mae(y_test, y_pred_3)
mse_3=mse(y_test, y_pred_3)

print(mae_3, mse_3)

model_results=[["model_1",mae_1.numpy(), mse_1.numpy()],["model_2",mae_1.numpy(),mse_2.numpy()],["model_3",mae_3.numpy(), mse_1.numpy()]]

all_results=pd.DataFrame(model_results, columns=["model_1", "mae_1", "mse_1"])
print(all_results)

model_2.save("best_model_SavedModel_format")

model_2.save("best_model_HDF5_format.h5")

loaded_SavedModel_format=tf.keras.models.load_model("/home/nax/Pytensor/best_model_SavedModel_format")

print(loaded_SavedModel_format.summary())
print(model_2.summary())

model_2_preds = model_2.predict(X_test)
loaded_SavedModel_format_preds = loaded_SavedModel_format.predict(X_test)

print(model_2_preds==loaded_SavedModel_format_preds)

loaded_h5_model = tf.keras.models.load_model("/home/nax/Pytensor/best_model_HDF5_format.h5")

print(loaded_h5_model.summary())
print(model_2.summary())

model_2_preds=model_2.predict(X_test)
loaded_h5_model_preds=loaded_h5_model.predict(X_test)
print(model_2_preds==loaded_h5_model_preds)