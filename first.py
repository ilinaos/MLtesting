import tensorflow as tf
import numpy as np
from tensorflow import keras

model = tf.keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])

model.compile(optimizer='sgd', loss='mean_squared_error')

x = np.round(np.array(np.random.uniform(-15.5, 13.9, 50), dtype=float), 3)
y = 2 * x - 3

model.fit(x, y, epochs=1000)

print(model.predict(np.array([4.0, -2.0, 17.1, -9.7, 0.0])))