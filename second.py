import tensorflow as tf
import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt

model = tf.keras.Sequential([
    keras.layers.Dense(units=64, activation='relu', input_shape=[1]),
    keras.layers.Dense(units=64, activation='relu'),
    keras.layers.Dense(units=1)
])

model.compile(optimizer=keras.optimizers.Adam(0.1), loss='mean_squared_error')

x = np.round(np.array(np.random.uniform(-10.0, 10.0, 30), dtype=float), 2)
y = 2 * x * x

learn_len = int(len(x)/5*4)
xlearn, xtest = x[:learn_len], x[learn_len:]
ylearn, ytest = y[:learn_len], y[learn_len:]

model.fit(xlearn, ylearn, epochs=100)
ypred = model.predict(xtest) # предсказанные значения

loss_value = model.evaluate(xtest, ytest)
print("Значение функции потерь на тестовых данных:", loss_value)

for predicted, test in zip(ypred, ytest):
    print(f'{predicted} --> {test}')

plt.figure()
plt.scatter(xtest, ypred, label='predicted')
plt.scatter(xtest, ytest, label='test')
plt.title('y = 2*x^2')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.savefig('plot.png')