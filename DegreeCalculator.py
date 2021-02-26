import matplotlib as plt
import numpy as np
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)


celsius_q = np.array([-40, -10, 0, 8, 15, 22, 38])
fahrenheit_a = np.array([-40, 14, 32, 46, 59, 72, 100])

for i, c in enumerate(celsius_q):
    print("{} degrees celsius = {} degrees fahrenheit".format, fahrenheit_a(i))

l0 = tf.keras.layers.Dense(units=1, input_shape=[1])
model = tf.keras.Sequential([l0])

model.compile(loss='mean_squared_error',
              optimizer=tf.keras.optimizersAdam(0.1))

history = model.fit(celsius_q, fahrenheit_a, epochs=500, verbose=false)
print('Finished training model')

plt.xlabel('Epoch Number')
plt.ylabel("Loss Magnitude")
plt.plot(history.history['loss'])

print(model.predict([100]))
print("These are the layer weights: {}".format(l0.get_weights()))
