import tensorflow as tf
import numpy as np

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
x_train = np.expand_dims(x_train, 3)
x_test = np.expand_dims(x_test, 3)

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, [5, 5], activation=tf.nn.relu, strides=1, padding='same'),
    tf.keras.layers.MaxPool2D((2, 2), strides=(2, 2)),
    tf.keras.layers.Conv2D(32, [5, 5], activation=tf.nn.relu, strides=1, padding='same'),
    tf.keras.layers.MaxPool2D((2, 2), strides=(2, 2)),
    tf.keras.layers.Conv2D(16, [3, 3], activation=tf.nn.relu, strides=1, padding='same'),
    tf.keras.layers.MaxPool2D((2, 2), strides=(2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5, batch_size=128)
model.summary()
score = model.evaluate(x_test, y_test)
print(score)
