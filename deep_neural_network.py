import tensorflow as tf
from tensorflow._api.v2.config import optimizer
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

model = Sequential([
    Flatten(input_shape = (28, 28)),
    Dense(500, activation="relu"),
    Dense(500, activation="relu"),
    Dense(500, activation="relu"),
    Dense(10, activation="softmax")
])

model.compile(
    optimizer=Adam(),
    loss="sparse_categorical_crossentropy",
    metrics = ['accuracy']
)

model.fit(
    x_train,
    y_train,
    batch_size=100,
    epochs=10,
    validation_split=0.1
)

test_loss, test_acc = model.evaluate(x_test, y_test)
print("Test accuracy:", test_acc)

# n_nodes_hl1 = 500
# n_nodes_hl2 = 500
# n_nodes_hl3 = 500
# n_classes = 10
# batch_size = 100

# x = tf.placeholder('float', [None, 784])
# y = tf.placeholder('float')

# def neural_network_model(data):
#     hidden_1_layer = {
#         'weights': tf.Variable(tf.random_normal([784, n_nodes_hl1])),
#         'biases': tf.Variable(tf.random_normal([n_nodes_hl1]))
#     }

#     hidden_2_layer = {
#         'weights': tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
#         'biases': tf.Variable(tf.random_normal([n_nodes_hl2]))
#     }

#     hidden_3_layer = {
#         'weights': tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3])),
#         'biases': tf.Variable(tf.random_normal([n_nodes_hl3]))
#     }

#     output_layer = {
#         'weights': tf.Variable(tf.random_normal([n_nodes_hl3, n_classes])),
#         'biases': tf.Variable(tf.random_normal([n_classes]))
#     }

#     l1 = tf.add(tf.matmul(data, hidden_1_layer['weights']), hidden_1_layer['biases'])
#     l1 = tf.nn.relu(l1)

#     l2 = tf.add(tf.matmul(l1, hidden_2_layer['weights']), hidden_2_layer['biases'])
#     l2 = tf.nn.relu(l2)

#     l3 = tf.add(tf.matmul(l2, hidden_3_layer['weights']), hidden_3_layer['biases'])
#     l3 = tf.nn.relu(l3)

#     output = tf.matmul(l3, output_layer['weights']) + output_layer['biases']

#     return output

# def train_neural_network(x):
#     prediction = neural_network_model(x)
#     cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))

#     optimizer = tf.train.AdamOptimizer().minimize(cost)

#     hm_epochs = 10
#     with tf.Session() as sess:
#         sess.run(tf.global_variables_initializer())

#         for epoch in range(hm_epochs):
#             epoch_loss = 0
#             for _ in range(int(mnist.train.num_examples/batch_size)):
#                 epoch_x, epoch_y = mnist.train.next_batch(batch_size)
#                 _, c = sess.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y})
#                 epoch_loss += c

#             print('Epoch ', epoch, 'completed out of ', hm_epochs,'loss: ', epoch_loss)
        
#         correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))

#         accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
#         print('Accuracy: ', accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))

# train_neural_network(x)