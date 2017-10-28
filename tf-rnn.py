import tensorflow as tf
from tensorflow.contrib import rnn

#import mnist dataset
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('./mnist/', one_hot=True)


#training parameters
learning_rate = 0.001
training_steps = 10000
batch_size = 128
display_step = 200

#network parameters
n_input = 28 # mnist image is 28x28
timesteps = 28 # input_data is the one row of image matrix at a time
n_hidden = 128 #just random number (size of lstm cell)
n_classes = 10

X = tf.placeholder(tf.float32, [None, timesteps, n_input])
Y = tf.placeholder(tf.float32, [None, n_classes])

weights = {
    'out': tf.Variable(tf.random_normal([n_hidden, n_classes])),
}
biases = {
    'out': tf.Variable(tf.random_normal([n_classes])),
}

def RNN(x, weights, biases):
    #current data(x) is (batch_size, timesteps, n_input) shape
    #need list of (batch_size, n_input) shape with size=timesteps

    x = tf.unstack(x, timesteps, 1)

    lstm_cell = rnn.BasicLSTMCell(n_hidden, forget_bias=1.0)

    outputs, state = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)

    return tf.add(tf.matmul(outputs[-1], weights['out']), biases['out'])

logits = RNN(X, weights, biases)
prediction = tf.nn.softmax(logits)

loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y))
train = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss_op)

correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

init = tf.global_variables_initializer()

with tf.Session() as sess:

    sess.run(init)

    for step in range(1, training_steps + 1):
        batch_x, batch_y = mnist.train.next_batch(batch_size)

        batch_x = batch_x.reshape([batch_size, timesteps, n_input])

        sess.run(train, feed_dict={X: batch_x, Y: batch_y})

        if step % display_step == 0 or step == 1:
            loss, acc = sess.run([loss_op, accuracy], feed_dict={X: batch_x, Y: batch_y})
            print("Step ", step, " of ", training_steps, "\nloss is ", loss, " accuracy is ", acc)


    print("Optimization finished!!!")

    #calculate accuracy for 128 test images
    len_test = 128
    test_data = mnist.test.images[:len_test].reshape([-1, timesteps, n_input])
    test_labels = mnist.test.labels[:len_test]
    print("Test accuracy is ", sess.run(accuracy, feed_dict={X: test_data, Y: test_labels}))