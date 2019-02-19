import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import mnist_inference
import numpy as np

BATCH_SIZE = 1000
LEARNING_RATE = 0.001
REGULARAZTION_RATE = 0.0001
TRAINING_STEPS = 110000

def train(mnist):
    x = tf.placeholder(tf.float32,
        [None,
         mnist_inference.IMAGE_SIZE,
         mnist_inference.IMAGE_SIZE,
         mnist_inference.NUM_CHANNELS],
        'x-input')
    y_ = tf.placeholder(tf.float32,
        [None, mnist_inference.NUM_LABELS],
        'y-input')
    keep_prob = tf.placeholder(tf.float32)

    regularizer = tf.contrib.layers.l2_regularizer(REGULARAZTION_RATE)

    y = mnist_inference.inference(x, keep_prob, regularizer)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1))
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    tf.add_to_collection('losses', cross_entropy_mean)
    loss = tf.add_n(tf.get_collection('losses'))

    train_step = tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss)

    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1)), tf.float32))

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        validation_feed = {
            x: np.reshape(mnist.validation.images,
                [-1, mnist_inference.IMAGE_SIZE,
                 mnist_inference.IMAGE_SIZE,
                 mnist_inference.NUM_CHANNELS]),
            y_:  mnist.validation.labels,
            keep_prob: 1.0}
        test_feed = {
            x: np.reshape(mnist.test.images,
                [-1, mnist_inference.IMAGE_SIZE,
                 mnist_inference.IMAGE_SIZE,
                 mnist_inference.NUM_CHANNELS]),
            y_:  mnist.test.labels,
            keep_prob: 1.0}
        for i in range(TRAINING_STEPS):
            xs, ys = mnist.train.next_batch(BATCH_SIZE)
            reshaped_xs = np.reshape(xs,
                [-1, mnist_inference.IMAGE_SIZE,
                 mnist_inference.IMAGE_SIZE,
                 mnist_inference.NUM_CHANNELS])
            sess.run(train_step, feed_dict={x: reshaped_xs, y_: ys, keep_prob: 0.5})
            if (i+1) % 100 == 0:
                val_loss, val_acc = sess.run([loss, accuracy], feed_dict=validation_feed)
                print('On the step %d, validation set loss is %g, validation set accuracy is %g.' %
                      (i+1, val_loss, val_acc))
        test_loss, test_acc = sess.run([loss, accuracy], feed_dict=test_feed)
        print('Test set loss is %g, test set accuracy is %g.' % (test_loss, test_acc))

def main(argv=None):
    mnist = input_data.read_data_sets('./MNIST_data', one_hot=True)
    print(mnist.train.images.shape, mnist.train.labels.shape)
    print(mnist.validation.images.shape, mnist.validation.labels.shape)
    print(mnist.test.images.shape, mnist.test.labels.shape)
    train(mnist)

if __name__ == '__main__':
    tf.app.run()