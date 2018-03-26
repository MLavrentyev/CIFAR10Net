import numpy as np
import tensorflow as tf
import data_processing


def conv_layer(input, channels_in, channels_out, name="conv"):
    with tf.name_scope(name):
        W = tf.Variable()
        b = tf.Variable()
        conv = tf.nn.conv2d(input, W, strides=[], padding="SAME")
        activation = tf.nn.relu(conv + b)

        tf.summary.histogram("weights", W)
        tf.summary.histogram("biases", b)
        tf.summary.histogram("act", activation)

    return activation


def fc_layer(input, channels_in, channels_out, name="fcl"):
    with tf.name_scope(name):
        W = tf.Variable()
        b = tf.Variable()
        ff = W * input + b
        activation = tf.nn.relu(ff)

    return activation



def neural_network(in_data, labels):
    pass


def main(unused_argv):
    pass


if __name__ == "__main__":
    tf.app.run()