import tensorflow as tf
import data_processing


def conv_layer(input, channels_in, channels_out, filter_shape, name="conv"):
    with tf.name_scope(name):
        W = tf.Variable(tf.truncated_normal(shape=[filter_shape[0], filter_shape[1], channels_in, channels_out],
                                            stddev=0.1))
        b = tf.Variable(tf.constant(0., shape=[channels_out]))
        conv = tf.nn.conv2d(input, W, strides=[1, 1, 1, 1], padding="SAME")
        activation = tf.nn.relu(conv + b)

        tf.summary.histogram("weights", W)
        tf.summary.histogram("biases", b)
        tf.summary.histogram("act", activation)

    return activation


def fc_layer(input, channels_in, channels_out, name="fcl"):
    with tf.name_scope(name):
        W = tf.Variable(tf.truncated_normal(shape=[channels_in, channels_out], stddev=0.1))
        b = tf.Variable(tf.constant(0., shape=[channels_out]))
        ff = tf.matmul(input, W) + b
        activation = tf.nn.relu(ff)

        tf.summary.histogram("weights", W)
        tf.summary.histogram("biases", b)
        tf.summary.histogram("act", activation)

    return activation



def neural_network(in_data, labels):
    conv1 = conv_layer(in_data, 3, 32, (6,6), name="conv1")
    pool1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

    conv2 = conv_layer(pool1, 32, 64, (6, 6), name="conv2")
    pool2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

    # in_fcl_size = pool2.reduce_prod()
    flattened = tf.reshape(pool2, [-1, 8*8*64], name="flattened")

    fcl1 = fc_layer(flattened, 8*8*64, 1024, name="fcl1")
    fcl2 = fc_layer(fcl1, 1024, 2048, name="fcl2")
    logits = fc_layer(fcl2, 2048, 10, name="logits")

    return logits


def main(unused_argv):
    sess = tf.Session()

    # Set up training data
    base_data_path = "data/cifar-10-batches-py/"
    data_files = ["data_batch_1", "data_batch_2", "data_batch_3", "data_batch_4", "data_batch_5"]
    data_batches = []
    label_batches = []
    for file in data_files:
        d, l = data_processing.load_batch_data(base_data_path + file)
        d = data_processing.reshape_image_data(d)
        l = data_processing.reshape_labels(l)
        data_batches.append(d)
        label_batches.append(l, 10)
    num_batches = len(data_batches)

    x = tf.placeholder(tf.float32, shape=[None, 32, 32, 3], name="x")
    y = tf.placeholder(tf.float32, shape=[None, 10], name="labels")

    logits = neural_network(x, y)

    # Backpropogate to optimize
    with tf.name_scope("xent"):
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y))
        tf.summary.scalar("xent", cross_entropy)
    with tf.name_scope("train"):
        train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    with tf.name_scope("accuracy"):
        correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.summary.scalar("accuracy", accuracy)

    sess.run(tf.global_variables_initializer())


    # Tensorboard writer
    writer = tf.summary.FileWriter("tensorboard/cifar10/0")
    writer.add_graph(sess.graph)
    merged_summary = tf.summary.merge_all()

    # Saver to save model
    saver = tf.train.Saver(max_to_keep=5, name="CIFAR10")


    # Train
    for i in range(12):
        data = data_batches[i % num_batches]
        labels = label_batches[i % num_batches]
        feed_dict = {x: data, y: labels}

        # Write to Tensorboard
        s = sess.run(merged_summary, feed_dict=feed_dict)
        writer.add_summary(s, i)

        # Print out accuracy
        if i % 2 == 0:
            train_accuracy = sess.run(accuracy, feed_dict=feed_dict)
            print("Step: %d, Training accuracy: %.2f" % (i, train_accuracy))
            saver.save(sess, "trained_models/cifar10", global_step = i)

        sess.run(train_step, feed_dict=feed_dict)

    sess.close()

if __name__ == "__main__":
    tf.app.run()