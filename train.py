import tensorflow as tf
import data_processing
import os


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
    # Determine file to store logs and model in
    trained_model_root_path = "trained_models/cifar10/"
    tensorboard_log_root_path = "tensorboard/cifar10/"

    file_n = 0
    while True:
        if os.path.exists(trained_model_root_path + str(file_n)) or \
            os.path.exists(tensorboard_log_root_path + str(file_n)):
            file_n += 1
        else:
            break

    # Initialize session
    sess_config = tf.ConfigProto(inter_op_parallelism_threads=1, intra_op_parallelism_threads=1)
    sess = tf.Session(config=sess_config)

    # Set up training data
    all_data, all_labels = data_processing.load_all_data("data/cifar-10-batches-py/data")
    all_data = data_processing.reshape_image_data(all_data)
    all_labels = data_processing.reshape_labels(all_labels, 10)
    print("Data loaded")
    x = tf.placeholder(tf.float32, shape=[None, 32, 32, 3], name="x")
    y = tf.placeholder(tf.float32, shape=[None, 10], name="labels")
    logits = neural_network(x, y)

    # Backpropogate to optimize
    with tf.name_scope("xent"):
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=y))
        tf.summary.scalar("xent", cross_entropy)
    with tf.name_scope("train"):
        train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    with tf.name_scope("accuracy"):
        correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.summary.scalar("accuracy", accuracy)

    sess.run(tf.global_variables_initializer())


    # Tensorboard writer
    writer = tf.summary.FileWriter(tensorboard_log_root_path + str(file_n))
    writer.add_graph(sess.graph)
    merged_summary = tf.summary.merge_all()

    # Saver to save model
    saver = tf.train.Saver(max_to_keep=5, name="CIFAR10")


    # Train
    batch_size = 100
    for i in range(100):
        #TODO: fix this
        data = data_processing.get_batch(all_data, batch_size, i*batch_size)
        labels = data_processing.get_batch(all_labels, batch_size, i*batch_size)
        feed_dict = {x: data, y: labels}

        # Write to Tensorboard
        s = sess.run(merged_summary, feed_dict=feed_dict)
        writer.add_summary(s, i)

        # Print out accuracy
        if i % 10 == 0:
            train_accuracy = sess.run(accuracy, feed_dict=feed_dict)
            print("Step: %d, Training accuracy: %.2f" % (i, train_accuracy))
            saver.save(sess, trained_model_root_path + str(file_n) + "/model", global_step = i)
        else:
            print("Step: %d" % (i))

        sess.run(train_step, feed_dict=feed_dict)

    sess.close()

if __name__ == "__main__":
    tf.app.run()