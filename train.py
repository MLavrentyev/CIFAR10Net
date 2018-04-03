import tensorflow as tf
import data_processing
import os


def conv_layer(input, channels_in, channels_out, filter_shape, name="conv"):
    with tf.name_scope(name):
        W = tf.Variable(tf.truncated_normal(shape=[filter_shape[0], filter_shape[1], channels_in, channels_out],
                                            stddev=0.1), name="W")
        b = tf.Variable(tf.constant(0., shape=[channels_out]), name="b")
        conv = tf.nn.conv2d(input, W, strides=[1, 1, 1, 1], padding="SAME")
        activation = tf.nn.leaky_relu(conv + b, 0.1, name="act")

        tf.summary.histogram("weights", W)
        tf.summary.histogram("biases", b)
        tf.summary.histogram("act", activation)

    return activation


def fc_layer(input, channels_in, channels_out, dropout_p=0., name="fcl"):
    with tf.name_scope(name):
        initializer = tf.contrib.layers.xavier_initializer()
        W = tf.Variable(initializer([channels_in, channels_out]), name="W")
        b = tf.Variable(tf.constant(0., shape=[channels_out]), name="b")
        ff = tf.matmul(input, W) + b
        activation = tf.sigmoid(ff, name="act")
        dropout = tf.nn.dropout(activation, 1 - dropout_p, name="dropout")

        tf.summary.histogram("weights", W)
        tf.summary.histogram("biases", b)
        tf.summary.histogram("act", activation)

    return dropout


def neural_network(n_channels_in):
    x = tf.placeholder(tf.float32, shape=[None, 32, 32, n_channels_in], name="x")
    y = tf.placeholder(tf.float32, shape=[None, 10], name="labels")

    conv1 = conv_layer(x, n_channels_in, 32, (6, 6), name="conv1")
    pool1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

    conv2 = conv_layer(pool1, 32, 32, (6, 6), name="conv2")
    pool2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

    conv3 = conv_layer(pool2, 32, 64, (6, 6), name="conv3")
    pool3 = tf.nn.max_pool(conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

    # in_fcl_size = pool2.reduce_prod()
    flattened = tf.reshape(pool3, [-1, 4*4*64], name="flattened")

    fcl1 = fc_layer(flattened, 4*4*64, 1024, dropout_p=0.2, name="fcl1")
    fcl2 = fc_layer(fcl1, 1024, 10, dropout_p=0., name="fcl2")

    global_step = tf.Variable(0, name="global_step", trainable=False)
    with tf.name_scope("xent"):
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=fcl2, labels=y), name="xent")
        tf.summary.scalar("xent", cross_entropy)
    with tf.name_scope("train"):
        optimizer = tf.train.AdamOptimizer(6e-4)
        train_step = optimizer.minimize(cross_entropy, global_step=global_step)
        tf.summary.scalar("learning_rate", optimizer._lr)
    with tf.name_scope("accuracy"):
        correct_prediction = tf.equal(tf.argmax(fcl2, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name="accuracy")
        tf.summary.scalar("accuracy", accuracy)

    return x, y, train_step, accuracy, global_step


def load_data(as_grayscale=False):
    all_data, all_labels = data_processing.load_all_data("data/cifar-10-batches-py/data")
    all_data = data_processing.reshape_image_data(all_data, to_grayscale=as_grayscale)
    all_data = data_processing.normalize_image_data(all_data)
    all_labels = data_processing.reshape_labels(all_labels, 10)
    print("Data loaded")

    return all_data, all_labels


def set_up_model(sess, base_save_path, model_num=None, grayscale=False):
    model_path = base_save_path + str(model_num) + "/"

    if type(model_num) == int and os.path.exists(model_path):
        graph_file = None
        for file in os.listdir(model_path):
            if file.endswith(".meta"):
                graph_file = file

        if graph_file:
            saver = tf.train.import_meta_graph(model_path + graph_file)
            saver.restore(sess, tf.train.latest_checkpoint(model_path))

            graph = tf.get_default_graph()
            [print(op.name) for op in graph.get_operations()]
            x = graph.get_tensor_by_name("x:0")
            y = graph.get_tensor_by_name("labels:0")
            train_step = graph.get_tensor_by_name("train_step:0")
            accuracy = graph.get_tensor_by_name("accuracy:0")
            global_step = graph.get_tensor_by_name("global_step:0")

            print("Model restored: %d" % (model_num))
        else:
            print("No model file found.")
            return
    else:
        n_channels = 1 if grayscale else 3
        x, y, train_step, accuracy, global_step = neural_network(n_channels)
        sess.run(tf.global_variables_initializer())

        model_num = 0
        saver = tf.train.Saver(max_to_keep=5, name="CIFAR10")
        while True:
            if os.path.exists(base_save_path + str(model_num)):
                model_num += 1
            else:
                break
        print("New model created: %d" % (model_num))

    return model_num, saver, x, y, train_step, accuracy, global_step


def train(sess, saver, model_num, train_step, accuracy, global_step, x, y,
          all_data, all_labels, batch_size, num_steps,
          tensorboard_log_root_path, trained_model_root_path):
    # Tensorboard writer
    writer = tf.summary.FileWriter(tensorboard_log_root_path + str(model_num))
    writer.add_graph(sess.graph)
    merged_summary = tf.summary.merge_all()

    # Training
    for i in range(num_steps):
        step = sess.run(global_step)

        data = data_processing.get_batch(all_data, batch_size, i * batch_size)
        labels = data_processing.get_batch(all_labels, batch_size, i * batch_size)
        feed_dict = {x: data, y: labels}

        # Write to Tensorboard
        s = sess.run(merged_summary, feed_dict=feed_dict)
        writer.add_summary(s, i)

        # Print out accuracy
        if i % 100 == 0:
            train_accuracy = sess.run(accuracy, feed_dict=feed_dict)
            print("Step: %d, Training accuracy: %.2f" % (step, train_accuracy))
            saver.save(sess, trained_model_root_path + str(model_num) + "/model", global_step=step)
        elif i % 10 == 0:
            print("Step: %d" % (step))

        sess.run(train_step, feed_dict=feed_dict)


def main(unused_argv):
    # Various settings here
    # TODO refactor and split off into seperate functions
    trained_model_root_path = "trained_models/cifar10/"
    tensorboard_log_root_path = "tensorboard/cifar10/"
    grayscale = False
    model_num = 42

    # Initialize session
    sess_config = tf.ConfigProto(inter_op_parallelism_threads=2, intra_op_parallelism_threads=2)
    sess = tf.Session(config=sess_config)
    sess.run(tf.global_variables_initializer())

    all_data, all_labels = load_data(as_grayscale=grayscale)
    model_num, saver, x, y, train_step, accuracy, global_step = set_up_model(sess, trained_model_root_path,
                                                                             model_num=model_num, grayscale=grayscale)
    train(sess, saver, model_num, train_step, accuracy, global_step, x, y,
          all_data, all_labels, 80, 100,
          tensorboard_log_root_path, trained_model_root_path)


    # Run validation set
    val_data, val_labels = data_processing.load_all_data("data/cifar-10-batches-py/test_batch")
    val_data = data_processing.reshape_image_data(val_data, to_grayscale=grayscale)
    val_labels = data_processing.reshape_labels(val_labels, 10)
    print("Test data loaded")

    test_accuracy = sess.run(accuracy, feed_dict={x: val_data, y: val_labels})
    print("Validation accuracy: ", test_accuracy)

    sess.close()

if __name__ == "__main__":
    tf.app.run()