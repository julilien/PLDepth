import tensorflow as tf
import tensorflow.python.keras.backend as keras_backend


def get_depth_relation(depth1, depth2, threshold=None):
    if threshold is None:
        if depth1 > depth2:
            return 1
        elif depth1 < depth2:
            return -1
        else:
            return 0
    else:
        epsilon = 1e-10

        if (depth1 + epsilon) / (depth2 + epsilon) >= 1 + threshold:
            return 1
        elif (depth1 + epsilon) / (depth2 + epsilon) <= 1 / (1 + threshold):
            return -1
        else:
            return 0


def get_depth_relation_tf(depth_values, threshold, dtype=tf.float32):
    relation = tf.where(
        tf.greater_equal(
            (depth_values[:, 0] + keras_backend.epsilon()) / (depth_values[:, 1] + keras_backend.epsilon()),
            1. + threshold),
        tf.constant(1, dtype),
        tf.where(tf.greater(1. / (1. + threshold),
                            (depth_values[:, 0] + keras_backend.epsilon()) / (
                                    depth_values[:, 1] + keras_backend.epsilon())),
                 tf.constant(-1, dtype),
                 tf.constant(0, dtype)))

    return tf.cast(relation, tf.int8)


def prepare_fully_fledged_loss_input(labels, logits, batch_size, ranking_size, debug=False):
    labels = tf.cast(labels, dtype=tf.float32)
    logits = tf.cast(logits, dtype=tf.float32)

    rankings = tf.reshape(labels, [batch_size, -1, ranking_size, 2])

    pred_maps = tf.reshape(logits, [batch_size, -1])
    point_coords = tf.reshape(rankings[:, :, :, 0], [batch_size, -1])
    if debug:
        point_coords = tf.compat.v1.Print(point_coords, [point_coords], "point_coords:", summarize=10)

    selected_depths = tf.gather(pred_maps, tf.cast(point_coords, dtype=tf.int32), axis=1, batch_dims=1)

    selected_depths = tf.reshape(selected_depths, [-1, ranking_size])

    if debug:
        selected_depths = tf.compat.v1.Print(selected_depths, [selected_depths], "selected_depths:", summarize=10)

    reshaped_labels = tf.reshape(rankings[:, :, :, 1], [-1, ranking_size])
    if debug:
        reshaped_labels = tf.compat.v1.Print(reshaped_labels, [reshaped_labels], "reshaped_labels:", summarize=10)

    return selected_depths, reshaped_labels
