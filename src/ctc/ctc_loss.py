import tensorflow as tf


@tf.function
def get_loss(y_actual, y_prediccion):
    b, f, _ = y_prediccion.get_shape()
    labels, long_labels, long_frames = y_actual
    long_frames = tf.fill([b, 1], f)

    long_labels = tf.reshape(long_labels, [-1])
    long_frames = tf.reshape(long_frames, [-1])

    loss = tf.nn.ctc_loss(labels, y_prediccion, long_labels,
            long_frames, logits_time_major=False, blank_index=-1)

    return tf.reduce_mean(loss)
