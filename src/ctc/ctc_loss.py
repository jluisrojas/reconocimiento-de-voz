import tensorflow as tf


def get_loss(y_actual, y_prediccion):
    labels, long_labels, long_frames = y_actual

    long_labels = tf.reshape(long_labels, [-1])
    print(long_labels.shape)
    long_frames = tf.reshape(long_frames, [-1])

    loss = tf.nn.ctc_loss(labels, y_prediccion, long_labels,
            long_frames, logits_time_major=False, blank_index=-1)

    return loss
