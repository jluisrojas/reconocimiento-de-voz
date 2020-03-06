import tensorflow as tf


def get_loss(y_actual, y_prediccion):
    labels = y_actual[0]
    long_labels = y_actual[1]
    long_frames = y_actual[2]

    loss = tf.nn.ctc_loss(labels, y_prediccion, long_labels,
            long_frames, logits_time_major=False, blank_index=-1)

    return loss
