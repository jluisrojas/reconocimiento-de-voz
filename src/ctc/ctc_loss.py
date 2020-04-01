import tensorflow as tf


@tf.function
def get_loss(y_actual, y_prediccion):
    labels, long_labels, long_frames = y_actual

    #long_labels = tf.reshape(long_labels, [-1])
    #long_frames = tf.reshape(long_frames, [-1])

    #loss = tf.nn.ctc_loss(labels, y_prediccion, long_labels, long_frames, logits_time_major=False, blank_index=-1)
    loss = tf.keras.backend.ctc_batch_cost(labels, y_prediccion, long_labels, long_frames)

    loss = tf.reduce_mean(loss)

    return loss
