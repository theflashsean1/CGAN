import tensorflow as tf


"""
Interface functions
"""


def conv2d(input_, num_feature_maps, filter_h=5, filter_w=5, stride=2, stddev=0.02,
           reuse=False, name="conv2d"):
    with tf.variable_scope(name, reuse=reuse):
        w = _weights('filter_weights',
                     [filter_h, filter_w, input_.get_shape()[3], num_feature_maps], stddev=stddev)
        b = _biases('filter_biases', [num_feature_maps])
        conv = tf.nn.bias_add(
            tf.nn.conv2d(input_, filter=w, strides=[1, stride, stride, 1], padding='SAME'),
            b
        )
    return conv


def deconv2d(input_, num_feature_maps, filter_h=5, filter_w=5, stride=2, stddev=0.02,
             reuse=False, name='deconv2d'):
    with tf.variable_scope(name, reuse=reuse):
        input_shape = input_.get_shape().as_list()
        w = _weights("filter_weights", [filter_h, filter_w, num_feature_maps, input_shape[3]], stddev=stddev)
        b = _biases('filter_biases', [num_feature_maps])
        deconv = tf.nn.bias_add(
            value=tf.nn.conv2d_transpose(input_, filter=w, strides=[1, stride, stride, 1], padding='SAME',
                                         output_shape=[
                                             input_shape[0],
                                             input_shape[1]*stride,
                                             input_shape[2]*stride, num_feature_maps
                                         ]),
            bias=b
        )
    return deconv


def fc(input_, output_dim, stddev=0.02, bias_init=0.0, reuse=False, name='fc'):
    """
    input_: ONLY 2D input, (batch_size, input_dim) is allowed
    output_dim:
    :return: 2D output (batch_size, output_dim)
    """
    with tf.variable_scope(name, reuse=reuse):
        w = _weights("fc_weights", [input_.get_shape()[1], output_dim], stddev=stddev)
        b = _biases("fc_biases", [output_dim], constant=bias_init)
        output_ = tf.nn.bias_add(tf.matmul(input_, w), b)
    return output_


def batch_norm(input_batch, is_training, epsilon=1e-5, decay=0.9, name_scope='bn', reuse=False):
    # TODO NEEDS MORE REVIEW ON THE PARAMETERS FOR batch_norm func  SCOPE???
    return tf.contrib.layers.batch_norm(
        inputs=input_batch,
        decay=decay,
        scale=True,
        scope=name_scope,
        reuse=reuse,
        updates_collections=None,
        is_training=is_training
    )


def leaky_relu(input_, slope):
    return tf.maximum(slope*input_, input_)


"""
The following helper functions should be only called within a tf scope
"""


def _weights(name, shape, mean=0.0, stddev=0.02):
    return tf.get_variable(name, shape, initializer=tf.truncated_normal_initializer(mean, stddev))


def _biases(name, shape, constant=0.0):
    return tf.get_variable(name, shape, initializer=tf.constant_initializer(constant))




