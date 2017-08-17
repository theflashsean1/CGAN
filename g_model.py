import tensorflow as tf
import ops


class Generator:
    def __init__(self, is_training, fc_dim=784, ngf=128, output_size=28, channel_dim=1, name='generator'):
        self._name = name
        self._is_training = is_training
        self._output_size = output_size
        self._reuse = False
        self._fc_dim = fc_dim
        self._ngf = ngf
        self._channel_dim = channel_dim
        self._variables = None

    def __call__(self, z, y):
        """
        :param z: 2D [batch_size, z_dim]
        :param y: 2D [batch_size, y_dim]
        :return:
        """
        batch_size, y_dim = y.get_shape().as_list()
        batch_size_, z_dim = z.get_shape().as_list()
        assert batch_size == batch_size_
        h1_size = int(self._output_size / 4)
        h2_size = int(self._output_size / 2)

        with tf.variable_scope(self._name):
            yb = tf.reshape(y, shape=[-1, 1, 1, y_dim])  # (100, 1, 1, 10)

            z = tf.concat([z, y], axis=1)  # (batch_size=100, y_dim+z_dim=110)
            h0 = tf.nn.relu(
                ops.batch_norm(
                    ops.fc(z, self._fc_dim, reuse=self._reuse, name='g_fc0'),
                    is_training=self._is_training,
                    reuse=self._reuse,
                    name_scope='g_bn0'
                )
            )
            h0 = tf.concat([h0, y], axis=1)  # (batch_size=100, fc_dim+y_dim=794)

            h1 = tf.nn.relu(
                ops.batch_norm(
                    ops.fc(h0, self._ngf*h1_size*h1_size, reuse=self._reuse, name='g_fc1'),
                    is_training=self._is_training,
                    reuse=self._reuse,
                    name_scope='g_bn1'
                )
            )
            h1 = tf.reshape(h1, shape=[-1, h1_size, h1_size, self._ngf])
            h1 = tf.concat([h1, yb*tf.ones([batch_size, h1_size, h1_size, y_dim])], axis=3)  # (100, 7, 7, 522)

            h2 = tf.nn.relu(
                ops.batch_norm(
                    ops.deconv2d(h1, self._ngf, reuse=self._reuse, name='g_conv2'),
                    is_training=self._is_training,
                    reuse=self._reuse,
                    name_scope='g_bn2'
                )
            )
            h2 = tf.concat([h2, yb*tf.ones([batch_size, h2_size, h2_size, y_dim])], axis=3)  # (100, 14, 14, 522)
            h3 = tf.nn.sigmoid(
                ops.deconv2d(h2, self._channel_dim, reuse=self._reuse, name='g_conv3')
            )  # TODO DIMENSION??? SHRINK
        self._reuse = True
        return h3  # (100, 28, 28, 1)

    @property
    def variables(self):
        if not self._variables:
            self._variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self._name)
        return self._variables

