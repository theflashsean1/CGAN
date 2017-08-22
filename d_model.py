import tensorflow as tf
import ops


class Discriminator:
    def __init__(self, is_training, fc_dim=784, ndf=64, input_size=28, name='discriminator'):
        self._name = name
        self._is_training = is_training
        self._input_size = input_size
        self._fc_dim = fc_dim
        self._ndf = ndf
        self._reuse = False
        self._variables = None

    def __call__(self, input_, y):
        batch_size, y_dim = y.get_shape().as_list()
        batch_size_, height, width, c_dim = input_.get_shape().as_list()
        assert batch_size == batch_size_
        assert (self._input_size == width) and (self._input_size == height)
        h0_size = int(self._input_size / 2)
        h1_size = int(self._input_size / 4)

        with tf.variable_scope(self._name):
            yb = tf.reshape(y, shape=[-1, 1, 1, y_dim])
            # dim(x) = (100, 28, 28, 11)
            x = tf.concat([input_, yb*tf.ones([batch_size, self._input_size, self._input_size, y_dim])], axis=3)
            h0 = ops.leaky_relu(
                ops.conv2d(x, c_dim + y_dim, reuse=self._reuse, name='d_conv0'),
                slope=0.2
            )
            h0 = tf.concat([h0, yb*tf.ones([batch_size, h0_size, h0_size, y_dim])], axis=3)  # (100, 14, 14, 21)

            h1 = ops.leaky_relu(
                ops.batch_norm(
                    ops.conv2d(h0, c_dim + self._ndf, reuse=self._reuse, name='d_conv1'),
                    is_training=self._is_training,
                    reuse=self._reuse,
                    name_scope='d_bn1'
                ),
                slope=0.2
            )
            h1 = tf.reshape(h1, [batch_size, h1_size*h1_size*(c_dim+self._ndf)])
            h1 = tf.concat([h1, y], axis=1)  # (100, 28*28*(1+64)+10)

            h2 = ops.leaky_relu(
                ops.batch_norm(
                    ops.fc(h1, self._fc_dim, reuse=self._reuse, name='d_fc2'),
                    is_training=self._is_training,
                    reuse=self._reuse,
                    name_scope='d_bn2'
                ),
                slope=0.2
            )
            h2 = tf.concat([h2, y], axis=1)  # (100, 794)
            # h3 = tf.nn.sigmoid(
            h3 = ops.fc(h2, 1, reuse=self._reuse, name='d_fc3')
            # )
        self._reuse = True
        return h3  # (100, 1)

    @property
    def variables(self):
        if not self._variables:
            self._variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                                scope=self._name)
        return self._variables


