import tensorflow as tf


class UNet2D:

    # down_weights = []
    # down_biases = []
    down_logit = []

    # up_weights = []
    # up_biases = []
    up_logit = []

    input_node = None
    output_node = None

    def __init__(self, input_size, depth=5, base_dim=64):
        self.depth = depth
        self.base_dim = base_dim
        self.input_size = input_size

        self.input_node = tf.placeholder(tf.float32, [None, input_size[0], input_size[1], input_size[2]])

        self.__create_down_layers()
        self.__create_up_layers()
        self.__create_output_layer()

    def __create_down_layers(self):
        self.down_logit = []

        for i in range(self.depth):
            with tf.name_scope('DOWN-{}'.format(i)):
                if i > 0:
                    y0 = self.down_logit[-1]
                    with tf.name_scope('POOL'):
                        y0 = tf.nn.max_pool(y0, [1, 2, 2, 1], [1, 2, 2, 1], padding='VALID')
                else:
                    y0 = self.input_node

                input_size = int(y0.shape[-1])
                output_size = self.base_dim*(2**i)

                with tf.name_scope('CONV'):
                    w1 = tf.Variable(tf.truncated_normal([3, 3, input_size, output_size], 0, 0.1), name='weights')
                    b1 = tf.Variable(tf.ones([output_size])/10, name='biases')
                    y1 = tf.nn.conv2d(y0, w1, strides=[1, 1, 1, 1], padding='SAME') + b1
                    y1 = tf.nn.relu(y1)

                with tf.name_scope('CONV'):
                    w2 = tf.Variable(tf.truncated_normal([3, 3, output_size, output_size], 0, 0.1), name='weights')
                    b2 = tf.Variable(tf.ones([output_size])/10, name='biases')
                    y2 = tf.nn.conv2d(y1, w2, strides=[1, 1, 1, 1], padding='SAME') + b2
                    y2 = tf.nn.relu(y2)

            self.down_logit.append(y2)

    def __create_up_layers(self):
        self.up_logit = []

        for i in range(self.depth - 2, -1, -1):
            with tf.name_scope('UP-{}'.format(i)):
                if i < self.depth - 2:
                    y0 = self.up_logit[-1]
                else:
                    y0 = self.down_logit[-1]

                input_size = int(y0.shape[-1])
                output_size = self.base_dim*(2**i)

                with tf.name_scope('UP-CONV'):
                    wd = tf.Variable(tf.truncated_normal([2, 2, output_size, input_size], 0, 0.1), name='weights')
                    bd = tf.Variable(tf.constant(0.1), name='biases')
                    yd = tf.nn.conv2d_transpose(y0, wd, tf.stack([tf.shape(y0)[0],
                                                                  tf.shape(y0)[1]*2,
                                                                  tf.shape(y0)[2]*2,
                                                                  tf.shape(y0)[3]//2]),
                                                strides=[1, 2, 2, 1],
                                                padding="SAME") + bd

                    yd = tf.concat([self.down_logit[i], yd], 3)

                with tf.name_scope('CONV'):
                    w1 = tf.Variable(tf.truncated_normal([3, 3, input_size, output_size], 0, 0.1), name='weights')
                    b1 = tf.Variable(tf.constant(0.1), name='biases')
                    y1 = tf.nn.conv2d(yd, w1, strides=[1, 1, 1, 1], padding='SAME') + b1
                    y1 = tf.nn.relu(y1)

                with tf.name_scope('CONV'):
                    w2 = tf.Variable(tf.truncated_normal([3, 3, output_size, output_size], 0, 0.1), name='weights')
                    b2 = tf.Variable(tf.constant(0.1), name='biases')
                    y2 = tf.nn.conv2d(y1, w2, strides=[1, 1, 1, 1], padding='SAME') + b2
                    y2 = tf.nn.relu(y2)

            self.up_logit.append(y2)

    def __create_output_layer(self):
        with tf.name_scope('OUTPUT'):
            y0 = self.up_logit[-1]
            input_size = int(y0.shape[-1])

            with tf.name_scope('CONV'):
                w1 = tf.Variable(tf.truncated_normal([1, 1, input_size, 2], 0, 0.1), name='weights')
                b1 = tf.Variable(tf.ones([2]) / 10, name='biases')
                y1 = tf.nn.conv2d(y0, w1, strides=[1, 1, 1, 1], padding='SAME') + b1

        self.output_node = y1
