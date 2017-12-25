import tensorflow as tf


class UNet2D:
    __down_params = []
    __down_logit = []
    __up_params = []
    __up_logit = []
    __input_node = None
    __output_node = None

    @property
    def input_size(self):
        return self.__input_size

    @property
    def depth(self):
        return self.__depth

    @property
    def base_dim(self):
        return self.__base_dim

    @property
    def params(self):
        return {'down': self.__down_params, 'up': self.__up_params}

    @property
    def logits(self):
        return {'down': self.__down_logit, 'up': self.__up_logit}

    @property
    def input_node(self):
        return self.__input_node

    @property
    def output_node(self):
        return self.__output_node

    def __init__(self, input_size, depth=5, base_dim=64):
        self.__depth = depth
        self.__base_dim = base_dim
        self.__input_size = input_size
        self.__input_node = tf.placeholder(tf.float32, [None, input_size[0], input_size[1], input_size[2]])
        self.__create_down_layers()
        self.__create_up_layers()
        self.__create_output_layer()

    def __create_down_layers(self):
        self.__down_logit = []

        for i in range(self.__depth):
            with tf.name_scope('DOWN-{}'.format(i)):
                if i > 0:
                    y0 = self.__down_logit[-1]
                    with tf.name_scope('POOL'):
                        y0 = tf.nn.max_pool(y0, [1, 2, 2, 1], [1, 2, 2, 1], padding='VALID')
                else:
                    y0 = self.__input_node

                input_size = int(y0.shape[-1])
                output_size = self.__base_dim * (2 ** i)

                with tf.name_scope('CONV'):
                    w1 = tf.Variable(tf.truncated_normal([3, 3, input_size, output_size], 0, 0.1), name='weights')
                    b1 = tf.Variable(tf.ones([output_size])/10, name='biases')
                    y1 = tf.nn.conv2d(y0, w1, strides=[1, 1, 1, 1], padding='SAME') + b1
                    y1 = tf.nn.relu(y1)
                    self.__down_params.append({'n': 'Ld-{}-CONV-{}'.format(i, 1), 'w': w1, 'p': b1})

                with tf.name_scope('CONV'):
                    w2 = tf.Variable(tf.truncated_normal([3, 3, output_size, output_size], 0, 0.1), name='weights')
                    b2 = tf.Variable(tf.ones([output_size])/10, name='biases')
                    y2 = tf.nn.conv2d(y1, w2, strides=[1, 1, 1, 1], padding='SAME') + b2
                    y2 = tf.nn.relu(y2)
                    self.__down_params.append({'n': 'Ld-{}-CONV-{}'.format(i, 2), 'w': w2, 'p': b2})
            self.__down_logit.append(y2)

    def __create_up_layers(self):
        self.__up_logit = []
        for i in range(self.__depth - 2, -1, -1):
            with tf.name_scope('UP-{}'.format(i)):
                if i < self.__depth - 2:
                    y0 = self.__up_logit[-1]
                else:
                    y0 = self.__down_logit[-1]

                input_size = int(y0.shape[-1])
                output_size = self.__base_dim * (2 ** i)
                with tf.name_scope('UP-CONV'):
                    wd = tf.Variable(tf.truncated_normal([2, 2, output_size, input_size], 0, 0.1), name='weights')
                    bd = tf.Variable(tf.constant(0.1), name='biases')
                    yd = tf.nn.conv2d_transpose(y0, wd, tf.stack([tf.shape(y0)[0],
                                                                  tf.shape(y0)[1]*2,
                                                                  tf.shape(y0)[2]*2,
                                                                  tf.shape(y0)[3]//2]),
                                                strides=[1, 2, 2, 1],
                                                padding="SAME") + bd
                    self.__up_params.append({'n': 'Lu-{}-UPCONV-{}'.format(i, 1), 'w': wd, 'p': bd})
                    yd = tf.concat([self.__down_logit[i], yd], 3)

                with tf.name_scope('CONV'):
                    w1 = tf.Variable(tf.truncated_normal([3, 3, input_size, output_size], 0, 0.1), name='weights')
                    b1 = tf.Variable(tf.constant(0.1), name='biases')
                    y1 = tf.nn.conv2d(yd, w1, strides=[1, 1, 1, 1], padding='SAME') + b1
                    y1 = tf.nn.relu(y1)
                    self.__up_params.append({'n': 'Lu-{}-CONV-{}'.format(i, 1), 'w': w1, 'p': b1})

                with tf.name_scope('CONV'):
                    w2 = tf.Variable(tf.truncated_normal([3, 3, output_size, output_size], 0, 0.1), name='weights')
                    b2 = tf.Variable(tf.constant(0.1), name='biases')
                    y2 = tf.nn.conv2d(y1, w2, strides=[1, 1, 1, 1], padding='SAME') + b2
                    y2 = tf.nn.relu(y2)
                    self.__up_params.append({'n': 'Lu-{}-CONV-{}'.format(i, 2), 'w': w2, 'p': b2})
            self.__up_logit.append(y2)

    def __create_output_layer(self):
        with tf.name_scope('OUTPUT'):
            y0 = self.__up_logit[-1]
            input_size = int(y0.shape[-1])
            with tf.name_scope('CONV'):
                w1 = tf.Variable(tf.truncated_normal([1, 1, input_size, 2], 0, 0.1), name='weights')
                b1 = tf.Variable(tf.ones([2]) / 10, name='biases')
                y1 = tf.nn.conv2d(y0, w1, strides=[1, 1, 1, 1], padding='SAME') + b1
            self.__up_params.append({'n': 'OUTPUT-CONV-{}'.format(1), 'w': w1, 'p': b1})
        self.__output_node = y1
