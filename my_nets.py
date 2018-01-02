import tensorflow as tf
import numpy as np


class UNet2D:
    __down_params = []
    __down_nodes = []
    __up_params = []
    __up_nodes = []
    __shared_feature_maps = []
    __input_node = None
    __output_node = None
    __output_node_logit = None

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
    def nodes(self):
        return {'down': self.__down_nodes, 'up': self.__up_nodes}

    @property
    def input_node(self):
        return self.__input_node

    @property
    def output_node(self):
        return self.__output_node

    @property
    def output_node_logit(self):
        return self.__output_node_logit

    def __init__(self, input_size, depth=5, base_dim=64):
        self.__depth = depth
        self.__base_dim = base_dim
        self.__input_size = input_size
        self.__input_node = tf.placeholder(tf.float32, [None, input_size[0], input_size[1], input_size[2]], name="input")
        self.__create_down_layers()
        self.__create_up_layers()
        self.__create_output_layer()

    def __create_down_layers(self):
        self.__down_nodes = []

        for i in range(self.__depth):
            with tf.name_scope('DOWN-{}'.format(i)):
                if i > 0:
                    y0 = self.__down_nodes[-1]
                else:
                    y0 = self.__input_node

                input_size = int(y0.shape[-1])
                output_size = self.__base_dim * (2 ** i)

                with tf.name_scope('CONV'):
                    w1 = tf.Variable(tf.truncated_normal([3, 3, input_size, output_size], 0, 0.1), name='weights')
                    b1 = tf.Variable(tf.constant(0.1, shape=[output_size]), name='biases')
                    y1 = tf.nn.conv2d(y0, w1, strides=[1, 1, 1, 1], padding='SAME') + b1
                with tf.name_scope('RELU'):
                    y1 = tf.nn.relu(y1)
                self.__down_params.append({'n': 'Ld-{}-CONV-{}'.format(i, 1), 'w': w1, 'p': b1})

                with tf.name_scope('CONV'):
                    w2 = tf.Variable(tf.truncated_normal([3, 3, output_size, output_size], 0, 0.1), name='weights')
                    b2 = tf.Variable(tf.constant(0.1, shape=[output_size]), name='biases')
                    y2 = tf.nn.conv2d(y1, w2, strides=[1, 1, 1, 1], padding='SAME') + b2
                with tf.name_scope('RELU'):
                    y2 = tf.nn.relu(y2)
                self.__down_params.append({'n': 'Ld-{}-CONV-{}'.format(i, 2), 'w': w2, 'p': b2})

                if i < self.__depth - 1:
                    with tf.name_scope('POOL'):
                        y3 = tf.nn.max_pool(y2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
                else:
                    y3 = y2

            self.__shared_feature_maps.append(y2)
            self.__down_nodes.append(y3)

    def __create_up_layers(self):
        self.__up_nodes = []
        for i in range(self.__depth - 2, -1, -1):
            with tf.name_scope('UP-{}'.format(i)):
                if i < (self.__depth - 2):
                    y0 = self.__up_nodes[-1]
                else:
                    y0 = self.__down_nodes[-1]

                input_size = int(y0.shape[-1])
                output_size = self.__base_dim * (2 ** i)
                with tf.name_scope('UP-CONV'):
                    wd = tf.Variable(tf.truncated_normal([2, 2, output_size, input_size], 0, 0.1), name='weights')
                    bd = tf.Variable(tf.constant(0.1, shape=[output_size]), name='biases')
                    yd = tf.nn.conv2d_transpose(y0, wd, tf.stack([tf.shape(y0)[0],
                                                                  tf.shape(y0)[1]*2,
                                                                  tf.shape(y0)[2]*2,
                                                                  tf.shape(y0)[3]//2]),
                                                strides=[1, 2, 2, 1],
                                                padding="SAME") + bd
                    yd = tf.concat([self.__shared_feature_maps[i], yd], 3)
                self.__up_params.append({'n': 'Lu-{}-UPCONV-{}'.format(i, 1), 'w': wd, 'p': bd})

                with tf.name_scope('CONV'):
                    w1 = tf.Variable(tf.truncated_normal([3, 3, input_size, output_size], 0, 0.1), name='weights')
                    b1 = tf.Variable(tf.constant(0.1, shape=[output_size]), name='biases')
                    y1 = tf.nn.conv2d(yd, w1, strides=[1, 1, 1, 1], padding='SAME') + b1
                with tf.name_scope('RELU'):
                    y1 = tf.nn.relu(y1)
                self.__up_params.append({'n': 'Lu-{}-CONV-{}'.format(i, 1), 'w': w1, 'p': b1})

                with tf.name_scope('CONV'):
                    w2 = tf.Variable(tf.truncated_normal([3, 3, output_size, output_size], 0, 0.1), name='weights')
                    b2 = tf.Variable(tf.constant(0.1, shape=[output_size]), name='biases')
                    y2 = tf.nn.conv2d(y1, w2, strides=[1, 1, 1, 1], padding='SAME') + b2
                with tf.name_scope('RELU'):
                    y2 = tf.nn.relu(y2)
                self.__up_params.append({'n': 'Lu-{}-CONV-{}'.format(i, 2), 'w': w2, 'p': b2})
            self.__up_nodes.append(y2)

    def __create_output_layer(self):
        with tf.name_scope('OUTPUT'):
            y0 = self.__up_nodes[-1]
            input_size = int(y0.shape[-1])
            with tf.name_scope('CONV'):
                w1 = tf.Variable(tf.truncated_normal([1, 1, input_size, 2], 0, 0.1), name='weights')
                b1 = tf.Variable(tf.constant(0.1, shape=[2]), name='biases')
                Ylogit = tf.nn.conv2d(y0, w1, strides=[1, 1, 1, 1], padding='SAME') + b1
            with tf.name_scope('SOFTMAX'):
                Y = tf.nn.softmax(Ylogit, name="output")
            self.__up_params.append({'n': 'OUTPUT-CONV-{}'.format(1), 'w': w1, 'p': b1})
        self.__output_node_logit = Ylogit
        self.__output_node = Y

