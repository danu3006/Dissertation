from typing import List, Any, Callable, Dict

import tensorflow as tf

__author__ = "Danu Kumanan"
__version__ = "1.0.0"


class ANN:
    """ Artificial Neural Network class to auto-generate all hidden layer sizes and weights """

    __n_nodes: List[int] = list()
    __hidden_layers: List[Dict] = list()
    __output_layer: Dict = dict()
    __layer_outputs: List[Any] = list()
    __training_data: List[Any] = list()

    __x = None
    __y = None

    def __init__(self, *, n_nodes: List[int] = list(), n_classes: int = int(), training_data):
        """ Will generate hidden layers based on `n_nodes` and `n_classes`

        :param n_nodes: A list of node sizes starting from input layer
        :type n_nodes: List[int]
        :param n_classes: An integer to specify the number of output classes
        :type n_classes: int
        """
        self.__n_nodes = n_nodes
        self.__hidden_layers = list()
        self.__output_layer = dict()
        self.__training_data = training_data

        self.__x = tf.placeholder('float', [None, self.__n_nodes[0]])
        self.__y = tf.placeholder('float')

        self.__n_nodes.append(n_classes)
        self._generate_layers()

    def _generate_layers(self):
        """ Generates all hidden and output layers based on class arguments """

        for layer, nodes in enumerate(self.__n_nodes[:-1]):
            self.__hidden_layers.append({
                'weights': tf.Variable(tf.random_normal([nodes, self.__n_nodes[layer + 1]])),
                'biases': tf.Variable(tf.random_normal([self.__n_nodes[layer + 1]]))
            })
        else:
            self.__output_layer = {
                'weights': tf.Variable(tf.random_normal([self.__n_nodes[-2], self.__n_nodes[-1]])),
                'biases': tf.Variable(tf.random_normal([self.__n_nodes[-1]]))
            }

    def get_hidden_layer(self, *, layer=int()):
        """ Gets and returns the dictionary associated with the relevant hidden layer

        :param layer: An integer > 0 denoting the hidden layer
        :type layer: int
        :return: A dictionary of the weights and biases
        :rtype: dict
        """
        try:
            return self.__hidden_layers[layer - 1]
        except IndexError:
            print('min=0, max=' + str(len(self.__hidden_layers) - 1))

    def get_output_layer(self):
        """ Gets and returns the output layer dictionary

        :return: The output layer dictionary
        :rtype: dict
        """
        return self.__output_layer

    def feed_forward(self, input_data, *, activation: Callable):
        """ Executes one feed-forward loop with given data and activation

        :param input_data: A tensor with input data to feed-forward
        :type input_data: tensor
        :param activation: The type of activation eg. RELU
        :type activation: Callable
        :return: A tensor containing the post-activation data of the network
        :rtype: tensor
        """
        for layer, data in enumerate(self.__hidden_layers):
            if layer is 0:
                pre_activation = tf.add(tf.matmul(input_data, data['weights']), data['biases'])
            else:
                pre_activation = tf.add(tf.matmul(self.__hidden_layers[layer - 1], data['weights']), data['biases'])

            self.__layer_outputs.append(activation(pre_activation))

        matrix_mult = tf.matmul(self.__layer_outputs[-1], self.__output_layer['weights'])
        return tf.add(matrix_mult, self.__output_layer['biases'])

    def train_network(self, input_data, learning_rate):
        pass
