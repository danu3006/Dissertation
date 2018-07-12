from tensorflow.examples.tutorials.mnist import input_data

# mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

n_nodes = [784, 500, 500, 500, ]
n_classes = 10
batch_size = 100

n_nodes.append(n_classes)

x = tf.placeholder('float', [None, 784])
y = tf.placeholder('float')


def neural_network(data):
    for layer, nodes in enumerate(n_nodes[1:-1]):
        globals()['hidden_layer_' + str(layer)] = {
            'weights': tf.Variable(tf.random_normal([nodes, n_nodes[layer + 1]])),
            'biases': tf.Variable(tf.random_normal([n_nodes[layer + 1]]))
        }


    return  hidden_layer_1['weights'],


with tf.Session() as sess:
    print(sess.run(neural_network('da')))
