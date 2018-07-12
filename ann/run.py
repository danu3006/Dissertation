from ann.src.ANN.ANN import ANN

from tensorflow import keras

(train_images, train_labels), (test_images, test_labels) = keras.datasets.mnist.load_data()

train_labels = train_labels[:1000]
test_labels = test_labels[:1000]

train_images = train_images[:1000].reshape(-1, 28 * 28) / 255.0
test_images = test_images[:1000].reshape(-1, 28 * 28) / 255.0

n_nodes = [784, 500, 500, 500]
n_classes = 10
batch_size = 100

neural_network = ANN(n_nodes=n_nodes, n_classes=n_classes, training_data=None)

print(neural_network.__init__.__doc__)
