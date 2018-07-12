from ann.src.ANN.ANN import ANN

n_nodes = [784, 500, 500, 500]
n_classes = 10
batch_size = 100

neural_network = ANN(n_nodes=n_nodes, n_classes=n_classes, training_data=None)

print(neural_network.__init__.__doc__)
