from typing import List

from .ANN import ANN

__author__ = "Danu Kumanan"
__version__ = "1.0.0"


class NCNN(ANN):

    def __init__(self, *, n_nodes: List[int] = list(), n_classes: int = int(), training_data):
        super().__init__(n_nodes=n_nodes, n_classes=n_classes, training_data=training_data)

    def train_network(self, input_data, learning_rate):
        pass
