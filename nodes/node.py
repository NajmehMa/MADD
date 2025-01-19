# nodes/node.py

class Node:
    def __init__(self, id, dataset):
        self.id = id
        self.dataset = dataset  # This will be an instance of one of the dataset classes

    def __repr__(self):
        return f"Node(id={self.id})"

