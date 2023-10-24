# edges/edge.py

class Edge:
    def __init__(self, id, model):
        self.id = id
        self.model = model  # This can be a reference or the actual model based on your need.

    def __repr__(self):
        return f"Edge(id={self.id})"
