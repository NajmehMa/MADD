import networkx as nx
from nodes.node import Node  # Adjust the import statement if your project structure is different.
from edges.edge import Edge  # Adjust the import statement if your project structure is different.

class Graph:
    def __init__(self):
        self.graph = nx.DiGraph()

    def add_node(self, node):
        self.graph.add_node(node.id, data=node)

    def add_edge(self, start_node, end_node, edge):
        self.graph.add_edge(start_node.id, end_node.id, data=edge)

    def display(self):
        print("Nodes:")
        for node in self.graph.nodes(data=True):
            print(node)

        print("Edges:")
        for start, end, edge in self.graph.edges(data=True):
            print(f"{start} -> {end}, {edge}")

    def train_edge_models_by_id(self, edge_id_list):

        for edge_id in edge_id_list:
            # Iterating over all edges to find the edge with the specific id
            for start, end, edge_data in self.graph.edges(data=True):
                edge = edge_data['data']

                # Check if the current edge id matches the given edge_id
                if edge.id == edge_id:
                    start_node_data = self.graph.nodes[start]['data']
                    dataset = start_node_data.dataset  # Extracting the dataset from the start node

                    # Assuming the model inside the edge has an input_dataset method and a train method
                    edge.model.input_dataset(dataset)
                    edge.model.train()
                    break  # Exit the loop once the model with the specific id is trained

    def fine_tune_edge_models_by_id(self, edge_id_list,learning_rate):
        for edge_id in edge_id_list:
            # Iterating over all edges to find the edge with the specific id
            for start, end, edge_data in self.graph.edges(data=True):
                edge = edge_data['data']

                # Check if the current edge id matches the given edge_id
                if edge.id == edge_id:
                    start_node_data = self.graph.nodes[start]['data']
                    dataset = start_node_data.dataset  # Extracting the dataset from the start node

                    # Assuming the model inside the edge has an input_dataset method and a train method
                    edge.model.input_dataset(dataset)
                    edge.model.fine_tune(learning_rate=learning_rate)
                    break  # Exit the loop once the model with the specific id is trained

    def test_edge_models_by_id(self, edge_id_list):
        test_results={}
        for edge_id in edge_id_list:
            # Iterating over all edges to find the edge with the specific id
            for start, end, edge_data in self.graph.edges(data=True):
                edge = edge_data['data']

                # Check if the current edge id matches the given edge_id
                if edge.id == edge_id:
                    start_node_data = self.graph.nodes[start]['data']
                    dataset = start_node_data.dataset  # Extracting the dataset from the start node

                    # Assuming the model inside the edge has an input_dataset method and a test method
                    edge.model.input_dataset(dataset)
                    results=edge.model.test()
                    test_results[edge_id]=results
                    break  # Exit the loop once the model with the specific id is tested
        return test_results # returns the results with thier edge node ids as the key
    def predict_edge_models_by_id(self, edge_id_list,inference_sample):
        test_results={}
        for idx,edge_id in enumerate(edge_id_list):
            # Iterating over all edges to find the edge with the specific id
            for start, end, edge_data in self.graph.edges(data=True):
                edge = edge_data['data']

                # Check if the current edge id matches the given edge_id
                if edge.id == edge_id:
                    start_node_data = self.graph.nodes[start]['data']
                    dataset = start_node_data.dataset  # Extracting the dataset from the start node

                    # Assuming the model inside the edge has an input_dataset method and a predict method
                    edge.model.input_dataset(dataset)
                    results=edge.model.predict(inference_sample)
                    inference_sample=results['preds']
                    test_results[edge_id] = results
                    break  # Exit the loop once the model with the specific id is called
        return test_results# returns the results with thier edge node ids as the key