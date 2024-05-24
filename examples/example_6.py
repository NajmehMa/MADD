from nodes.node import Node
from graphs.graph import Graph
from edges.edge import Edge

# Import datasets
from data.mri_brain_image_segmentation_dataset import MRIBrainImageSegmentationDataset
from data.skull_image_segmentation_dataset import SkullImageSegmentationDataset
from data.age_classification_dataset import AgeDataset
from data.image_classification_dataset import ImageClassificationDataset
from data.gene_classification_dataset import GeneDataset
from data.surface_classification_dataset import SurfaceDataset
from data.thickness_classification_dataset import ThicknessDataset
from data.cognitive_classification_dataset import CognitiveDataset
from data.volume_classification_dataset import VolumeDataset
from data.pet_normalized_meas_dataset import PETNormalizedMeasDataset
# Import models
from models.brain_image_classification_model import BrainImageClassificationModel
from models.mri_brain_image_segmentation_model import MRIBrainImageSegmentationModel
from models.skull_image_segmentation_model import SkullImageSegmentationModel
from models.age_classification_model import AgeClassificationModel
from models.gene_classification_model import GeneClassificationModel
from models.surface_classification_model import SurfaceClassificationModel
from models.thickness_classification_model import ThicknessClassificationModel
from models.cognitive_classification_model import CognitiveClassificationModel
from models.volume_classification_model import VolumeClassificationModel
from models.pet_normalized_meas_classification_model import PETNormalizedMeasClassificationModel

# Initialize the graph
my_graph = Graph()

# Add nodes (datasets)
node1 = Node(1, VolumeDataset(data='/home/azargari/modular-metalearning/final_scripts/MADD/datasets/volume_dataset/'))
node2 = Node(2, MRIBrainImageSegmentationDataset(data='/home/azargari/modular-metalearning/final_scripts/MADD/datasets/mri_image_segmentation_dataset/'))
my_graph.add_node(node1)
my_graph.add_node(node2)

# Add an edge (model) between the nodes
edge1 = Edge(3, VolumeClassificationModel(ckpt="/home/azargari/modular-metalearning/final_scripts/MADD/chekpoints/volume_classifier_ckpt.pt"
                                       ,batch_size=2, val_batch_size=2,learning_rate=1e-3))
my_graph.add_edge(node1, node2, edge1)

# Display the graph
my_graph.display()


edge_id_list = [3]  # Replace with the id of the edge you want to train
my_graph.train_edge_models_by_id(edge_id_list)
my_graph.fine_tune_edge_models_by_id(edge_id_list,learning_rate=1e-4)
test_results=my_graph.test_edge_models_by_id(edge_id_list)
print(test_results)
predict_results=my_graph.predict_edge_models_by_id(edge_id_list,inference_sample="/home/azargari/modular-metalearning/final_scripts/MADD/inference_samples/volume_samples/inference_samples_1.csv")
print(predict_results)
