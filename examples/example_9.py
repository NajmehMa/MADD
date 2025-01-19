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
node1 = Node(1, SkullImageSegmentationDataset(data='/home/azargari/modular-metalearning/final_scripts/MADD/datasets/mri_skull_image_segmentation_dataset/'))
node2 = Node(3, MRIBrainImageSegmentationDataset(data='/home/azargari/modular-metalearning/final_scripts/MADD/datasets/mri_image_segmentation_dataset/'))
node3 = Node(5, ImageClassificationDataset(data='/home/azargari/modular-metalearning/final_scripts/MADD/datasets/mri_image_classification_dataset/'))
my_graph.add_node(node1)
my_graph.add_node(node2)
my_graph.add_node(node3)


# Add an edge (model) between the nodes
edge1 = Edge(2, SkullImageSegmentationModel(ckpt="/home/azargari/modular-metalearning/final_scripts/MADD/chekpoints/mri_skull_image_segmentation_ckpt.pt"
                                       ,batch_size=16, val_batch_size=16,learning_rate=1e-4))

edge2 = Edge(4, MRIBrainImageSegmentationModel(ckpt="/home/azargari/modular-metalearning/final_scripts/MADD/chekpoints/mri_brain_image_segmentation_ckpt.pt"
                                       ,batch_size=16, val_batch_size=16,learning_rate=1e-4))

edge3 = Edge(6, BrainImageClassificationModel(ckpt="/home/azargari/modular-metalearning/final_scripts/MADD/chekpoints/segmented_mri_image_classifier_ckpt.pt"
                                       ,batch_size=16, val_batch_size=16,learning_rate=1e-4))

my_graph.add_edge(node1, node2, edge1)
my_graph.add_edge(node2, node3, edge2)
my_graph.add_edge(node3, node3, edge3)
# Display the graph
my_graph.display()


edge_id_list = [2,4,6] # Replace with the id of the edges you want to train
my_graph.train_edge_models_by_id(edge_id_list)
my_graph.fine_tune_edge_models_by_id(edge_id_list,learning_rate=1e-4)
test_results=my_graph.test_edge_models_by_id(edge_id_list)
print(test_results)
predict_results=my_graph.predict_edge_models_by_id(edge_id_list,inference_sample='/home/azargari/modular-metalearning/final_scripts/MADD/inference_samples/mri_skull_image_segmentation_samples/inference_samples_1/')
print(predict_results)