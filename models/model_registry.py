from models.hierarchical_rnn import HierarchicalRNN
from models.graph_nn import GraphNN
from models.temporal_transformer import TemporalTransformer
from models.kmeans_base_model import KMeansBaseModel

def get_model(model_name):
    registry = {
        "hierarchical_rnn": HierarchicalRNN,
        "graph_nn": GraphNN,
        "temporal_transformer": TemporalTransformer,
        "kmeans_base_model": KMeansBaseModel
    }
    return registry[model_name]