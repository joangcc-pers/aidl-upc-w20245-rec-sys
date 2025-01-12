from models.hierarchical_rnn import HierarchicalRNN
from models.graph_nn import GraphNN
from models.temporal_transformer import TemporalTransformer

def get_model(model_name):
    registry = {
        "hierarchical_rnn": HierarchicalRNN,
        "graph_nn": GraphNN,
        "temporal_transformer": TemporalTransformer,
    }
    return registry[model_name]