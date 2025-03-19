import sys
sys.path.append('../')  # Ajusta la ruta si es necesario
from pycore.tikzeng import *  # Importa las herramientas para dibujar redes neuronales


def create_ggnn_attention_diagram():
    arch = [
        to_head('..'),
        to_cor(),
        to_begin(),

        # Capa de Node Embedding
        to_Conv(name="embedding",
                        s_filer=100, # hidden_dim
                        n_filer=321, # 5 ja que passem 5 embeddings (category, sub_category, elements, brand i product_id). 64 ja que es la embedding dim. 64*5 + 1 del price tensor
                        offset="(0,0,0)", to="(0,0,0)", 
                        width=10, height=8, depth=8, 
                        caption="Node Embedding"),
                        
        
        # Capa GNN
        to_Conv(name="gnn", s_filer=100, n_filer=321, 
                        offset="(2,0,0)", to="(embedding-east)", 
                        width=10, height=8, depth=8, 
                        caption="GGNN"),
        
        to_connection("embedding", "gnn"),
        
        # Embedding del penúltimo ítem
        to_Conv(name="penultimate_item", s_filer=100, n_filer=321, 
                offset="(0,-10,0)", to="(gnn-east)", 
                width=10, height=8, depth=8, caption="Penultimate Item Emb. (through Scatter Max)"),
        
        # 🔹 Conexión desde Node Embedding hacia Penultimate Item Embedding
        to_connection("gnn", "penultimate_item"),


        # 🔹 Mecanismo de Self-Attention para el penúltimo ítem        
        to_Conv(name="attention_fc", s_filer=1, n_filer=321, 
                offset="(2,0,0)", to="(penultimate_item-east)", 
                width=10, height=8, depth=1, 
                caption="Self-Attention FC"),
        # 🔹 Conexión del penúltimo ítem a la atención (ahora es válida)
        to_connection("penultimate_item", "attention_fc"),

        # Nueva capa para representar la expansión de embeddings
        to_Conv(name="expand_last_item_embeddings", s_filer=100, n_filer=321, 
                offset="(2,0,0)", to="(attention_fc-east)", 
                width=10, height=8, depth=8, 
                caption="Expand Embeddings"),

        # Conectar Self-Attention FC a la expansión de embeddings


        to_connection("attention_fc", "expand_last_item_embeddings"),
        to_Sum(name="add_embeddings",
               offset="(2,10,0)", to="(expand_last_item_embeddings-east)"
               ),

        # Conectar la capa expandida a la suma final
        to_connection("expand_last_item_embeddings", "add_embeddings"),
        to_connection("gnn", "add_embeddings"),

        # Attentional Aggregation prework: dropout
        
        # Attentional Aggregation prework: Linear Layer
        to_Conv(name="attentional_aggregation_fc1", s_filer=100, n_filer=321, 
                offset="(1,0,0)", to="(add_embeddings-east)", 
                width=10, height=8, depth=8, 
                caption="FC1 (with preliminary dropout)"),
        to_connection("add_embeddings", "attentional_aggregation_fc1"),

        # Attentional Aggregation prework: ReLU
        to_ConvSoftMax(name="relu", s_filer=100,
                offset="(2,0,0)", to="(attentional_aggregation_fc1-east)", 
                width=10, height=8, depth=8, 
                caption="ReLU"),
        to_connection("attentional_aggregation_fc1", "relu"),

        # Attentional Aggregation prework: Linear Layer
        to_ConvSoftMax(name="attentional_aggregation_fc2",
                s_filer=100, 
                offset="(2,0,0)", to="(relu-east)", 
                width=10, height=8, depth=8, 
                caption="FC2"),
        to_connection("relu", "attentional_aggregation_fc2"),
        # # Attentional Aggregation
        # to_ConvSoftMax(name="attentional_aggregation_fc3", s_filer=100,
        #         offset="(2,0,0)", to="(attentional_aggregation_fc2-east)", 
        #         width=10, height=8, depth=8, 
        #         caption="Attentional Aggregation"),
        # to_connection("attentional_aggregation_fc2", "attentional_aggregation_fc3"),
        
        # Capa completamente conectada para puntuaciones
        to_Conv(name="fc", s_filer=71441, n_filer=1, 
                offset="(3,0,0)", to="(attentional_aggregation_fc2-east)", 
                width=10, height=6, depth=6, caption="Fully Connected: scores for each product"),
        
        # to_connection("attentional_aggregation_fc3", "fc"),
        to_connection("attentional_aggregation_fc2", "fc"),
        
        to_end()
    ]
    
    return arch


def main():
    arch = create_ggnn_attention_diagram()
    to_generate(arch, "./pyexamples/ggnn_att_agg_arch.tex")
    
if __name__ == "__main__":
    main()