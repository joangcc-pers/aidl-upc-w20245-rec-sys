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

        # 🔹 Mecanismo de Self-Attention para todos los ítems menos el penúltimo
        to_Conv(name="attention_fc1", s_filer=100, n_filer=321, 
                        offset="(2,0,0)", to="(gnn-east)", 
                        width=10, height=8, depth=8, 
                        caption="Self-Attention FC1"),
        
        to_connection("gnn", "attention_fc1"),
        
        # Embedding del penúltimo ítem
        to_Conv(name="penultimate_item", s_filer=100, n_filer=321, 
                offset="(0,-10,0)", to="(gnn-east)", 
                width=10, height=8, depth=8, caption="Penultimate Item Emb. (through Scatter Max)"),
        
        # 🔹 Conexión desde Node Embedding hacia Penultimate Item Embedding
        to_connection("gnn", "penultimate_item"),


        # 🔹 Mecanismo de Self-Attention para el penúltimo ítem        
        to_Conv(name="attention_fc2", s_filer=1, n_filer=321, 
                offset="(2,0,0)", to="(penultimate_item-east)", 
                width=10, height=8, depth=1, 
                caption="Self-Attention FC2"),
        # 🔹 Conexión del penúltimo ítem a la atención (ahora es válida)
        to_connection("penultimate_item", "attention_fc2"),


        # Nueva capa para representar la expansión de embeddings
        to_Conv(name="expand_last_item_embeddings", s_filer=100, n_filer=321, 
                offset="(2,0,0)", to="(attention_fc2-east)", 
                width=10, height=8, depth=8, 
                caption="Expand Embeddings"),

        # Conectar Self-Attention FC2 a la expansión de embeddings


        to_connection("attention_fc2", "expand_last_item_embeddings"),
        to_Sum(name="add_embeddings",
               offset="(8,-4,0)", to="(attention_fc1-east)"
               ),

        # Conectar la capa expandida a la suma final
        to_connection("expand_last_item_embeddings", "add_embeddings"),

        to_connection("attention_fc1", "add_embeddings"),
        # to_connection("attention_fc2", "add_embeddings"),


        # Cálculo del score de atención
        to_ConvSoftMax(name="attention_score_fc", s_filer=1,
                #        n_filer=321, 
                        offset="(1,0,0)", to="(add_embeddings-east)", 
                        width=10, height=6, depth=1, 
                        caption="Sigmoid Attention Score (followed by Dropout)"),
        
        to_connection("add_embeddings", "attention_score_fc"),
        
        # Normalización de pesos
        to_ConvSoftMax(name="scatter_softmax", s_filer=1, 
                offset="(1,0,0)", to="(attention_score_fc-east)", 
                width=20, height=6, depth=1, caption="Scatter Softmax"),
        
        to_connection("attention_score_fc", "scatter_softmax"),

        # Cálculo de embeddings de sesión
        to_Conv(name="weighted_session", s_filer=100, n_filer=256, 
                offset="(1,4,0)", to="(scatter_softmax-east)", 
                width=20, height=6, depth=6, caption="Weighted Session Emb."),
        
        to_connection("scatter_softmax", "weighted_session"),
        to_connection("attention_fc1", "weighted_session"),
        
        to_Conv(name="scatter_sum", s_filer=100, n_filer=256, 
                offset="(2,0,0)", to="(weighted_session-east)", 
                width=20, height=6, depth=6, caption="Scatter Sum: session representation"),
        
        to_connection("weighted_session", "scatter_sum"),
        
        # Capa completamente conectada para puntuaciones
        to_Conv(name="fc", s_filer=71441, n_filer=1, 
                offset="(2,0,0)", to="(scatter_sum-east)", 
                width=10, height=6, depth=6, caption="Fully Connected: scores for each product"),
        
        to_connection("scatter_sum", "fc"),
        
        to_end()
    ]
    
    return arch


def main():
    arch = create_ggnn_attention_diagram()
    to_generate(arch, "./pyexamples/ggnn_attention_architecture.tex")
    
if __name__ == "__main__":
    main()