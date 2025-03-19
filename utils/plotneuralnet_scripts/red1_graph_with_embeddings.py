import sys
sys.path.append('../')  # Ajusta la ruta si es necesario
from pycore.tikzeng import *  # Importa las herramientas para dibujar redes neuronales

def create_srgnn_architecture():
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
                        
        
        # Capa GNN Gru Graph Layer
        to_Conv(name="gnn", s_filer=100, n_filer=321, 
                        offset="(2,0,0)", to="(embedding-east)", 
                        width=10, height=8, depth=8, 
                        caption="GGNN"),
        
        to_connection("embedding", "gnn"),
        
        # Global Mean Pooling Layer
        to_Pool(name="pool", offset="(2,0,0)", to="(gnn-east)", 
                width=5, height=40, depth=40, opacity=0.5, 
                caption="Global Mean Pool (followed by Dropout)"),
        
        # Fully Connected Layer
        to_SoftMax(name="fc", s_filer=100, offset="(2,0,0)", to="(pool-east)", 
                width=1.5, height=3, depth=25, opacity=0.8, caption="FC Layer"),
        
        # Output Layer
        to_SoftMax(name="output", s_filer=71441, offset="(2,0,0)", to="(fc-east)", 
                width=1, height=4, depth=40, caption="Output"),
        
        # Conexiones (sin input) 
        
        to_connection("embedding", "gnn"),
        to_connection("gnn", "pool"),
        to_connection("pool", "fc"),
        to_connection("fc", "output"),
        
        to_end()
    ]

    return arch

if __name__ == '__main__':
    arch = create_srgnn_architecture()
    to_generate(arch, "./pyexamples/ggnn.tex")
