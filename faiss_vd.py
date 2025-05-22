# Importing necessary libraries
import faiss
import numpy as np

# FAISS Vector Database functions
def create_faiss_index(embedding_dim):
    index = faiss.IndexFlatL2(embedding_dim)
    return index

def add_embeddings_to_faiss(index, embeddings):
    # Her embedding için tip kontrolü yap
    embeddings_np = np.array([
        embedding.cpu().numpy() if hasattr(embedding, 'cpu') else embedding
        for embedding in embeddings
    ])
    index.add(embeddings_np)
    return index
