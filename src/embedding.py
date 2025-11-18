from sentence_transformers import SentenceTransformer
from src.config import EMBEDDING_MODEL

def get_embedding_model():
    """
    Returns the sentence transformer model.
    """
    return SentenceTransformer(EMBEDDING_MODEL)

def generate_embeddings(text_data, model):
    """
    Generates embeddings for the given text data.
    """
    embeddings = []
    for filename, text in text_data:
        embedding = model.encode(text)
        embeddings.append((filename, text, embedding))  # Store filename, text, and embedding
    return embeddings
