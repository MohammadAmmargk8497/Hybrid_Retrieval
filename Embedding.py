from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import chromadb

def generate_embeddings(text_data, model):
    embeddings = []
    for filename, text in text_data:
        embedding = model.encode(text)
        embeddings.append((filename, text, embedding))  # Store filename, text, and embedding
    return embeddings

def setup_chroma():
    # Initialize Chroma client and create or get the collection
    client = chromadb.Client()
    collection = client.get_or_create_collection(name="pdf_embeddings")
    return collection

def store_embeddings_in_chroma(collection, embeddings):
    for filename, text, embedding in embeddings:
        collection.add(
            documents=[text],       # Store the document text, not the embedding
            embeddings=[embedding], # Store the actual embedding (vector)
            metadatas=[{"filename": filename}], # Store metadata (filename)
            ids=[filename]          # Use filename as a unique identifier
        )
    print("Embeddings have been stored in Chroma DB!")

def create_faiss_index(embeddings):
    dimension = len(embeddings[0][1])  # Get the dimensionality of embeddings
    index = faiss.IndexFlatL2(dimension)  # Create a FAISS index
    
    # Convert embeddings into a numpy array
    all_embeddings = np.array([embedding[1] for embedding in embeddings]).astype('float32')
    
    # Add embeddings to the FAISS index
    index.add(all_embeddings)
    
    return index