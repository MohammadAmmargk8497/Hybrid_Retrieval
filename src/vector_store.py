import chromadb
from tqdm import tqdm
import logging

def setup_chroma(persist_directory: str):
    """
    Initialize Chroma persistent client and collection.
    """
    client = chromadb.PersistentClient(persist_directory)
    collection = client.get_or_create_collection(name="pdf_embeddings")
    return client, collection

def store_new_pdfs_in_chroma(collection, text_data: list):
    """
    Store PDF chunks in ChromaDB. Adjust batch_size if needed for performance.
    """
    total_chunks = len(text_data)
    batch_size = 1000
    for start_idx in tqdm(range(0, total_chunks, batch_size), desc="Storing PDFs"):
        end_idx = start_idx + batch_size
        batch = text_data[start_idx:end_idx]

        ids = [chunk_id for chunk_id, _, _ in batch]
        documents = [chunk_text for _, chunk_text, _ in batch]
        metadatas = [metadata for _, _, metadata in batch]

        try:
            collection.add(
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
            logging.info(f"Added documents {start_idx + 1} to {min(end_idx, total_chunks)} out of {total_chunks}.")
        except Exception as e:
            logging.error(f"Error adding batch {start_idx + 1} to {min(end_idx, total_chunks)}: {e}")

    logging.info(f"Stored {total_chunks} new PDF chunks in Chroma DB.")


