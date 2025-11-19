import os

# ==========================
# Configuration
# ==========================

# Define directories
PDF_DIRECTORY = '/Users/ammar/Downloads/'           # Directory containing your PDFs
PERSIST_DIRECTORY = '/Users/ammar/Desktop/ProjectX' # Directory for ChromaDB and tracking files

FAILED_PDFS_PATH = os.path.join(PERSIST_DIRECTORY, 'failed_pdfs.txt')
PROCESSED_PDFS_PATH = os.path.join(PERSIST_DIRECTORY, 'processed_pdfs.txt')
LOG_FILE = os.path.join(PERSIST_DIRECTORY, 'pdf_processing.log')
BM25_CORPUS_PATH = os.path.join(PERSIST_DIRECTORY, 'bm25_corpus.json')
BM25_MODEL_PATH = os.path.join(PERSIST_DIRECTORY, 'bm25_model.pkl')


# Chunking configuration
CHUNK_SIZE = 20000
CHUNK_OVERLAP = 500

# Embedding model
EMBEDDING_MODEL = 'all-MiniLM-L6-v2'

# Search configuration
TOP_K = 5