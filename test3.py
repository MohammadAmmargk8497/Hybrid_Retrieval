import os
import re
import platform
import subprocess
import logging
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from PyPDF2 import PdfReader
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from tqdm import tqdm

# ==========================
# Configuration
# ==========================

# Define directories
PDF_DIRECTORY = '/Users/ammar/Downloads/'           # Directory containing your PDFs
PERSIST_DIRECTORY = '/Users/ammar/Desktop/ProjectX' # Directory for ChromaDB and tracking files

FAILED_PDFS_PATH = os.path.join(PERSIST_DIRECTORY, 'failed_pdfs.txt')
PROCESSED_PDFS_PATH = os.path.join(PERSIST_DIRECTORY, 'processed_pdfs.txt')
LOG_FILE = os.path.join(PERSIST_DIRECTORY, 'pdf_processing.log')

# Chunking configuration
CHUNK_SIZE = 20000
CHUNK_OVERLAP = 500

# Logging configuration
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format='%(asctime)s:%(levelname)s:%(message)s'
)

# Ensure directories exist
os.makedirs(PERSIST_DIRECTORY, exist_ok=True)

# ==========================
# Helper Functions
# ==========================

def clean_text(text: str) -> str:
    """
    Remove non-ASCII characters and extra whitespace from extracted text.
    """
    cleaned_text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
    return cleaned_text

def load_pdfs_from_directory(directory: str):
    """
    Return a list of all PDF filenames in the given directory.
    """
    return [f for f in os.listdir(directory) if f.lower().endswith('.pdf')]

def load_failed_pdfs(filepath: str) -> set:
    """
    Loads the list of failed PDFs from the specified file and returns a set of filenames.
    """
    failed_pdfs = set()
    if os.path.exists(filepath):
        with open(filepath, 'r') as f:
            for line in f:
                pdf = line.strip()
                if pdf:
                    failed_pdfs.add(pdf)
        logging.info(f"Loaded {len(failed_pdfs)} failed PDFs from {filepath}.")
    else:
        logging.info(f"No existing failed PDFs found at {filepath}.")
    return failed_pdfs

def save_failed_pdfs(failed_pdfs: list, filepath: str):
    """
    Appends the list of failed PDFs to the specified file.
    """
    if not failed_pdfs:
        return
    with open(filepath, 'a') as f:
        for pdf in failed_pdfs:
            f.write(f"{pdf}\n")
    logging.info(f"Added {len(failed_pdfs)} failed PDFs to {filepath}.")

def load_processed_pdfs(filepath: str) -> set:
    """
    Loads the list of processed PDFs from the specified file.
    """
    processed_pdfs = set()
    if os.path.exists(filepath):
        with open(filepath, 'r') as f:
            for line in f:
                pdf = line.strip()
                if pdf:
                    processed_pdfs.add(pdf)
        logging.info(f"Loaded {len(processed_pdfs)} processed PDFs from {filepath}.")
    else:
        logging.info(f"No existing processed PDFs found at {filepath}.")
    return processed_pdfs

def save_processed_pdfs(processed_pdfs: list, filepath: str):
    """
    Appends the list of processed PDFs to the specified file.
    """
    if not processed_pdfs:
        return
    with open(filepath, 'a') as f:
        for pdf in processed_pdfs:
            f.write(f"{pdf}\n")
    logging.info(f"Added {len(processed_pdfs)} processed PDFs to {filepath}.")

def extract_text_from_pdfs(directory: str, filenames: list, chunk_size:int=CHUNK_SIZE, chunk_overlap:int=CHUNK_OVERLAP):
    """
    Extract, clean, and chunk text from the specified PDFs. Uses LangChain to load and split PDFs.
    """
    text_data = []
    failed_pdfs = []
    success_pdfs = []

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""]
    )

    for filename in filenames:
        filepath = os.path.join(directory, filename)
        loader = PyPDFLoader(filepath)
        try:
            documents = loader.load()
            if not documents:
                logging.warning(f"No documents found in {filename}. Skipping.")
                failed_pdfs.append(filename)
                continue

            full_text = " ".join([doc.page_content for doc in documents])
            cleaned_text = clean_text(full_text)
            
            if not cleaned_text:
                logging.warning(f"No text found in {filename}. Skipping.")
                failed_pdfs.append(filename)
                continue

            chunks = text_splitter.split_text(cleaned_text)
            for idx, chunk in enumerate(chunks):
                chunk_id = f"{filename}_chunk_{idx+1}"
                metadata = {"source": filename}
                text_data.append((chunk_id, chunk, metadata))
            
            logging.info(f"Processed {filename}: {len(chunks)} chunks created.")
            success_pdfs.append(filename)

        except Exception as e:
            logging.error(f"Error processing {filename}: {e}. Skipping.")
            failed_pdfs.append(filename)
            continue

    return text_data, failed_pdfs, success_pdfs

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

def search_in_chroma(collection, query: str, top_k:int=5):
    """
    Perform a search in ChromaDB and return top_k results.
    """
    results = collection.query(
        query_texts=[query],
        n_results=top_k,
        include=['documents', 'metadatas', 'distances']
    )
    return results

def open_pdf(file_path:str):
    """
    Open PDF file with the system's default viewer.
    """
    try:
        if platform.system() == 'Windows':
            os.startfile(file_path)
        elif platform.system() == 'Darwin':
            subprocess.call(['open', file_path])
        else:
            subprocess.call(['xdg-open', file_path])
        logging.info(f"Opened PDF: {file_path}")
    except Exception as e:
        logging.error(f"Failed to open {file_path}: {e}")

def display_search_results(results: dict, max_chars:int=200, directory:str=None, top_k:int=5):
    """
    Display search results and automatically open top_k PDFs.
    """
    if 'documents' in results and 'metadatas' in results and 'distances' in results:
        documents = results['documents'][0]
        metadatas = results['metadatas'][0]
        distances = results['distances'][0]

        opened_pdfs = set()
        
        for idx, (doc, meta, distance) in enumerate(zip(documents, metadatas, distances)):
            source = meta.get('source', 'Unknown')
            snippet = doc[:max_chars] + '...' if len(doc) > max_chars else doc
            print(f"Document: {source}, Score: {distance}")
            print(f"Snippet: {snippet}\n")

            if idx < top_k and directory:
                if source not in opened_pdfs:
                    file_path = os.path.join(directory, source)
                    if os.path.exists(file_path):
                        open_pdf(file_path)
                        opened_pdfs.add(source)
                    else:
                        logging.warning(f"File not found: {file_path}")
    else:
        logging.info("No results found or unexpected result structure.")

# ==========================
# Main Application Entry Point
# ==========================

def main():
    logging.info("Starting PDF processing application...")

    # Setup ChromaDB
    client, collection = setup_chroma(PERSIST_DIRECTORY)
    logging.info("Chroma DB setup complete.")

    # Retrieve existing docs from ChromaDB
    try:
        collection_data = collection.get(limit=1000000)
        existing_docs = collection_data.get('ids', [])
    except Exception as e:
        logging.error(f"Error retrieving existing document IDs: {e}")
        existing_docs = []

    existing_docs_set = set(existing_docs)
    logging.info(f"Existing documents in Chroma DB: {len(existing_docs_set)}")

    failed_pdfs_set = load_failed_pdfs(FAILED_PDFS_PATH)
    logging.info(f"Failed documents in Chroma DB: {len(failed_pdfs_set)}")

    processed_pdfs_set = load_processed_pdfs(PROCESSED_PDFS_PATH)
    logging.info(f"Processed documents in Chroma DB: {len(processed_pdfs_set)}")

    all_filenames = load_pdfs_from_directory(PDF_DIRECTORY)
    logging.info(f"Total PDFs in directory: {len(all_filenames)}")

    # Determine new PDFs (not processed before and not failed)
    new_filenames = [f for f in all_filenames if f not in processed_pdfs_set and f not in failed_pdfs_set]
    logging.info(f"New PDFs to process: {len(new_filenames)}")

    # Extract text from new PDFs
    if new_filenames:
        logging.info("Extracting text from new PDFs...")
        text_data, failed_pdfs, success_pdfs = extract_text_from_pdfs(PDF_DIRECTORY, new_filenames)
        logging.info(f"Extracted text from {len(text_data)} new PDFs.")

        # Save failed PDFs
        if failed_pdfs:
            logging.info(f"Saving {len(failed_pdfs)} failed PDFs.")
            save_failed_pdfs(failed_pdfs, FAILED_PDFS_PATH)

        # Store new PDFs in ChromaDB
        if text_data:
            logging.info("Storing new PDFs in Chroma DB...")
            store_new_pdfs_in_chroma(collection, text_data)
            # Save processed PDFs
            unique_processed_pdfs = list(set(success_pdfs))
            save_processed_pdfs(unique_processed_pdfs, PROCESSED_PDFS_PATH)
        else:
            logging.info("No valid text data extracted from new PDFs. Skipping storage.")
    else:
        logging.info("No new PDFs to process. Skipping text extraction and storage.")

    # Persist ChromaDB changes
    try:
        client.persist()
        logging.info("Chroma DB has been persisted to disk.")
    except Exception as e:
        logging.error(f"Error persisting Chroma DB to disk: {e}")

    # Perform search based on user query
    query = input("Enter your search query: ").strip()
    if not query:
        logging.info("Empty query provided. Exiting search.")
        return
    logging.info(f"Searching for '{query}'...")
    
    results = search_in_chroma(collection, query)
    
    print("\nSearch Results:")
    display_search_results(results, directory=PDF_DIRECTORY, top_k=5)

    logging.info("Application finished successfully.")

if __name__ == "__main__":
    main()
