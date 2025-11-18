import os
import re
import logging
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.config import CHUNK_SIZE, CHUNK_OVERLAP

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
