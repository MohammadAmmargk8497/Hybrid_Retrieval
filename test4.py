import os
import re
import platform
import subprocess
import logging
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from PyPDF2 import PdfReader
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from tqdm import tqdm

# ==========================
# Configuration (initial defaults)
# ==========================
PDF_DIRECTORY = '/Users/ammar/Downloads/'
PERSIST_DIRECTORY = '/Users/ammar/Desktop/ProjectX'
FAILED_PDFS_PATH = os.path.join(PERSIST_DIRECTORY, 'failed_pdfs.txt')
PROCESSED_PDFS_PATH = os.path.join(PERSIST_DIRECTORY, 'processed_pdfs.txt')
LOG_FILE = os.path.join(PERSIST_DIRECTORY, 'pdf_processing.log')

CHUNK_SIZE = 20000
CHUNK_OVERLAP = 500

# Setup logging
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format='%(asctime)s:%(levelname)s:%(message)s'
)

# ==========================
# Helper Functions (Same as before)
# ==========================

def clean_text(text):
    cleaned_text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
    return cleaned_text

def load_failed_pdfs(filepath):
    failed_pdfs = set()
    if os.path.exists(filepath):
        with open(filepath, 'r') as f:
            for line in f:
                pdf = line.strip()
                if pdf:
                    failed_pdfs.add(pdf)
        logging.info(f"Loaded {len(failed_pdfs)} failed PDFs from {filepath}.")
    else:
        logging.info(f"No existing failed PDFs at {filepath}.")
    return failed_pdfs

def save_failed_pdfs(failed_pdfs, filepath):
    if not failed_pdfs:
        return
    with open(filepath, 'a') as f:
        for pdf in failed_pdfs:
            f.write(f"{pdf}\n")
    logging.info(f"Added {len(failed_pdfs)} failed PDFs to {filepath}.")

def load_processed_pdfs(filepath):
    processed_pdfs = set()
    if os.path.exists(filepath):
        with open(filepath, 'r') as f:
            for line in f:
                pdf = line.strip()
                if pdf:
                    processed_pdfs.add(pdf)
        logging.info(f"Loaded {len(processed_pdfs)} processed PDFs from {filepath}.")
    else:
        logging.info(f"No existing processed PDFs at {filepath}.")
    return processed_pdfs

def save_processed_pdfs(processed_pdfs, filepath):
    if not processed_pdfs:
        return
    with open(filepath, 'a') as f:
        for pdf in processed_pdfs:
            f.write(f"{pdf}\n")
    logging.info(f"Added {len(processed_pdfs)} processed PDFs to {filepath}.")

def extract_text_from_pdfs(directory, filenames, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP):
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
                logging.warning(f"No documents in {filename}.")
                failed_pdfs.append(filename)
                continue
            full_text = " ".join([doc.page_content for doc in documents])
            cleaned_text = clean_text(full_text)
            if not cleaned_text:
                logging.warning(f"No text in {filename}.")
                failed_pdfs.append(filename)
                continue

            chunks = text_splitter.split_text(cleaned_text)
            for idx, chunk in enumerate(chunks):
                chunk_id = f"{filename}_chunk_{idx+1}"
                metadata = {"source": filename}
                text_data.append((chunk_id, chunk, metadata))
            
            logging.info(f"Processed {filename}: {len(chunks)} chunks.")
            success_pdfs.append(filename)

        except Exception as e:
            logging.error(f"Error processing {filename}: {e}")
            failed_pdfs.append(filename)
            continue
    return text_data, failed_pdfs, success_pdfs

def setup_chroma(persist_directory):
    client = chromadb.PersistentClient(persist_directory)
    collection = client.get_or_create_collection(name="pdf_embeddings")
    return client, collection

def store_new_pdfs_in_chroma(collection, text_data):
    total_chunks = len(text_data)
    batch_size = 1000
    for start_idx in tqdm(range(0, total_chunks, batch_size), desc="Storing PDFs"):
        end_idx = start_idx + batch_size
        batch = text_data[start_idx:end_idx]
        ids = [c_id for c_id, _, _ in batch]
        documents = [t for _, t, _ in batch]
        metadatas = [m for _, _, m in batch]
        try:
            collection.add(
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
            logging.info(f"Added docs {start_idx+1}-{min(end_idx, total_chunks)} of {total_chunks}.")
        except Exception as e:
            logging.error(f"Error adding batch: {e}")
    logging.info(f"Stored {total_chunks} new PDF chunks in Chroma DB.")

def search_in_chroma(collection, query, top_k=5):
    results = collection.query(
        query_texts=[query],
        n_results=top_k,
        include=['documents', 'metadatas', 'distances']
    )
    return results

def open_pdf(file_path):
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

def display_search_results(results, output_widget, directory=None, top_k=5, max_chars=200):
    output_widget.delete('1.0', tk.END)
    if 'documents' in results and 'metadatas' in results and 'distances' in results:
        documents = results['documents'][0]
        metadatas = results['metadatas'][0]
        distances = results['distances'][0]

        opened_pdfs = set()
        
        for idx, (doc, meta, distance) in enumerate(zip(documents, metadatas, distances)):
            source = meta.get('source', 'Unknown')
            snippet = doc[:max_chars] + '...' if len(doc) > max_chars else doc
            output_widget.insert(tk.END, f"Document: {source}, Score: {distance}\nSnippet: {snippet}\n\n")

            # Optionally open top_k PDFs automatically:
            if idx < top_k and directory:
                if source not in opened_pdfs:
                    file_path = os.path.join(directory, source)
                    if os.path.exists(file_path):
                        open_pdf(file_path)
                        opened_pdfs.add(source)
                    else:
                        output_widget.insert(tk.END, f"File not found: {file_path}\n")
    else:
        output_widget.insert(tk.END, "No results found or unexpected result structure.\n")

def process_pdfs(pdf_directory, persist_directory, output_widget):
    # Setup Chroma
    client, collection = setup_chroma(persist_directory)

    # Load existing docs
    try:
        collection_data = collection.get(limit=1000000)
        existing_docs = collection_data.get('ids', [])
    except Exception as e:
        logging.error(f"Error retrieving existing docs: {e}")
        existing_docs = []
    existing_docs_set = set(existing_docs)

    failed_pdfs_set = load_failed_pdfs(FAILED_PDFS_PATH)
    processed_pdfs_set = load_processed_pdfs(PROCESSED_PDFS_PATH)

    all_files = [f for f in os.listdir(pdf_directory) if f.lower().endswith('.pdf')]
    new_filenames = [f for f in all_files if f not in processed_pdfs_set and f not in failed_pdfs_set]

    output_widget.insert(tk.END, f"Found {len(all_files)} PDFs, {len(new_filenames)} new to process.\n")
    if new_filenames:
        text_data, failed_pdfs, success_pdfs = extract_text_from_pdfs(pdf_directory, new_filenames)
        output_widget.insert(tk.END, f"Extracted text from {len(text_data)} PDFs.\n")

        if failed_pdfs:
            save_failed_pdfs(failed_pdfs, FAILED_PDFS_PATH)

        if text_data:
            output_widget.insert(tk.END, "Storing new PDFs in Chroma DB...\n")
            store_new_pdfs_in_chroma(collection, text_data)
            unique_processed = list(set(success_pdfs))
            save_processed_pdfs(unique_processed, PROCESSED_PDFS_PATH)
        else:
            output_widget.insert(tk.END, "No valid text extracted. Skipping storage.\n")

    client.persist()
    output_widget.insert(tk.END, "Chroma DB persisted to disk.\n")

def run_search(query, pdf_directory, persist_directory, output_widget):
    client, collection = setup_chroma(persist_directory)
    results = search_in_chroma(collection, query)
    display_search_results(results, output_widget, directory=pdf_directory)

# ==========================
# GUI
# ==========================

def browse_pdf_directory(pdf_dir_entry):
    dirpath = filedialog.askdirectory()
    if dirpath:
        pdf_dir_entry.delete(0, tk.END)
        pdf_dir_entry.insert(0, dirpath)

def browse_persist_directory(persist_dir_entry):
    dirpath = filedialog.askdirectory()
    if dirpath:
        persist_dir_entry.delete(0, tk.END)
        persist_dir_entry.insert(0, dirpath)

def main():
    root = tk.Tk()
    root.title("PDF Processing and Search Application")

    # PDF Directory
    tk.Label(root, text="PDF Directory:").grid(row=0, column=0, sticky="e", padx=5, pady=5)
    pdf_dir_entry = tk.Entry(root, width=50)
    pdf_dir_entry.grid(row=0, column=1, padx=5, pady=5)
    pdf_dir_entry.insert(0, PDF_DIRECTORY)
    tk.Button(root, text="Browse", command=lambda: browse_pdf_directory(pdf_dir_entry)).grid(row=0, column=2, padx=5, pady=5)

    # Persist Directory
    tk.Label(root, text="Persist Directory:").grid(row=1, column=0, sticky="e", padx=5, pady=5)
    persist_dir_entry = tk.Entry(root, width=50)
    persist_dir_entry.grid(row=1, column=1, padx=5, pady=5)
    persist_dir_entry.insert(0, PERSIST_DIRECTORY)
    tk.Button(root, text="Browse", command=lambda: browse_persist_directory(persist_dir_entry)).grid(row=1, column=2, padx=5, pady=5)

    # Output widget (scrolled text)
    output_widget = scrolledtext.ScrolledText(root, width=80, height=20)
    output_widget.grid(row=2, column=0, columnspan=3, padx=5, pady=5)

    # Process PDFs Button
    tk.Button(root, text="Process PDFs", command=lambda: process_pdfs(pdf_dir_entry.get(), persist_dir_entry.get(), output_widget)).grid(row=3, column=0, padx=5, pady=5, sticky="e")

    # Search Entry and Button
    tk.Label(root, text="Search Query:").grid(row=3, column=1, sticky="e", padx=5, pady=5)
    query_entry = tk.Entry(root, width=30)
    query_entry.grid(row=3, column=2, padx=5, pady=5, sticky="w")

    tk.Button(root, text="Search", command=lambda: run_search(query_entry.get(), pdf_dir_entry.get(), persist_dir_entry.get(), output_widget)).grid(row=4, column=2, padx=5, pady=5, sticky="e")

    root.mainloop()

if __name__ == "__main__":
    main()
