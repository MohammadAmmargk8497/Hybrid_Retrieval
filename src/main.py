import os
import logging
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext
import platform
import subprocess
import json
import pickle
from rank_bm25 import BM25Okapi

from src.config import (
    PDF_DIRECTORY,
    PERSIST_DIRECTORY,
    FAILED_PDFS_PATH,
    PROCESSED_PDFS_PATH,
    LOG_FILE,
    TOP_K,
    BM25_CORPUS_PATH,
    BM25_MODEL_PATH,
)
from src.pdf_processing import (
    load_failed_pdfs,
    save_failed_pdfs,
    load_processed_pdfs,
    save_processed_pdfs,
    extract_text_from_pdfs,
    load_pdfs_from_directory,
)
from src.vector_store import setup_chroma, store_new_pdfs_in_chroma
from src.search import hybrid_search

# Setup logging
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format='%(asctime)s:%(levelname)s:%(message)s'
)

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

def display_search_results(results: list, output_widget, directory=None, top_k:int=5, max_chars=200):
    output_widget.delete('1.0', tk.END)
    
    opened_pdfs = set()
    
    for idx, result in enumerate(results):
        doc = result['document']
        meta = result['metadata']
        score = result['score']
        
        source = meta.get('source', 'Unknown')
        snippet = doc[:max_chars] + '...' if len(doc) > max_chars else doc
        output_widget.insert(tk.END, f"Document: {source}, Score: {score}\nSnippet: {snippet}\n\n")

        # Optionally open top_k PDFs automatically:
        if idx < top_k and directory:
            if source not in opened_pdfs:
                file_path = os.path.join(directory, source)
                if os.path.exists(file_path):
                    open_pdf(file_path)
                    opened_pdfs.add(source)
                else:
                    output_widget.insert(tk.END, f"File not found: {file_path}\n")

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

    all_files = load_pdfs_from_directory(pdf_directory)
    new_filenames = [f for f in all_files if f not in processed_pdfs_set and f not in failed_pdfs_set]

    output_widget.insert(tk.END, f"Found {len(all_files)} PDFs, {len(new_filenames)} new to process.\n")
    
    new_files_processed = False
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
            new_files_processed = True
        else:
            output_widget.insert(tk.END, "No valid text extracted. Skipping storage.\n")

    # Create and save BM25 model if it doesn't exist or if new files were processed
    if new_files_processed or not os.path.exists(BM25_MODEL_PATH):
        output_widget.insert(tk.END, "Creating and saving BM25 model...\n")
        logging.info("Creating and saving BM25 model...")
        corpus_data = collection.get()
        corpus = corpus_data['documents']
        if corpus:
            tokenized_corpus = [doc.split(" ") for doc in corpus]
            bm25 = BM25Okapi(tokenized_corpus)
            
            with open(BM25_MODEL_PATH, 'wb') as f:
                pickle.dump(bm25, f)
            
            with open(BM25_CORPUS_PATH, 'w') as f:
                json.dump(corpus, f)
            
            output_widget.insert(tk.END, "BM25 model created and saved.\n")
            logging.info("BM25 model created and saved.")
        else:
            output_widget.insert(tk.END, "Corpus is empty. Skipping BM25 model creation.\n")
            logging.info("Corpus is empty. Skipping BM25 model creation.")

    output_widget.insert(tk.END, "PDF processing complete.\n")

def run_search(query, pdf_directory, persist_directory, output_widget):
    client, collection = setup_chroma(persist_directory)
    
    # Load BM25 model
    try:
        with open(BM25_MODEL_PATH, 'rb') as f:
            bm25_model = pickle.load(f)
    except FileNotFoundError:
        output_widget.insert(tk.END, "BM25 model not found. Please process PDFs first.\n")
        return
        
    # Get all documents from the collection to form the corpus
    corpus_data = collection.get()
    corpus = corpus_data['documents']
    metadatas = corpus_data['metadatas']
    
    results = hybrid_search(collection, bm25_model, corpus, metadatas, query, top_k=TOP_K)
    display_search_results(results, output_widget, directory=pdf_directory, top_k=TOP_K)


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
