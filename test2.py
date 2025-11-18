import os
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from PyPDF2 import PdfReader
from chromadb.config import Settings
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import re
import os
import subprocess
import platform
from tqdm import tqdm

# Function to clean the extracted text
def clean_text(text):
    """
    Removes non-ASCII characters and extra whitespace from the extracted text.
    """
    cleaned_text = re.sub(r'[^\x00-\x7F]+', ' ', text)  # Remove non-ASCII characters
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()  # Remove extra spaces
    return cleaned_text

# Function to extract text from PDF files
def extract_text_from_pdfs(directory, filenames, chunk_size=20000, chunk_overlap=500):
    """
    Extracts and cleans text from the specified PDF files in the given directory.
    Splits the text into chunks using LangChain's RecursiveCharacterTextSplitter.
    
    Returns:
        text_data (list): List of tuples containing (chunk_id, chunk_text, metadata)
        failed_pdfs (list): List of filenames that failed to extract text
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
                print(f"No documents found in {filename}. Skipping.")
                failed_pdfs.append(filename)
                continue
            
            # Combine all pages' content
            full_text = " ".join([doc.page_content for doc in documents])
            cleaned_text = clean_text(full_text)
            
            if not cleaned_text:
                print(f"No text found in {filename}. Skipping.")
                failed_pdfs.append(filename)
                continue
            
            # Split the text into chunks
            chunks = text_splitter.split_text(cleaned_text)
            for idx, chunk in enumerate(chunks):
                chunk_id = f"{filename}_chunk_{idx+1}"
                metadata = {"source": filename}
                text_data.append((chunk_id, chunk, metadata))
            
            print(f"Processed {filename}: {len(chunks)} chunks created.")
            success_pdfs.append(filename)
            print(filename)
        
        except Exception as e:
            print(f"Error processing {filename}: {e}. Skipping.")
            failed_pdfs.append(filename)
            continue
    
    return text_data, failed_pdfs, success_pdfs

# Function to setup Chroma DB with persistent storage on disk
def setup_chroma(persist_directory):
    """
    Initializes the Chroma persistent client with the specified directory
    and retrieves or creates the 'pdf_embeddings' collection.
    """
    # Initialize Chroma PersistentClient with the specified path
    client = chromadb.PersistentClient(
        
            persist_directory
        )
    
    
    # Retrieve or create the collection without supplying an embedding function
    collection = client.get_or_create_collection(name="pdf_embeddings")
    return client, collection

# Function to store the new PDFs into Chroma DB
def store_new_pdfs_in_chroma(collection, text_data):
    """
    Adds new PDF chunks to the ChromaDB collection.
    
    Parameters:
        collection: The ChromaDB collection object.
        text_data (list): List of tuples containing (chunk_id, chunk_text, metadata)
    """
    for chunk_id, chunk_text, metadata in text_data:
        collection.add(
            documents=[chunk_text],          # Store the chunk text
            metadatas=[metadata],            # Metadata containing the source filename
            ids=[chunk_id]                   # Use chunk ID as unique ID
        )
    print(f"Stored {len(text_data)} new PDF chunks in Chroma DB.")


# Function to perform a search based on the query in Chroma DB
def search_in_chroma(collection, query, top_k=5):
    """
    Performs a search in ChromaDB based on the query and retrieves the top_k results.
    """
    results = collection.query(
        query_texts=[query],                  # Query as text
        n_results=top_k,                      # Number of top results to retrieve
        include=['documents', 'metadatas', 'distances']  # Include necessary fields
    )
    return results

def open_pdf(file_path):
    """
    Opens a PDF file using the default PDF viewer based on the operating system.
    
    Parameters:
        file_path (str): The absolute path to the PDF file.
    """
    try:
        if platform.system() == 'Windows':
            os.startfile(file_path)
        elif platform.system() == 'Darwin':  # macOS
            subprocess.call(['open', file_path])
        else:  # Linux and others
            subprocess.call(['xdg-open', file_path])
        print(f"Opened PDF: {file_path}")
    except Exception as e:
        print(f"Failed to open {file_path}: {e}")


# Function to display search results correctly
def display_search_results(results, max_chars=200, directory=None, top_k=5):
    """
    Displays the search results with filenames, scores, and text snippets.
    Also opens the PDFs corresponding to the top_k results automatically.
    
    Parameters:
        results (dict): The results returned by ChromaDB's query.
        max_chars (int): Maximum number of characters to display from each document.
        directory (str): The directory where PDFs are stored.
        top_k (int): Number of top results to open automatically.
    """
    if 'documents' in results and 'metadatas' in results and 'distances' in results:
        # Assuming one query, so access the first element in each list
        documents = results['documents'][0]
        metadatas = results['metadatas'][0]
        distances = results['distances'][0]
        
        # To keep track of opened PDFs to avoid opening the same PDF multiple times
        opened_pdfs = set()
        
        for idx, (doc, meta, distance) in enumerate(zip(documents, metadatas, distances)):
            source = meta.get('source', 'Unknown')
            snippet = doc[:max_chars] + '...' if len(doc) > max_chars else doc
            print(f"Document: {source}, Score: {distance}")
            print(f"Snippet: {snippet}\n")
            
            # Open the PDF if within the top_k results and not already opened
            if idx < top_k and directory:
                if source not in opened_pdfs:
                    file_path = os.path.join(directory, source)
                    if os.path.exists(file_path):
                        open_pdf(file_path)
                        opened_pdfs.add(source)
                    else:
                        print(f"File not found: {file_path}")
    else:
        print("No results found or unexpected result structure.")


# Function to save failed PDFs to disk
def save_failed_pdfs(failed_pdfs, filepath):
    """
    Appends the list of failed PDFs to the specified file.
    
    Parameters:
        failed_pdfs (list): List of filenames that failed to extract text.
        filepath (str): Path to the file where failed PDFs are stored.
    """
    if not failed_pdfs:
        return  # No failed PDFs to save
    
    with open(filepath, 'a') as f:
        for pdf in failed_pdfs:
            f.write(f"{pdf}\n")
    print(f"Added {len(failed_pdfs)} failed PDFs to {filepath}.")

# Function to load failed PDFs from a file
def load_failed_pdfs(filepath):
    """
    Loads the list of failed PDFs from the specified file.
    Returns a set of filenames.
    
    Parameters:
        filepath (str): Path to the file where failed PDFs are stored.
    
    Returns:
        set: A set containing the filenames of failed PDFs.
    """
    failed_pdfs = set()
    if os.path.exists(filepath):
        with open(filepath, 'r') as f:
            for line in f:
                pdf = line.strip()
                if pdf:
                    failed_pdfs.add(pdf)
        print(f"Loaded {len(failed_pdfs)} failed PDFs from {filepath}.")
    else:
        print(f"No existing failed PDFs found at {filepath}.")
    return failed_pdfs


# Function to save processed PDFs to disk
def save_processed_pdfs(processed_pdfs, filepath):
    """
    Appends the list of processed PDFs to the specified file.
    
    Parameters:
        processed_pdfs (list): List of filenames that were successfully processed.
        filepath (str): Path to the file where processed PDFs are stored.
    """
    if not processed_pdfs:
        return  # No processed PDFs to save
    
    with open(filepath, 'a') as f:
        for pdf in processed_pdfs:
            f.write(f"{pdf}\n")
    print(f"Added {len(processed_pdfs)} processed PDFs to {filepath}.")

# Function to load processed PDFs from the file
def load_processed_pdfs(filepath):
    """
    Loads the list of processed PDFs from the specified file.
    
    Parameters:
        filepath (str): Path to the file where processed PDFs are stored.
    
    Returns:
        set: A set containing the filenames of processed PDFs.
    """
    processed_pdfs = set()
    if os.path.exists(filepath):
        with open(filepath, 'r') as f:
            for line in f:
                pdf = line.strip()
                if pdf:
                    processed_pdfs.add(pdf)
        print(f"Loaded {len(processed_pdfs)} processed PDFs from {filepath}.")
    else:
        print(f"No existing processed PDFs found at {filepath}.")
    return processed_pdfs

# Main function to execute the workflow
def main():
    # Replace these paths with your actual directories
    directory = '/Users/ammar/Downloads/'         # Path containing your PDFs
    persist_directory = '/Users/ammar/Desktop/ProjectX/'   # Path to store the Chroma DB
    failed_pdfs_path = os.path.join(persist_directory, 'failed_pdfs.txt')    # Path to store failed PDFs
    processed_pdfs_path = os.path.join(persist_directory, 'processed_pdfs.txt')  # Path to store processed PDFs

    # Ensure persist_directory exists
    os.makedirs(persist_directory, exist_ok=True)

    # Step 1: Setup Chroma with persistent storage and get the collection
    print("Setting up Chroma DB...")
    client, collection = setup_chroma(persist_directory)

    # Step 2: Retrieve existing document IDs from ChromaDB
    try:
        # Fetch all existing document IDs without using 'include' parameter
        collection_data = collection.get(limit=1000000)  # Set a high limit to retrieve all IDs
        existing_docs = collection_data.get('ids', [])   # Retrieve 'ids' from the result
    except Exception as e:
        print(f"Error retrieving existing document IDs: {e}")
        existing_docs = []

    existing_docs_set = set(existing_docs)
    print(f"Existing documents in Chroma DB: {len(existing_docs_set)}")
    # Debugging: Uncomment the next line to see existing document IDs
    # print(f"Existing document IDs: {existing_docs_set}")

    # Step 3: Load failed PDFs
    failed_pdfs_set = load_failed_pdfs(failed_pdfs_path)
    print(f"Failed documents in Chroma DB: {len(failed_pdfs_set)}")

    # Step 4: Load processed PDFs
    processed_pdfs_set = load_processed_pdfs(processed_pdfs_path)
    print(f"Processed documents in Chroma DB: {len(processed_pdfs_set)}")

    # Step 5: List all PDF filenames in the directory
    all_filenames = [f for f in os.listdir(directory) if f.lower().endswith('.pdf')]
    print(f"Total PDFs in directory: {len(all_filenames)}")

    # Step 6: Determine which PDFs are new (not in ChromaDB and not failed)
    new_filenames = [f for f in all_filenames if f not in processed_pdfs_set and f not in failed_pdfs_set]
    print(f"New PDFs to process: {len(new_filenames)}")

    # Step 7: Extract text from new PDFs
    if new_filenames:
        print("Extracting text from new PDFs...")
        text_data, failed_pdfs, success_pdfs = extract_text_from_pdfs(directory, new_filenames)
        print(f"Extracted text from {len(text_data)} new PDFs.")

        # Step 8: Save failed PDFs to disk
        if failed_pdfs:
            print(f"Saving {len(failed_pdfs)} failed PDFs to {failed_pdfs_path}.")
            save_failed_pdfs(failed_pdfs, failed_pdfs_path)

        # Step 9: Store new PDFs in ChromaDB
        if text_data:
            print("Storing new PDFs in Chroma DB...")
            store_new_pdfs_in_chroma(collection, text_data)  # Adjust batch_size as needed
            # After successful storage, mark these PDFs as processed
            # processed_pdfs = [filename for filename, _, _ in text_data]
            unique_processed_pdfs = list(set(success_pdfs))
            save_processed_pdfs(unique_processed_pdfs, processed_pdfs_path)
        else:
            print("No valid text data extracted from new PDFs. Skipping storage.")
    else:
        print("No new PDFs to process. Skipping text extraction and storage.")

    
    print("Chroma DB has been persisted to disk.")
    
    # Step 7: Perform search based on user query
    query = input("Enter your search query: ").strip()
    if not query:
        print("Empty query provided. Exiting search.")
        return
    print(f"Searching for '{query}'...")
    
    results = search_in_chroma(collection, query)
    
    # Step 8: Display the search results
    print("\nSearch Results:")
    display_search_results(results, directory=directory, top_k=5)

if __name__ == "__main__":
    main()
