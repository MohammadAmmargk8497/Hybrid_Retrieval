from sentence_transformers import SentenceTransformer
from pdf_reader import extract_text_from_pdfs
from Embedding import generate_embeddings, create_faiss_index, setup_chroma, store_embeddings_in_chroma
from Searcher import search_documents, search_in_chroma
import chromadb








def main():
    directory = '/Users/ammar/Downloads/'
    
    # Step 1: Extract text from PDFs
    text_data = extract_text_from_pdfs(directory)
    
    # Step 2: Generate embeddings using Sentence Transformer
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = generate_embeddings(text_data, model)
    
    # Step 3: Setup Chroma and store embeddings
    collection = setup_chroma()
    store_embeddings_in_chroma(collection, embeddings)

    # Step 4: Perform search using Chroma DB
    query = input("Enter your search query: ")
    results = search_in_chroma(collection, query, model)

    for result in results['documents']:
        # Each result in 'documents' is a dictionary containing document data
        filename = result['metadata']['filename']
        score = result['score']
        print(f"Document: {filename}, Score: {score}")
    
    print("Search Results:")
    for result in results['documents']:
        filename = result['metadata']['filename']
        print(f"Document: {filename}, Score: {result['score']}")

if __name__ == "__main__":
    main()