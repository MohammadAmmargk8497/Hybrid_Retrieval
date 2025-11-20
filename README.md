# Local Hybrid PDF Search

This project is a local hybrid PDF retrieval system for searching through a collection of PDF documents. It extracts text from PDFs, generates embeddings, and uses a hybrid search approach to find relevant documents based on a user's query.
## Motivation
For the past couple of months I have been reading and downloading extensive research papers in Artificial Intelligence. With more than 20-30 papers each week being published on Arxiv it is becoming ever more complex to track and read them. So whenever I see a new **Banger** paper I go ahead and download it to read it later. But the problem is I am a messy person and all those papers with weird file names keep stacking in my downloads folder. Searching the paper in the downloads becomes a blind search task with very bad time complexity(yeah yeah, I do use Hueristic search but still). So I developed this dope method to just search up using Vector Search coupled with BM25. Saves me a ton of time🤓.
## Features

- **PDF Text Extraction:** Extracts text from PDF files in a specified directory.
- **Hybrid Search:** Utilizes a hybrid search approach combining semantic search (Vector Search with ChromaDB) and keyword search (BM25).
- **Reciprocal Rank Fusion (RRF):** Combines the results from the two search methods to provide more relevant and robust search results.
- **Vector Embeddings:** Generates vector embeddings for the extracted text using Sentence Transformers.
- **Vector Store:** Uses ChromaDB to store and search through the vector embeddings.
- **GUI:** A simple Tkinter-based graphical user interface to interact with the application.
- **Dependency Management:** Uses Poetry for managing project dependencies.

## Project Structure

```
.
├── pyproject.toml
├── poetry.lock
├── src
│   ├── __init__.py
│   ├── config.py
│   ├── embedding.py
│   ├── main.py
│   ├── pdf_processing.py
│   ├── search.py
│   └── vector_store.py
└── ...
```

## Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd <repository-name>
    ```

2.  **Install Poetry:**
    If you don't have Poetry installed, follow the instructions on the [official website](https://python-poetry.org/docs/#installation).

3.  **Install dependencies:**
    ```bash
    poetry install
    ```

## Configuration

The application can be configured by editing the `src/config.py` file.

-   `PDF_DIRECTORY`: The directory where your PDF files are located.
-   `PERSIST_DIRECTORY`: The directory where the ChromaDB database and BM25 models will be stored.
-   Other parameters like chunk size, overlap, and embedding model can also be configured.

## How to Run

To run the application, execute the following command from the root of the project:

```bash
poetry run python -m src.main
```

This will open the GUI.

1.  **Set Directories:**
    -   Verify that the `PDF Directory` and `Persist Directory` are set correctly. You can use the "Browse" buttons to change them.

2.  **Process PDFs:**
    -   Click the "Process PDFs" button to start the PDF processing. The application will scan the PDF directory, extract text from new PDFs, generate embeddings, and store them in the vector database. It will also create and save a BM25 model for keyword search.

3.  **Search:**
    -   Enter your search query in the "Search Query" field and click the "Search" button.
    -   The application will perform a hybrid search and the results will be displayed in the text area, including the document name, a snippet of the text, and a relevance score.
    -   The top 5 matching PDF files will be automatically opened.

## Hybrid Search

The hybrid search approach combines two different search methods to improve the relevance of the search results:

-   **Vector Search:** This is a semantic search that understands the meaning and context of the query. It is performed using ChromaDB and Sentence Transformers embeddings.
-   **BM25:** This is a keyword-based search that ranks documents based on the terms they contain.
-   **Reciprocal Rank Fusion (RRF):** The results from both search methods are combined using RRF. This technique re-ranks the documents based on their position in each of the search results, providing a more robust and relevant final ranking.

## Next Steps

-   **Unit Tests:** Add unit tests for the core functionality.
-   **Error Handling:** Improve error handling and provide more informative messages to the user.
-   **UI/UX:** Enhance the user interface for a better user experience.
-   **Packaging:** Package the application for easier distribution.