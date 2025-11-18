# Local RAG PDF Search

This project is a local RAG (Retrieval-Augmented Generation) system for searching through a collection of PDF documents. It extracts text from PDFs, generates embeddings, and uses a vector database to find relevant documents based on a user's query.

## Features

- **PDF Text Extraction:** Extracts text from PDF files in a specified directory.
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
-   `PERSIST_DIRECTORY`: The directory where the ChromaDB database will be stored.
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
    -   Click the "Process PDFs" button to start the PDF processing. The application will scan the PDF directory, extract text from new PDFs, generate embeddings, and store them in the vector database.

3.  **Search:**
    -   Enter your search query in the "Search Query" field and click the "Search" button.
    -   The search results will be displayed in the text area, including the document name, a snippet of the text, and a relevance score.
    -   The top 5 matching PDF files will be automatically opened.

## Next Steps

-   **Unit Tests:** Add unit tests for the core functionality.
-   **Error Handling:** Improve error handling and provide more informative messages to the user.
-   **UI/UX:** Enhance the user interface for a better user experience.
-   **Packaging:** Package the application for easier distribution.
