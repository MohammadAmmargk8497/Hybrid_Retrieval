import os
from PyPDF2 import PdfReader
from PyPDF2 import PdfReader
from PyPDF2.errors import PdfReadError

def extract_text_from_pdfs(directory):
    text_data = []
    for filename in os.listdir(directory):
        if filename.endswith('.pdf'):
            filepath = os.path.join(directory, filename)
            with open(filepath, 'rb') as file:
                try:
                    reader = PdfReader(file)
                    # Check if the PDF is encrypted
                    if reader.is_encrypted:
                        # Try decrypting the PDF with an empty password (if no password is required)
                        if reader.decrypt(''):
                            print(f"Decrypted {filename} successfully.")
                        else:
                            print(f"Failed to decrypt {filename}.")
                            continue  # Skip the file if decryption fails
                    
                    text = ""
                    # Extract text from each page
                    for page in reader.pages:
                        text += page.extract_text()
                    text_data.append((filename, text))
                except PdfReadError as e:
                    print(f"Error reading {filename}: {e}")
                    continue  # Skip unreadable PDFs
    return text_data