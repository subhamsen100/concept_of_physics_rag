# import os
# from langchain.document_loaders import PyPDFLoader
# from langchain.embeddings import HuggingFaceEmbeddings
# from langchain.vectorstores import Chroma
# from langchain.text_splitter import RecursiveCharacterTextSplitter

# # Constants
# CHROMA_PERSIST_DIR = "./chroma_db"
# EMBEDDING_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"

# def embed_pdf(pdf_path):
#     # Check if PDF exists
#     if not os.path.exists(pdf_path):
#         raise FileNotFoundError(f"PDF not found at: {pdf_path}")

#     print("Loading PDF...")
#     # Load PDF and process it
#     loader = PyPDFLoader(pdf_path)
#     documents = loader.load()

#     print("Splitting PDF into chunks...")
#     # Split text into manageable chunks
#     text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
#     texts = text_splitter.split_documents(documents)

#     print("Creating embeddings and saving to ChromaDB...")
#     # Create embeddings and save to ChromaDB
#     embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
#     vectorstore = Chroma.from_documents(
#         documents=texts,
#         embedding=embeddings,
#         persist_directory=CHROMA_PERSIST_DIR,
#     )
#     vectorstore.persist()
#     print(f"Vector database created and saved to {CHROMA_PERSIST_DIR}")

# if __name__ == "__main__":
#     pdf_path = "input.pdf"  # Replace with your PDF file path
#     embed_pdf(pdf_path)
import os
import PyPDF2
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Constants
CHROMA_PERSIST_DIR = "./chroma_db"  # Path to store Chroma vector database
EMBEDDING_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"  # Embedding model from HuggingFace

# Function to load and extract text from PDF
def load_pdf(file_path):
    """
    Load a PDF file and extract its text content.
    """
    text = ""
    with open(file_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        for page_num in range(len(reader.pages)):
            page = reader.pages[page_num]
            text += page.extract_text() or ""  # Ensure text is extracted
    return text

# Function to split text into chunks with overlap
def split_text_into_chunks(text, chunk_size=500, overlap=50):
    """
    Split text into chunks of a specified size with overlap.
    """
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
    return text_splitter.split_text(text)

# Function to create and persist a Chroma vector store from PDF
def create_chroma_vector_store(pdf_path, persist_dir=CHROMA_PERSIST_DIR):
    """
    Create and persist a Chroma vector store with embeddings for the text in a PDF.
    """
    # Step 1: Load and extract text from the PDF
    pdf_text = load_pdf(pdf_path)

    # Step 2: Split text into chunks
    chunks = split_text_into_chunks(pdf_text)

    # Step 3: Initialize the embedding model
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

    # Step 4: Convert chunks into Document objects
    documents = [Document(page_content=chunk) for chunk in chunks]

    # Step 5: Create a Chroma vector store from documents and embeddings
    vectorstore = Chroma.from_documents(documents, embedding=embeddings, persist_directory=persist_dir)

    # Step 6: Persist the Chroma vector store
    vectorstore.persist()
    print(f"Chroma vector store created and saved at {persist_dir}")

    return vectorstore

# Example usage
pdf_path = "input.pdf"  # Replace with your PDF file path
create_chroma_vector_store(pdf_path)
