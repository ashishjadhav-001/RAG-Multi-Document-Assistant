import os
from langchain_community.vectorstores import Chroma
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings  # or your embedding model


# ✅ Initialize embedding model (change if needed)
embedding_model = OpenAIEmbeddings()


def load_and_split_pdfs(pdf_paths):
    documents = []

    for pdf in pdf_paths:
        loader = PyPDFLoader(pdf)
        documents.extend(loader.load())

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    return text_splitter.split_documents(documents)


def create_vectorstore(pdf_paths, persist_directory=None):
    """
    Creates or loads a Chroma vector store safely.
    """

    # ✅ Fix 1: Use safe default directory (Render-compatible)
    if persist_directory is None:
        persist_directory = "/tmp/chroma_db"

    # ✅ Fix 2: Ensure directory exists
    os.makedirs(persist_directory, exist_ok=True)

    # ✅ Fix 3: Initialize vectorstore
    vectorstore = Chroma(
        persist_directory=persist_directory,
        embedding_function=embedding_model
    )

    # ✅ Fix 4: Only add documents if DB is empty
    try:
        existing_count = vectorstore._collection.count()
    except Exception:
        existing_count = 0

    if existing_count == 0:
        if not pdf_paths:
            raise ValueError("No PDF files provided to create vector store.")

        documents = load_and_split_pdfs(pdf_paths)

        if not documents:
            raise ValueError("No content extracted from PDFs.")

        vectorstore.add_documents(documents)
        vectorstore.persist()

    return vectorstore


def load_vectorstore(persist_directory=None):
    """
    Loads an existing vectorstore safely.
    """

    if persist_directory is None:
        persist_directory = "/tmp/chroma_db"

    if not os.path.exists(persist_directory):
        raise FileNotFoundError("Vector DB not found. Create it first.")

    vectorstore = Chroma(
        persist_directory=persist_directory,
        embedding_function=embedding_model
    )

    return vectorstore