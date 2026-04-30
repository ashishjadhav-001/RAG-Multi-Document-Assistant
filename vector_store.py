import os
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_mistralai import MistralAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

# ✅ Load API key
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")

if not MISTRAL_API_KEY:
    raise ValueError("❌ MISTRAL_API_KEY not found")

# ✅ Embedding model
embedding_model = MistralAIEmbeddings(
    api_key=MISTRAL_API_KEY,
    model="mistral-embed"
)


def load_and_split_pdfs(pdf_paths):
    documents = []

    for pdf in pdf_paths:
        loader = PyPDFLoader(pdf)
        docs = loader.load()

        # 🔥 Add clean metadata
        for doc in docs:
            doc.metadata["file_name"] = os.path.basename(pdf)
            doc.metadata["page"] = doc.metadata.get("page", "N/A")

        documents.extend(docs)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    return text_splitter.split_documents(documents)


def create_vectorstore(pdf_paths, persist_directory=None):

    if persist_directory is None:
        persist_directory = "/tmp/chroma_db"

    os.makedirs(persist_directory, exist_ok=True)

    vectorstore = Chroma(
        persist_directory=persist_directory,
        embedding_function=embedding_model
    )

    try:
        existing_count = vectorstore._collection.count()
    except Exception:
        existing_count = 0

    if existing_count == 0:
        if not pdf_paths:
            raise ValueError("No PDF files provided.")

        documents = load_and_split_pdfs(pdf_paths)

        if not documents:
            raise ValueError("No content extracted from PDFs.")

        vectorstore.add_documents(documents)
        vectorstore.persist()

    return vectorstore


def load_vectorstore(persist_directory=None):

    if persist_directory is None:
        persist_directory = "/tmp/chroma_db"

    if not os.path.exists(persist_directory):
        raise FileNotFoundError("Vector DB not found.")

    return Chroma(
        persist_directory=persist_directory,
        embedding_function=embedding_model
    )