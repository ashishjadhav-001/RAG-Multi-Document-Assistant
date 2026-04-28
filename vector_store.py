from dotenv import load_dotenv
load_dotenv()

from langchain_community.vectorstores import Chroma
from langchain_mistralai import MistralAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os

def create_vectorstore(pdf_paths: list, persist_directory: str):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=100
    )

    embedding_model = MistralAIEmbeddings(model="mistral-embed")

    vectorstore = Chroma(
        persist_directory=persist_directory,
        embedding_function=embedding_model
    )

    all_docs = []

    for path in pdf_paths:
        loader = PyPDFLoader(path)
        docs = loader.load()

        filename = os.path.basename(path)

        for doc in docs:
            doc.metadata["source"] = filename

        all_docs.extend(docs)

    chunks = splitter.split_documents(all_docs)

    vectorstore.add_documents(chunks)

    vectorstore.persist()

    return vectorstore