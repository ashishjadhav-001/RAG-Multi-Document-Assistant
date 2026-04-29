# rag_pipeline.py

import os
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate


def load_vectorstore(persist_directory, embedding_model):
    """
    Safely load Chroma vector store
    """
    if not os.path.exists(persist_directory):
        raise FileNotFoundError("Vector DB not found. Please create it first.")

    vectorstore = Chroma(
        persist_directory=persist_directory,
        embedding_function=embedding_model
    )

    return vectorstore


def get_qa_chain(vectorstore, llm):
    """
    Create Retrieval QA chain using your existing LLM
    """

    prompt_template = """
    You are a helpful AI assistant.

    Answer ONLY from the provided context.
    If the answer is not present, say:
    "I couldn't find the answer in the provided documents."

    Context:
    {context}

    Question:
    {question}

    Answer:
    """

    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )

    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4}
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,  # ✅ your existing model
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=False,
        chain_type_kwargs={"prompt": prompt}
    )

    return qa_chain


def ask_question(query, persist_directory, llm, embedding_model):
    """
    Main function (uses your existing models)
    """

    try:
        vectorstore = load_vectorstore(
            persist_directory,
            embedding_model
        )

        qa_chain = get_qa_chain(
            vectorstore,
            llm
        )

        result = qa_chain.invoke({"query": query})

        return result["result"]

    except Exception as e:
        return f"❌ Error: {str(e)}"