# rag_pipeline.py

import os
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_mistralai import ChatMistralAI, MistralAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

# ✅ Load API key
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")

if not MISTRAL_API_KEY:
    raise ValueError("❌ MISTRAL_API_KEY not found in environment variables")

# ✅ Embedding model
embedding_model = MistralAIEmbeddings(
    api_key=MISTRAL_API_KEY,
    model="mistral-embed"
)

# ✅ LLM
llm = ChatMistralAI(
    api_key=MISTRAL_API_KEY,
    model="mistral-small",
    temperature=0
)


def load_vectorstore(persist_directory):
    if not os.path.exists(persist_directory):
        raise FileNotFoundError("Vector DB not found. Please create it first.")

    return Chroma(
        persist_directory=persist_directory,
        embedding_function=embedding_model
    )


def get_qa_chain(vectorstore):

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

    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )


def ask_question(query, persist_directory):
    try:
        vectorstore = load_vectorstore(persist_directory)
        qa_chain = get_qa_chain(vectorstore)

        result = qa_chain.invoke({"query": query})

        answer = result["result"]

        # ✅ Extract ONLY PDF names (no page numbers)
        sources = set()

        for doc in result["source_documents"]:
            file_name = doc.metadata.get("file_name")

            # fallback if metadata missing
            if not file_name:
                source_path = doc.metadata.get("source", "")
                file_name = os.path.basename(source_path)

            sources.add(file_name)

        return answer, list(sources)

    except Exception as e:
        return f"❌ Error: {str(e)}", []