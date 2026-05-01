# rag_pipeline.py

import os
from dotenv import load_dotenv

from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from langchain_mistralai import ChatMistralAI, MistralAIEmbeddings

# ✅ Load environment variables
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


# =========================
# Load Vector Store
# =========================
def load_vectorstore(persist_directory):
    if not os.path.exists(persist_directory):
        raise FileNotFoundError("❌ Vector DB not found. Please create it first.")

    return Chroma(
        persist_directory=persist_directory,
        embedding_function=embedding_model
    )


# =========================
# Create Runnable QA Chain
# =========================
def get_qa_chain(retriever):

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

    prompt = PromptTemplate.from_template(prompt_template)

    # ✅ Runnable pipeline
    qa_chain = (
        {
            "context": retriever,
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return qa_chain


# =========================
# Ask Question Function
# =========================
def ask_question(query, persist_directory):
    try:
        # Load DB
        vectorstore = load_vectorstore(persist_directory)

        # Create retriever
        retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 4}
        )

        # Get relevant documents (for sources)
        docs = retriever.invoke(query)

        # Create chain
        qa_chain = get_qa_chain(retriever)

        # Get answer
        answer = qa_chain.invoke(query)

        # Extract sources
        sources = set()

        for doc in docs:
            file_name = doc.metadata.get("file_name")

            if not file_name:
                source_path = doc.metadata.get("source", "")
                file_name = os.path.basename(source_path)

            sources.add(file_name)

        return answer, list(sources)

    except Exception as e:
        return f"❌ Error: {str(e)}", []