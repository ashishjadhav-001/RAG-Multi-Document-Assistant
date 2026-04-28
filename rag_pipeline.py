from dotenv import load_dotenv
load_dotenv()

from langchain_mistralai import ChatMistralAI, MistralAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate

embedding_model = MistralAIEmbeddings(model="mistral-embed")

def get_vectorstore(persist_directory: str):
    return Chroma(
        persist_directory=persist_directory,
        embedding_function=embedding_model
    )

llm = ChatMistralAI(model="mistral-small-2603")

prompt = ChatPromptTemplate.from_messages([
    ("system",
     """You are a helpful document assistant.

Use the context to answer the question.

If the answer is partially available, answer using available information.

Only say "I could not find this in the document" if NOTHING relevant exists.

Try your best to extract useful meaning from context.

Be clear and helpful."""
     ),
    ("human",
     """Context:
{context}

Question:
{question}
""")
])

def ask_question(query: str, persist_directory: str):
    vectorstore = get_vectorstore(persist_directory)

    results = vectorstore.similarity_search_with_score(query, k=6)
    results = sorted(results, key=lambda x: x[1])

    docs = []

    # 🔥 relaxed filtering
    for doc, score in results:
        if score < 0.5:
            docs.append(doc)

    # 🔥 fallback if nothing found
    if not docs:
        docs = [doc for doc, _ in results[:3]]

    docs = docs[:3]

    context = "\n\n".join([doc.page_content for doc in docs])

    final_prompt = prompt.invoke({
        "context": context,
        "question": query
    })

    response = llm.invoke(final_prompt)

    sources = set()
    for doc in docs:
        source = doc.metadata.get("source", None)
        if source:
            sources.add(source)

    source_text = "\n".join(sources)

    return f"""{response.content}

📄 Source:
{source_text}
"""