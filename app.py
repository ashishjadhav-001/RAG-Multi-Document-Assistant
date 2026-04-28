import streamlit as st

st.set_page_config(page_title="RAG Book Assistant", layout="wide")

from dotenv import load_dotenv
import os

from vector_store import create_vectorstore
from rag_pipeline import ask_question

load_dotenv()

st.title("📚 RAG Book Assistant")
st.write("Upload PDFs and ask questions from the documents")

# -------------------------------
# Session State
# -------------------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "current_files" not in st.session_state:
    st.session_state.current_files = []

if "persist_directory" not in st.session_state:
    st.session_state.persist_directory = "chroma_db/main_db"
    os.makedirs(st.session_state.persist_directory, exist_ok=True)

if "file_paths" not in st.session_state:
    st.session_state.file_paths = []

if "db_ready" not in st.session_state:
    st.session_state.db_ready = False

# -------------------------------
# Sidebar
# -------------------------------
with st.sidebar:
    st.header("⚙️ Settings")

    if st.button("🧹 Clear Chat", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()

    if st.session_state.current_files:
        st.markdown("**📄 Files in DB:**")
        for f in st.session_state.current_files:
            st.markdown(f"- {f}")

# -------------------------------
# Upload PDFs
# -------------------------------
uploaded_files = st.file_uploader(
    "Upload PDF files",
    type="pdf",
    accept_multiple_files=True
)

if uploaded_files:
    os.makedirs("temp_files", exist_ok=True)

    new_files_added = False

    for file in uploaded_files:
        path = f"temp_files/{file.name}"

        if file.name not in st.session_state.current_files:
            with open(path, "wb") as f:
                f.write(file.read())

            st.session_state.file_paths.append(path)
            st.session_state.current_files.append(file.name)
            new_files_added = True

    if new_files_added:
        st.session_state.db_ready = False
        st.success("✅ New PDFs added. Please update the database.")

# -------------------------------
# Update Vector DB
# -------------------------------
if st.session_state.file_paths:
    if not st.session_state.db_ready:
        if st.button("Update Vector Database"):
            with st.spinner("Updating database..."):
                create_vectorstore(
                    pdf_paths=st.session_state.file_paths,
                    persist_directory=st.session_state.persist_directory
                )
                st.session_state.db_ready = True

            st.success("✅ Database ready!")

# -------------------------------
# Chat Section
# -------------------------------
if st.session_state.db_ready:
    st.divider()
    st.success(f"📄 Using {len(st.session_state.current_files)} files")

    query = st.chat_input("Ask questions from your PDFs...")

    if query:
        with st.spinner("Thinking..."):
            response = ask_question(
                query=query,
                persist_directory=st.session_state.persist_directory
            )

            st.session_state.chat_history.append(("user", query))
            st.session_state.chat_history.append(("ai", response))

    for role, message in st.session_state.chat_history:
        if role == "user":
            st.chat_message("user").write(message)
        else:
            st.chat_message("assistant").write(message)

else:
    st.info("📌 Upload PDFs and click 'Update Vector Database' to start.")