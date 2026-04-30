import streamlit as st
import os
from dotenv import load_dotenv

from vector_store import create_vectorstore
from rag_pipeline import ask_question

st.set_page_config(page_title="RAG Book Assistant", layout="wide")

load_dotenv()

st.title("📚 RAG Book Assistant")
st.write("Upload PDFs and ask questions from the documents")

# -------------------------------
# Session State Initialization
# -------------------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "current_files" not in st.session_state:
    st.session_state.current_files = []

if "persist_directory" not in st.session_state:
    st.session_state.persist_directory = "/tmp/chroma_db"
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
    os.makedirs("/tmp/temp_files", exist_ok=True)

    new_files_added = False

    for file in uploaded_files:
        path = f"/tmp/temp_files/{file.name}"

        if file.name not in st.session_state.current_files:
            with open(path, "wb") as f:
                f.write(file.read())

            st.session_state.file_paths.append(path)
            st.session_state.current_files.append(file.name)
            new_files_added = True

    if new_files_added:
        st.session_state.db_ready = False
        st.success("✅ New PDFs added. Click below to update database.")

# -------------------------------
# Update Vector DB
# -------------------------------
st.divider()

if st.session_state.file_paths:

    st.write(f"📂 Files ready: {len(st.session_state.file_paths)}")

    if st.button("🚀 Update Vector Database", use_container_width=True):

        try:
            with st.spinner("Updating database..."):
                create_vectorstore(
                    pdf_paths=st.session_state.file_paths,
                    persist_directory=st.session_state.persist_directory
                )

            st.session_state.db_ready = True
            st.rerun()

        except Exception as e:
            st.session_state.db_ready = False
            st.error(f"❌ Error while creating DB:\n{str(e)}")

else:
    st.warning("⚠️ Upload PDFs first")

# -------------------------------
# DEBUG
# -------------------------------
st.caption(f"DEBUG → DB Ready: {st.session_state.db_ready}")

# -------------------------------
# Chat Section
# -------------------------------
if st.session_state.db_ready:
    st.divider()
    st.success(f"📄 Using {len(st.session_state.current_files)} files")

    query = st.chat_input("Ask questions from your PDFs...")

    if query:
        try:
            with st.spinner("Thinking..."):
                response, sources = ask_question(
                    query=query,
                    persist_directory=st.session_state.persist_directory
                )

            # Store chat history
            st.session_state.chat_history.append(("user", query))
            st.session_state.chat_history.append(("ai", response, sources))

        except Exception as e:
            st.error(f"❌ Error while answering:\n{str(e)}")

    # -------------------------------
    # Display Chat
    # -------------------------------
    for item in st.session_state.chat_history:
        role = item[0]

        if role == "user":
            st.chat_message("user").write(item[1])

        else:
            response = item[1]
            sources = item[2]

            with st.chat_message("assistant"):
                st.write(response)

                if sources:
                    st.markdown("**📄 Source PDFs:**")
                    for s in sources:
                        st.markdown(f"- {s}")

else:
    st.info("📌 Upload PDFs and click 'Update Vector Database' to start.")