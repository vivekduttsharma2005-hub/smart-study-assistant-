import streamlit as st
from rag_pipeline import load_pdf, create_vectorstore_from_docs, create_qa_chain

st.set_page_config(page_title="Smart Study Assistant", layout="wide")
st.title("📚 Smart Study Assistant (RAG)")

# Session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None

# Upload PDFs
uploaded_files = st.file_uploader(
    "Upload PDFs",
    type="pdf",
    accept_multiple_files=True
)

# Process PDFs
if uploaded_files and st.session_state.qa_chain is None:
    with st.spinner("Processing PDFs..."):
        try:
            all_docs = []

            for file in uploaded_files:
                path = f"temp_{file.name}"
                with open(path, "wb") as f:
                    f.write(file.read())

                docs = load_pdf(path)
                all_docs.extend(docs)

            if not all_docs:
                st.error("❌ No text extracted from PDFs")
                st.stop()

            vectorstore = create_vectorstore_from_docs(all_docs)
            st.session_state.qa_chain = create_qa_chain(vectorstore)

            st.success("✅ PDFs processed successfully!")

        except Exception as e:
            st.error(f"❌ Error: {e}")
            st.stop()

# Show chat
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# Input
query = st.chat_input("Ask something from PDFs")

if query:
    if not st.session_state.qa_chain:
        st.warning("Upload PDFs first")
        st.stop()

    st.session_state.messages.append({"role": "user", "content": query})

    with st.chat_message("user"):
        st.write(query)

    with st.spinner("Thinking..."):
        result = st.session_state.qa_chain({"query": query})

    answer = result["result"]
    sources = result["source_documents"]

    st.session_state.messages.append({"role": "assistant", "content": answer})

    with st.chat_message("assistant"):
        st.write(answer)

        if sources:
            st.markdown("### 📄 Sources")
            for doc in sources:
                st.write(doc.page_content[:200] + "...")