import os
from dotenv import load_dotenv
load_dotenv()

# ✅ Updated imports (no deprecated ones)
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq


# Load PDF
def load_pdf(path):
    loader = PyPDFLoader(path)
    return loader.load()


# Create vector DB
def create_vectorstore_from_docs(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    chunks = splitter.split_documents(docs)

    if not chunks:
        raise ValueError("No text found")

    embeddings = HuggingFaceEmbeddings()

    return FAISS.from_documents(chunks, embeddings)


# Create QA chain (modern way)
def create_qa_chain(vectorstore):
    llm = ChatGroq(
        groq_api_key=os.getenv("GROQ_API_KEY"),
        model_name="llama-3.1-8b-instant"
    )

    retriever = vectorstore.as_retriever()

    prompt = ChatPromptTemplate.from_template("""
Answer ONLY from the given context.

Context:
{context}

Question:
{question}
""")

    def qa_chain(inputs):
        query = inputs["query"]

        docs = retriever.invoke(query)

        if not docs:
            return {
                "result": "No relevant information found.",
                "source_documents": []
            }

        context = "\n".join([d.page_content for d in docs])

        response = llm.invoke(
            prompt.format(context=context, question=query)
        )

        return {
            "result": response.content,
            "source_documents": docs
        }

    return qa_chain