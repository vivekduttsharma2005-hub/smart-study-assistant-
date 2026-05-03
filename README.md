# 🤖 AI Smart Study Assistant (RAG)

## 📌 Problem

Students waste time searching through large PDFs and notes.

## 🚀 Solution

Upload your study material and ask questions. AI retrieves relevant content and generates accurate answers.

## 🛠️ Tech Stack

* LLM: Groq (LLaMA 3)
* Embeddings: HuggingFace
* Vector DB: FAISS
* Framework: LangChain
* UI: Streamlit

## 📖 How It Works

1. Upload PDF
2. Text is split into chunks
3. Converted into embeddings
4. Stored in FAISS
5. Query retrieves top chunks
6. LLM generates answer

## 💬 Prompts Used

* Context-based answering
* Strict grounding (no hallucination)
* Structured output

## 🧪 Sample Inputs

* What is machine learning?
* Explain neural networks
* Define supervised learning

## 🏃 Run Locally

pip install -r requirements.txt
streamlit run app.py

## 📸 Screenshots

<img width="1910" height="526" alt="002" src="https://github.com/user-attachments/assets/f00e2ba3-c67a-4624-8a1f-d11e45344336" />


## 👤 Author

Vivek Dutt Sharma
