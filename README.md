# Rag_on_llm
# 📚 RAG AI Document Assistant

A **Retrieval-Augmented Generation (RAG) system** that allows users to ask questions about their documents.  
The system retrieves relevant information from stored documents and generates answers using a large language model.

This project combines **vector search, embeddings, and LLM reasoning** to provide accurate answers grounded in your data.

---

# 🚀 Features

✔ Multi-document support (PDF, TXT, CSV, DOCX, Excel, JSON)  
✔ Automatic document chunking  
✔ Fast vector similarity search  
✔ Context-aware LLM responses  
✔ Chat-style web interface  
✔ Source citation in answers  
✔ Local embeddings (no API cost)  

---

# 🧠 Technologies Used

This project uses the following technologies:

- **Python**
- **FAISS** for vector search
- **Sentence Transformers** for embeddings
- **Groq API** for LLM inference
- **Streamlit** for the web interface
- **LangChain** document utilities

### Models

Embedding Model  
`all-MiniLM-L6-v2`

LLM Model  
`llama-3.1-8b-instant`

---

# 📂 Project Structure


RAG_ON_LLM
│
├── data/ # Documents to index
│
├── src/
│ ├── init.py
│ ├── data_loader.py # Load documents
│ ├── embedding.py # Chunk + embedding pipeline
│ ├── vectorstore.py # FAISS vector database
│ └── search.py # RAG retrieval + LLM generation
│
├── faiss_store/ # Stored vector index
│
├── app.py # Streamlit web interface
├── requirements.txt
├── .env
└── README.md


---

# ⚙️ Installation

### 1️⃣ Clone the repository

```bash
git clone https://github.com/yourusername/rag-ai-assistant.git

cd rag-ai-assistant
2️⃣ Create a virtual environment
python -m venv rag

Activate environment

Windows

rag\Scripts\activate

Mac/Linux

source rag/bin/activate
3️⃣ Install dependencies
pip install -r requirements.txt
🔑 Environment Variables

Create a .env file in the root directory.

GROQ_API_KEY=your_groq_api_key_here

You can get an API key from:

https://console.groq.com

▶️ Running the Application

Start the web application:

streamlit run app.py

Then open:

http://localhost:8501
📄 Adding Documents

Place documents inside the data/ directory.

Example:

data/
  machine_learning.pdf
  notes.txt
  dataset.csv

The system automatically loads and indexes them.

🔍 How the System Works

The RAG pipeline follows these steps:

User Question
      ↓
Text Embedding
      ↓
Vector Similarity Search
      ↓
Retrieve Relevant Context
      ↓
LLM Generation
      ↓
Answer + Sources
🧪 Testing the System

You can test the pipeline directly from Python.

Example:

from src.search import RAGSearch

rag = RAGSearch()

question = "What is attention mechanism?"

answer = rag.search_and_answer(question)

print(answer)
📊 Example Query

Input

What is attention mechanism?

Output

The attention mechanism allows neural networks to focus on the most relevant
parts of an input sequence when generating outputs.

Sources:
data/machine_learning.pdf
🎨 Web Interface

The Streamlit interface provides:

• Chat-style conversation
• Retrieval-based answers
• Source citations
• Query history

📈 Future Improvements

Planned enhancements:

Hybrid search (BM25 + Vector)

Reranking models

Streaming responses

Document upload UI

Multi-user chat sessions

Evaluation metrics for RAG

🤝 Contributing

Contributions are welcome.

Steps:

Fork the repository

Create a new branch

Submit a pull request

📜 License

This project is licensed under the MIT License.

⭐ Acknowledgements

Special thanks to the open-source community and the teams behind:

LangChain

FAISS

Sentence Transformers

Streamlit

Groq

👨‍💻 Author

Your Name

GitHub: https://github.com/Om700-create

LinkedIn: https://www.linkedin.com/in/narayan-bhandari/
