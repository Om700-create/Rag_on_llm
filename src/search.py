import os
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage

from src.vectorstore import FaissVectorStore
from src.data_loader import load_all_documents

load_dotenv()


class RAGSearch:

    def __init__(self):

        persist_dir = "faiss_store"

        self.vectorstore = FaissVectorStore(persist_dir)

        faiss_path = os.path.join(persist_dir, "faiss.index")

        if not os.path.exists(faiss_path):

            print("[INFO] Building vector store")

            docs = load_all_documents("data")

            self.vectorstore.build_from_documents(docs)

        else:

            print("[INFO] Loading vector store")

            self.vectorstore.load()

        self.llm = ChatGroq(
            api_key=os.getenv("GROQ_API_KEY"),
            model_name="llama-3.1-8b-instant",
            temperature=0.1
        )

        print("[INFO] Groq LLM ready")

    def search_and_answer(self, query):

        results = self.vectorstore.query(query, top_k=3)

        context = "\n\n".join([r["text"] for r in results])

        if not context:

            return "No relevant context found."

        prompt = f"""
Use the following context to answer the question.

Context:
{context}

Question:
{query}

Answer:
"""

        response = self.llm.invoke([
            HumanMessage(content=prompt)
        ])

        return response.content


if __name__ == "__main__":

    rag = RAGSearch()

    question = "What is attention mechanism?"

    answer = rag.search_and_answer(question)

    print("\nAnswer:\n")

    print(answer)