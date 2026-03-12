import os
import faiss
import pickle
import numpy as np

from sentence_transformers import SentenceTransformer

from src.embedding import EmbeddingPipeline
from src.data_loader import load_all_documents


class FaissVectorStore:

    def __init__(self, persist_dir="faiss_store", embedding_model="all-MiniLM-L6-v2"):

        self.persist_dir = persist_dir

        os.makedirs(self.persist_dir, exist_ok=True)

        self.model = SentenceTransformer(embedding_model)

        self.index = None

        self.metadata = []

        print("[INFO] FAISS Vector Store initialized")

    def build_from_documents(self, documents):

        pipe = EmbeddingPipeline()

        chunks = pipe.chunk_documents(documents)

        embeddings = pipe.embed_chunks(chunks)

        dim = embeddings.shape[1]

        self.index = faiss.IndexFlatL2(dim)

        self.index.add(embeddings)

        self.metadata = [{"text": c.page_content} for c in chunks]

        self.save()

    def save(self):

        faiss.write_index(
            self.index,
            os.path.join(self.persist_dir, "faiss.index")
        )

        with open(os.path.join(self.persist_dir, "metadata.pkl"), "wb") as f:

            pickle.dump(self.metadata, f)

        print("[INFO] Vector store saved")

    def load(self):

        self.index = faiss.read_index(
            os.path.join(self.persist_dir, "faiss.index")
        )

        with open(os.path.join(self.persist_dir, "metadata.pkl"), "rb") as f:

            self.metadata = pickle.load(f)

        print("[INFO] Vector store loaded")

    def query(self, query_text, top_k=3):

        query_embedding = self.model.encode([query_text]).astype("float32")

        distances, indices = self.index.search(query_embedding, top_k)

        results = []

        for idx, dist in zip(indices[0], distances[0]):

            results.append({
                "text": self.metadata[idx]["text"],
                "score": float(dist)
            })

        return results


if __name__ == "__main__":

    docs = load_all_documents("../data")

    store = FaissVectorStore()

    store.build_from_documents(docs)

    store.load()

    print(store.query("What is attention mechanism?"))