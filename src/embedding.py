from typing import List
import numpy as np

from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema import Document

from src.data_loader import load_all_documents


class EmbeddingPipeline:

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        chunk_size: int = 1000,
        chunk_overlap: int = 200
    ):

        self.model = SentenceTransformer(model_name)

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        print(f"[INFO] Loaded embedding model: {model_name}")

    def chunk_documents(self, documents: List[Document]):

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )

        chunks = splitter.split_documents(documents)

        print(f"[INFO] Created {len(chunks)} chunks")

        return chunks

    def embed_chunks(self, chunks: List[Document]):

        texts = [chunk.page_content for chunk in chunks]

        embeddings = self.model.encode(
            texts,
            show_progress_bar=True
        )

        embeddings = np.array(embeddings).astype("float32")

        print(f"[INFO] Embedding shape: {embeddings.shape}")

        return embeddings


if __name__ == "__main__":

    docs = load_all_documents("../data")

    pipe = EmbeddingPipeline()

    chunks = pipe.chunk_documents(docs)

    embeddings = pipe.embed_chunks(chunks)

    print("\nExample embedding vector:\n")

    print(embeddings[0])