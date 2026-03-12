from pathlib import Path
from typing import List

from langchain.schema import Document
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    CSVLoader,
    Docx2txtLoader,
    JSONLoader
)

from langchain_community.document_loaders.excel import UnstructuredExcelLoader


def load_all_documents(data_dir: str) -> List[Document]:

    data_path = Path(data_dir).resolve()

    if not data_path.exists():
        raise ValueError(f"Data directory does not exist: {data_path}")

    print(f"[INFO] Loading documents from: {data_path}")

    documents = []

    loaders = {
        ".pdf": PyPDFLoader,
        ".txt": TextLoader,
        ".csv": CSVLoader,
        ".docx": Docx2txtLoader,
        ".xlsx": UnstructuredExcelLoader,
        ".json": JSONLoader
    }

    for file_path in data_path.glob("**/*"):

        suffix = file_path.suffix.lower()

        if suffix in loaders:

            print(f"[INFO] Loading {file_path}")

            try:
                loader = loaders[suffix](str(file_path))

                docs = loader.load()

                documents.extend(docs)

                print(f"[INFO] Loaded {len(docs)} documents")

            except Exception as e:

                print(f"[ERROR] Failed loading {file_path}: {e}")

    print(f"[INFO] Total documents loaded: {len(documents)}")

    return documents


if __name__ == "__main__":

    docs = load_all_documents("../data")

    print("Example document:\n")

    print(docs[0])