"""
ingest.py
Reads the scraped .txt files from data/raw/, splits them into chunks,
embeds them with a free local embedding model, and stores everything
in a persistent Chroma vector database at ./chroma_db/.

Run this once after scraper.py, and again any time you add/update docs.

Usage:
    python ingest.py
"""

import os
import glob

from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

RAW_DATA_DIR = "data/raw"
PERSIST_DIR = "chroma_db"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50


def load_documents():
    docs = []
    filepaths = glob.glob(os.path.join(RAW_DATA_DIR, "*.txt"))

    if not filepaths:
        raise FileNotFoundError(
            f"No .txt files found in {RAW_DATA_DIR}/. Run scraper.py first."
        )

    for filepath in filepaths:
        loader = TextLoader(filepath, encoding="utf-8")
        loaded = loader.load()

        # Pull the SOURCE_URL line we wrote in scraper.py into metadata,
        # so the chatbot can cite where an answer came from later.
        for doc in loaded:
            first_line = doc.page_content.split("\n", 1)[0]
            if first_line.startswith("SOURCE_URL:"):
                doc.metadata["source_url"] = first_line.replace("SOURCE_URL:", "").strip()
        docs.extend(loaded)

    print(f"Loaded {len(docs)} documents from {RAW_DATA_DIR}/")
    return docs


def chunk_documents(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    print(f"Split into {len(chunks)} chunks (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")
    return chunks


def build_vector_store(chunks):
    print(f"Loading embedding model: {EMBEDDING_MODEL} (first run downloads it, ~90MB)")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    print("Embedding chunks and writing to Chroma — this can take a minute...")
    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=PERSIST_DIR,
    )
    vectordb.persist()
    print(f"Done. Vector store saved to ./{PERSIST_DIR}/")


if __name__ == "__main__":
    documents = load_documents()
    chunks = chunk_documents(documents)
    build_vector_store(chunks)
