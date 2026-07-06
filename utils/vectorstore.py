from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings


def create_vectorstore(chunks):
    """
    Create a Chroma vector database from document chunks.
    """

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="chroma_db"
    )

    return vectorstore