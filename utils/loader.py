from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader
)


def load_document(file_path):
    """
    Load a PDF or DOCX document.

    Args:
        file_path (str): Path to the uploaded document.

    Returns:
        list: List of LangChain Document objects.
    """

    if file_path.endswith(".pdf"):
        loader = PyPDFLoader(file_path)

    elif file_path.endswith(".docx"):
        loader = Docx2txtLoader(file_path)

    else:
        raise ValueError("Unsupported file format")

    documents = loader.load()

    return documents