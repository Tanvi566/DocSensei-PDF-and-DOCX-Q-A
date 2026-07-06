from langchain_text_splitters import RecursiveCharacterTextSplitter


def split_documents(documents,
                    chunk_size=500,
                    chunk_overlap=50):
    """
    Split documents into smaller chunks.

    Args:
        documents: LangChain documents
        chunk_size: Size of each chunk
        chunk_overlap: Overlap between chunks

    Returns:
        List of split document chunks.
    """

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    chunks = splitter.split_documents(documents)

    return chunks