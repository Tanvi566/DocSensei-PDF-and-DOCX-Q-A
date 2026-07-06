from utils.loader import load_document
from utils.splitter import split_documents
from utils.vectorstore import create_vectorstore

docs = load_document("data/sample.pdf")

chunks = split_documents(docs)

vectorstore = create_vectorstore(chunks)

print("Vector database created successfully!")

print(f"Total chunks stored: {vectorstore._collection.count()}")