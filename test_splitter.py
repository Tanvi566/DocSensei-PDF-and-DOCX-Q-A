from utils.loader import load_document
from utils.splitter import split_documents

docs = load_document("data/sample.pdf")

chunks = split_documents(docs)

print(f"Pages: {len(docs)}")
print(f"Chunks: {len(chunks)}")

print("\nFirst Chunk:\n")
print(chunks[0].page_content)

print("\nMetadata:")
print(chunks[0].metadata)