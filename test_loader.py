from utils.loader import load_document

docs = load_document("data/sample.pdf")

print(f"Pages loaded: {len(docs)}")

print("\nFirst page:\n")

print(docs[0].page_content[:500])

print("\nMetadata:")

print(docs[0].metadata)