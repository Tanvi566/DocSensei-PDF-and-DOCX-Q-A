from utils.loader import load_document
from utils.splitter import split_documents
from utils.vectorstore import create_vectorstore
from utils.qa import ask_question

docs = load_document("data/sample.pdf")

chunks = split_documents(docs)

vectorstore = create_vectorstore(chunks)

question = input("Ask a question: ")

answer, sources = ask_question(vectorstore, question)

print("\nAnswer:\n")
print(answer)

print("\nSources:\n")

for source in sources:
    print(source.metadata)