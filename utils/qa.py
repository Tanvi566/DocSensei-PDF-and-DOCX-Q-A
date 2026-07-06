from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()


def ask_question(vectorstore, question, top_k):
    """
    Retrieve relevant chunks and ask Gemini.
    """

    # Retrieve top 3 most relevant chunks
    docs = vectorstore.similarity_search(
    question,
    k=top_k
    )

    context = "\n\n".join(
        [doc.page_content for doc in docs]
    )

    prompt = f"""
You are DocSensei.

Answer ONLY using the context below.

If the answer cannot be found in the context,
reply exactly:

I don't know based on the uploaded document.

Context:
{context}

Question:
{question}

Answer:
"""

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0
    )

    response = llm.invoke(prompt)

    return response.content, docs