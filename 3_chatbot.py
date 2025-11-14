# Local RAG Chatbot using Ollama + Chroma
from dotenv import load_dotenv
import os
from langchain_chroma import Chroma
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document

load_dotenv()

embeddings = OllamaEmbeddings(model=os.getenv("EMBEDDING_MODEL"))

db = Chroma(
    collection_name=os.getenv("COLLECTION_NAME"),
    embedding_function=embeddings,
    persist_directory=os.getenv("DATABASE_LOCATION"),
)

print("Loaded LLM_MODEL:", os.getenv("LLM_MODEL"))

llm = OllamaLLM(model=os.getenv("LLM_MODEL"))

#RAG Chatbot
def rag_chatbot(question):
    #To Retrieve the top 3 most relevant chunks
    docs = db.similarity_search(question, k=3)

    context = "\n\n".join([d.page_content for d in docs])

    template = """You are a helpful AI assistant that answers questions based only on the following context.

    Context:
    {context}

    Question:
    {question}

    Answer concisely using only the context above.
    """

    prompt = PromptTemplate(template=template, input_variables=["context", "question"])
    formatted_prompt = prompt.format(context=context, question=question)

    answer = llm.invoke(formatted_prompt)
    return answer


print("\n Local RAG Chatbot ready! Ask anything (type 'exit' to quit)\n")

while True:
    question = input("You: ")
    if question.lower() in ["exit", "quit", "bye"]:
        print(" Goodbye!")
        break

    response = rag_chatbot(question)
    print(f"\nAI: {response}\n")
