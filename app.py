import streamlit as st
from langchain_ollama import OllamaLLM
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
import os
from dotenv import load_dotenv


load_dotenv()

# Initialize models and vector DB
embeddings = OllamaEmbeddings(model=os.getenv("EMBEDDING_MODEL"))
db = Chroma(
    collection_name=os.getenv("COLLECTION_NAME"),
    embedding_function=embeddings,
    persist_directory=os.getenv("DATABASE_LOCATION"),
)
llm = OllamaLLM(model=os.getenv("LLM_MODEL"))

# Streamlit UI
st.set_page_config(page_title="💬 Local RAG Chatbot", page_icon="🤖")
st.title("💬 Local RAG Chatbot with Ollama + LangChain")

# Chat state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).markdown(msg["content"])

# User input
if question := st.chat_input("Ask a question..."):
    st.chat_message("user").markdown(question)
    st.session_state.messages.append({"role": "user", "content": question})

    # Retrieve relevant documents
    docs = db.similarity_search(question, k=3)
    context = "\n\n".join([d.page_content for d in docs])

    prompt = f"Answer the question based on the context:\n\n{context}\n\nQuestion: {question}"
    response = llm.invoke(prompt)

    st.chat_message("assistant").markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})
