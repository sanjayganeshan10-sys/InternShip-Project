import os
import streamlit as st
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

st.set_page_config(page_title="AI RAG Assistant", page_icon="🤖", layout="centered")
st.title("🤖 Advanced RAG Knowledge Assistant")
st.caption("A production-grade context-aware intelligence platform")

load_dotenv()
DB_DIR = "./chroma_db"

@st.cache_resource
def initialize_rag_pipeline():
    """Initializes and caches the heavy database and model setups."""
    if not os.path.exists(DB_DIR):
        return None
    
    embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2-preview")
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3)
    
    system_instruction = """
    You are an advanced, context-aware AI Assistant. Your goal is to provide insightful, clear, and conversational responses to the user.
    You are equipped with a local knowledge base context. Use it naturally to answer questions accurately without giving rigid refusals.
    Context:
    {context}
    """
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_instruction),
        ("human", "{input}")
    ])
    
    vector_store = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    return create_retrieval_chain(vector_store.as_retriever(search_kwargs={"k": 2}), question_answer_chain)

rag_chain = initialize_rag_pipeline()

if rag_chain is None:
    st.error("Database directory not found! Please run 'python updater.py' first to build the index.")
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! I am your advanced knowledge assistant. Ask me anything about your data documents!"}
    ]
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if user_query := st.chat_input("Ask a question..."):
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)
        
    with st.chat_message("assistant"):
        with st.spinner("Analyzing context indexes..."):
            try:
                response = rag_chain.invoke({"input": user_query})
                output_text = response["answer"]
                st.markdown(output_text)
                st.session_state.messages.append({"role": "assistant", "content": output_text})
            except Exception as e:
                st.error(f"Execution Error: {e}")