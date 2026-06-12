import os
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

load_dotenv()

DB_DIR = "./chroma_db"

embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2-preview")
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3)

system_instruction = """
You are an advanced, context-aware AI Assistant. Your goal is to provide insightful, clear, and conversational responses to the user.

You are equipped with a local knowledge base context. Use it naturally:
1. If the user asks general questions or greets you (like 'hi', 'hello', 'what can you do'), respond politely, warmly, and explain your purpose dynamically without saying "I don't know".
2. When answering specific questions about health, recovery, or nutrition, rely on the provided context to offer structured, professional guidelines.
3. If the user asks for something outside the bounds of your text document, gracefully redirect them or explain what you can help with based on your system, keeping the conversation fluid and natural.

Context:
{context}
"""

prompt = ChatPromptTemplate.from_messages([
    ("system", system_instruction),
    ("human", "{input}")
])

if not os.path.exists(DB_DIR):
    print("Database directory not found! Run updater.py first.")
    exit()

vector_store = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)

question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(vector_store.as_retriever(search_kwargs={"k": 2}), question_answer_chain)

print("\n=== AI Chatbot Assistant Active ===")
print("Type 'exit' to close.\n")

while True:
    user_input = input("You: ")
    if user_input.strip().lower() == 'exit':
        print("Goodbye!")
        break
        
    if not user_input.strip():
        continue

    try:
        response = rag_chain.invoke({"input": user_input})
        print(f"\nBot: {response['answer']}\n")
    except Exception as e:
        print(f"\nAn error occurred: {e}\n")