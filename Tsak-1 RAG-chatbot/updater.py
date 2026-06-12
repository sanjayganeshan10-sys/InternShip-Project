import os
import time
import schedule
from dotenv import load_dotenv

load_dotenv()

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings  
from langchain_community.vectorstores import Chroma

DB_DIR = "./chroma_db"
SOURCE_FILE = "knowledge_source.txt"

embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2-preview")

def fetch_and_update_knowledge():
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Checking for updates...")
    
    if not os.path.exists(SOURCE_FILE):
        print(f"Source file {SOURCE_FILE} not found.")
        return

    try:
        loader = TextLoader(SOURCE_FILE)
        documents = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        docs = text_splitter.split_documents(documents)
        
        if not docs:
            print("Knowledge source is empty. Standing by...")
            return

        vector_store = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)
        vector_store.add_documents(docs)
        
        print(f"Success! Vector DB updated via Google API with {len(docs)} chunks.")

    except Exception as e:
        print(f"An error occurred: {e}")

schedule.every(1).minutes.do(fetch_and_update_knowledge)

if __name__ == "__main__":
    print("Continuous knowledge base updater (Google Mode) is running...")
    fetch_and_update_knowledge()
    
    while True:
        schedule.run_pending()
        time.sleep(1)