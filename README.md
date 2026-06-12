# Local Knowledge Retrieval-Augmented Generation (RAG) Chatbot

An advanced, context-aware AI assistant built using a local vector database architecture (ChromaDB) integrated with the Google Gemini LLM framework. The system intelligently indexes custom corporate or medical documentation datasets and answers queries strictly utilizing the provided context.

## 🚀 Features
- **Semantic Search Vector Pipeline:** Converts raw text documents into high-dimensional vector embeddings using `gemini-embedding-2-preview`.
- **Local Persistence Storage:** Uses ChromaDB to store indexed knowledge locally, minimizing API latency and costs.
- **Context-Guided Responses:** Leverages `gemini-2.5-flash` with engineered prompt templates to enforce boundaries and prevent AI hallucinations.
- **Interactive Terminal Interface:** Features automated greeting routing, intent recognition, and structured fallback responses.

## 🛠️ Tech Stack
- **Language:** Python 3.10+
- **LLM API:** Google Gemini Framework
- **Vector Database:** ChromaDB
- **Orchestration:** LangChain / Native Google GenAI SDK

## ⚙️ Installation & Setup

1. **Clone the repository and navigate into the project folder:**
```bash
   cd local-rag-chatbot