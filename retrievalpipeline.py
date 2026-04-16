import os
import uuid
import numpy as np
import chromadb
from typing import List, Dict
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage

# Load environment variables
load_dotenv()


# ---------------------------
# EMBEDDING MANAGER
# ---------------------------
class EmbeddingManager:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def embed(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.array([])

        return self.model.encode(
            texts,
            batch_size=32,
            show_progress_bar=False,
            normalize_embeddings=True
        )


# ---------------------------
# VECTOR STORE
# ---------------------------
class ConversationVectorStore:
    def __init__(self, collection_name="chat_memory", persist_dir="./chat_db"):
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.collection = self.client.get_or_create_collection(name=collection_name)

    def add_messages(self, messages: List[Dict[str, str]], embeddings: np.ndarray):
        if not messages or len(messages) != len(embeddings):
            return

        ids = [f"msg_{uuid.uuid4().hex}" for _ in messages]
        documents = [m["content"] for m in messages]
        metadatas = [{"role": m["role"]} for m in messages]

        self.collection.add(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
            embeddings=embeddings.tolist()
        )

    def search(self, query_embedding: np.ndarray, top_k: int = 5):
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k
        )

        if not results.get("documents") or not results["documents"][0]:
            return []

        return [
            {
                "content": doc,
                "role": meta.get("role", "unknown"),
                "score": round(1 - dist, 4)
            }
            for doc, meta, dist in zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0]
            )
        ]


# ---------------------------
# RETRIEVER
# ---------------------------
class ConversationRetriever:
    def __init__(self, store: ConversationVectorStore, embedder: EmbeddingManager):
        self.store = store
        self.embedder = embedder

    def retrieve(self, query: str, top_k: int = 5):
        if not query.strip():
            return []

        query_embedding = self.embedder.embed([query])[0]
        return self.store.search(query_embedding, top_k)


# ---------------------------
# LLM SETUP
# ---------------------------
def get_llm():
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("Missing GROQ_API_KEY in environment variables.")

    return ChatGroq(
        groq_api_key=api_key,
        model_name="llama-3.3-70b-versatile",
        temperature=0.7,
        max_tokens=1000
    )


# ---------------------------
# RAG FUNCTION
# ---------------------------
def rag_simple(query: str, retriever: ConversationRetriever, llm, top_k: int = 3):
    results = retriever.retrieve(query, top_k=top_k)

    if not results:
        return "No relevant memory found."

    context = "\n".join(
        f"{r['role']}: {r['content']}" for r in results
    )

    prompt = f"""
You are a helpful assistant.

Use the conversation context to answer the question clearly and concisely.

Context:
{context}

Question: {query}

Answer:
"""

    response = llm.invoke([HumanMessage(content=prompt)])
    return response.content


# ---------------------------
# MAIN LOOP
# ---------------------------
def main():
    embedder = EmbeddingManager()
    store = ConversationVectorStore()
    retriever = ConversationRetriever(store, embedder)
    llm = get_llm()

    print("Chat started (type 'exit' to quit)\n")

    while True:
        query = input("Ask: ").strip()

        if query.lower() == "exit":
            break

        # Generate answer
        answer = rag_simple(query, retriever, llm)

        print("\nAnswer:\n")
        print(answer)
        print("\n" + "-" * 50 + "\n")

        # Store conversation
        messages = [
            {"role": "user", "content": query},
            {"role": "assistant", "content": answer},
        ]

        texts = [m["content"] for m in messages]
        embeddings = embedder.embed(texts)

        store.add_messages(messages, embeddings)


if __name__ == "__main__":
    main()