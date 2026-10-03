import chromadb
import ollama

# Connect to the same vector database you built in ingest.py
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="my_documents")

def ask_question(question):
    # RETRIEVAL: find the most relevant chunks
    results = collection.query(
        query_texts=[question],
        n_results=3
    )

    retrieved_chunks = results["documents"][0]
    retrieved_metadata = results["metadatas"][0]  # e.g. [{"source": "sample.txt"}, ...]

    # Pair each chunk with its source, so we can show it later
    context_parts = []
    sources_used = []
    for chunk, meta in zip(retrieved_chunks, retrieved_metadata):
        source = meta.get("source", "unknown")
        context_parts.append(f"[Source: {source}]\n{chunk}")
        if source not in sources_used:
            sources_used.append(source)

    context = "\n\n---\n\n".join(context_parts)

    print("\n--- RETRIEVED CHUNKS ---")
    for i, chunk in enumerate(retrieved_chunks):
        print(f"\n[Chunk {i+1}]:\n{chunk}")
    print("\n--- END CHUNKS ---\n")

    # GENERATION: ask the local model to answer using only the retrieved context
    prompt = f"""Answer the question using only the context below. If the answer isn't in the context, say so.

Context:
{context}

Question: {question}

Answer:"""

    response = ollama.chat(
        model="llama3.2",
        messages=[{"role": "user", "content": prompt}]
    )

    answer = response["message"]["content"]

    # Append sources ourselves - reliable, doesn't depend on the model remembering
    sources_line = "\n\nSources: " + ", ".join(sources_used)

    return answer + sources_line

if __name__ == "__main__":
    while True:
        q = input("Ask a question (or type 'quit'): ")
        if q.lower() == "quit":
            break
        answer = ask_question(q)
        print("\n" + answer + "\n")