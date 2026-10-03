import os
import re
import chromadb

# Connect to the same vector database that file.py uses
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="my_documents")

def read_docs(folder_path="./docs"):
    """Read all .txt files from the docs folder."""
    documents = []
    filenames = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            filepath = os.path.join(folder_path, filename)
            with open(filepath, "r", encoding="utf-8") as f:
                documents.append(f.read())
                filenames.append(filename)
    return documents, filenames

def chunk_text(text, max_words=300):
    """Split text into paragraph-based chunks."""
    raw_paragraphs = re.split(r"\n\s*\n", text)

    chunks = []
    current_chunk = []
    current_word_count = 0

    for para in raw_paragraphs:
        para = para.strip()
        if not para:
            continue

        para_word_count = len(para.split())

        # If adding this paragraph would blow past max_words, start a new chunk
        if current_word_count + para_word_count > max_words and current_chunk:
            chunks.append(" ".join(current_chunk))
            current_chunk = []
            current_word_count = 0

        current_chunk.append(para)
        current_word_count += para_word_count

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks

def main():
    documents, filenames = read_docs()

    if not documents:
        print("No .txt files found in ./docs — add some files there first.")
        return

    for doc_text, filename in zip(documents, filenames):
        # Remove old chunks for this specific file, in case it shrank or changed
        collection.delete(where={"source": filename})

        chunks = chunk_text(doc_text)
        chunk_ids = [f"{filename}-chunk-{i}" for i in range(len(chunks))]
        chunk_metadatas = [{"source": filename} for _ in chunks]

        # upsert = update if the ID already exists, insert if it's new
        collection.upsert(
            documents=chunks,
            ids=chunk_ids,
            metadatas=chunk_metadatas
        )

        print(f"Ingested {len(chunks)} chunks from {filename}")

    print("Done.")

if __name__ == "__main__":
    main()