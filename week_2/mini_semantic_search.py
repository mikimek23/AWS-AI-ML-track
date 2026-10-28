import pandas as pd
import json
import re
import chromadb
def load_faqs(path:str)->pd.DataFrame:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    extracted_data=[]
    for item in data["rows"]:
        row_content = item["row"]
        extracted_data.append(
            {"question": row_content["question"], "answer": row_content["answer"]}
        )
    df = pd.DataFrame(extracted_data)
    return df


def clean_text(text: str) -> str:
    text = re.sub(r"<[^>]+>", " ", str(text))   # remove HTML tags
    text = re.sub(r"\s+", " ", text)            # collapse extra whitespace/newlines
    return text.strip()
def clean_dataframe(df:pd.DataFrame)->pd.DataFrame:
    df=df.dropna(subset=["question","answer"]).copy()
    df["question"] = df["question"].apply(clean_text)
    df["answer"] = df["answer"].apply(clean_text)
    df = df[(df["question"] != "") & (df["answer"] != "")]
    df = df.drop_duplicates(subset=["question", "answer"]).reset_index(drop=True)
    return df

def chunk_text(text: str, size: int = 50) -> list[str]:
    """Split text into pieces of at most `size` words."""
    words = text.split()
    return [" ".join(words[i:i + size]) for i in range(0, len(words), size)]
def build_chunks(df: pd.DataFrame):
    """Turn each FAQ row into one or more chunks, keeping track of where each came from."""
    ids, documents, metadatas = [], [], []
    for row_id, row in df.iterrows():
        full_text = f"{row['question']} {row['answer']}"
        for chunk_id, piece in enumerate(chunk_text(full_text)):
            ids.append(f"{row_id}-{chunk_id}")        # unique id, e.g. "12-0"
            documents.append(piece)                   # the text that gets embedded
            metadatas.append({"question": row["question"], "answer": row["answer"]})
    return ids, documents, metadatas

def build_collection(ids, documents, metadatas):
    client = chromadb.PersistentClient(path="./chroma_db")
    try:
        client.delete_collection("faqs")
    except Exception:
        pass
 
    collection = client.create_collection(
        name="faqs",
        metadata={"hnsw:space": "cosine"},
    )

    batch = 500
    for i in range(0, len(ids), batch):
        collection.add(
            ids=ids[i:i + batch],
            documents=documents[i:i + batch],
            metadatas=metadatas[i:i + batch],
        )
    return collection
def search(collection, query: str, top_k: int = 3):
    results = collection.query(query_texts=[query], n_results=top_k)
 
    docs = results["documents"][0]
    metas = results["metadatas"][0]
    dists = results["distances"][0]
 
    print(f"\nTop {len(docs)} matches for: '{query}'\n")
    for rank, (doc, meta, dist) in enumerate(zip(docs, metas, dists), start=1):
        similarity = 1 - dist   # cosine similarity: closer to 1 = more similar
        print(f"{rank}. Similarity: {similarity:.2f}")
        print(f"   Q: {meta['question']}")
        print(f"   A: {meta['answer']}\n")
 
 
def main():
    df = load_faqs('MakTek_FCustomer_support_faqs_dataset.json')
    df = clean_dataframe(df)
    print(f"Loaded {len(df)} clean FAQs.")
 
    ids, documents, metadatas = build_chunks(df)
    print(f"Created {len(documents)} chunks. Embedding (first run downloads the model)...")
 
    collection = build_collection(ids, documents, metadatas)
    print("Done. Ready to search!\n")
 
    while True:
        query = input("Ask a question (or type 'exit'): ").strip()
        if query.lower() in {"exit", "quit", ""}:
            break
        search(collection, query)
 
 
if __name__ == "__main__":
    main()