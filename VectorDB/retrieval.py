#!/usr/bin/env python3
"""
retrieval.py (v2) — interactive ChromaDB query tool (aligned with textExtract.py v8.3 + embeddings v2)

Features
--------
✓ Auto-detects the embedding model from embedding_model.txt (ensures dim consistency)
✓ Supports CLI args for top-k, collection name, and persist directory
✓ Normalizes query embeddings for cosine similarity (matches stored vectors)
✓ Displays source, section, cosine similarity, and snippet for quick inspection
✓ Gracefully handles missing files, empty results, and Chroma load errors
"""

import argparse
import json
import os
from pathlib import Path
import chromadb
from sentence_transformers import SentenceTransformer
import numpy as np


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_article_metadata(jsonl_path: Path) -> dict:
    """Load article metadata keyed by article_id."""
    if not jsonl_path.exists():
        print(f"⚠️ Articles file not found: {jsonl_path}")
        return {}
    articles = {}
    with open(jsonl_path, "r", encoding="utf-8") as fh:
        for line in fh:
            try:
                rec = json.loads(line)
                aid = rec.get("article_id") or rec.get("id")
                if aid:
                    articles[aid] = rec
            except json.JSONDecodeError:
                continue
    return articles


def load_embed_model(model_file: Path, fallback: str) -> str:
    """Read the model name from embedding_model.txt if available."""
    if model_file.exists():
        name = model_file.read_text(encoding="utf-8").strip()
        print(f"🧠 Detected embedding model from file: {name}")
        return name
    else:
        print(f"⚠️ No embedding_model.txt found, using fallback model: {fallback}")
        return fallback


def embed_query(text: str, model: SentenceTransformer) -> np.ndarray:
    """Encode and L2-normalize a query."""
    emb = model.encode([text], normalize_embeddings=True)
    return emb


def run_query(query: str, coll, model: SentenceTransformer, articles: dict, top_k: int):
    """Execute a semantic search query against a Chroma collection."""
    query_emb = embed_query(query, model)

    results = coll.query(query_embeddings=query_emb, n_results=top_k)

    # No results?
    if not results or not results.get("documents") or not results["documents"][0]:
        print("⚠️ No results found.")
        return

    print(f"\n🔍 Top {top_k} results for: “{query}”")
    print("-" * 80)

    for i, (doc, dist, meta) in enumerate(
        zip(results["documents"][0], results["distances"][0], results["metadatas"][0])
    ):
        similarity = 1 - dist  # since Chroma stores cosine *distance*
        print(f"\nResult {i+1}")
        print(f"Cosine similarity: {similarity:.3f}")
        print(f"Section: {meta.get('section', 'N/A')}")
        print(f"Source:  {meta.get('source_relpath', 'N/A')}")
        print(f"Text: {doc[:800]}{' …' if len(doc) > 800 else ''}")

        # Optional: small reference snippet if available
        art_id = meta.get("article_id")
        if art_id and art_id in articles:
            refs = articles[art_id].get("references_text", "")
            if refs:
                lines = refs.splitlines()[:2]
                print("References:")
                for line in lines:
                    print("   ", line[:120])
                if len(refs.splitlines()) > 2:
                    print("   ...")

    # Show metadata for top hit
    print("\n📌 Top hit metadata:")
    print(json.dumps(results["metadatas"][0][0], indent=2, ensure_ascii=False))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Interactive ChromaDB retrieval console.")
    parser.add_argument("--persist-dir", type=str, default="chroma_store",
                        help="Directory containing persistent Chroma collections.")
    parser.add_argument("--collection", type=str, default="msk_chunks",
                        help="Collection name to query.")
    parser.add_argument("--top-k", type=int, default=5,
                        help="Number of results to display (default: 5).")
    parser.add_argument("--articles", type=str, default="MSKArticlesINDEX/all_articles.jsonl",
                        help="Path to all_articles.jsonl for metadata lookup.")
    parser.add_argument("--model-fallback", type=str, default="mixedbread-ai/mxbai-embed-large-v1",
                        help="Fallback model name if embedding_model.txt is missing.")
    args = parser.parse_args()

    persist_dir = Path(args.persist_dir)
    collection_name = args.collection
    articles_path = Path(args.articles)
    model_file = Path("embeddings") / "embedding_model.txt"

    print(f"🔧 Persist dir: {persist_dir}")
    print(f"🔧 Collection:  {collection_name}")
    print(f"🔧 Metadata:    {articles_path}")

    model_name = load_embed_model(model_file, args.model_fallback)

    # Load model
    print("\n🚀 Loading embedding model…")
    model = SentenceTransformer(model_name)

    # Initialize Chroma client
    client = chromadb.PersistentClient(path=str(persist_dir))
    try:
        coll = client.get_collection(collection_name)
    except Exception as e:
        print(f"❌ Could not load collection '{collection_name}': {e}")
        print("💡 Try rebuilding your Chroma index or check --persist-dir path.")
        return

    print("✅ Collection loaded successfully.")
    print(f"ℹ️  Stored vectors: {coll.count() if hasattr(coll, 'count') else 'unknown'}\n")

    articles = load_article_metadata(articles_path)

    # Interactive loop
    print("💬 Enter a query (type 'q' or 'quit' to exit)\n")
    while True:
        try:
            query = input("🧠 Query: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n👋 Exiting.")
            break
        if not query:
            continue
        if query.lower() in {"q", "quit", "exit"}:
            print("👋 Exiting.")
            break
        run_query(query, coll, model, articles, args.top_k)
        print("\n" + "-" * 80 + "\n")


if __name__ == "__main__":
    main()
