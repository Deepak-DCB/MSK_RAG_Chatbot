#!/usr/bin/env python3
"""
embeddings.py (v3 — readable refactor, compatible with textExtract.py v8.5.1+)

Generate dual L2-normalized embeddings for RAG retrieval:
  • embeddings.npy          (from 'embed_text': clean text with header context)
  • embeddingsImages.npy    (from 'text_with_images': includes image alt/caption text)

Features:
  ✓ Works with any SentenceTransformer model (default: mixedbread-ai/mxbai-embed-large-v1)
  ✓ L2-normalizes embeddings for cosine similarity scoring
  ✓ Robust handling of empty/malformed rows (preserves row count with placeholder vectors)
  ✓ Auto-detects GPU; supports --device cpu/cuda override
  ✓ Accepts chunks.parquet or chunks.jsonl (auto-fallback)
  ✓ Records model name for reproducibility

Usage:
    python embeddings.py --input chunks.parquet --output-dir ./embeddings
    python embeddings.py -i ./data/ -d ./out -m BAAI/bge-base-en-v1.5
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

# ----------------------------- Logging ------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s %(message)s",
    handlers=[logging.StreamHandler()]
)
log = logging.getLogger("embeddings")

# ----------------------------- Constants ----------------------------------------

# Dynamically infer project root (two levels up from this file)
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_INPUT_PATH = PROJECT_ROOT / "MSKArticlesINDEX" / "chunks.parquet"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "embeddings"
DEFAULT_MODEL = "mixedbread-ai/mxbai-embed-large-v1"
DEFAULT_BATCH_SIZE = 128

EMPTY_TEXT_PLACEHOLDER = "[EMPTY]"
EMPTY_TEXT_PATTERNS = {"", "none", "nan", "[]", "{}", "null"}

# NEW: control whether to also generate image-augmented embeddings
IMAGE_EMBEDDING = False  # If False, only 'embed_text' is embedded

# ----------------------------- Data loading -------------------------------------


def find_chunks_file(input_path: Path) -> Path:
    """
    Locate chunks.parquet or chunks.jsonl given a file path or directory.
    Raises FileNotFoundError if no valid file is found.
    """
    candidates: List[Path] = []

    if input_path.is_file():
        candidates = [input_path]
    elif input_path.is_dir():
        # Try common filenames in directory
        candidates = [
            input_path / "chunks.parquet",
            input_path / "chunks.jsonl",
        ]
    else:
        # Try alternate extensions if user specified wrong suffix
        if input_path.suffix.lower() == ".parquet":
            candidates = [input_path, input_path.with_suffix(".jsonl")]
        elif input_path.suffix.lower() in {".jsonl", ".json"}:
            candidates = [input_path, input_path.with_suffix(".parquet")]
        else:
            candidates = [input_path]

    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            return candidate

    tried = ", ".join(str(c) for c in candidates)
    raise FileNotFoundError(
        f"Could not locate chunks file. Tried: {tried}\n"
        f"Ensure the file exists or specify a valid path."
    )


def load_chunks_dataframe(input_path: Path) -> pd.DataFrame:
    """
    Load chunks DataFrame from Parquet or JSONL.
    Auto-detects format from file extension.
    """
    chunks_file = find_chunks_file(input_path)
    suffix = chunks_file.suffix.lower()

    log.info("📂 Loading chunks from: %s", chunks_file)

    if suffix == ".parquet":
        return pd.read_parquet(chunks_file)
    elif suffix in {".jsonl", ".json"}:
        return pd.read_json(chunks_file, lines=True, orient="records")
    else:
        raise ValueError(f"Unsupported file format: {suffix}. Use .parquet or .jsonl")


def extract_text_column(df: pd.DataFrame, column_name: str) -> List[str]:
    """
    Extract a text column as a list of strings.
    Validates column existence and converts to string dtype.
    """
    if column_name not in df.columns:
        available = ", ".join(df.columns)
        raise ValueError(
            f"Column '{column_name}' not found in DataFrame.\n"
            f"Available columns: {available}"
        )

    texts = df[column_name].astype(str).tolist()
    log.info("📝 Extracted %d rows from column '%s'", len(texts), column_name)
    return texts


# ----------------------------- Text sanitization --------------------------------


def sanitize_text_for_embedding(text: str) -> str:
    """
    Clean a text string for embedding encoding.
    Replaces empty/null-like values with a placeholder to maintain row alignment.
    """
    if not isinstance(text, str):
        return EMPTY_TEXT_PLACEHOLDER

    cleaned = text.strip()
    if cleaned.lower() in EMPTY_TEXT_PATTERNS:
        return EMPTY_TEXT_PLACEHOLDER

    return cleaned


def sanitize_texts(texts: List[str]) -> List[str]:
    """Sanitize a list of texts, replacing empties with placeholder."""
    return [sanitize_text_for_embedding(t) for t in texts]


# ----------------------------- Embedding generation -----------------------------


def encode_with_normalization(
    model: SentenceTransformer,
    texts: List[str],
    batch_size: int,
    show_progress: bool = True
) -> np.ndarray:
    """
    Encode texts using SentenceTransformer and L2-normalize for cosine similarity.

    Args:
        model: Pre-loaded SentenceTransformer model
        texts: List of text strings (must be pre-sanitized)
        batch_size: Batch size for encoding
        show_progress: Show progress bar during encoding

    Returns:
        L2-normalized embeddings of shape (len(texts), embedding_dim)
    """
    log.info("🔄 Encoding %d texts (batch_size=%d)...", len(texts), batch_size)

    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=show_progress,
        convert_to_numpy=True,
        normalize_embeddings=False  # We'll normalize manually for clarity
    )

    # Convert to float32 for storage efficiency
    embeddings = embeddings.astype(np.float32)

    # L2-normalize: recommended for BGE, MXBAI, and cosine similarity scoring
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)  # Avoid division by zero
    embeddings = embeddings / norms

    mean_norm = float(np.linalg.norm(embeddings, axis=1).mean())
    log.info("✅ Encoded %d embeddings (dim=%d, mean_norm=%.3f)",
             embeddings.shape[0], embeddings.shape[1], mean_norm)

    return embeddings


def generate_embeddings_for_column(
    model: SentenceTransformer,
    df: pd.DataFrame,
    column_name: str,
    batch_size: int
) -> np.ndarray:
    """
    Extract text from a DataFrame column, sanitize, encode, and normalize.

    Returns:
        L2-normalized embeddings as numpy array
    """
    texts = extract_text_column(df, column_name)
    sanitized = sanitize_texts(texts)

    empty_count = sum(1 for t in sanitized if t == EMPTY_TEXT_PLACEHOLDER)
    if empty_count > 0:
        log.warning("⚠️ Found %d empty/null texts (will use placeholder embeddings)", empty_count)

    return encode_with_normalization(model, sanitized, batch_size)


# ----------------------------- Model & device setup -----------------------------


def setup_device(device_choice: str, model: SentenceTransformer) -> SentenceTransformer:
    """
    Configure model device (CPU or CUDA) based on user choice and availability.

    Args:
        device_choice: 'auto', 'cpu', or 'cuda'
        model: SentenceTransformer model to move

    Returns:
        Model on the chosen device
    """
    if device_choice == "cpu":
        model = model.to("cpu")
        log.info("🟡 Using CPU (user-specified)")
        return model

    # Try to use CUDA if requested or auto-detected
    try:
        import torch
        cuda_available = torch.cuda.is_available()
    except ImportError:
        cuda_available = False

    if device_choice == "cuda":
        if cuda_available:
            model = model.to("cuda")
            log.info("🟢 Using CUDA (user-specified)")
        else:
            log.warning("⚠️ CUDA requested but not available. Falling back to CPU.")
            model = model.to("cpu")
    else:
        # Auto-detect
        if cuda_available:
            model = model.to("cuda")
            log.info("🟢 Using CUDA (auto-detected)")
        else:
            model = model.to("cpu")
            log.info("🟡 Using CPU (auto-selected)")

    return model


def load_embedding_model(model_name: str, device: str) -> SentenceTransformer:
    """
    Load a SentenceTransformer model and configure device.

    Args:
        model_name: HuggingFace model identifier
        device: 'auto', 'cpu', or 'cuda'

    Returns:
        Configured SentenceTransformer model
    """
    log.info("🚀 Loading model: %s", model_name)
    model = SentenceTransformer(model_name)
    model = setup_device(device, model)
    return model


# ----------------------------- Output saving ------------------------------------


def save_embeddings(
    embeddings: np.ndarray,
    output_dir: Path,
    filename: str,
    column_name: str
) -> Path:
    """
    Save embeddings to .npy file with validation.

    Args:
        embeddings: Numpy array to save
        output_dir: Output directory
        filename: Output filename (e.g., 'embeddings.npy')
        column_name: Source column name (for logging)

    Returns:
        Path to saved file
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / filename

    np.save(output_path, embeddings)

    log.info(
        "💾 Saved '%s' embeddings: %d × %d → %s",
        column_name, embeddings.shape[0], embeddings.shape[1], output_path
    )

    return output_path


def record_model_metadata(model_name: str, output_dir: Path) -> None:
    """
    Save model name to a text file for reproducibility tracking.

    Args:
        model_name: Name of the embedding model used
        output_dir: Directory to save metadata
    """
    metadata_file = output_dir / "embedding_model.txt"
    metadata_file.write_text(model_name, encoding="utf-8")
    log.info("🧠 Model metadata saved: %s", metadata_file)


# ----------------------------- Main pipeline ------------------------------------


def generate_dual_embeddings(
    input_path: Path,
    output_dir: Path,
    model_name: str,
    batch_size: int,
    device: str
) -> None:
    """
    Main pipeline: Load chunks, generate embeddings, save outputs.

    Args:
        input_path: Path to chunks.parquet or chunks.jsonl (or directory)
        output_dir: Directory to save embeddings
        model_name: SentenceTransformer model identifier
        batch_size: Encoding batch size
        device: Device selection ('auto', 'cpu', or 'cuda')
    """
    # 1. Load model
    model = load_embedding_model(model_name, device)

    # 2. Load data
    df = load_chunks_dataframe(input_path)
    log.info("📊 Loaded DataFrame: %d rows, %d columns", len(df), len(df.columns))

    # 3. Generate embeddings for 'embed_text' (clean text with headers)
    log.info("\n🧩 Processing 'embed_text' column...")
    embed_text_vectors = generate_embeddings_for_column(
        model, df, "embed_text", batch_size
    )
    save_embeddings(embed_text_vectors, output_dir, "embeddings.npy", "embed_text")

    # 4. Optionally generate embeddings for 'text_with_images'
    if IMAGE_EMBEDDING:
        log.info("\n🧩 Processing 'text_with_images' column...")
        text_with_images_vectors = generate_embeddings_for_column(
            model, df, "text_with_images", batch_size
        )
        save_embeddings(
            text_with_images_vectors, output_dir, "embeddingsImages.npy", "text_with_images"
        )
    else:
        log.info("⏭ Skipping 'text_with_images' embeddings (IMAGE_EMBEDDING=False)")

    # 5. Save model metadata for tracking
    record_model_metadata(model_name, output_dir)

    log.info("\n✨ Embedding generation complete.")


# ----------------------------- CLI ----------------------------------------------


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate L2-normalized embeddings for RAG retrieval",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "-i", "--input",
        type=Path,
        default=DEFAULT_INPUT_PATH,
        help="Path to chunks.parquet, chunks.jsonl, or directory containing them"
    )

    parser.add_argument(
        "-d", "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to save embeddings and metadata"
    )

    parser.add_argument(
        "-m", "--model-name",
        type=str,
        default=DEFAULT_MODEL,
        help="SentenceTransformer model (e.g., BAAI/bge-base-en-v1.5)"
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Batch size for encoding (higher = faster but more memory)"
    )

    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Device selection: auto-detect, force CPU, or force CUDA"
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging"
    )

    return parser.parse_args()


def main() -> None:
    """Entry point for embeddings generation."""
    args = parse_arguments()

    if args.debug:
        log.setLevel(logging.DEBUG)

    try:
        generate_dual_embeddings(
            input_path=args.input,
            output_dir=args.output_dir,
            model_name=args.model_name,
            batch_size=args.batch_size,
            device=args.device
        )
    except Exception as e:
        log.exception("❌ Fatal error: %s", e)
        raise


if __name__ == "__main__":
    main()
