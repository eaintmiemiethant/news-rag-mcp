from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Corpus + index paths
RAG_CORPUS_PATH = PROJECT_ROOT / "data" / "processed" / "rag_corpus.jsonl"
RAG_INDEX_PATH = PROJECT_ROOT / "models" / "rag" / "corpus.index"
RAG_METADATA_PATH = PROJECT_ROOT / "models" / "rag" / "corpus_metadata.jsonl"

# Embedding model (SentenceTransformers)
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Summariser model
SUMMARISER_MODEL_NAME = "facebook/bart-large-cnn"
MAX_INPUT_TOKENS = 1024
MAX_SUMMARY_TOKENS = 128
TOP_K_RETRIEVAL = 5
