import sys
from pathlib import Path

CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_FILE.parents[1]   # one level up from scripts/
sys.path.insert(0, str(PROJECT_ROOT))

from rag.summarizer import NewsSummariser

def main():
    article_text = """
    The president of El Salvador announced new security measures following a surge in arrests
    linked to gang activity. Human rights groups have raised concerns about arbitrary detention
    and overcrowded prisons, as thousands of suspected gang members have been detained in recent months.
    """

    print("Loading RAG summariser (this may take a moment the first time)...")
    summariser = NewsSummariser()

    print("\n=== VANILLA SUMMARY (no retrieval) ===")
    vanilla = summariser.summarize_vanilla(article_text)
    print(vanilla)

    print("\n=== RAG SUMMARY (with retrieval) ===")
    rag = summariser.summarize_rag(article_text, debug=True)
    print(rag)

if __name__ == "__main__":
    main()
