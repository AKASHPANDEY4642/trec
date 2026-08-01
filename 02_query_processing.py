"""
Step 2: Query Processing — Extract Focused Queries from Long Narratives
========================================================================
The TREC RAG 2026 narratives are 2-3 sentences long. Raw BM25 with the
full narrative text can be noisy. This script tries several strategies
to create better search queries:

1. First-sentence extraction (simple but effective)
2. Keyword extraction (TF-IDF style)
3. Extracting the core question from the narrative

Usage:
    python 02_query_processing.py

Output:
    tmp/processed_queries.json  — Cleaned queries for each narrative
"""

import sys
import re
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from config import QUERIES_FILE, TMP_DIR


def load_queries(filepath: Path) -> list[tuple[str, str]]:
    """Load test narratives from TSV file."""
    queries = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t", 1)
            if len(parts) == 2:
                queries.append((parts[0].strip(), parts[1].strip()))
    return queries


def extract_first_sentence(text: str) -> str:
    """Extract the first sentence as a query."""
    # Split on sentence-ending punctuation followed by space
    sentences = re.split(r'(?<=[.!?])\s+', text)
    if sentences:
        return sentences[0]
    return text


def extract_core_question(text: str) -> str:
    """
    Extract the core question or information need from the narrative.
    Looks for question sentences (containing '?') and key phrases.
    """
    sentences = re.split(r'(?<=[.!?])\s+', text)
    
    # Find question sentences
    questions = [s for s in sentences if '?' in s]
    
    if questions:
        # Use the first question, but also include context from first sentence
        first_sentence = sentences[0] if sentences else ""
        if first_sentence not in questions:
            return first_sentence + " " + " ".join(questions[:2])
        return " ".join(questions[:2])
    
    # No explicit question — use first two sentences
    return " ".join(sentences[:2])


def extract_keywords(text: str, max_words: int = 30) -> str:
    """
    Extract key terms from the narrative using simple heuristics.
    Removes common stop words and keeps content words.
    """
    # Common English stop words
    stop_words = {
        'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you',
        "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself',
        'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her',
        'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them',
        'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom',
        'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are',
        'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having',
        'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if',
        'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for',
        'with', 'about', 'against', 'between', 'through', 'during', 'before',
        'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out',
        'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once',
        'here', 'there', 'when', 'where', 'why', 'how', 'all', 'both',
        'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor',
        'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't',
        'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now',
        'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't",
        'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn',
        "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't",
        'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't",
        'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren',
        "weren't", 'won', "won't", 'wouldn', "wouldn't",
        'help', 'want', 'need', 'would', 'could', 'might', 'also', 'like',
        'keep', 'make', 'get', 'know', 'think', 'understand', 'tell',
        'whether', 'much', 'many', 'something', 'anything', 'everything',
        'really', 'actually', 'especially', 'specifically', 'particularly',
        'im', "i'm", "i've", "i'd", "i'll", 'ive', 'id',
    }
    
    # Tokenize and filter
    words = re.findall(r'\b[a-zA-Z]+(?:[-/][a-zA-Z]+)*\b', text.lower())
    content_words = [w for w in words if w not in stop_words and len(w) > 2]
    
    # Keep unique words in order, limited to max_words
    seen = set()
    unique_words = []
    for w in content_words:
        if w not in seen:
            seen.add(w)
            unique_words.append(w)
            if len(unique_words) >= max_words:
                break
    
    return " ".join(unique_words)


def create_hybrid_query(narrative_id: str, narrative_text: str) -> dict:
    """
    Create multiple query variants for a single narrative.
    Returns a dict with different query strategies.
    """
    return {
        "narrative_id": narrative_id,
        "original": narrative_text,
        "first_sentence": extract_first_sentence(narrative_text),
        "core_question": extract_core_question(narrative_text),
        "keywords": extract_keywords(narrative_text, max_words=25),
        # Best default: first sentence + keywords gives BM25 good signal
        "combined": extract_first_sentence(narrative_text) + " " + extract_keywords(narrative_text, max_words=15)
    }


def main():
    print("=" * 70)
    print("TREC RAG 2026 — Step 2: Query Processing")
    print("=" * 70)
    
    if not QUERIES_FILE.exists():
        print(f"ERROR: Queries file not found: {QUERIES_FILE}")
        print(f"Run: git clone https://github.com/TREC-RAG/trec-rag-data.git")
        sys.exit(1)
    
    queries = load_queries(QUERIES_FILE)
    print(f"Loaded {len(queries)} narratives\n")
    
    processed = []
    for narrative_id, narrative_text in queries:
        result = create_hybrid_query(narrative_id, narrative_text)
        processed.append(result)
    
    # Save processed queries
    output_path = TMP_DIR / "processed_queries.json"
    output_path.parent.mkdir(exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(processed, f, indent=2)
    
    print(f"Saved {len(processed)} processed queries to {output_path}")
    
    # Show examples
    print(f"\n--- Example (first query) ---")
    ex = processed[0]
    print(f"ID: {ex['narrative_id']}")
    print(f"Original ({len(ex['original'])} chars): {ex['original'][:120]}...")
    print(f"First sentence: {ex['first_sentence'][:120]}...")
    print(f"Keywords: {ex['keywords'][:120]}...")
    print(f"Combined: {ex['combined'][:120]}...")
    
    # Stats
    orig_lens = [len(q["original"]) for q in processed]
    combined_lens = [len(q["combined"]) for q in processed]
    print(f"\n--- Query Length Stats ---")
    print(f"Original:  avg={sum(orig_lens)/len(orig_lens):.0f} chars, "
          f"min={min(orig_lens)}, max={max(orig_lens)}")
    print(f"Combined:  avg={sum(combined_lens)/len(combined_lens):.0f} chars, "
          f"min={min(combined_lens)}, max={max(combined_lens)}")
    
    print(f"\nNext: Run 03_bm25_retrieve_processed.py to retrieve with processed queries")


if __name__ == "__main__":
    main()
