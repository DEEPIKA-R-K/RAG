
import os
import time
from pathlib import Path

import fitz
import pandas as pd
from dotenv import load_dotenv
from groq import Groq
from rouge_score import rouge_scorer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

BASE_DIR = Path(__file__).resolve().parent
DOCUMENTS_DIR = BASE_DIR / "documents"
EVALUATION_FILE = BASE_DIR / "evaluation.csv"
TOP_K = 5
CHUNK_WORDS = 250
CHUNK_OVERLAP = 40

load_dotenv(BASE_DIR / ".env")


def load_pdf_chunks(folder=DOCUMENTS_DIR):
    """Extract PDF text and split each page into overlapping word chunks."""
    chunks = []
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)

    for pdf_path in sorted(folder.glob("*.pdf")):
        try:
            with fitz.open(pdf_path) as pdf:
                for page_number, page in enumerate(pdf, start=1):
                    text = " ".join(page.get_text().split())
                    if not text:
                        continue
                    words = text.split()
                    step = max(1, CHUNK_WORDS - CHUNK_OVERLAP)
                    for start in range(0, len(words), step):
                        part = " ".join(words[start:start + CHUNK_WORDS])
                        if part:
                            chunks.append({
                                "text": part,
                                "source": pdf_path.name,
                                "page": page_number,
                            })
                        if start + CHUNK_WORDS >= len(words):
                            break
        except Exception as exc:
            print(f"Warning: could not read {pdf_path.name}: {exc}")
    return chunks


def build_search_index(chunks):
    """Build the TF-IDF index once and reuse it for all queries."""
    if not chunks:
        raise ValueError("No readable PDF text found in the documents folder.")
    vectorizer = TfidfVectorizer(
        stop_words="english", max_features=100_000, ngram_range=(1, 2)
    )
    matrix = vectorizer.fit_transform([item["text"] for item in chunks])
    return vectorizer, matrix


def search_documents(query, chunks, vectorizer, matrix, top_k=TOP_K):
    """Return the highest-scoring text chunks for a query."""
    query_vector = vectorizer.transform([query])
    scores = cosine_similarity(query_vector, matrix).ravel()
    indices = scores.argsort()[::-1][:top_k]
    return [
        {**chunks[i], "score": float(scores[i])}
        for i in indices if scores[i] > 0
    ]


def create_groq_client():
    """Create a Groq client from GROQ_API_KEY in .env."""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY is missing from your .env file.")
    return Groq(api_key=api_key)


def generate_summary(query, retrieved_chunks, client, length="Short"):
    """Generate a summary grounded in retrieved passages."""
    if not retrieved_chunks:
        return "No relevant document passages found."

    context = "\n\n".join(
        f"Source: {item['source']}, Page: {item['page']}\n{item['text']}"
        for item in retrieved_chunks
    )
    instructions = {
        "Short": "Write 3 to 5 sentences.",
        "Medium": "Write 2 short paragraphs.",
        "Long": "Write a detailed summary with key points.",
    }
    model = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "Use only the supplied passages. Do not invent facts. "
                    "If the answer is not supported, say the documents do not "
                    "contain enough information. Mention source filenames and pages."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Question: {query}\n"
                    f"Instructions: {instructions.get(length, instructions['Short'])}\n"
                    f"Passages:\n{context}"
                ),
            },
        ],
        temperature=0,
    )
    return response.choices[0].message.content or ""


def parse_sources(value):
    """Read relevant source filenames separated by |."""
    if pd.isna(value):
        return set()
    return {x.strip() for x in str(value).split("|") if x.strip()}


def evaluate():
    if not EVALUATION_FILE.exists():
        raise FileNotFoundError(
            "Create evaluation.csv with columns: query,relevant_source,reference_summary"
        )

    tests = pd.read_csv(EVALUATION_FILE)
    required = {"query", "relevant_source", "reference_summary"}
    missing = required - set(tests.columns)
    if missing:
        raise ValueError(f"Missing evaluation.csv columns: {', '.join(sorted(missing))}")

    chunks = load_pdf_chunks()
    vectorizer, matrix = build_search_index(chunks)
    client = create_groq_client()
    scorer = rouge_scorer.RougeScorer(
        ["rouge1", "rougeL"], use_stemmer=True
    )

    retrieval_rows = []
    summary_rows = []

    print(f"Loaded {len(chunks)} chunks from {DOCUMENTS_DIR}")
    print(f"Testing {len(tests)} queries with Top-{TOP_K} retrieval.")
    print("ROUGE evaluation makes live Groq API calls.\n")

    for index, row in tests.iterrows():
        query = str(row["query"]).strip()
        if not query:
            continue

        relevant = parse_sources(row["relevant_source"])

        start = time.perf_counter()
        found = search_documents(query, chunks, vectorizer, matrix, TOP_K)
        search_seconds = time.perf_counter() - start
        found_sources = [item["source"] for item in found]

        hits = sum(source in relevant for source in found_sources)
        precision = hits / TOP_K
        recall = hits / len(relevant) if relevant else 0.0

        reciprocal_rank = 0.0
        for rank, item in enumerate(found, start=1):
            if item["source"] in relevant:
                reciprocal_rank = 1 / rank
                break

        retrieval_rows.append({
            "query": query,
            f"Precision@{TOP_K}": precision,
            f"Recall@{TOP_K}": recall,
            "Reciprocal Rank": reciprocal_rank,
            "Search Time (seconds)": search_seconds,
            "Retrieved Sources": " | ".join(found_sources),
        })

        print(f"[{index + 1}/{len(tests)}] {query}")
        print(f"  Retrieved: {', '.join(found_sources) or 'none'}")
        print(f"  Search time: {search_seconds:.4f}s")

        reference = row["reference_summary"]
        if pd.isna(reference) or not str(reference).strip():
            print("  ROUGE skipped: reference_summary is empty.")
            continue

        try:
            start = time.perf_counter()
            generated = generate_summary(query, found, client, "Short")
            summary_seconds = time.perf_counter() - start
            rouge = scorer.score(str(reference), generated)

            summary_rows.append({
                "query": query,
                "ROUGE-1 Precision": rouge["rouge1"].precision,
                "ROUGE-1 Recall": rouge["rouge1"].recall,
                "ROUGE-1 F1": rouge["rouge1"].fmeasure,
                "ROUGE-L F1": rouge["rougeL"].fmeasure,
                "Summary Time (seconds)": summary_seconds,
                "Generated Summary": generated,
                "Reference Summary": str(reference),
            })
            print(
                f"  ROUGE-1 F1: {rouge['rouge1'].fmeasure:.3f}; "
                f"ROUGE-L F1: {rouge['rougeL'].fmeasure:.3f}; "
                f"summary time: {summary_seconds:.2f}s"
            )
        except Exception as exc:
            print(f"  Summary evaluation failed: {exc}")

    retrieval_df = pd.DataFrame(retrieval_rows)
    summary_df = pd.DataFrame(summary_rows)

    if not retrieval_df.empty:
        retrieval_df.to_csv(BASE_DIR / "retrieval_results.csv", index=False)
        print("\n=== RETRIEVAL RESULTS ===")
        print(f"Mean Precision@{TOP_K}: {retrieval_df[f'Precision@{TOP_K}'].mean():.3f}")
        print(f"Mean Recall@{TOP_K}: {retrieval_df[f'Recall@{TOP_K}'].mean():.3f}")
        print(f"Mean Reciprocal Rank: {retrieval_df['Reciprocal Rank'].mean():.3f}")
        print(f"Average search time: {retrieval_df['Search Time (seconds)'].mean():.4f}s")
        print("Saved retrieval_results.csv")

    if not summary_df.empty:
        summary_df.to_csv(BASE_DIR / "summary_results.csv", index=False)
        print("\n=== SUMMARY RESULTS (ROUGE) ===")
        print(f"Mean ROUGE-1 F1: {summary_df['ROUGE-1 F1'].mean():.3f}")
        print(f"Mean ROUGE-L F1: {summary_df['ROUGE-L F1'].mean():.3f}")
        print(f"Average summary time: {summary_df['Summary Time (seconds)'].mean():.2f}s")
        print("Saved summary_results.csv")

    print("\nReminder: ROUGE measures text overlap, not factual correctness.")
    print("Manually check generated summaries for accuracy and unsupported claims.")


if __name__ == "__main__":
    evaluate()
