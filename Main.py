
import os
from pathlib import Path

import fitz
import streamlit as st
from dotenv import load_dotenv
from groq import Groq
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))


def load_pdfs(folder="documents"):
    chunks = []

    Path(folder).mkdir(exist_ok=True)

    for pdf in Path(folder).glob("*.pdf"):
        with fitz.open(pdf) as doc:
            for page_number, page in enumerate(doc):
                text = page.get_text().strip()

                if text:
                    chunks.append({
                        "text": text,
                        "source": pdf.name,
                        "page": page_number + 1,
                    })

    return chunks


def search_documents(chunks, query, top_k=3):
    if not chunks:
        return []

    texts = [item["text"] for item in chunks]

    vectorizer = TfidfVectorizer(
        stop_words="english"
    )

    matrix = vectorizer.fit_transform(texts + [query])
    scores = cosine_similarity(
        matrix[-1], matrix[:-1]
    ).flatten()

    indices = scores.argsort()[::-1][:top_k]

    return [
        chunks[i]
        for i in indices
        if scores[i] > 0
    ]


def summarize(query, results, length):
    if not results:
        return "No relevant document passages found."

    context = "\n\n".join(
        f"Source: {item['source']}, "
        f"Page: {item['page']}\n{item['text']}"
        for item in results
    )

    instructions = {
        "Short": "Write 3 to 5 sentences.",
        "Medium": "Write 2 short paragraphs.",
        "Long": "Write a detailed summary with key points.",
    }

    response = client.chat.completions.create(
        model=os.getenv(
                   "GROQ_MODEL",
                   "llama-3.3-70b-versatile"
                ),
        messages=[
            {
                "role": "system",
                "content": (
                    "Answer using only the provided document passages. "
                    "Do not invent facts. If the answer is not supported "
                    "by the passages, say that the documents do not "
                    "contain enough information. Cite source filenames "
                    "and page numbers when possible."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Question: {query}\n"
                    f"Instructions: {instructions[length]}\n"
                    f"Documents:\n{context}"
                ),
            },
        ],
        temperature=0,
    )

    return response.choices[0].message.content


st.title("Document Search & Summarization")

chunks = load_pdfs()

if not chunks:
    st.info("Add PDF files to the documents folder.")
else:
    st.success(f"Loaded {len(chunks)} PDF pages.")

    query = st.text_input("Ask a question about your documents")
    length = st.selectbox(
        "Summary length", ["Short", "Medium", "Long"]
    )
    top_k = st.slider("Number of results", 1, 10, 3)

    if st.button("Search and Summarize") and query.strip():
        with st.spinner("Searching documents..."):
            results = search_documents(
                chunks, query, top_k
            )

            if results:
                answer = summarize(query, results, length)

                st.subheader("Summary")
                st.write(answer)

                st.subheader("Retrieved Sources")
                for item in results:
                    st.write(
                        f"**{item['source']} — "
                        f"Page {item['page']}**"
                    )
                    st.write(item["text"][:500])
            else:
                st.warning("No matching passages found.")
