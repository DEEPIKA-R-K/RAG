# RAG

## Abstract

This project presents a document search and summarization application that helps users retrieve relevant information from PDF documents and generate concise summaries using a Large Language Model (LLM).

The system uses Python for document processing, TF-IDF and cosine similarity for information retrieval, Groq API for LLM-based summarization, and Streamlit for the user interface.

The project evaluates retrieval performance using Precision@K, Recall@K, and Mean Reciprocal Rank (MRR). Summary quality is evaluated using ROUGE metrics and manual inspection. Search and summarization latency are recorded to assess efficiency.

The objective is to build a practical system that retrieves relevant passages, generates summaries grounded in the retrieved content, and provides source information for verification.

**Keywords:** Information Retrieval, Document Search, Text Summarization, Large Language Models, TF-IDF, Cosine Similarity, ROUGE.

---

## 1. Introduction

### 1.1 Background

Organizations and individuals often need to extract information from large collections of documents. Reading each document manually can be time-consuming, particularly when users need answers to specific questions.

Document search and automated summarization can reduce the effort required to locate and understand relevant information.

### 1.2 Problem Statement

The objective is to develop an application that accepts user questions, identifies relevant passages from a document collection, and generates summaries using an LLM.

The application should provide relevant search results, understandable summaries, source traceability, and measurable evaluation results.

### 1.3 Project Objectives

The main objectives are:

1. Extract text from PDF documents.
2. Preprocess and divide extracted text into manageable passages.
3. Retrieve relevant passages for a user query.
4. Generate summaries using an LLM.
5. Provide source information for verification.
6. Evaluate retrieval accuracy and summary quality.
7. Measure search and summarization latency.
8. Provide an accessible interface for interacting with the system.

---

## 2. Data Preprocessing

### 2.1 Document Collection

The input corpus consists of PDF documents stored in the project's document directory.

Dataset details:

- Number of PDF files: 13
- Total pages processed: 922 PDF pages.

The documents should be representative of the questions the system is expected to answer.

### 2.2 Text Extraction

PyMuPDF is used to extract text from PDF pages.

The extracted text is associated with metadata such as the source filename and page number. This metadata supports source identification when displaying search results.

### 2.3 Text Cleaning

The extracted text is normalized by removing unnecessary whitespace and consolidating line breaks.

This preprocessing makes the text more consistent for subsequent retrieval.

The original document files are retained so that retrieved passages can be traced back to their sources.

### 2.4 Text Chunking

Long page contents are divided into smaller text chunks.

In the current evaluation implementation, the configured chunk size is 250 words with an overlap of 40 words.

The overlap helps preserve context when a relevant passage crosses a chunk boundary.

Chunking also limits the amount of text processed during each retrieval operation.

### 2.5 TF-IDF Representation

Term Frequency-Inverse Document Frequency (TF-IDF) represents each text chunk numerically.

Term frequency measures how frequently a term occurs in a document, while inverse document frequency reduces the influence of terms that appear across many documents.

The resulting vectors are used to compare document passages with user queries.

### 2.6 Preprocessing Justification

The preprocessing steps serve the following purposes:

- Text extraction makes PDF content searchable.
- Whitespace normalization reduces unnecessary text variation.
- Chunking supports passage-level retrieval.
- Overlapping chunks help preserve context.
- Source metadata supports traceability.
- TF-IDF provides a straightforward and computationally practical representation for lexical search.

A limitation is that text extraction may not work properly for scanned PDFs without OCR. TF-IDF may also fail to retrieve semantically related passages when the query and document use different vocabulary.

---

## 3. System Methodology

### 3.1 System Architecture

The application follows this workflow:

PDF Documents  
↓  
Text Extraction  
↓  
Text Cleaning  
↓  
Chunking and Metadata Assignment  
↓  
TF-IDF Index Construction  
↓  
User Query  
↓  
Cosine Similarity Retrieval  
↓  
Top-K Relevant Passages  
↓  
Groq LLM Summarization  
↓  
Summary and Source Display

### 3.2 Document Retrieval

The system uses TF-IDF vectors and cosine similarity to rank passages according to their lexical similarity to the query.

Cosine similarity is calculated as:

cosine similarity(A, B) = (A · B) / (||A|| × ||B||)

Here, A represents the query vector and B represents a document chunk vector.

A higher score indicates greater similarity between the vectors.

The system selects the highest-scoring passages, subject to the configured Top-K limit, and passes them to the summarization component.

### 3.3 Summarization

The Groq API provides access to the selected LLM.

The summarization prompt includes:

- The user's question.
- The requested summary length.
- The retrieved document passages.
- Instructions to avoid unsupported claims and acknowledge insufficient evidence.

The application can support short, medium, and long summaries through length-specific instructions.

The summary is generated from the retrieved passages rather than requiring the model to process the entire document collection.

### 3.4 Source Attribution

Each retrieved passage retains its source filename and page number.

Displaying this information helps users locate the original text and verify whether the generated summary is supported by the document.

Source metadata improves traceability, but it does not guarantee that every statement in a summary is correct.

### 3.5 User Interface

Streamlit provides the interface for entering questions and displaying results.

Depending on the implemented interface, users can select summary length, specify the number of results, view retrieved passages, and inspect the generated summary.

---

## 4. Evaluation Methodology

### 4.1 Evaluation Dataset

A manually verified evaluation dataset is created in `evaluation.csv`.

Each test case contains:

- `query`: the question submitted to the system.
- `relevant_source`: the expected relevant document filename or filenames.
- `reference_summary`: a human-written summary used for comparison.

The expected source labels and reference summaries must be checked against the original documents before evaluation.

Dataset details:

- Number of evaluation questions: 5
- Number of relevant source labels: 5
- Number of reference summaries: 5

### 4.2 Retrieval Metrics

#### Precision@K

Precision@K measures the proportion of the top K retrieved results that are relevant.

Precision@K = Relevant retrieved results / K

A higher value indicates that more retrieved results are relevant.

#### Recall@K

Recall@K measures the proportion of known relevant sources retrieved within the top K results.

Recall@K = Relevant retrieved sources / Total known relevant sources

A higher value indicates better coverage of the known relevant sources.

#### Mean Reciprocal Rank

Mean Reciprocal Rank (MRR) evaluates the ranking of the first relevant result across test queries.

For each query, the reciprocal rank is the inverse of the position of the first relevant result. If no relevant result is retrieved, the reciprocal rank is zero.

MRR = Mean of reciprocal ranks across queries.

Higher MRR indicates that relevant results tend to appear nearer the top.

**Evaluation consideration:** If several chunks come from the same PDF, chunk-level results and unique-document results are not equivalent. The evaluation should clearly specify which unit is being measured.

### 4.3 Summary Evaluation

ROUGE metrics compare generated summaries against human-written reference summaries.

ROUGE-1 measures unigram overlap.

ROUGE-L measures overlap based on the longest common subsequence.

The evaluation records precision, recall, and F1 where applicable.

ROUGE scores measure textual similarity rather than factual correctness. Manual inspection is therefore also required.

### 4.4 Human Evaluation

A sample of generated summaries should be reviewed using the following dimensions:

- Correctness: Are the statements supported by the source?
- Relevance: Does the summary answer the question?
- Completeness: Are the important points included?
- Faithfulness: Does the summary avoid unsupported claims?

Each dimension can be scored on a scale from 1 to 5 using a consistent rubric.

### 4.5 Efficiency Evaluation

Search and summarization latency are measured separately using a timer.

Search latency measures the time taken to retrieve passages.

Summarization latency measures the time taken to obtain the LLM response.

The measurements are recorded across the evaluation questions.

For a fair performance comparison, the hardware, corpus, retrieval configuration, and model should be documented.

---

## 5. Evaluation Results

### 5.1 Retrieval Results

Insert the actual output produced by `evaluate.py`.

| Metric | Measured result |
|---|---:|
| Mean Precision@K | 0.120 |
| Mean Recall@K |  0.600 |
| Mean Reciprocal Rank | 0.200 |
| Average search latency | 0.0024 seconds |

### 5.2 Summary Results

| Metric | Measured result |
|---|---:|
| Mean ROUGE-1 F1 | 0.078 |
| Mean ROUGE-L F1 | 0.078 |
| Average summarization latency | 1.65 seconds |

### 5.3 Results Interpretation

Discuss the actual results after running the evaluation.

Consider:

- Whether relevant sources appear near the top of the retrieval results.
- Whether the system misses relevant documents.
- Whether the generated summaries accurately reflect the retrieved passages.
- Whether summaries omit important details.
- Whether latency is acceptable for the intended use case.

Do not report expected or hypothetical values as measured results.

### 5.4 Example Queries

Include 5–10 real test cases.

For each test case, document:

1. The user's question.
2. The expected relevant source.
3. The retrieved source and page.
4. The generated summary.
5. A brief assessment of correctness and relevance.

---

## 6. Challenges and Solutions

### 6.1 PDF Text Extraction

**Challenge:** PDF files may contain inconsistent formatting, line breaks, or scanned images.

**Solution:** Normalize extracted text and identify documents that require OCR.

### 6.2 Retrieval Relevance

**Challenge:** TF-IDF relies primarily on word overlap and may miss passages expressed using different terminology.

**Solution:** Evaluate retrieval using manually verified questions and relevant-source labels. Consider semantic embeddings or hybrid retrieval as future improvements.

### 6.3 Summary Hallucinations

**Challenge:** An LLM may generate unsupported information or omit relevant details.

**Solution:** Provide retrieved passages as context, instruct the model to acknowledge missing evidence, and manually inspect generated summaries.

These measures reduce risk but do not eliminate hallucinations.

### 6.4 API Errors

**Challenge:** Model availability, invalid model names, authentication problems, rate limits, and network failures may interrupt summarization.

**Solution:** Use a model available to the configured Groq account, store API credentials securely, and implement appropriate error handling.

### 6.5 Evaluation Quality

**Challenge:** Automated metrics may not capture factual correctness or the practical usefulness of a summary.

**Solution:** Combine retrieval metrics and ROUGE with manual evaluation using a consistent rubric.

---

## 7. Scalability and Efficiency

The current TF-IDF approach is suitable as a baseline for lexical document retrieval.

For larger corpora, performance can be improved by:

- Building the TF-IDF index once and reusing it.
- Persisting the index between application runs.
- Caching extracted document text.
- Avoiding unnecessary LLM calls.
- Limiting the context passed to the LLM.
- Evaluating search latency as the corpus grows.

Semantic search or a hybrid retrieval system may improve relevance for queries that use different vocabulary from the documents.

Scalability claims should be supported by experiments using different corpus sizes. Results from a small dataset should not be treated as evidence of performance at very large scale.

---

## 8. Limitations

The project has several limitations:

1. TF-IDF does not fully represent semantic meaning.
2. Scanned documents may require OCR.
3. LLM-generated summaries can contain errors.
4. ROUGE does not establish factual correctness.
5. Results depend on document quality and evaluation-set coverage.
6. API latency and availability depend on the external service.
7. Performance on large corpora requires additional benchmarking.

---

## 9. Future Improvements

Potential future enhancements include:

- Hybrid retrieval using lexical search and semantic embeddings.
- Better chunking and ranking strategies.
- Persistent indexing and document caching.
- Expanded evaluation datasets.
- Automated factual-consistency checks.
- More comprehensive human evaluation.
- Automated unit and integration tests.
- Support for additional document formats.
- Performance benchmarking on larger corpora.

---

## 10. Conclusion

This project demonstrates an approach to searching PDF documents and generating summaries using lexical retrieval and a Large Language Model.

The preprocessing pipeline converts PDF content into searchable passages. TF-IDF and cosine similarity rank passages against user queries, while the Groq API generates summaries from retrieved context.

The evaluation framework measures retrieval quality, summary text overlap, and processing latency.

The effectiveness of the final system should be determined from the measured evaluation results and manual review of generated summaries. Future improvements can focus on semantic retrieval, stronger factual validation, and efficient indexing.

---

## Appendix A: Project Setup

Document the exact setup instructions from the project's README.

Include:

- Python version.
- Dependency installation.
- Environment variable configuration.
- Document directory structure.
- Application launch command.
- Evaluation command.

## Appendix B: Project Files

List the actual files used for document processing, retrieval, summarization, interface development, and evaluation.

## Appendix C: References

Include the official documentation for the libraries and services actually used, such as Python, PyMuPDF, scikit-learn, Streamlit, Groq, and ROUGE Score.
