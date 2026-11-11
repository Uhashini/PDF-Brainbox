# PDF Brainbox

**PDF Brainbox** is an enterprise-grade Retrieval-Augmented Generation (RAG) and document intelligence web application built with Streamlit, FAISS, BM25, and Mistral AI. It allows users to upload single or multiple educational documents (PDF, DOCX, TXT, images) and explore advanced AI tools including Hybrid Q&A, Agentic Multi-Query Expansion, real-time RAG Triad Evaluation, Concept Knowledge Graphs, Multi-Document Comparison, auto-generated Quizzes, Study Notes, Presentation Slides, and 3D Interactive Flashcards.

**Live Demo**: [https://pdf-brainbox.streamlit.app](https://pdf-brainbox.streamlit.app)

---

## Key Features

- **Multi-Document Support & Global Auto-Indexing**: Upload single or multiple files (PDF, DOCX, TXT, Images) from any page.
- **Hybrid Search Architecture**: Combines Dense Vector Search (FAISS) and Sparse Keyword Search (BM25) using Reciprocal Rank Fusion (RRF).
- **Agentic Multi-Query Expansion**: Re-writes vague prompts into 3 semantic query variations to maximize context retrieval recall.
- **RAG Triad Evaluator Engine**: Real-time LLM-as-a-Judge diagnostics measuring Faithfulness, Answer Relevance, and Context Precision.
- **Vector Semantic Caching**: Sub-10ms response times for semantically similar queries to optimize API token cost.
- **NLP Document Analytics & Knowledge Graph**: Computes Flesch Reading Ease, Lexical Diversity, and extracts interactive physics concept knowledge graphs.
- **Multi-Document Comparison Engine**: Calculates semantic embedding overlap percentage and shared versus unique topic breakdowns between documents.
- **Auto-Generated Study Tools**:
  - Auto-generated multiple-choice quizzes with progress scoring.
  - Interactive 3D flip flashcards with Anki CSV export.
  - Bulleted study notes with Markdown (.md) and Text (.txt) downloads.
  - Presentation slide outlines with PowerPoint (.pptx) download.

---

## Tech Stack

| Component | Technology |
|---|---|
| Frontend | Streamlit |
| Backend | Python |
| Document Parsing | PyPDF2, python-docx, Pillow, pytesseract |
| Dense Vector Index | FAISS |
| Sparse Keyword Search | Rank-BM25 |
| LLM & Embeddings | Mistral AI (`mistral-embed`, `open-mistral-7b`) |
| Data & Math | NumPy |
| Network Visualizer | vis.js |

---

## How It Works

1. **Document Ingestion**: Extracts text from single or multiple uploaded files and splits content into overlapping chunks.
2. **Hybrid Indexing**: Generates dense vector embeddings via Mistral AI and indexes them in FAISS, while building a parallel BM25 sparse keyword index.
3. **Retrieval & Fusion**: Queries execute dense FAISS search and sparse BM25 search fused using Reciprocal Rank Fusion (RRF). Optional Multi-Query Expansion generates semantic prompt variations.
4. **Semantic Caching**: Checks query vector cosine similarity against cached questions before invoking LLM generation.
5. **Context Generation**: Mistral AI synthesizes accurate answers strictly grounded in retrieved document chunks.
6. **Evaluation & Telemetry**: LLM-as-a-Judge calculates Faithfulness, Answer Relevance, and Context Precision scores alongside latency breakdowns.

---

## Evaluation Metrics & Empirical Ablation Benchmark

The RAG pipeline is benchmarked using an automated evaluation framework comparing **Dense Vector Search (FAISS)**, **Sparse Keyword Search (BM25)**, **Hybrid Fusion (RRF)**, and **Multi-Query Expansion**:

| Retrieval Strategy | Hit Rate @ 3 | MRR | Context Precision | Faithfulness (Anti-Hallucination) | Answer Relevance | Retrieval Latency |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Dense Only (FAISS)** | 80.0% | 0.567 | 53.2% | 92.4% | 85.0% | `2.88 ms` |
| **Sparse Only (BM25)** | 80.0% | 0.700 | 53.2% | 92.4% | 85.0% | `1.06 ms` |
| **Hybrid (FAISS + BM25 + RRF)** | **80.0%** | **0.667** | **53.2%** | **92.4%** | **85.0%** | **`0.98 ms`** |
| **Hybrid + Multi-Query Exp.** | 60.0% | 0.600 | 50.0% | 89.2% | 82.5% | `2.23 ms` |

### Running the Evaluation Suite & CI/CD Tests

1. **Run the CI/CD Quality Gate**:
   ```bash
   python tests/test_rag_evaluation.py
   ```
   *Asserts Faithfulness >= 85%, Answer Relevance >= 85%, Reciprocal Rank Fusion ranking math, and sub-15ms Semantic Cache retrieval.*

2. **Run the Retrieval Ablation Benchmark**:
   ```bash
   python evaluation/benchmark.py --mode fast
   ```
   *Generates `evaluation/benchmark_report.md` and `evaluation/benchmark_report.json`.*

3. **Interview Preparation**:
   *Check out [`evaluation/INTERVIEW_GUIDE.md`](evaluation/INTERVIEW_GUIDE.md) for in-depth answers to top RAG evaluation questions.*

---

## Installation

To run the application locally:

### 1. Clone the repository

```bash
git clone https://github.com/Uhashini/PDF-Brainbox.git
cd PDF-Brainbox
```

### 2. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Set API Key & Run

```bash
export MISTRAL_API_KEY="your_api_key_here"  # On Windows PowerShell: $env:MISTRAL_API_KEY="your_api_key_here"
streamlit run app.py
```
