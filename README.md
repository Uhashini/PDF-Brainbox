# PDF Brainbox

**PDF Brainbox** is a Streamlit web application that uses Retrieval-Augmented Generation (RAG) to help users interact intelligently with PDF content. Upload educational slides, books, or notes in PDF format and explore features like Q&A, quizzes, summaries, flashcards, and more — all powered by modern AI.

**Live Demo**: [https://pdf-brainbox.streamlit.app](https://pdf-brainbox.streamlit.app)

---

## Features

- PDF reading  
- AI-powered Q&A chatbot  
- Automatic topic summaries  
- Auto-generated multiple-choice quizzes  
- Interactive flashcards  
- Study guide and structured notes

---

## Tech Stack

| Component       | Technology         |
|----------------|--------------------|
| Frontend       | Streamlit          |
| Backend        | Python             |
| PDF Parsing    | PyPDF2             |
| Embeddings     | FAISS + Mistral AI |
| Data Handling  | NumPy              |

---

## How It Works

1. **PDF Upload**: The app extracts and processes section-based content using `PyPDF2`.
2. **Embedding & Indexing**: It generates vector embeddings and indexes them using FAISS.
3. **RAG Querying**: Questions are answered using relevant chunks retrieved from the vector store and passed to Mistral AI.
4. **Content Generation**: Summaries, quizzes, and flashcards are created using context-aware prompts.
5. **Streamlit Interface**: The app displays all outputs through an interactive and responsive interface.

---

## Installation

To run the app locally:

### 1. Clone the repository

```bash
git clone https://github.com/Uhashini/PDF-Brainbox.git
cd PDF-Brainbox
```

### 2. Create a virtual environment (optional)

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit app

```bash
streamlit run app.py
```


