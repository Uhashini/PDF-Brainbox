import streamlit as st
from PyPDF2 import PdfReader
import numpy as np
import faiss
from docx import Document
from PIL import Image
import pytesseract
from pptx import Presentation
from pptx.util import Inches
import streamlit.components.v1 as components
import urllib.parse
import firebase_admin
from firebase_admin import credentials, firestore
import db
import os
import datetime
import json
import time
import io
import csv

# Import Advanced Data Science & RAG Logic
from logic.advanced_rag import (
    HybridRetriever, 
    SemanticCache, 
    RAGDiagnostics, 
    DocumentAnalytics, 
    MultiQueryExpander, 
    MultiDocumentComparator,
    safe_parse_json
)

# Initialize Firebase (handling optional secrets gracefully)
try:
    if "FIREBASE" in st.secrets and "service_account" in st.secrets["FIREBASE"]:
        cred_dict = json.loads(st.secrets["FIREBASE"]["service_account"])
        cred = credentials.Certificate(cred_dict)
        if not firebase_admin._apps:
            firebase_admin.initialize_app(cred)
        db_firestore = firestore.client()
    else:
        db_firestore = None
except Exception as e:
        db_firestore = None

st.set_page_config(
    page_title="PDF Brainbox - Advanced RAG & AI Analytics",      
    page_icon="logo.png",           
    initial_sidebar_state="expanded"
)

# Custom Design System (Clean Glassmorphism & Professional Badge System)
st.markdown("""
<style>
/* Modern Glassmorphism & UI Styling */
.badge-triad {
    background: linear-gradient(135deg, rgba(99, 102, 241, 0.25), rgba(168, 85, 247, 0.25));
    color: #a855f7;
    padding: 6px 14px;
    border-radius: 20px;
    font-weight: 700;
    border: 1px solid rgba(168, 85, 247, 0.4);
    display: inline-block;
    font-size: 0.9rem;
}
.badge-faith {
    background: rgba(34, 197, 94, 0.15);
    color: #4ade80;
    padding: 4px 12px;
    border-radius: 20px;
    font-weight: 600;
    border: 1px solid rgba(34, 197, 94, 0.3);
    display: inline-block;
    font-size: 0.82rem;
    margin-right: 6px;
}
.badge-rel {
    background: rgba(59, 130, 246, 0.15);
    color: #60a5fa;
    padding: 4px 12px;
    border-radius: 20px;
    font-weight: 600;
    border: 1px solid rgba(59, 130, 246, 0.3);
    display: inline-block;
    font-size: 0.82rem;
    margin-right: 6px;
}
.badge-prec {
    background: rgba(168, 85, 247, 0.15);
    color: #c084fc;
    padding: 4px 12px;
    border-radius: 20px;
    font-weight: 600;
    border: 1px solid rgba(168, 85, 247, 0.3);
    display: inline-block;
    font-size: 0.82rem;
}
.badge-cache {
    background: rgba(234, 179, 8, 0.15);
    color: #fde047;
    padding: 5px 14px;
    border-radius: 20px;
    font-weight: 600;
    border: 1px solid rgba(234, 179, 8, 0.4);
    display: inline-block;
    font-size: 0.85rem;
}
</style>
""", unsafe_allow_html=True)

# Initialize Session State Variables
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "username" not in st.session_state:
    st.session_state.username = ""
if "semantic_cache" not in st.session_state:
    st.session_state.semantic_cache = SemanticCache()
if "eval_history" not in st.session_state:
    st.session_state.eval_history = []
if "retrieval_mode" not in st.session_state:
    st.session_state.retrieval_mode = "Hybrid (FAISS + BM25 + RRF)"

query_params = st.query_params
email = query_params.get("email", [None])[0]

def fake_email(username):
    return f"{username}@pdfbrainbox.local"
    
if email and not st.session_state.authenticated:
    st.session_state.authenticated = True
    st.session_state.username = email

# Authentication UI
if not st.session_state.authenticated:
    st.title("Login to PDF Brainbox")
    tab1, tab2 = st.tabs(["Login", "Signup"])

    with tab1:
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        if st.button("Login"):
            user_email = fake_email(username)
            if db.verify_user(user_email, password):
                st.success("Login successful!")
                st.session_state.authenticated = True
                st.session_state.username = username
                st.rerun()
            else:
                st.error("Invalid username or password")

        st.markdown("---")
        st.subheader("Or Login with Google")
        st.markdown(
            '<a href="https://uhashini.github.io/pdf-login-page/" target="_blank">'
            '<button style="padding: 0.5rem 1rem; cursor: pointer;">Login with Google</button>'
            '</a>',
            unsafe_allow_html=True
        )

    with tab2:
        new_username = st.text_input("New Username")
        new_password = st.text_input("New Password", type="password")
        if st.button("Create Account"):
            new_email = fake_email(new_username)
            if db.create_user(new_email, new_password):
                st.success("Account created! Please log in.")
            else:
                st.error("Username already exists or error creating account.")

    st.stop()

st.sidebar.write(f"User: **{st.session_state.username}**")

if st.sidebar.button("Logout"):
    st.session_state.authenticated = False
    st.session_state.username = ""
    st.markdown("""
        <meta http-equiv="refresh" content="0;url=https://pdf-brainbox.streamlit.app/" />
    """, unsafe_allow_html=True)
    st.stop()

# Header Logo & Title
logo = "logo.png"
col1, col2 = st.columns([2, 8])

with col1:
    if os.path.exists(logo):
        st.image(logo, width=120)

with col2:
    st.markdown("<h1 style='margin: 0; padding-top: 10px;'>PDF Brainbox</h1>", unsafe_allow_html=True)
    st.caption("Intelligent RAG Platform powered by Hybrid Search, LLM Evaluation & Knowledge Graphs")

# Clean Navigation Bar (No emojis)
tool_map = {
    "Home": "Home",
    "Q&A": "Q&A",
    "Analytics & Graph": "Analytics & Graph",
    "RAG Benchmark": "RAG Benchmark",
    "Compare Docs": "Compare Documents",
    "Quiz": "Quiz",
    "Slides": "Slides",
    "Notes": "Notes",
    "Flashcards": "Flashcards"
}
tool_labels = list(tool_map.keys())

if "current_tool" not in st.session_state:
    st.session_state.current_tool = "Home"

# Top Navigation Bar
st.markdown("---")
selected_label = st.radio(
    "Navigation Bar",
    tool_labels,
    index=tool_labels.index(st.session_state.current_tool) if st.session_state.current_tool in tool_labels else 0,
    horizontal=True,
    label_visibility="collapsed"
)
st.session_state.current_tool = selected_label
page = tool_map[selected_label]

# Sidebar Quick Navigation Selector
sidebar_choice = st.sidebar.selectbox(
    "Quick Tool Selector", 
    tool_labels,
    index=tool_labels.index(st.session_state.current_tool)
)
if sidebar_choice != st.session_state.current_tool:
    st.session_state.current_tool = sidebar_choice
    st.rerun()

# File uploader in sidebar (Supports Multi-Document & Any-Page Uploads)
uploaded_files = st.sidebar.file_uploader(
    "Upload Document(s)", 
    type=["pdf", "docx", "txt", "png", "jpg", "jpeg"],
    accept_multiple_files=True
)

def extract_text_from_file(file):
    file_type = file.name.split('.')[-1].lower()

    if file_type == 'pdf':
        pdf_reader = PdfReader(file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() or ""
        return text

    elif file_type == 'docx':
        doc = Document(file)
        return "\n".join([para.text for para in doc.paragraphs])

    elif file_type == 'txt':
        return file.read().decode('utf-8')

    elif file_type in ['png', 'jpg', 'jpeg']:
        try:
            if os.name == 'nt' and os.path.exists(r"C:\Program Files\Tesseract-OCR\tesseract.exe"):
                pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
            image = Image.open(file)
            return pytesseract.image_to_string(image)
        except Exception as e:
            return f"OCR Extraction Error: {e}"

    else:
        return ""

# JSON Parsing Helper
import re

def safe_parse_json(text_content):
    if not isinstance(text_content, str):
        return text_content
    clean_text = text_content.strip()
    clean_text = re.sub(r"^```(?:json)?", "", clean_text, flags=re.MULTILINE)
    clean_text = re.sub(r"```$", "", clean_text, flags=re.MULTILINE).strip()
    try:
        return json.loads(clean_text)
    except Exception:
        pass
    sb, eb = clean_text.find('['), clean_text.rfind(']')
    if sb != -1 and eb > sb:
        try:
            return json.loads(clean_text[sb:eb + 1])
        except Exception:
            pass
    sb, eb = clean_text.find('{'), clean_text.rfind('}')
    if sb != -1 and eb > sb:
        try:
            return json.loads(clean_text[sb:eb + 1])
        except Exception:
            pass
    raise ValueError(f"Could not parse JSON output: {text_content[:150]}")

# Mistral API Configuration & Dual SDK Compatibility Wrapper (v1 & v0)
def get_api_key():
    env_key = os.getenv("MISTRAL_API_KEY")
    if env_key:
        return env_key
    try:
        if "MISTRAL_API_KEY" in st.secrets:
            return st.secrets["MISTRAL_API_KEY"]
    except Exception:
        pass
    return "Y70bo7Bnkil7MgiZ3VdOdWwH3edP9UK4"

api_key = get_api_key()
embed_model = "mistral-embed"
chat_model = "open-mistral-7b"

class UnifiedMistralClient:
    def __init__(self, key):
        self.key = key
        self.client_v1 = None
        self.client_v0 = None
        self.err_log = []
        
        try:
            from mistralai import Mistral
            self.client_v1 = Mistral(api_key=key)
        except Exception as e1:
            self.err_log.append(f"s1 error: {e1}")

        if not self.client_v1 and not self.client_v0:
            try:
                from mistralai import MistralClient
                self.client_v0 = MistralClient(api_key=key)
            except Exception as e2:
                self.err_log.append(f"s2 error: {e2}")

        if not self.client_v1 and not self.client_v0:
            try:
                from mistralai.client import MistralClient
                self.client_v0 = MistralClient(api_key=key)
            except Exception as e3:
                self.err_log.append(f"s3 error: {e3}")

        if not self.client_v1 and not self.client_v0:
            try:
                import mistralai
                if hasattr(mistralai, "Mistral"):
                    cls = getattr(mistralai, "Mistral")
                    self.client_v1 = cls(api_key=key)
                elif hasattr(mistralai, "MistralClient"):
                    cls = getattr(mistralai, "MistralClient")
                    self.client_v0 = cls(api_key=key)
            except Exception as e4:
                self.err_log.append(f"s4 error: {e4}")

    def get_embedding(self, txt):
        if self.client_v1:
            res = self.client_v1.embeddings.create(model=embed_model, inputs=[txt])
            return res.data[0].embedding
        elif self.client_v0:
            res = self.client_v0.embeddings(model=embed_model, input=[txt])
            return res.data[0].embedding
        else:
            raise RuntimeError(f"Mistral SDK client failed: {self.err_log}")

    def chat_complete(self, message, model_name=None):
        target = model_name or chat_model
        messages = [{"role": "user", "content": message}]
        
        if self.client_v1:
            try:
                res = self.client_v1.chat.complete(model=target, messages=messages)
                return res.choices[0].message.content
            except Exception as e:
                if target != "open-mistral-7b":
                    res = self.client_v1.chat.complete(model="open-mistral-7b", messages=messages)
                    return res.choices[0].message.content
                raise e
        elif self.client_v0:
            from mistralai.models.chat_completion import ChatMessage
            msg_objs = [ChatMessage(role="user", content=message)]
            try:
                res = self.client_v0.chat(model=target, messages=msg_objs)
                return res.choices[0].message.content
            except Exception as e:
                if target != "open-mistral-7b":
                    res = self.client_v0.chat(model="open-mistral-7b", messages=msg_objs)
                    return res.choices[0].message.content
                raise e
        else:
            raise RuntimeError(f"Mistral SDK client failed: {self.err_log}")

unified_mistral = UnifiedMistralClient(api_key)

def get_text_embedding(txt):
    try:
        return unified_mistral.get_embedding(txt)
    except Exception as e:
        st.error(f"Mistral Embedding API Error: {e}. Check API Key or limits on console.mistral.ai.")
        st.stop()

def mistral_chat(user_message, is_json=False):
    try:
        return unified_mistral.chat_complete(user_message)
    except Exception as e:
        return f"Mistral Chat API Error: {e}"

def log_pdf_upload(user_id, file_name):
    if db_firestore:
        try:
            db_firestore.collection("userActivity").document(user_id).collection("uploads").add({
                "filename": file_name,
                "timestamp": datetime.datetime.now()
            })
        except Exception:
            pass

# Global Multi-Document Indexer (Executes on ANY page whenever files are uploaded/updated)
if uploaded_files:
    current_sig = "-".join([f"{f.name}_{f.size}" for f in uploaded_files])
    
    if st.session_state.get("uploaded_sig") != current_sig:
        with st.sidebar.status("Indexing documents...", expanded=True) as status:
            all_chunks = []
            all_text_list = []
            doc_summaries = []
            
            for file in uploaded_files:
                st.write(f"Processing `{file.name}`...")
                text = extract_text_from_file(file)
                if text.strip():
                    chunk_size = 512
                    chunk_overlap = 50
                    file_chunks = []
                    for i in range(0, len(text), chunk_size - chunk_overlap):
                        chunk_content = f"[Doc: {file.name}] {text[i: i + chunk_size]}"
                        file_chunks.append(chunk_content)
                    
                    all_chunks.extend(file_chunks)
                    all_text_list.append(f"--- Document: {file.name} ---\n{text}")
                    doc_summaries.append({
                        "File Name": file.name,
                        "Size (KB)": round(len(file.getbuffer()) / 1024, 2),
                        "Chunks": len(file_chunks)
                    })
                    log_pdf_upload(user_id=st.session_state.username, file_name=file.name)
            
            if all_chunks:
                st.write("Generating embeddings & indexing...")
                text_embeddings = np.array([get_text_embedding(c) for c in all_chunks])
                d = text_embeddings.shape[1]
                index = faiss.IndexFlatL2(d)
                index.add(text_embeddings)
                
                hybrid_retriever = HybridRetriever(all_chunks, index, text_embeddings)
                
                st.session_state.chunks = all_chunks
                st.session_state.index = index
                st.session_state.embeddings = text_embeddings
                st.session_state.hybrid_retriever = hybrid_retriever
                st.session_state.full_text = "\n\n".join(all_text_list)
                st.session_state.doc_summaries = doc_summaries
                st.session_state.uploaded_sig = current_sig
                st.session_state.doc_name = uploaded_files[0].name if len(uploaded_files) == 1 else f"{len(uploaded_files)} Documents"
                
                status.update(label=f"Indexed {len(uploaded_files)} Doc(s) ({len(all_chunks)} chunks)", state="complete")

# Display Active Documents in Sidebar
if "doc_summaries" in st.session_state and st.session_state.doc_summaries:
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"**Indexed Documents ({len(st.session_state.doc_summaries)}):**")
    for doc in st.session_state.doc_summaries:
        st.sidebar.caption(f"• `{doc['File Name']}` ({doc['Chunks']} chunks)")

# HOME PAGE
if page == "Home":
    st.title("Document Indexing & RAG Dashboard")
    st.markdown("Tip: Upload single or multiple documents anytime in the sidebar on ANY page.")

    if "doc_summaries" in st.session_state and st.session_state.doc_summaries:
        st.subheader("Active Indexed Documents")
        st.dataframe(st.session_state.doc_summaries, use_container_width=True)

        st.success(f"Total **{len(st.session_state.chunks)} chunks** indexed across **{len(st.session_state.doc_summaries)} document(s)** using Dense FAISS & Sparse BM25.")
        st.subheader("Extracted Text Preview")
        st.code(st.session_state.full_text[:1000] + "...")
    else:
        st.info("Upload your PDF, DOCX, TXT, or Image files using the sidebar to start indexing.")

    st.markdown("---")
    st.subheader("Explore AI Tools & Features")
    col_t1, col_t2, col_t3 = st.columns(3)

    with col_t1:
        st.markdown("#### Document Q&A")
        st.caption("Ask questions with Hybrid Search (BM25 + FAISS RRF) & RAG Triad Diagnostics.")
        if st.button("Launch Q&A Engine ->", use_container_width=True):
            st.session_state.current_tool = "Q&A"
            st.rerun()

        st.markdown("#### Compare Documents")
        st.caption("Upload a second PDF to calculate Semantic Overlap % & topic divergence.")
        if st.button("Compare 2 PDFs ->", use_container_width=True):
            st.session_state.current_tool = "Compare Docs"
            st.rerun()

        st.markdown("#### Study Notes")
        st.caption("Generate structured bulleted study notes & download as Markdown.")
        if st.button("Generate Notes ->", use_container_width=True):
            st.session_state.current_tool = "Notes"
            st.rerun()

    with col_t2:
        st.markdown("#### Analytics & Graph")
        st.caption("Extract Flesch Readability statistics & 2D/3D physics Knowledge Graphs.")
        if st.button("Launch Analytics & Graph ->", use_container_width=True):
            st.session_state.current_tool = "Analytics & Graph"
            st.rerun()

        st.markdown("#### Interactive Quiz")
        st.caption("Auto-generate multiple choice quizzes from document content.")
        if st.button("Start Quiz ->", use_container_width=True):
            st.session_state.current_tool = "Quiz"
            st.rerun()

        st.markdown("#### Presentation Slides")
        st.caption("Generate presentation outlines & download directly as PowerPoint (.pptx).")
        if st.button("Generate Slides ->", use_container_width=True):
            st.session_state.current_tool = "Slides"
            st.rerun()

    with col_t3:
        st.markdown("#### RAG Benchmark")
        st.caption("View real-time evaluation logs, Faithfulness metrics, & Latency charts.")
        if st.button("View Telemetry Dashboard ->", use_container_width=True):
            st.session_state.current_tool = "RAG Benchmark"
            st.rerun()

        st.markdown("#### Study Flashcards")
        st.caption("Interactive 3D CSS flip flashcards & Anki CSV Deck Exporter.")
        if st.button("Study Flashcards ->", use_container_width=True):
            st.session_state.current_tool = "Flashcards"
            st.rerun()

# Q&A PAGE
elif page == "Q&A":
    st.title("Advanced Question Answering Engine")
    st.markdown("Powered by **Hybrid Search (BM25 + FAISS RRF)**, **Agentic Multi-Query Expansion**, **Semantic Caching**, and **LLM Evaluation**.")

    if "chunks" not in st.session_state or "hybrid_retriever" not in st.session_state:
        st.warning("Please upload a document in the sidebar to get started.")
    else:
        with st.expander("Advanced Retrieval & Search Controls", expanded=False):
            col_m1, col_m2 = st.columns(2)
            with col_m1:
                retrieval_mode = st.radio(
                    "Retrieval Strategy",
                    ["Hybrid (FAISS + BM25 + RRF)", "Dense Vector (FAISS)", "Sparse Keyword (BM25)"]
                )
                st.session_state.retrieval_mode = retrieval_mode
                enable_multi_query = st.checkbox("Enable Agentic Multi-Query Expansion", value=False)
            with col_m2:
                top_k = st.slider("Top Chunks Retrieved ($k$)", 1, 5, 3)

        if "messages" not in st.session_state:
            st.session_state.messages = []

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                if "eval" in message:
                    ev = message["eval"]
                    badge_html = f"""
                    <div style="margin-top: 8px;">
                      <span class="badge-triad">Triad Score: {ev['triad_score']}/1.0</span>
                      <span class="badge-faith">Faithfulness: {ev['faithfulness']}</span>
                      <span class="badge-rel">Relevance: {ev['answer_relevance']}</span>
                      <span class="badge-prec">Precision: {ev['context_precision']}</span>
                    </div>
                    """
                    st.markdown(badge_html, unsafe_allow_html=True)
                    st.caption(f"Latency: `{ev['latency_sec']}s` | Strategy: `{ev.get('mode', 'Hybrid')}`")

        if question := st.chat_input("Ask a question about the document..."):
            with st.chat_message("user"):
                st.markdown(question)
            st.session_state.messages.append({"role": "user", "content": question})

            t_start = time.time()
            question_emb = get_text_embedding(question)

            # 1. Semantic Cache Lookup
            cached_res = st.session_state.semantic_cache.lookup(question_emb, threshold=0.93)
            if cached_res:
                ans = cached_res["answer"]
                latency = round(time.time() - t_start, 4)
                with st.chat_message("assistant"):
                    st.markdown(ans)
                    st.markdown(f'<span class="badge-cache">Semantic Cache Hit (Similarity: {cached_res["similarity"]:.3f}) — Latency: {latency}s</span>', unsafe_allow_html=True)
                st.session_state.messages.append({"role": "assistant", "content": ans})
            else:
                query_list = [question]
                if enable_multi_query:
                    with st.spinner("Agentic Multi-Query Expansion generating variations..."):
                        variations = MultiQueryExpander.expand_query(question, mistral_chat)
                        query_list = [question] + variations
                        st.info(f"Generated Query Variations: {variations}")

                mode_key = "hybrid" if "Hybrid" in retrieval_mode else ("sparse" if "Sparse" in retrieval_mode else "dense")
                t_ret_start = time.time()
                
                all_chunks = []
                for q in query_list:
                    q_emb = get_text_embedding(q)
                    ret_chunks = st.session_state.hybrid_retriever.search(
                        query=q,
                        query_embedding=q_emb,
                        top_k=top_k,
                        mode=mode_key
                    )
                    all_chunks.extend(ret_chunks)
                
                seen = set()
                retrieved_chunks = [c for c in all_chunks if not (c in seen or seen.add(c))][:top_k * 2]
                t_retrieval = time.time() - t_ret_start

                context = "\n---\n".join(retrieved_chunks)
                prompt = f"""
Context information from document:
---------------------
{context}
---------------------
Given the context information above, answer the query concisely and accurately.
Query: {question}
Answer:
"""
                t_gen_start = time.time()
                answer = mistral_chat(prompt)
                t_generation = time.time() - t_gen_start
                total_latency = round(time.time() - t_start, 2)

                eval_metrics = RAGDiagnostics.evaluate(
                    query=question,
                    context=context,
                    answer=answer,
                    mistral_chat_fn=mistral_chat
                )
                eval_metrics["latency_sec"] = total_latency
                eval_metrics["retrieval_latency"] = round(t_retrieval, 3)
                eval_metrics["generation_latency"] = round(t_generation, 3)
                eval_metrics["query"] = question
                eval_metrics["mode"] = f"{retrieval_mode} + MultiQuery" if enable_multi_query else retrieval_mode

                st.session_state.semantic_cache.add(question, question_emb, answer, eval_metrics)
                st.session_state.eval_history.append(eval_metrics)

                with st.chat_message("assistant"):
                    st.markdown(answer)

                    with st.expander("RAG Diagnostics & Evaluation Metrics", expanded=True):
                        st.markdown(f"""
                        <div style="margin-bottom: 12px;">
                          <span class="badge-triad">Triad Score: {eval_metrics['triad_score']} / 1.0</span>
                          <span class="badge-faith">Faithfulness: {eval_metrics['faithfulness']}</span>
                          <span class="badge-rel">Relevance: {eval_metrics['answer_relevance']}</span>
                          <span class="badge-prec">Precision: {eval_metrics['context_precision']}</span>
                        </div>
                        """, unsafe_allow_html=True)
                        st.markdown(f"**Evaluator Feedback:** *\"{eval_metrics['reasoning']}\"*")
                        st.caption(f"Latency Split: Total `{total_latency}s` (Retrieval `{round(t_retrieval,3)}s` + Generation `{round(t_generation,3)}s`) | Strategy: `{retrieval_mode}`")

                st.session_state.messages.append({"role": "assistant", "content": answer, "eval": eval_metrics})

# ANALYTICS & KNOWLEDGE GRAPH PAGE
elif page == "Analytics & Graph":
    st.title("Document Analytics & Knowledge Graph")
    st.markdown("Extract structural NLP insights, readability indices, and interactive entity relation maps.")

    if "full_text" not in st.session_state:
        st.warning("Please upload a document in the sidebar to get started.")
    else:
        text = st.session_state.full_text

        stats = DocumentAnalytics.compute_stats(text)
        st.subheader("Readability & Lexical Metrics")

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Word Count", stats["num_words"])
        m2.metric("Sentences", stats["num_sentences"])
        m3.metric("Lexical Diversity", f"{stats['lexical_diversity']}%")
        m4.metric("Flesch Reading Ease", f"{stats['reading_ease']} / 100")

        st.markdown("---")
        st.subheader("Concept Knowledge Graph")
        st.markdown("Extracting Entity-Relationship triples to map key document concepts.")

        if st.button("Generate Concept Knowledge Graph") or "knowledge_triples" in st.session_state:
            if "knowledge_triples" not in st.session_state:
                with st.spinner("Extracting entity triples via LLM..."):
                    st.session_state.knowledge_triples = DocumentAnalytics.extract_knowledge_graph(text, mistral_chat)

            triples = st.session_state.knowledge_triples
            st.write("### Identified Knowledge Triples")
            st.dataframe(triples, use_container_width=True)

            nodes_dict = {}
            edges_list = []
            node_id = 1

            for t in triples:
                src = str(t.get("source", "A"))
                rel = str(t.get("relation", "relates to"))
                tgt = str(t.get("target", "B"))

                if src not in nodes_dict:
                    nodes_dict[src] = node_id
                    node_id += 1
                if tgt not in nodes_dict:
                    nodes_dict[tgt] = node_id
                    node_id += 1

                edges_list.append({
                    "from": nodes_dict[src],
                    "to": nodes_dict[tgt],
                    "label": rel,
                    "arrows": "to"
                })

            nodes_list = [
                {"id": v, "label": k, "color": "#60a5fa" if i % 2 == 0 else "#4ade80", "shape": "dot", "size": 22} 
                for i, (k, v) in enumerate(nodes_dict.items())
            ]

            vis_html = f"""
            <!DOCTYPE html>
            <html>
            <head>
              <script type="text/javascript" src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
              <style>
                #network-canvas {{
                  width: 100%;
                  height: 450px;
                  border: 1px solid rgba(255,255,255,0.15);
                  background-color: #0f172a;
                  border-radius: 12px;
                }}
              </style>
            </head>
            <body>
              <div id="network-canvas"></div>
              <script type="text/javascript">
                var nodes = new vis.DataSet({json.dumps(nodes_list)});
                var edges = new vis.DataSet({json.dumps(edges_list)});
                var container = document.getElementById('network-canvas');
                var data = {{ nodes: nodes, edges: edges }};
                var options = {{
                  nodes: {{ font: {{ color: '#ffffff', size: 14 }}, borderWidth: 2 }},
                  edges: {{ font: {{ color: '#94a3b8', size: 12, align: 'middle' }}, color: {{ color: '#475569', highlight: '#38bdf8' }} }},
                  physics: {{ stabilization: true, barnesHut: {{ gravitationalConstant: -2500 }} }}
                }};
                var network = new vis.Network(container, data, options);
              </script>
            </body>
            </html>
            """

            st.write("### Interactive 2D/3D Graph Physics Network")
            st.caption("Drag nodes around, click concepts, and scroll to zoom in/out.")
            components.html(vis_html, height=470)

# RAG BENCHMARK PAGE
elif page == "RAG Benchmark":
    st.title("RAG Evaluation Benchmark Dashboard")
    st.markdown("Quantitative telemetry dashboard tracking real-time evaluation logs across queries.")

    if not st.session_state.eval_history:
        st.info("No query evaluation logs recorded yet. Ask questions in the **Q&A** tab to populate the telemetry benchmark.")
    else:
        history = st.session_state.eval_history
        avg_triad = round(np.mean([h["triad_score"] for h in history]), 2)
        avg_faith = round(np.mean([h["faithfulness"] for h in history]), 2)
        avg_rel = round(np.mean([h["answer_relevance"] for h in history]), 2)
        avg_prec = round(np.mean([h["context_precision"] for h in history]), 2)
        avg_lat = round(np.mean([h["latency_sec"] for h in history]), 2)

        st.subheader("Session Average Metrics")
        b1, b2, b3, b4, b5 = st.columns(5)
        b1.metric("Overall Triad Score", f"{avg_triad} / 1.0")
        b2.metric("Faithfulness", avg_faith)
        b3.metric("Answer Relevance", avg_rel)
        b4.metric("Context Precision", avg_prec)
        b5.metric("Avg Latency", f"{avg_lat}s")

        st.markdown("---")
        st.subheader("Telemetry Visual Analytics")
        
        chart_data = {
            "Faithfulness": [h["faithfulness"] for h in history],
            "Answer Relevance": [h["answer_relevance"] for h in history],
            "Context Precision": [h["context_precision"] for h in history]
        }
        st.write("#### Metric Score Trend across Queries")
        st.bar_chart(chart_data)

        latency_data = {
            "Total Latency (s)": [h["latency_sec"] for h in history],
            "Retrieval Latency (s)": [h.get("retrieval_latency", 0) for h in history],
            "Generation Latency (s)": [h.get("generation_latency", 0) for h in history]
        }
        st.write("#### Latency Performance Breakdown (Seconds)")
        st.line_chart(latency_data)

        st.markdown("---")
        st.subheader("Query Telemetry Log")
        log_table = []
        for h in history:
            log_table.append({
                "Query": h.get("query", ""),
                "Strategy": h.get("mode", ""),
                "Triad Score": h.get("triad_score", 0),
                "Faithfulness": h.get("faithfulness", 0),
                "Relevance": h.get("answer_relevance", 0),
                "Precision": h.get("context_precision", 0),
                "Latency (s)": h.get("latency_sec", 0)
            })
        st.dataframe(log_table, use_container_width=True)

# COMPARE DOCUMENTS PAGE
elif page == "Compare Documents":
    st.title("Multi-Document Comparison & Semantic Similarity")
    st.markdown("Compare two documents side-by-side to compute embedding cosine similarity, shared topics, and comparative Q&A.")

    if "full_text" not in st.session_state or "embeddings" not in st.session_state:
        st.warning("Please upload your primary document in the sidebar to get started.")
    else:
        text_a = st.session_state.full_text
        embeddings_a = st.session_state.embeddings
        doc_a_name = st.session_state.get("doc_name", "Primary Document")

        st.info(f"**Document A:** `{doc_a_name}` (Indexed)")

        uploaded_b = st.file_uploader("Upload Second Document (Document B)", type=["pdf", "docx", "txt"])

        if uploaded_b is not None:
            text_b = extract_text_from_file(uploaded_b)
            if text_b.strip():
                with st.spinner("Embedding Document B for comparison..."):
                    chunk_size = 512
                    chunks_b = [text_b[i:i+chunk_size] for i in range(0, len(text_b), chunk_size)]
                    embeddings_b = np.array([get_text_embedding(c) for c in chunks_b])

                sim_score = MultiDocumentComparator.compute_similarity(embeddings_a, embeddings_b)

                st.subheader("Document Similarity Metrics")
                c1, c2 = st.columns(2)
                c1.metric("Semantic Overlap", f"{sim_score}%")
                if sim_score > 75:
                    c2.success("High Topic Alignment")
                elif sim_score > 45:
                    c2.info("Moderate Semantic Similarity")
                else:
                    c2.warning("Distinct/Divergent Content")

                st.markdown("---")
                st.subheader("Comparative Topic Extraction")
                if st.button("Extract Shared & Unique Topics") or "topic_comparison" in st.session_state:
                    if "topic_comparison" not in st.session_state:
                        with st.spinner("Comparing concepts across documents..."):
                            st.session_state.topic_comparison = MultiDocumentComparator.compare_topics(text_a, text_b, mistral_chat)

                    comp = st.session_state.topic_comparison
                    st.write(f"**Summary:** *\"{comp.get('comparison_summary', '')}\"*")

                    tc1, tc2, tc3 = st.columns(3)
                    with tc1:
                        st.markdown("### Shared Topics")
                        for t in comp.get("shared_topics", []):
                            st.write(f"- {t}")
                    with tc2:
                        st.markdown(f"### Unique to {doc_a_name}")
                        for t in comp.get("unique_to_doc_a", []):
                            st.write(f"- {t}")
                    with tc3:
                        st.markdown(f"### Unique to {uploaded_b.name}")
                        for t in comp.get("unique_to_doc_b", []):
                            st.write(f"- {t}")

# QUIZ PAGE
elif page == "Quiz":
    st.title("Quiz from Document")

    if "chunks" not in st.session_state:
        st.warning("Please upload a document in the sidebar to get started.")
    else:
        chunks = st.session_state.chunks
        full_text = " ".join(chunks)
        num_questions = st.slider("Select number of questions", 1, 5, 3)

        if st.button("Generate Quiz") or "quiz_data" in st.session_state:
            if "quiz_data" not in st.session_state:
                quiz_prompt = f"""
From the following text, generate {num_questions} multiple choice questions in this JSON format:
[
  {{
    "question": "What is ...?",
    "options": ["A", "B", "C", "D"],
    "answer": "B"
  }}
]
Only return valid JSON without markdown codeblocks:
{full_text[:3000]}
"""
                try:
                    with st.spinner("Generating quiz questions..."):
                        quiz_str = mistral_chat(quiz_prompt, is_json=True)
                        st.session_state.quiz_data = safe_parse_json(quiz_str)
                except Exception as e:
                    st.error(f"Error generating quiz: {e}")
                    st.stop()

            quiz_data = st.session_state.quiz_data
            if "quiz_answers" not in st.session_state:
                st.session_state.quiz_answers = {}

            for i, q in enumerate(quiz_data):
                st.markdown(f"**Q{i+1}. {q['question']}**")
                user_choice = st.radio(
                    label="",
                    options=q["options"],
                    key=f"q_{i}"
                )
                st.session_state.quiz_answers[f"q_{i}"] = {
                    "selected": user_choice,
                    "correct": q["answer"]
                }

            if st.button("Submit Quiz Answers"):
                st.subheader("Results")
                correct_count = 0
                for i in range(len(quiz_data)):
                    selected = st.session_state.quiz_answers[f"q_{i}"]["selected"]
                    correct = st.session_state.quiz_answers[f"q_{i}"]["correct"]
                    if selected == correct:
                        st.success(f"Correct - Q{i+1}")
                        correct_count += 1
                    else:
                        st.error(f"Incorrect - Q{i+1} (Correct: {correct})")

                st.progress(correct_count / len(quiz_data))
                st.info(f"Score: {correct_count} / {len(quiz_data)}")

# NOTES PAGE
elif page == "Notes":
    st.title("Automated Study Notes")
    if "chunks" not in st.session_state:
        st.warning("Please upload a document in the sidebar to get started.")
    else:
        full_text = " ".join(st.session_state.chunks)
        if st.button("Generate Study Notes") or "generated_notes" in st.session_state:
            if "generated_notes" not in st.session_state:
                prompt = f"""
Create structured, bulleted study notes from the following content:
{full_text[:4000]}
Include key definitions and core summary points.
"""
                with st.spinner("Generating study notes..."):
                    st.session_state.generated_notes = mistral_chat(prompt)

            notes = st.session_state.generated_notes
            st.markdown(notes)
            
            c1, c2 = st.columns(2)
            c1.download_button("Download Notes (.txt)", notes, file_name="study_notes.txt", mime="text/plain")
            c2.download_button("Download Markdown (.md)", notes, file_name="study_notes.md", mime="text/markdown")

# SLIDES PAGE
elif page == "Slides":
    st.title("Presentation Slides Generator")
    if "chunks" not in st.session_state:
        st.warning("Please upload a document in the sidebar to get started.")
    else:
        full_text = " ".join(st.session_state.chunks)
        if st.button("Generate Presentation Outline") or "generated_slides_text" in st.session_state:
            if "generated_slides_text" not in st.session_state:
                prompt = f"""
Create clear presentation slide titles and bullet points from the content:
{full_text[:4000]}
Format slide titles clearly separated by double newlines.
"""
                with st.spinner("Creating slides..."):
                    st.session_state.generated_slides_text = mistral_chat(prompt)

            slides_text = st.session_state.generated_slides_text
            st.markdown(slides_text)

            def generate_pptx_from_text(text):
                prs = Presentation()
                slide_layout = prs.slide_layouts[1]
                slides = text.strip().split("\n\n")
                for slide in slides:
                    lines = slide.strip().split("\n")
                    if not lines:
                        continue
                    title = lines[0]
                    content = lines[1:]
                    slide_obj = prs.slides.add_slide(slide_layout)
                    slide_obj.shapes.title.text = title
                    body = slide_obj.placeholders[1]
                    body.text = "\n".join(content)
                pptx_path = "presentation.pptx"
                prs.save(pptx_path)
                return pptx_path

            pptx_file_path = generate_pptx_from_text(slides_text)
            with open(pptx_file_path, "rb") as f:
                st.download_button("Download Presentation (.pptx)", f, file_name="presentation.pptx")

# FLASHCARDS PAGE
elif page == "Flashcards":
    st.title("Interactive 3D Study Flashcards")
    if "chunks" not in st.session_state:
        st.warning("Please upload a document in the sidebar to get started.")
    else:
        full_text = " ".join(st.session_state.chunks)
        if st.button("Generate Flashcards") or "flashcards" in st.session_state:
            if "flashcards" not in st.session_state:
                flashcard_prompt = f"""
Create 5 study flashcards in JSON format:
[
  {{"question": "What is ...?", "answer": "..."}}
]
Text:
{full_text[:3000]}
"""
                try:
                    with st.spinner("Generating flashcards..."):
                        fc_str = mistral_chat(flashcard_prompt, is_json=True)
                        st.session_state.flashcards = safe_parse_json(fc_str)
                except Exception as e:
                    st.error(f"Error creating flashcards: {e}")
                    st.stop()

            flashcards = st.session_state.flashcards
            if "flashcard_index" not in st.session_state:
                st.session_state.flashcard_index = 0

            idx = st.session_state.flashcard_index
            card = flashcards[idx]
            st.subheader(f"Flashcard {idx + 1} of {len(flashcards)}")

            # 3D Flip Card Component
            flip_card_html = f"""
            <!DOCTYPE html>
            <html>
            <head>
            <style>
            .flip-card {{
              background-color: transparent;
              width: 100%;
              height: 220px;
              perspective: 1000px;
              cursor: pointer;
            }}
            .flip-card-inner {{
              position: relative;
              width: 100%;
              height: 100%;
              text-align: center;
              transition: transform 0.6s;
              transform-style: preserve-3d;
            }}
            .flip-card:hover .flip-card-inner {{
              transform: rotateY(180deg);
            }}
            .flip-card-front, .flip-card-back {{
              position: absolute;
              width: 100%;
              height: 100%;
              -webkit-backface-visibility: hidden;
              backface-visibility: hidden;
              border-radius: 14px;
              padding: 24px;
              box-sizing: border-box;
              display: flex;
              flex-direction: column;
              justify-content: center;
              align-items: center;
            }}
            .flip-card-front {{
              background: linear-gradient(135deg, #1e293b, #0f172a);
              color: white;
              border: 1px solid #3b82f6;
            }}
            .flip-card-back {{
              background: linear-gradient(135deg, #065f46, #047857);
              color: white;
              transform: rotateY(180deg);
              border: 1px solid #10b981;
            }}
            </style>
            </head>
            <body>
            <div class="flip-card">
              <div class="flip-card-inner">
                <div class="flip-card-front">
                  <h3 style="margin:0; color:#60a5fa;">Question</h3>
                  <p style="font-size:1.1rem; font-weight:500;">{card['question']}</p>
                  <small style="color:#94a3b8;">(Hover or tap to reveal answer)</small>
                </div>
                <div class="flip-card-back">
                  <h3 style="margin:0; color:#34d399;">Answer</h3>
                  <p style="font-size:1.1rem;">{card['answer']}</p>
                </div>
              </div>
            </div>
            </body>
            </html>
            """
            components.html(flip_card_html, height=240)

            # Anki CSV Deck Exporter
            csv_buffer = io.StringIO()
            writer = csv.writer(csv_buffer)
            writer.writerow(["Front", "Back"])
            for fc in flashcards:
                writer.writerow([fc.get("question", ""), fc.get("answer", "")])
            anki_csv_bytes = csv_buffer.getvalue()

            st.download_button(
                "Download Anki Deck (.csv)", 
                anki_csv_bytes, 
                file_name="pdf_brainbox_anki_deck.csv", 
                mime="text/csv"
            )

            col1, col2 = st.columns(2)
            with col1:
                if st.button("Previous") and st.session_state.flashcard_index > 0:
                    st.session_state.flashcard_index -= 1
                    st.rerun()
            with col2:
                if st.button("Next") and st.session_state.flashcard_index < len(flashcards) - 1:
                    st.session_state.flashcard_index += 1
                    st.rerun()
