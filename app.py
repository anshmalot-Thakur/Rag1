#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""NCERT Class 10 Science RAG Chatbot"""

import os
import sys
import streamlit as st

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeEmbeddings, PineconeVectorStore
from langchain_classic.chains.retrieval_qa.base import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from pinecone import Pinecone, ServerlessSpec
import time

try:
    import pytesseract
except ImportError:
    pytesseract = None

# ── Load environment ──────────────────────────────────────────────────────────
load_dotenv()

PINECONE_API_KEY   = os.getenv("PINECONE_API_KEY")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

REQUIRED_KEYS = {"OPENROUTER_API_KEY": "OpenRouter LLM API", "PINECONE_API_KEY": "Pinecone Vector DB"}
missing_keys  = [k for k in REQUIRED_KEYS if not os.getenv(k)]

# ── Config ────────────────────────────────────────────────────────────────────
_BASE         = os.path.dirname(os.path.abspath(__file__))
PDF_FOLDER    = os.path.join(_BASE, "science")
IMAGES_FOLDER = os.path.join(_BASE, "science_images")

PINECONE_API_KEY    = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = "ncert-science"

EMBED_MODEL  = "multilingual-e5-large"
LLM_MODEL    = "google/gemini-2.0-flash-001"
RETRIEVER_K  = 10
CHUNK_SIZE   = 800
CHUNK_OVERLAP= 200

# ── Embeddings ────────────────────────────────────────────────────────────────
def get_embeddings():
    return PineconeEmbeddings(pinecone_api_key=PINECONE_API_KEY, model=EMBED_MODEL)

# ── Prompt ────────────────────────────────────────────────────────────────────
prompt_template = """You are the BEST NCERT Class 10 Science tutor in the world.
A student has asked you a question. Using the NCERT context provided, write COMPLETE, STRUCTURED STUDY NOTES.

Your answer MUST follow this format when relevant:

## [Topic Title]

**Definition / What is it?**
Write a clear, complete definition in 2-4 sentences.

**Key Equation / Chemical Reaction** (if applicable)
Write the equation using proper notation, for example:
  6CO₂ + 6H₂O → C₆H₁₂O₆ + 6O₂  (in presence of sunlight + chlorophyll)

**How it works / Process**
Explain step by step, clearly and completely.

**Important Components / Raw Materials**
List what is needed and why each matters.

**Key Points to Remember**
- 4-6 bullet points a student must memorize for exams.

**Real-life Examples / Significance**
Why this matters. 1-2 practical examples.

RULES:
- Be THOROUGH. A student should ace their exam using only your answer.
- Use correct scientific terminology but explain it simply.
- Format equations on their own line.
- Do NOT add inline [Source: ...] citations.
- Skip any section for which the context has no information.

Context from NCERT Class 10 Science:
{context}

Student's Question: {question}

Structured Answer:"""

prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="NCERT 10 Science", page_icon="📚", layout="centered")

if missing_keys:
    st.error("Missing API keys: " + ", ".join(missing_keys))
    st.stop()

st.title("📚 NCERT Class 10 Science")
st.caption("Ask any question — get structured study notes with textbook pages")
st.divider()

# ── Connect to Pinecone (fast, no upload) ─────────────────────────────────────
@st.cache_resource
def load_data():
    """Connect to existing Pinecone index — instant each startup."""
    try:
        pc    = Pinecone(api_key=PINECONE_API_KEY)
        embed = get_embeddings()

        index_names = [idx.name for idx in pc.list_indexes().indexes]
        if PINECONE_INDEX_NAME not in index_names:
            return "Pinecone index not found. Run upload_to_pinecone.py first."

        stats = pc.Index(PINECONE_INDEX_NAME).describe_index_stats()
        if stats.get("total_vector_count", 0) == 0:
            return "Pinecone index is empty. Run upload_to_pinecone.py first."

        db = PineconeVectorStore(
            index_name=PINECONE_INDEX_NAME,
            embedding=embed,
            pinecone_api_key=PINECONE_API_KEY
        )
        return db

    except Exception as e:
        return f"Connection error: {e}"

# ── RAG chain ─────────────────────────────────────────────────────────────────
def init_rag_chain():
    db = load_data()

    if isinstance(db, str):
        st.error(f"Database error: {db}")
        return None

    retriever = db.as_retriever(search_kwargs={"k": RETRIEVER_K})

    llm = ChatOpenAI(
        model=LLM_MODEL,
        openai_api_key=OPENROUTER_API_KEY,
        openai_api_base="https://openrouter.ai/api/v1",
        temperature=0.4,
        max_tokens=4000,
        top_p=0.9,
        default_headers={
            "HTTP-Referer": "http://localhost:8501",
            "X-Title": "NCERT Science Bot",
        }
    )

    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True
    )

# ── Image lookup ──────────────────────────────────────────────────────────────
def get_page_images(source_docs, max_images=4):
    """Return paths of textbook page images matching retrieved chunks."""
    images, seen = [], set()
    for doc in source_docs:
        src  = doc.metadata.get("source", "")
        page = doc.metadata.get("page", None)
        if page is None:
            continue
        page_int = int(page)
        key      = (src, page_int)
        if key in seen:
            continue
        seen.add(key)
        pdf_stem = os.path.splitext(src)[0]
        img_path = os.path.join(IMAGES_FOLDER, pdf_stem, f"page_{page_int}.jpg")
        if os.path.exists(img_path):
            images.append({"path": img_path, "src": src, "page": page_int})
        if len(images) >= max_images:
            break
    return images

# ── Typo / fuzzy query correction ─────────────────────────────────────────────
import difflib

SCIENCE_VOCAB = [
    "photosynthesis","respiration","digestion","nutrition","transportation",
    "excretion","reproduction","heredity","evolution","electricity",
    "magnetism","magnetic","reflection","refraction","lens","mirror",
    "acid","base","salt","chemical","reaction","metal","nonmetal",
    "carbon","oxygen","hydrogen","nitrogen","iron","copper","zinc",
    "atom","molecule","element","compound","mixture","solution",
    "cell","tissue","organ","organism","ecosystem","food chain",
    "chlorophyll","chloroplast","mitochondria","nucleus","membrane",
    "newton","force","motion","gravity","pressure","energy","work","power",
    "current","voltage","resistance","circuit","ohm","watt","ampere",
    "gene","chromosome","dna","mutation","variation","natural selection",
    "fossil","carbon dating","speciation","embryology",
    "ozone","greenhouse","pollution","biodiversity","conservation",
    "nephron","kidney","lung","heart","blood","plasma","haemoglobin",
    "hormone","enzyme","neuron","reflex","brain","spinal cord",
    "alloy","corrosion","galvanization","oxidation","reduction",
    "periodic table","valency","atomic number","mass number","isotope",
    "soap","detergent","ester","alcohol","aldehyde","ketone",
    "electromagnetic","induction","generator","motor","transformer",
    "resistance","conductor","insulator","semiconductor","diode","transistor",
]

def suggest_correction(query: str):
    """Return corrected query if typos are detected, else return None."""
    words = query.lower().split()
    corrections = []
    changed = False
    for word in words:
        clean = word.strip("?.,!")
        matches = difflib.get_close_matches(clean, SCIENCE_VOCAB, n=1, cutoff=0.75)
        if matches and matches[0] != clean:
            corrections.append(matches[0])
            changed = True
        else:
            corrections.append(clean)
    if changed:
        return " ".join(corrections)
    return None

# ── Main UI ───────────────────────────────────────────────────────────────────
st.markdown("""
<style>
.presentation-card {
    background: #f8f9fa;
    border-left: 4px solid #4a90d9;
    border-radius: 8px;
    padding: 1rem 1.2rem;
    margin-bottom: 1rem;
}
.page-label {
    font-size: 0.75rem;
    color: #888;
    text-align: center;
    margin-top: 4px;
}
</style>
""", unsafe_allow_html=True)

raw_query = st.text_input(
    "Ask your question:",
    placeholder="e.g. What is photosynthesis?",
    label_visibility="collapsed"
)

# ── Typo detection ──
if raw_query.strip():
    correction = suggest_correction(raw_query.strip())
    if correction and correction.lower() != raw_query.strip().lower():
        col1, col2 = st.columns([3, 1])
        with col1:
            st.warning(f"🔍 Did you mean: **{correction}**?")
        with col2:
            use_corrected = st.button("✅ Use corrected")
        query = correction if use_corrected else raw_query.strip()
    else:
        query = raw_query.strip()
else:
    query = ""

if query:
    try:
        qa_chain = init_rag_chain()

        if qa_chain is not None:
            with st.spinner("Thinking..."):
                response    = qa_chain.invoke({"query": query})
            answer      = response["result"]
            source_docs = response.get("source_documents", [])
            page_images = get_page_images(source_docs)

            st.divider()

            # ── PRESENTATION LAYOUT ──────────────────────────────────────────
            # Split answer into sections by ## headings for interleaved layout
            import re
            sections = re.split(r'(?=^##\s)', answer, flags=re.MULTILINE)
            sections = [s.strip() for s in sections if s.strip()]

            img_index = 0  # track which image to show next

            for s_idx, section in enumerate(sections):
                if s_idx == 0 and page_images:
                    # First section: text LEFT, first page image RIGHT (like a book opening)
                    col_text, col_img = st.columns([3, 2], gap="large")
                    with col_text:
                        st.markdown(section)
                    with col_img:
                        img = page_images[img_index]
                        st.image(img["path"], use_container_width=True)
                        st.markdown(
                            f"<p class='page-label'>📄 {img['src']} • Page {img['page']+1}</p>",
                            unsafe_allow_html=True
                        )
                        img_index += 1
                elif s_idx % 2 == 1 and img_index < len(page_images):
                    # Alternate sections: image LEFT, text RIGHT
                    col_img, col_text = st.columns([2, 3], gap="large")
                    with col_img:
                        img = page_images[img_index]
                        st.image(img["path"], use_container_width=True)
                        st.markdown(
                            f"<p class='page-label'>📄 {img['src']} • Page {img['page']+1}</p>",
                            unsafe_allow_html=True
                        )
                        img_index += 1
                    with col_text:
                        st.markdown(section)
                else:
                    # No image for this section — full width
                    st.markdown(section)

            # ── Any leftover images not used above ──────────────────────────
            remaining = page_images[img_index:]
            if remaining:
                st.divider()
                st.markdown("**📐 More textbook pages:**")
                cols = st.columns(min(len(remaining), 2))
                for i, img in enumerate(remaining):
                    with cols[i % 2]:
                        st.image(img["path"], use_container_width=True)
                        st.markdown(
                            f"<p class='page-label'>📄 {img['src']} • Page {img['page']+1}</p>",
                            unsafe_allow_html=True
                        )

            elif not page_images and not os.path.exists(IMAGES_FOLDER):
                st.info("Run extract_page_images.py once to enable textbook page previews.")

            # ── Sources collapsible ──────────────────────────────────────────
            if source_docs:
                seen2, refs = set(), []
                for doc in source_docs:
                    src  = doc.metadata.get("source", "Unknown")
                    page = doc.metadata.get("page", "?")
                    key  = (src, page)
                    if key not in seen2:
                        seen2.add(key)
                        p_display = int(page) + 1 if str(page).replace(".", "").isdigit() else page
                        refs.append(f"📄 **{src}** — Page {p_display}")
                with st.expander("📚 Sources", expanded=False):
                    for r in refs:
                        st.markdown(r)

    except Exception as e:
        st.error(f"Error: {e}")

