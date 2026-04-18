#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
📚 NCERT 10 Science - RAG Chat Engine
Simple, fast question answering for NCERT Science content
"""
 
import os
import sys
import time
import streamlit as st
 
# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
 
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
 
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_pinecone import PineconeVectorStore
from langchain_classic.chains.retrieval_qa.base import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from pinecone import Pinecone, ServerlessSpec
try:
    import pytesseract
except ImportError:
    pytesseract = None
 
# ============================================
# LOAD API KEYS FROM STREAMLIT SECRETS
# ============================================
# Add these to your .streamlit/secrets.toml:
#
# OPENROUTER_API_KEY = "sk-or-..."
# PINECONE_API_KEY   = "pcsk_..."
#
OPENROUTER_API_KEY = st.secrets.get("OPENROUTER_API_KEY", "")
PINECONE_API_KEY   = st.secrets.get("PINECONE_API_KEY", "")
 
REQUIRED_KEYS = {
    "OPENROUTER_API_KEY": OPENROUTER_API_KEY,
    "PINECONE_API_KEY":   PINECONE_API_KEY,
}
 
# ============================================
# CONFIGURATION
# ============================================
CHUNK_SIZE   = 800
CHUNK_OVERLAP = 200
RETRIEVER_K  = 5
EMBED_MODEL  = "nomic-embed-text"       # Local Ollama embeddings
LLM_MODEL    = "google/gemini-2.0-flash-001"
 
# ============================================
# PINECONE CLOUD CONFIGURATION
# ============================================
PINECONE_INDEX_NAME  = "ncert-science"
PINECONE_ENVIRONMENT = "us-east-1"
 
# ============================================
# PROMPT TEMPLATE
# ============================================
prompt_template = """You are an NCERT Class 10 Science expert tutor. Your goal is to help students understand concepts clearly and accurately.
 
IMPORTANT GUIDELINES:
1. **Provide Complete Answers** - Don't just give one-liners. Explain the concept thoroughly with examples from the context.
2. **Use Context First** - Base your answer on the provided NCERT materials. If context doesn't have information, say so explicitly.
3. **Cite All Sources** - For EVERY important fact, include: [Source: filename | Page X | Type]
4. **Quality Over Strict Rules** - Be helpful to students while maintaining accuracy from the materials.
5. **Handle Different Content Types**:
   - From PDF text: Direct quotes are best
   - From Images/OCR: Acknowledge if OCR quality is uncertain
6. **Check Page Numbers** - The page numbers in source metadata are your reference - use them exactly as provided.
7. **Multi-part Answers** - If question has multiple parts, answer each one with proper citations.
8. **When Unsure** - Say "This information is not clearly available in the provided materials" rather than guessing.
 
ANSWER STYLE: Be educational and clear. Help students learn, not just answer mechanically.
 
Context from NCERT textbook:
{context}
 
Question from student: {question}
 
Your Answer:"""
 
prompt = PromptTemplate(
    template=prompt_template,
    input_variables=["context", "question"]
)
 
# ============================================
# PAGE CONFIGURATION
# ============================================
st.set_page_config(
    page_title="📚 NCERT 10 Science",
    layout="centered",
    initial_sidebar_state="collapsed"
)
 
# Check for missing API keys
missing_keys = [k for k, v in REQUIRED_KEYS.items() if not v]
if missing_keys:
    st.error("❌ **Missing Required API Keys in Streamlit Secrets!**")
    for key in missing_keys:
        st.error(f"- `{key}` is not set. Add it to `.streamlit/secrets.toml`.")
    st.info(
        "**How to fix:** Create `.streamlit/secrets.toml` in your project root:\n\n"
        "```toml\n"
        'OPENROUTER_API_KEY = "sk-or-..."\n'
        'PINECONE_API_KEY   = "pcsk_..."\n'
        "```"
    )
    st.stop()
 
st.title("📚 NCERT Class 10 Science")
st.markdown("*Ask any question about NCERT Class 10 Science*")
st.divider()
 
# ============================================
# LOAD EXISTING PINECONE INDEX
# (No re-upload — connects to the index you
#  already built with this same index name)
# ============================================
@st.cache_resource
def load_vectorstore():
    """
    Connect to the existing Pinecone index.
    Documents must already be uploaded to 'ncert-science'.
    This function never re-uploads — it only reads.
    """
    try:
        pc = Pinecone(api_key=PINECONE_API_KEY)
 
        # Verify the index actually exists
        try:
            indexes_response = pc.list_indexes()
            if hasattr(indexes_response, 'names'):
                index_names = indexes_response.names()
            elif hasattr(indexes_response, 'indexes'):
                index_names = [
                    idx.name if hasattr(idx, 'name') else idx
                    for idx in indexes_response.indexes
                ]
            elif isinstance(indexes_response, list):
                index_names = [
                    idx.name if hasattr(idx, 'name') else idx
                    for idx in indexes_response
                ]
            else:
                index_names = []
        except Exception:
            index_names = []
 
        if PINECONE_INDEX_NAME not in index_names:
            st.error(
                f"❌ Pinecone index **'{PINECONE_INDEX_NAME}'** not found. "
                "Please upload your documents first (run the ingestion script locally)."
            )
            return None
 
        # Connect to the existing index — no document upload
        embed = OllamaEmbeddings(model=EMBED_MODEL)
        db = PineconeVectorStore.from_existing_index(
            index_name=PINECONE_INDEX_NAME,
            embedding=embed,
        )
        return db
 
    except Exception as e:
        st.error(f"❌ Failed to connect to Pinecone: {str(e)[:120]}")
        return None
 
 
def init_rag_chain():
    """Build the RAG chain on top of the existing vector store."""
    db = load_vectorstore()
    if db is None:
        return None
 
    retriever = db.as_retriever(
        search_kwargs={"k": RETRIEVER_K}
    )
 
    llm = ChatOpenAI(
        model=LLM_MODEL,
        openai_api_key=OPENROUTER_API_KEY,
        openai_api_base="https://openrouter.ai/api/v1",
        temperature=0.5,
        max_tokens=2500,
        top_p=0.9,
        default_headers={
            "HTTP-Referer": "https://your-app.streamlit.app",  # update with your app URL
            "X-Title": "NCERT Science Bot",
        }
    )
 
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True
    )
 
    return qa_chain
 
 
# ============================================
# MAIN QUERY INTERFACE
# ============================================
query = st.text_input(
    "Ask your question:",
    placeholder="What is photosynthesis?",
    label_visibility="collapsed"
)
 
if query.strip():
    try:
        qa_chain = init_rag_chain()
 
        if qa_chain is None:
            st.error("Could not initialise the QA chain. Check the errors above.")
        else:
            with st.spinner("Thinking..."):
                response = qa_chain.invoke({"query": query})
            answer = response["result"]
            st.markdown(answer)
 
    except Exception as e:
        st.error(f"Error: {str(e)[:120]}")
 
