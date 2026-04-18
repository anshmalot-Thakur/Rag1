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
import shutil

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
from groq import Groq
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from openai import OpenAI
from langchain_openai import ChatOpenAI
from pinecone import Pinecone, ServerlessSpec
try:
    import pytesseract
except ImportError:
    pytesseract = None

load_dotenv()

# Validate required API keys at startup
REQUIRED_ENV_KEYS = {
    "OPENROUTER_API_KEY": "OpenRouter LLM API",
    "PINECONE_API_KEY": "Pinecone Vector DB API"
}

missing_keys = [k for k in REQUIRED_ENV_KEYS.keys() if not os.getenv(k)]

# Auto-clear cache if database missing (helps with rebuilds)
if not os.path.exists(r"C:\Users\DELL LAPTOP\Downloads\AI_Bot\science"):
    st.cache_resource.clear()

client = OpenAI(
  base_url="https://openrouter.io/api/v1",
  api_key=os.getenv("OPENROUTER_API_KEY"),
)

# ============================================
# TEST EMBEDDING FUNCTION (DEBUG)
# ============================================
def test_embeddings():
    """Test if Ollama embeddings are working correctly."""
    try:
        embed = OllamaEmbeddings(model=EMBED_MODEL)
        # Test embedding a simple query
        test_vec = embed.embed_query("gene mutation heredity")
        return {
            "status": "✅ Embedding Working",
            "vector_size": len(test_vec),
            "sample": test_vec[:5]
        }
    except Exception as e:
        return {
            "status": "❌ Embedding Failed",
            "error": str(e),
            "solution": "Make sure Ollama is running: ollama serve"
        }


# ============================================
# CONFIGURATION
# ============================================
PDF_FOLDER = r"C:\Users\DELL LAPTOP\Downloads\AI_Bot\science"
CHUNK_SIZE = 800  # Smaller chunks for better relevance
CHUNK_OVERLAP = 200  # More overlap for better context
RETRIEVER_K = 5  # Get more diverse results (increased from 3)
EMBED_MODEL = "nomic-embed-text"  # Local embeddings
LLM_MODEL = "google/gemini-2.0-flash-001"  # 

# ============================================
# PINECONE CLOUD CONFIGURATION
# ============================================
PINECONE_API_KEY = os.secrets("PINECONE_API_KEY")
PINECONE_INDEX_NAME = "ncert-science"  # Your index name
PINECONE_ENVIRONMENT = "us-east-1"  # Default region 

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
if missing_keys:
    st.error("❌ **Missing Required API Keys!**")
    for key in missing_keys:
        st.error(f"- {key}: {REQUIRED_ENV_KEYS[key]}")
    st.stop()

st.title("📚 NCERT Class 10 Science")
st.markdown("*Ask any question about NCERT Class 10 Science*")
st.divider()

# ============================================
# MULTI-MODAL DATA PROCESSING (TEXT + IMAGE OCR)
# ============================================

def extract_images_ocr(pdf_path, file_name):
    """Extract text from images in PDF using OCR with proper metadata and UTF-8 encoding."""
    if pytesseract is None:
        return []
    
    try:
        from pdf2image import convert_from_path
        images_text = []
        
        pages = convert_from_path(pdf_path)
        for page_num, page in enumerate(pages):
            text = pytesseract.image_to_string(page)
            if text.strip():
                # Clean UTF-8 encoding
                text = text.encode('utf-8', errors='replace').decode('utf-8')
            
                # IMPORTANT: Store actual page number (0-indexed page_num = page in file)
                images_text.append({
                    'text': f"[DIAGRAM/IMAGE from {file_name}]\n{text}",
                    'page': page_num,
                    'source': file_name,
                    'type': 'image_ocr'
                })
        
        return images_text
    except Exception as e:
        print(f"OCR failed for {pdf_path}: {str(e)}")
        return []

@st.cache_resource
def load_data():
    """Load PDFs and upload to Pinecone vector cloud database - silent mode."""
    
    if not PINECONE_API_KEY:
        return None
    
    all_docs = []
    
    if not os.path.exists(PDF_FOLDER):
        return None
    
    pdf_files = [f for f in os.listdir(PDF_FOLDER) if f.endswith(".pdf")]
    if not pdf_files:
        return None
    
    skipped_pdfs = []
    
    for idx, file in enumerate(pdf_files):
        pdf_path = os.path.join(PDF_FOLDER, file)
        
        try:
            # Extract TEXT from PDF with proper metadata
            loader = PyPDFLoader(pdf_path)
            pages = loader.load()
            
            for page in pages:
                if page.page_content.strip():
                    # Clean UTF-8 encoding
                    page.page_content = page.page_content.encode('utf-8', errors='replace').decode('utf-8')
                    
                    # Ensure metadata has proper source info
                    page.metadata['source'] = file
                    page.metadata['source_type'] = 'text'
                    # Ensure page is integer for proper indexing
                    if 'page' not in page.metadata or not isinstance(page.metadata['page'], int):
                        page.metadata['page'] = 0
                    all_docs.append(page)
            
            # Extract TEXT from IMAGES using OCR with proper file tracking
            image_texts = extract_images_ocr(pdf_path, file)
            for img_data in image_texts:
                # Clean UTF-8 encoding for OCR content
                ocr_text = img_data['text'].encode('utf-8', errors='replace').decode('utf-8')
                
                doc = Document(
                    page_content=ocr_text,
                    metadata={
                        'source': file,
                        'page': int(img_data['page']),  # Ensure page is integer
                        'source_type': 'image_ocr'
                    }
                )
                all_docs.append(doc)
            
        except Exception as e:
            error_msg = str(e)
            skipped_pdfs.append((file, error_msg[:60]))
    
    if not all_docs:
        return None
    
    # Split into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    documents = text_splitter.split_documents(all_docs)
    
    try:
        # Initialize Pinecone
        pc = Pinecone(api_key=PINECONE_API_KEY)
        
        # Initialize embeddings
        embed = OllamaEmbeddings(model=EMBED_MODEL)
        
        # Clean content for UTF-8 compatibility
        for doc in documents:
            doc.page_content = doc.page_content.encode('utf-8', errors='replace').decode('utf-8')
        
        # Check if index exists, create if not
        try:
            # List existing indexes with better error handling
            indexes_response = pc.list_indexes()
            
            if hasattr(indexes_response, 'names'):
                index_names = indexes_response.names()
            elif hasattr(indexes_response, 'indexes'):
                index_names = [idx.name if hasattr(idx, 'name') else idx for idx in indexes_response.indexes]
            elif isinstance(indexes_response, list):
                index_names = [idx.name if hasattr(idx, 'name') else idx for idx in indexes_response]
            else:
                index_names = []
                
        except Exception as e:
            index_names = []
        
        # Create index if not exists
        if PINECONE_INDEX_NAME not in index_names:
            try:
                pc.create_index(
                    name=PINECONE_INDEX_NAME,
                    dimension=768,  # nomic-embed-text dimension
                    metric="cosine",
                    spec=ServerlessSpec(cloud="aws", region="us-east-1")
                )
                time.sleep(5)  # Wait for index to initialize
            except Exception as e:
                if "already exists" not in str(e):
                    pass
        
        # Upload documents to Pinecone
        try:
            db = PineconeVectorStore.from_documents(
                documents,
                embed,
                index_name=PINECONE_INDEX_NAME
            )
            return db
        except Exception as upload_err:
            return None
                
    except Exception as e:
        return None

def init_rag_chain():
    """Initialize RAG chain - NOT cached to allow different retrieval results per query."""
    db = load_data()
    if db is None:
        return None
    
    # Use higher k for multi-modal data (text + images) - gets more diverse results
    retriever = db.as_retriever(
        search_kwargs={"k": RETRIEVER_K}
    )
    
    # Use ChatOpenAI (LangChain's version) instead of the raw OpenAI client
    llm = ChatOpenAI(
        model=LLM_MODEL,
        openai_api_key=st.secrets("OPENROUTER_API_KEY"),
        openai_api_base="https://openrouter.ai/api/v1",
        temperature=0.5,  # Balanced: factual but natural explanations
        max_tokens=2500,  # Allow longer, detailed answers with proper citations
        top_p=0.9,  # Better response diversity
        default_headers={
            "HTTP-Referer": "http://localhost:8501",
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

# Question input
query = st.text_input(
    "Ask your question:",
    placeholder="What is photosynthesis?",
    label_visibility="collapsed"
)

# Process query
if query.strip():
    try:
        qa_chain = init_rag_chain()
        
        if qa_chain is None:
            st.error("Error loading database")
        else:
            response = qa_chain.invoke({"query": query})
            answer = response["result"]
            
            # Display answer only
            st.markdown(answer)
    
    except Exception as e:
        st.error(f"Error: {str(e)[:50]}")
