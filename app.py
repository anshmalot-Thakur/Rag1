#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import streamlit as st
import time

# =========================
# 🔐 LOAD SECRETS
# =========================
OPENROUTER_API_KEY = st.secrets["OPENROUTER_API_KEY"]
PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]
LANGSMITH_API_KEY = st.secrets.get("LANGSMITH_API_KEY", "")

# Set env for libraries
os.environ["OPENAI_API_KEY"] = OPENROUTER_API_KEY
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
if LANGSMITH_API_KEY:
    os.environ["LANGCHAIN_API_KEY"] = LANGSMITH_API_KEY

# =========================
# 📦 IMPORTS
# =========================
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from pinecone import Pinecone, ServerlessSpec

# =========================
# ⚙️ CONFIG
# =========================
INDEX_NAME = "ncert-science"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 200
RETRIEVER_K = 4

# =========================
# 🎯 STREAMLIT UI
# =========================
st.set_page_config(page_title="📚 NCERT Science RAG", layout="centered")
st.title("📚 NCERT Class 10 Science")
st.markdown("Ask questions from your uploaded PDFs")

# =========================
# 📄 FILE UPLOAD
# =========================
uploaded_file = st.file_uploader("Upload your NCERT PDF", type="pdf")

# =========================
# 🔧 EMBEDDINGS
# =========================
embeddings = OpenAIEmbeddings(
    api_key=OPENROUTER_API_KEY,
    base_url="https://openrouter.ai/api/v1"
)

# =========================
# ☁️ PINECONE INIT
# =========================
pc = Pinecone(api_key=PINECONE_API_KEY)

def ensure_index():
    indexes = pc.list_indexes()
    index_names = [i.name for i in indexes]

    if INDEX_NAME not in index_names:
        pc.create_index(
            name=INDEX_NAME,
            dimension=1536,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
        time.sleep(5)

ensure_index()

# =========================
# 📚 PROCESS PDF
# =========================
@st.cache_resource
def process_pdf(file):
    with open("temp.pdf", "wb") as f:
        f.write(file.read())

    loader = PyPDFLoader("temp.pdf")
    pages = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )

    docs = splitter.split_documents(pages)

    vectorstore = PineconeVectorStore.from_documents(
        docs,
        embeddings,
        index_name=INDEX_NAME
    )

    return vectorstore

# =========================
# 🧠 LLM
# =========================
llm = ChatOpenAI(
    model="openai/gpt-4o-mini",
    openai_api_key=OPENROUTER_API_KEY,
    openai_api_base="https://openrouter.ai/api/v1",
    temperature=0.4,
    max_tokens=2000
)

# =========================
# 📝 PROMPT TEMPLATE
# =========================
prompt_template = """You are an NCERT Class 10 Science tutor.

Use the context below to answer clearly.

Context:
{context}

Question:
{question}

Answer:"""

prompt = PromptTemplate(
    template=prompt_template,
    input_variables=["context", "question"]
)

# =========================
# 🔗 QA CHAIN
# =========================
def get_qa_chain(vectorstore):
    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": RETRIEVER_K}),
        chain_type="stuff",
        chain_type_kwargs={"prompt": prompt}
    )

# =========================
# 💬 QUERY INPUT
# =========================
query = st.text_input("Ask your question")

# =========================
# ⚡ RUN
# =========================
if uploaded_file:
    st.success("PDF uploaded successfully!")

    vectorstore = process_pdf(uploaded_file)

    if query:
        with st.spinner("Thinking..."):
            qa_chain = get_qa_chain(vectorstore)
            result = qa_chain.run(query)

        st.markdown("### 📌 Answer")
        st.write(result)

elif query:
    st.warning("Please upload a PDF first")
