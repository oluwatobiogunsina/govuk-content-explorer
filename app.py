import streamlit as st
from utils import fetch_govuk_page
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# Load embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

def embed_texts(texts):
    return model.encode(texts, show_progress_bar=False).tolist()

def chunk_text(text, max_tokens=400):
    paragraphs = text.split('\n')
    chunks = []
    current_chunk = []
    current_length = 0
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        length = len(para.split())
        if current_length + length > max_tokens:
            chunks.append('\n'.join(current_chunk))
            current_chunk = [para]
            current_length = length
        else:
            current_chunk.append(para)
            current_length += length
    if current_chunk:
        chunks.append('\n'.join(current_chunk))
    return chunks

# --- Streamlit UI ---
st.title("🔎 GOV.UK Multi-Page Data Scout")

urls_input = st.text_area("Enter one or more GOV.UK page URLs (one per line):")
urls = [u.strip() for u in urls_input.splitlines() if u.strip()]

if urls and st.button("🚀
