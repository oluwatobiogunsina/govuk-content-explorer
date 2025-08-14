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

# --- Streamlit App Starts Here ---
st.title("🔎 GOV.UK Multi-Page Data Scout")

urls_input = st.text_area("Enter one or more GOV.UK page URLs (one per line):")
urls = [u.strip() for u in urls_input.splitlines() if u.strip()]

if urls and st.button("🚀 Process Pages"):
    try:
        with st.spinner("Fetching and processing content..."):
            all_chunks = []
            for url in urls:
                st.write(f"📄 Processing: {url}")
                text = fetch_govuk_page(url, follow_links=False)
                chunks = chunk_text(text)
                all_chunks.extend(chunks)

            embeddings = embed_texts(all_chunks)
            index = faiss.IndexFlatL2(len(embeddings[0]))
            index.add(np.array(embeddings).astype("float32"))

            st.session_state["chunks"] = all_chunks
            st.session_state["index"] = index

        st.success("✅ All pages processed! Ask a question below:")

    except Exception as e:
        st.error(f"🚨 Error: {e}")

# Only show Q&A if data has been processed
if "index" in st.session_state and "chunks" in st.session_state:
    query = st.text_input("Ask a question about the content:")
    if query:
        query_embedding = embed_texts([query])[0]
        D, I = st.session_state["index"].search(np.array([query_embedding]).astype("float32"), 5)

        st.subheader("🔍 Top Matching Content")
        for i in I[0]:
            st.markdown(st.session_state["chunks"][i])
            st.markdown("---")
