import streamlit as st
from utils import fetch_govuk_page
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# Load the sentence transformer model
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
st.title("🔎 GOV.UK Data Scout")

url = st.text_input("Enter a GOV.UK page URL to fetch content:")

if url:
    try:
        with st.spinner("Fetching and processing content..."):
            text = fetch_govuk_page(url, follow_links=False)
            chunks = chunk_text(text)
            embeddings = embed_texts(chunks)
            index = faiss.IndexFlatL2(len(embeddings[0]))
            index.add(np.array(embeddings).astype("float32"))

            st.session_state["chunks"] = chunks
            st.session_state["index"] = index

        st.success("✅ Page processed successfully! Ask a question below:")

        query = st.text_input("Ask a question about the page content:")
        if query and "index" in st.session_state:
            query_embedding = embed_texts([query])[0]
            D, I = st.session_state["index"].search(np.array([query_embedding]).astype("float32"), 5)

            st.subheader("🔎 Top Matching Chunks")
            for i in I[0]:
                st.markdown(st.session_state["chunks"][i])
                st.markdown("---")

    except Exception as e:
        st.error(f"Error: {e}")
