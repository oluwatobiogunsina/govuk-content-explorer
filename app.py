import streamlit as st
from utils import fetch_govuk_page, chunk_text, embed_texts
import faiss
import numpy as np

st.set_page_config(page_title="GOV.UK Data Scout", layout="wide")
st.title("🔎 GOV.UK Data Scout")
st.write("Paste one or more GOV.UK URLs below to fetch and chunk content for search.")

urls_input = st.text_area("GOV.UK URLs (one per line):", height=150)
process = st.button("🚀 Process Pages")

if process and urls_input:
    urls = [url.strip() for url in urls_input.splitlines() if url.strip()]
    all_chunks = []
    for url in urls:
        try:
            text = fetch_govuk_page(url)
            chunks = chunk_text(text)
            all_chunks.extend(chunks)
        except Exception as e:
            st.error(f"Error fetching {url}: {e}")

    if all_chunks:
        embeddings = embed_texts(all_chunks)
        index = faiss.IndexFlatL2(len(embeddings[0]))
        index.add(np.array(embeddings).astype("float32"))
        st.session_state["chunks"] = all_chunks
        st.session_state["index"] = index
        st.success("✅ All pages processed! Ask a question below:")
    else:
        st.error("❌ No valid content found.")
        
query = st.text_input("Ask a question:")
if query and "index" in st.session_state:
    q_embed = embed_texts([query])[0]
    D, I = st.session_state["index"].search(np.array([q_embed]).astype("float32"), 5)
    st.subheader("🔍 Most Relevant Chunks")
    for i in I[0]:
        st.markdown(st.session_state["chunks"][i])
        st.markdown("---")
