import streamlit as st
from utils import fetch_govuk_page, chunk_text, embed_texts
import numpy as np
import faiss

st.set_page_config(page_title="GOV.UK Data Scout")
st.title("🔍 GOV.UK Data Scout")

st.markdown("Paste one or more GOV.UK URLs to extract and semantically search structured content.")

urls = st.text_area("Enter GOV.UK URLs (one per line):", height=150)

if st.button("Process Pages"):
    url_list = [url.strip() for url in urls.splitlines() if url.strip()]
    all_chunks = []

    for url in url_list:
        try:
            text = fetch_govuk_page(url)
            chunks = chunk_text(text)
            all_chunks.extend(chunks)
            st.success(f"✅ {url} processed ({len(chunks)} chunks)")
        except Exception as e:
            st.error(f"❌ Error processing {url}: {e}")

    if all_chunks:
        embeddings = embed_texts(all_chunks)
        index = faiss.IndexFlatL2(len(embeddings[0]))
        index.add(np.array(embeddings).astype("float32"))

        st.session_state["chunks"] = all_chunks
        st.session_state["index"] = index
        st.success("🚀 Pages embedded. You can now ask a question below.")

# Semantic search
if "index" in st.session_state and "chunks" in st.session_state:
    query = st.text_input("🔍 Ask a question about the content:")

    if query:
        query_embedding = embed_texts([query])[0]
        D, I = st.session_state["index"].search(np.array([query_embedding]).astype("float32"), 3)

        st.subheader("🔎 Most Relevant Chunks")
        for i in I[0]:
            chunk = st.session_state["chunks"][i]
            st.markdown(f"**Chunk #{i+1}:**\n\n{chunk}")

