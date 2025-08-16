import streamlit as st
from utils import process_pages_with_chunks, get_top_matches

st.set_page_config(page_title="GOV.UK Data Scout", layout="wide")

st.title("🕵️ GOV.UK Data Scout")

with st.expander("ℹ️ What this app does", expanded=True):
    st.markdown("""
This tool fetches and processes content from GOV.UK pages, breaking it down into manageable **semantic chunks** using a local embedding model. These chunks are stored as **vector embeddings** for semantic search.

It allows you to:
- ✅ Understand how structured and AI-friendly GOV.UK content is
- 🔎 Examine how clearly the page communicates without web layout
- 🧠 Experiment with how well the content performs in **LLM-powered retrieval tools**
""")

with st.expander("💡 What you can learn from this", expanded=False):
    st.markdown("""
- How well the content chunks stand on their own without web layout
- Whether metadata and structure help LLMs determine relevance
- Where improvements like clearer headings, dates, or metadata could improve retrieval
- Whether long pages should be broken down into shorter, more focused units
""")

if "urls" not in st.session_state:
    st.session_state.urls = []

url_input = st.text_input("Enter a GOV.UK URL and press Add:", "")
if st.button("➕ Add URL") and url_input:
    st.session_state.urls.append(url_input)

if st.session_state.urls:
    st.markdown("### ✅ URLs to process:")
    for url in st.session_state.urls:
        st.write(f"- {url}")

if st.session_state.urls and st.button("🚀 Process GOV.UK Pages"):
    with st.spinner("Fetching and chunking content..."):
        try:
            chunked_pages = process_pages_with_chunks(st.session_state.urls)
            st.session_state.chunked_pages = chunked_pages
            st.success("✅ All pages processed!")
        except Exception as e:
            st.error(f"❌ Error processing pages: {e}")
            st.session_state.chunked_pages = {}

# Show chunks
if "chunked_pages" in st.session_state:
    st.markdown("## 📄 Chunked Content by Page")
    for url, chunks in st.session_state.chunked_pages.items():
        st.markdown(f"### 🔗 {url}")
        for i, chunk in enumerate(chunks):
            with st.expander(f"Chunk {i+1}", expanded=False):
                st.write(chunk)

# Ask a question
st.markdown("---")
st.markdown("## 🤖 Ask a Question")
query = st.text_input("Enter a question about the content")

if query and "chunked_pages" in st.session_state:
    all_chunks = []
    for page_chunks in st.session_state.chunked_pages.values():
        all_chunks.extend(page_chunks)

    if all_chunks:
        with st.spinner("🔍 Searching..."):
            top_chunks = get_top_matches(query, all_chunks)
            st.markdown("### 🔎 Top Relevant Chunks")
            for i, chunk in enumerate(top_chunks, 1):
                with st.expander(f"Match {i}"):
                    st.write(chunk)
