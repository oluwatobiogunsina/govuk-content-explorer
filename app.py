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

# Input box for multiple URLs
st.markdown("### 📥 Paste GOV.UK URLs (one per line):")
url_input = st.text_area("Enter one or more GOV.UK URLs", placeholder="https://www.gov.uk/example-page\nhttps://www.gov.uk/another-page")

# Process button
if st.button("🚀 Process Pages"):
    urls = [u.strip() for u in url_input.splitlines() if u.strip()]
    if urls:
        with st.spinner("Fetching and chunking content..."):
            try:
                chunked_pages = process_pages_with_chunks(urls)
                st.session_state.chunked_pages = chunked_pages
                st.success("✅ All pages processed!")
            except Exception as e:
                st.error(f"❌ Error processing pages: {e}")
                st.session_state.chunked_pages = {}
    else:
        st.warning("Please enter at least one valid GOV.UK URL.")

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
