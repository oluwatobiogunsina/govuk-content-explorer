import streamlit as st
from utils import fetch_govuk_page, chunk_text, embed_texts, get_top_matches

# --- Page Setup ---
st.set_page_config(page_title="GOV.UK Data Scout", layout="wide")

st.title("🔎 GOV.UK Data Scout")

with st.expander("ℹ️ What this tool does"):
    st.markdown("""
    This tool lets you explore how content from GOV.UK is structured and retrieved using AI techniques.

    **How it works:**
    - You enter one or more GOV.UK URLs
    - The tool fetches the content (and optionally a few internal links)
    - It chunks the content into smaller blocks for semantic search
    - It generates vector embeddings
    - You can ask a question and retrieve the most relevant chunks

    **Why this matters:**
    - Helps assess how AI-friendly GOV.UK content is
    - Shows how clarity and metadata affect retrieval
    - Simulates what LLMs do during semantic search
    """)

# --- User Inputs ---
urls_input = st.text_area("Enter one or more GOV.UK URLs (one per line):", height=200)
follow_links = st.checkbox("Follow internal GOV.UK links?", value=True)

question = st.text_input("Ask a question (optional):")

# --- Session State for content ---
if "all_chunks" not in st.session_state:
    st.session_state.all_chunks = []
    st.session_state.all_sources = []
    st.session_state.chunk_ids = []
    st.session_state.embeddings = []

# --- Process Pages ---
if st.button("🚀 Process Pages"):
    if not urls_input.strip():
        st.warning("Please enter at least one GOV.UK URL.")
    else:
        urls = [u.strip() for u in urls_input.strip().split('\n') if u.strip()]
        all_chunks = []
        all_sources = []
        chunk_ids = []

        with st.spinner("Fetching and processing..."):
            for url in urls:
                try:
                    raw = fetch_govuk_page(url, follow_links=follow_links)
                    chunks = chunk_text(raw, max_tokens=300)
                    all_chunks.extend(chunks)
                    all_sources.extend([url] * len(chunks))
                    chunk_ids.extend([f"{url}--{i}" for i in range(len(chunks))])

                    st.markdown(f"✅ Processed: {url} ({len(chunks)} chunks)")
                except Exception as e:
                    st.error(f"Error with {url}: {e}")

        if all_chunks:
            embeddings = embed_texts(all_chunks)

            st.session_state.all_chunks = all_chunks
            st.session_state.all_sources = all_sources
            st.session_state.chunk_ids = chunk_ids
            st.session_state.embeddings = embeddings

            st.success("✅ All content chunked and embedded.")

# --- Semantic Search ---
if question and st.session_state.embeddings:
    st.markdown("### 🔍 Top Matching Chunks")
    top_matches = get_top_matches(question, st.session_state.all_chunks, st.session_state.embeddings, top_n=3)
    for match_text, score in top_matches:
        match_index = st.session_state.all_chunks.index(match_text)
        chunk_id = st.session_state.chunk_ids[match_index]
        st.markdown(f"**Score: {score:.2f}** — [Jump to chunk](#{chunk_id})")
        st.markdown(f"> {match_text[:300]}...")

# --- Display All Chunks with Anchors ---
st.markdown("### 📚 Full List of Chunks")
for i, (chunk, url, chunk_id) in enumerate(zip(st.session_state.all_chunks, st.session_state.all_sources, st.session_state.chunk_ids)):
    st.markdown(f'<div id="{chunk_id}"></div>', unsafe_allow_html=True)
    st.markdown(f"**Chunk {i+1} from [{url}]({url})**")
    st.markdown(chunk)
    st.markdown("---")
