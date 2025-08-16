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
    - The tool fetches the main content (and optionally a few internal links)
    - It chunks the content into smaller blocks for semantic search
    - It generates vector embeddings to simulate how an AI model understands the text
    - You can ask a question and retrieve the most relevant chunks

    **Why this matters:**
    - Helps assess how AI-friendly GOV.UK content is
    - Lets you evaluate clarity, metadata use, and structure
    - Demonstrates how improvements to structure and tagging might support AI tools like chat assistants
    """)

# --- User Inputs ---
urls_input = st.text_area("Enter one or more GOV.UK URLs (one per line):", height=200)
follow_links = st.checkbox("Follow internal GOV.UK links on each page?", value=True)

question = st.text_input("Ask a question about the content (optional):")

# --- Process Pages ---
if st.button("🚀 Process Pages"):
    if not urls_input.strip():
        st.warning("Please enter at least one GOV.UK URL.")
    else:
        urls = [u.strip() for u in urls_input.strip().split('\n') if u.strip()]
        all_chunks = []
        all_sources = []

        with st.spinner("Fetching and processing content..."):
            for url in urls:
                try:
                    raw_content = fetch_govuk_page(url, follow_links=follow_links)
                    chunks = chunk_text(raw_content, max_tokens=300)
                    all_chunks.extend(chunks)
                    all_sources.extend([url] * len(chunks))

                    st.markdown(f"### ✅ Fetched and chunked: {url}")
                    for i, chunk in enumerate(chunks):
                        with st.expander(f"Chunk {i+1}"):
                            st.markdown(chunk)
                except Exception as e:
                    st.error(f"Error processing {url}: {e}")

        if not all_chunks:
            st.warning("No content was successfully processed.")
        else:
            embeddings = embed_texts(all_chunks)
            st.success("✅ All content processed and embedded.")

            if question:
                st.markdown("### 🔍 Top Matching Chunks")
                top_matches = get_top_matches(question, all_chunks, embeddings, top_n=3)
                for i, (chunk, score) in enumerate(top_matches, 1):
                    st.markdown(f"**Match {i} (Score: {score:.2f})**")
                    st.markdown(chunk)
