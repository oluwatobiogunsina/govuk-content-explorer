import streamlit as st
from utils import fetch_govuk_page, chunk_text, embed_texts, get_top_matches

# Initialize session state variables if not already set
if "chunked_pages" not in st.session_state:
    st.session_state["chunked_pages"] = {}
if "embeddings" not in st.session_state:
    st.session_state["embeddings"] = []
if "all_chunks" not in st.session_state:
    st.session_state["all_chunks"] = []


st.set_page_config(page_title="GOV.UK Data Scout", layout="wide")
st.title("🔍 GOV.UK Data Scout")
st.markdown("""
### 🧠 About This Tool

This tool helps evaluate how GOV.UK content performs in AI-powered retrieval systems, especially those using large language models (LLMs) and semantic search. It:

- **Fetches content** from a GOV.UK page (and selected internal links)  
- **Splits the content into semantic chunks** — similar to how AI models parse and retrieve text  
- **Generates local embeddings** of each chunk for fast, similarity-based search  
- **Allows you to ask a question** and see the top-matching content chunks
""")

with st.expander("💡 What You Can Learn"):
    st.markdown("""
    **1. Content structure**  
    See how your content is chunked. Are the chunks focused and meaningful on their own? This helps assess whether your content works when consumed out of its original context.

    **2. Metadata gaps**  
    Look at what’s included or missing in each chunk. Consider what metadata (e.g. audience, date, type) could help an AI system filter or rank it.

    **3. Retrieval experience**  
    Try asking a question and reviewing the retrieved chunks. Are they precise and trustworthy? If not, what improvements could help?
    """)

# Input and process URLs
urls_input = st.text_area("Paste one or more GOV.UK URLs (one per line):")
follow_links = st.checkbox("Follow internal links from each page?", value=False)

if st.button("🚀 Process pages"):
    urls = [u.strip() for u in urls_input.splitlines() if u.strip()]
    if urls:
        all_chunks = []
        chunked_pages = {}

        with st.spinner("Fetching and chunking content..."):
            for url in urls:
                try:
                    content = fetch_govuk_page(url, follow_links=follow_links)
                    chunks = chunk_text(content)
                    chunked_pages[url] = chunks
                    all_chunks.extend(chunks)
                except Exception as e:
                    st.error(f"Error fetching {url}: {e}")

        if all_chunks:
            embeddings = embed_texts(all_chunks)
            st.session_state["chunks"] = all_chunks
            st.session_state["embeddings"] = embeddings
            st.session_state["chunked_pages"] = chunked_pages
            st.success("✅ All pages processed! Ask a question below.")
        else:
            st.warning("No valid content found.")

# Search interface
if "chunks" in st.session_state and "embeddings" in st.session_state:
    query = st.text_input("Ask a question to test semantic retrieval:")
    if query:
        top_matches = get_top_matches(query, st.session_state["chunks"], st.session_state["embeddings"])
        st.subheader("🔎 Top Matching Chunks")
        for match, score in top_matches:
            st.markdown(f"**Score:** {score:.2f}")
            st.code(match, language="markdown")

    st.markdown("---")
    with st.expander("📄 View chunks by page"):
        for url, chunks in st.session_state["chunked_pages"].items():
            st.markdown(f"### 🔗 {url}")
            for i, chunk in enumerate(chunks):
                with st.expander(f"Chunk {i+1}"):
                    st.code(chunk, language="markdown")

