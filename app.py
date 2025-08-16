import streamlit as st
from utils import fetch_govuk_page, chunk_text, embed_texts, get_top_matches

st.set_page_config(page_title="GOV.UK Data Scout", layout="wide")
st.title("🔍 GOV.UK Data Scout")
st.markdown("""
### 🧠 About This Tool

This tool helps evaluate how GOV.UK content performs in AI-powered retrieval systems, especially those using large language models (LLMs) and semantic search. It:

- **Fetches content** from a GOV.UK page (and selected internal links)  
- **Splits the content into semantic chunks** — similar to how AI models parse and retrieve text  
- **Generates local embeddings** of each chunk for fast, similarity-based search  
- **Allows you to ask a question** and see the top-matching content chunks

---

### 💡 What You Can Learn

**1. Content structure**  
See how your content is chunked. Are the chunks focused and meaningful on their own? This helps assess whether your content works when consumed out of its original context — critical for AI tools.

**2. AI-friendliness**  
Try asking a question. Do the retrieved chunks answer it well? This shows how well your content performs in a retrieval-augmented generation (RAG) system, where relevant information is passed to an LLM to answer user queries.

**3. Metadata and clarity gaps**  
You can infer where metadata might help improve retrieval accuracy. If irrelevant chunks are shown, it may be due to a lack of:
- Clear headings or structure
- Time- or audience-specific labelling
- Consistent terminology

These insights help you identify opportunities to improve the **AI readiness** of your content.
""")


# Session state for storing page content and embeddings
if "chunks" not in st.session_state:
    st.session_state.chunks = []
if "embeddings" not in st.session_state:
    st.session_state.embeddings = []

# Input
urls_input = st.text_area("Enter GOV.UK page URLs (one per line):", height=150)
follow_links = st.checkbox("Follow internal links on each page", value=False)

# Process Pages
if urls_input and st.button("🚀 Process Pages"):
    urls = [url.strip() for url in urls_input.splitlines() if url.strip()]
    all_chunks = []

    with st.spinner("Fetching and processing content..."):
        for url in urls:
            try:
                content = fetch_govuk_page(url, follow_links=follow_links)
                chunks = chunk_text(content, max_tokens=250)  # Limit chunk size
                all_chunks.extend(chunks)
                st.success(f"✅ Processed {url} ({len(chunks)} chunks)")
            except Exception as e:
                st.error(f"❌ Failed to process {url}: {e}")

    # Store in session state
    st.session_state.chunks = all_chunks
    st.session_state.embeddings = embed_texts(all_chunks)
    st.success("✅ All pages processed and embeddings generated!")

# Search / Ask a question
if st.session_state.chunks and st.session_state.embeddings:
    st.markdown("---")
    st.subheader("🤖 Ask a question about the content")
    question = st.text_input("Your question:")

    if question:
        results = get_top_matches(
            question,
            st.session_state.chunks,
            st.session_state.embeddings,
            top_n=3
        )
        st.markdown("### 🧠 Top Relevant Chunks")
        for i, (chunk, score) in enumerate(results, 1):
            st.markdown(f"**{i}.** (Relevance: {score:.2f})")
            st.code(chunk, language="markdown")


