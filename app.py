import streamlit as st
from utils import fetch_govuk_page_with_links, chunk_text

st.set_page_config(page_title="GOV.UK Data Scout", layout="wide")

st.title("🔎 GOV.UK Data Scout")
st.markdown("Paste one or more GOV.UK URLs below (one per line). We'll extract and chunk the content including a few internal links.")

# Text area for URLs
urls_input = st.text_area("Enter GOV.UK URLs (one per line):", height=200)
process_button = st.button("🚀 Process Pages")

if process_button and urls_input:
    urls = [u.strip() for u in urls_input.splitlines() if u.strip()]
    all_chunks = []

    with st.spinner("Fetching and chunking content..."):
        for url in urls:
            st.subheader(f"🔗 {url}")
            try:
                # Fetch full content including some internal links
                full_text = fetch_govuk_page_with_links(url, follow_links=True, max_links=3)
                chunks = chunk_text(full_text, max_words=150)

                for i, chunk in enumerate(chunks):
                    st.markdown(f"**Chunk {i+1}:**")
                    st.code(chunk, language='markdown')
                    all_chunks.append(chunk)

            except Exception as e:
                st.error(f"Failed to process {url}: {e}")

    if all_chunks:
        st.success("✅ Done! All chunks generated.")
