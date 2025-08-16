import requests
from bs4 import BeautifulSoup
import re
from urllib.parse import urljoin
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# --- Content Fetching and Cleaning ---

def fetch_govuk_page(url, follow_links=False, max_links=3):
    """
    Fetches structured content from a GOV.UK page. Optionally follows internal GOV.UK links.
    """
    def clean_and_format(soup):
        parts = []
        for elem in soup.find_all(['h1', 'h2', 'h3', 'p', 'ul', 'ol', 'li']):
            if elem.name in ['h1', 'h2', 'h3']:
                level = int(elem.name[1])
                parts.append(f"{'#' * level} {elem.get_text(strip=True)}")
            elif elem.name == 'p':
                parts.append(elem.get_text(strip=True))
            elif elem.name in ['ul', 'ol']:
                for li in elem.find_all('li'):
                    parts.append(f"- {li.get_text(strip=True)}")
        return '\n'.join(part for part in parts if part).strip()

    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        main = soup.find('main')
        if not main:
            raise ValueError("No <main> content found")
            # Remove non-content sections that may still be inside <main>
                non_content_selectors = [
                    '.gem-c-contextual-sidebar',
                    '.gem-c-related-navigation',
                    'nav',
                    'aside',
                    'header',
                    'footer'
                ]
                
                for selector in non_content_selectors:
                    for tag in main.select(selector):
                        tag.decompose()

        content = clean_and_format(main)

        if follow_links:
            internal_links = [
                urljoin(url, a['href'])
                for a in main.find_all('a', href=True)
                if a['href'].startswith('/') and 'mailto:' not in a['href']
            ]
            visited = set()
            for link in internal_links[:max_links]:
                if link not in visited:
                    visited.add(link)
                    try:
                        sub_content = fetch_govuk_page(link, follow_links=False)
                        content += f"\n\n---\n\n## Linked Page: {link}\n\n{sub_content}"
                    except Exception as e:
                        print(f"Skipped link {link}: {e}")
        return content

    except Exception as e:
        raise RuntimeError(f"Failed to fetch or parse page: {e}")

# --- Chunking ---

def chunk_text(text, max_tokens=200):
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

# --- Embeddings and Similarity Search ---

model = SentenceTransformer("all-MiniLM-L6-v2")

def embed_texts(texts):
    return model.encode(texts, show_progress_bar=False).tolist()

def get_top_matches(query, texts, embeddings, top_n=3):
    query_embedding = model.encode([query])
    similarities = cosine_similarity(query_embedding, embeddings)[0]
    top_indices = similarities.argsort()[::-1][:top_n]
    results = [(texts[i], similarities[i]) for i in top_indices]
    return results
