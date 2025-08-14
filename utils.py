import requests
from bs4 import BeautifulSoup
import re
from urllib.parse import urljoin
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')

def fetch_govuk_page(url, follow_links=False, max_links=3):
    def clean_and_format(soup):
        parts = []
        for elem in soup.find_all(['h1', 'h2', 'h3', 'h4', 'p', 'ul', 'ol', 'li', 'a']):
            if elem.name in ['h1', 'h2', 'h3']:
                level = int(elem.name[1])
                parts.append(f"{'#' * level} {elem.get_text(strip=True)}")
            elif elem.name == 'p':
                parts.append(elem.get_text(strip=True))
            elif elem.name in ['ul', 'ol']:
                for li in elem.find_all('li'):
                    parts.append(f"- {li.get_text(strip=True)}")
            elif elem.name == 'a' and elem.get('href'):
                href = elem['href']
                full_url = urljoin(url, href)
                link_text = elem.get_text(strip=True)
                parts.append(f"[{link_text}]({full_url})")
        return '\n'.join(part for part in parts if part).strip()

    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        main = soup.find('main')
        if not main:
            raise ValueError("No <main> content found")
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

def chunk_text(text, max_words=400):
    lines = text.split('\n')
    chunks = []
    current_chunk = []
    current_length = 0
    for line in lines:
        line = line.strip()
        if not line:
            continue
        line_words = line.split()
        line_length = len(line_words)
        if current_length + line_length > max_words:
            chunks.append('\n'.join(current_chunk))
            current_chunk = [line]
            current_length = line_length
        else:
            current_chunk.append(line)
            current_length += line_length
    if current_chunk:
        chunks.append('\n'.join(current_chunk))
    return chunks

def embed_texts(texts):
    return model.encode(texts, show_progress_bar=False).tolist()
