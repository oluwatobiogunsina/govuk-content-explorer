import requests
from bs4 import BeautifulSoup
import re
from sentence_transformers import SentenceTransformer

# Fetch and clean GOV.UK page content
def fetch_govuk_page(url):
    response = requests.get(url)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, 'html.parser')

    # Remove unwanted elements
    for script_or_style in soup(['script', 'style']):
        script_or_style.decompose()

    # Extract main content
    main_content = soup.find('main')
    if not main_content:
        raise ValueError("No <main> content found on the page.")

    text = main_content.get_text(separator='\n')
    text = re.sub(r'\n+', '\n', text).strip()

    return text

# Chunk text into ~400-token blocks
def chunk_text(text, max_tokens=400):
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

# Load local model and create embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')

def embed_texts(texts):
    return model.encode(texts, show_progress_bar=False).tolist()
