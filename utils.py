import requests
from bs4 import BeautifulSoup
import re

def fetch_govuk_page(url):
    """
    Fetches and returns cleaned text from a GOV.UK page.
    """
    response = requests.get(url)
    response.raise_for_status()  # Raise an error for bad responses

    soup = BeautifulSoup(response.text, 'html.parser')

    # Remove script and style elements
    for script_or_style in soup(['script', 'style']):
        script_or_style.decompose()

    # GOV.UK main content is usually in 'main' tag
    main_content = soup.find('main')
    if not main_content:
        raise ValueError("No <main> content found on the page.")

    # Get visible text and clean whitespace
    text = main_content.get_text(separator='\n')
    text = re.sub(r'\n+', '\n', text)  # collapse multiple newlines
    text = text.strip()

    return text

def chunk_text(text, max_tokens=400):
    """
    Splits the text into chunks of ~max_tokens size (roughly).
    """
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

    # Catch any leftovers
    if current_chunk:
        chunks.append('\n'.join(current_chunk))

    return chunks

import os
from dotenv import load_dotenv
import openai

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

def embed_texts(texts):
    """
    Takes a list of texts and returns their embeddings using OpenAI.
    """
    response = openai.embeddings.create(
        model="text-embedding-3-small",
        input=texts
    )
    embeddings = [e.embedding for e in response.data]
    return embeddings

from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')

def embed_texts(texts):
    """
    Takes a list of texts and returns their local sentence embeddings.
    """
    return model.encode(texts, show_progress_bar=False).tolist()
