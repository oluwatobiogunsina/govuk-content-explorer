import requests
from bs4 import BeautifulSoup
import re
from urllib.parse import urljoin
from typing import List

def fetch_page_text(url):
    """
    Fetch visible text from <main> content of a GOV.UK page.
    Removes scripts, styles, headers, links etc.
    """
    response = requests.get(url)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, 'html.parser')

    main = soup.find('main')
    if not main:
        raise ValueError(f"No <main> tag found in {url}")

    # Remove scripts and styles
    for tag in main(['script', 'style']):
        tag.decompose()

    # Remove link text and headers (optional, can be kept if needed)
    for tag in main(['a', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6']):
        tag.decompose()

    # Get plain text
    text = main.get_text(separator="\n")
    text = re.sub(r'\n+', '\n', text).strip()
    return text


def fetch_govuk_page_with_links(url, follow_links=True, max_links=3):
    """
    Fetch text content from a GOV.UK page + selected internal links.
    Returns a single combined string of text.
    """
    combined_text = fetch_page_text(url)

    if follow_links:
        try:
            response = requests.get(url)
            soup = BeautifulSoup(response.text, 'html.parser')
            main = soup.find('main')
            links = [
                urljoin(url, a['href'])
                for a in main.find_all('a', href=True)
                if a['href'].startswith('/') and 'mailto:' not in a['href']
            ]
            visited = set()
            for link in links[:max_links]:
                if link not in visited:
                    visited.add(link)
                    try:
                        combined_text += "\n\n" + fetch_page_text(link)
                    except Exception as e:
                        print(f"Skipped linked page: {link} — {e}")
        except Exception as e:
            print(f"Failed to re-parse main for links — {e}")

    return combined_text


def chunk_text(text: str, max_words=150) -> List[str]:
    """
    Splits a long string into smaller chunks based on word count.
    Returns a list of chunked strings.
    """
    words = text.split()
    chunks = []
    for i in range(0, len(words), max_words):
        chunk = ' '.join(words[i:i + max_words])
        chunks.append(chunk.strip())
    return chunks
