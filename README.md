# 🔍 GOV.UK Content Explorer

A tool for analysing GOV.UK pages to understand how they perform in AI-powered retrieval systems — especially those powered by large language models (LLMs) and semantic search.

This app helps content designers, publishers, and product teams explore whether GOV.UK content is structured and annotated in ways that make it effective for use in tools like GOV.UK Chat, AI copilots, and retrieval-augmented generation (RAG) systems.

---

## 🚀 What It Does

- ✅ Fetches the content of one or more GOV.UK pages (with optional link following)
- ✂️ Splits content into **semantic-style chunks**, as LLM tools would
- 🧠 Generates **vector embeddings** locally (no OpenAI key needed)
- ❓ Lets you ask a question and view the **most relevant chunks**
- 📄 Displays full chunk lists by page to help audit structure and clarity

---

## 💡 What You Can Learn

### 1. Content Structure
See how your content is chunked. Are the chunks understandable and useful in isolation? This is essential when content is consumed out of its original web context.

### 2. AI-Friendliness
Try asking a question. If the top chunks answer well, your content is likely effective in semantic search systems like GOV.UK Chat.

### 3. Metadata & Clarity Gaps
Poor chunk performance might reveal:
- Weak or missing headings
- Repetition or lack of clarity
- Missing metadata (e.g. time sensitivity or intended audience)

These are clues for improving GOV.UK content for use in future AI services.

## 🛠️ Tools used
| Tool                                                                | Purpose                                                           |
| ------------------------------------------------------------------- | ----------------------------------------------------------------- |
| **[Streamlit](https://streamlit.io/)**                              | Web app framework for building the interactive interface          |
| **[BeautifulSoup](https://www.crummy.com/software/BeautifulSoup/)** | HTML parsing and content extraction from GOV.UK pages             |
| **[Requests](https://docs.python-requests.org/)**                   | Fetching content from GOV.UK URLs                                 |
| **[SentenceTransformers](https://www.sbert.net/)**                  | Generating semantic embeddings using the `all-MiniLM-L6-v2` model |
| **[scikit-learn](https://scikit-learn.org/)**                       | Calculating cosine similarity for semantic search                 |
| **[Markdown](https://www.markdownguide.org/)**                      | Rendering structured content and chunks in readable format        |
| **Python 3.x**                                                      | Programming language used across all components                   |

## 🛠️ Installation

To run locally:

```bash
git clone https://github.com/YOUR_USERNAME/govuk-content-explorer.git
cd govuk-content-explorer
pip install -r requirements.txt
streamlit run app.py
