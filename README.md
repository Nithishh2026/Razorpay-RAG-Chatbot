# Razorpay Docs Assistant — RAG Chatbot

A retrieval-augmented generation (RAG) chatbot that answers questions about
Razorpay's payment products using **only** Razorpay's public documentation
as its knowledge source — built to explore how RAG can reduce support
ticket load and prevent LLM hallucination in a domain-specific setting.

> This is an independent educational project and is not affiliated with,
> endorsed by, or built in partnership with Razorpay.

## Live Demo

_Add your deployed Streamlit link here once deployed._

## Architecture

```
User question
     │
     ▼
Embed question (sentence-transformers, local)
     │
     ▼
Similarity search in Chroma vector DB  ──►  Top-k relevant doc chunks
     │
     ▼
Chunks + question → Groq LLM (Llama 3.1) with grounding instructions
     │
     ▼
Answer + cited source pages shown in Streamlit UI
```

## Tech Stack

| Component      | Tool                                      |
|-----------------|--------------------------------------------|
| Data source     | Razorpay public docs (scraped)             |
| Chunking        | LangChain `RecursiveCharacterTextSplitter` |
| Embeddings      | `sentence-transformers/all-MiniLM-L6-v2`   |
| Vector database | Chroma (local, persistent)                 |
| LLM             | Groq (Llama 3.1 8B Instant)                |
| UI              | Streamlit                                  |

## Setup

1. **Clone and install dependencies**
   ```bash
   git clone <your-repo-url>
   cd razorpay-rag-bot
   pip install -r requirements.txt
   ```

2. **Get a free Groq API key**
   Sign up at [console.groq.com](https://console.groq.com/keys), then:
   ```bash
   cp .env.example .env
   # paste your key into .env
   ```

3. **Scrape the docs**
   ```bash
   python scraper.py
   ```
   This saves ~50-60 pages of Razorpay docs to `data/raw/`.

4. **Build the vector database**
   ```bash
   python ingest.py
   ```
   This chunks the docs, generates embeddings, and stores them in `chroma_db/`.

5. **Run the app**
   ```bash
   streamlit run app.py
   ```

## Evaluation

_Fill this in once you've tested it — this section matters a lot for
your resume/portfolio credibility. Example format:_

| # | Question | Correct? | Notes |
|---|----------|----------|-------|
| 1 | How long do refunds take to process? | ✅ | Matched docs exactly |
| 2 | What payment methods does Razorpay support? | ✅ | |
| 3 | Does Razorpay support subscriptions? | ⚠️ | Partially correct, missed one plan type |
| ... | | | |

**Result: X/20 correct** — note any patterns in what it got wrong (e.g.
very recent features, ambiguous questions) and what you'd improve next.

## Possible Improvements

- Add reranking (Cohere Rerank / BGE Reranker) to improve retrieval precision
- Swap Chroma for Pinecone/Qdrant for a cloud-hosted, production-style setup
- Add conversation memory for multi-turn follow-up questions
- Add a feedback button (👍/👎) on answers to track real accuracy over time

## Deployment

Deploy for free on [Streamlit Community Cloud](https://streamlit.io/cloud):
1. Push this repo to GitHub
2. Connect the repo on Streamlit Cloud
3. Add `GROQ_API_KEY` as a secret in the app settings
4. Deploy — note: you'll need to commit `chroma_db/` or re-run ingestion
   on deploy, since it's built locally

## Disclaimer

Built as a personal learning project to demonstrate RAG architecture.
Not an official Razorpay product. All Razorpay content belongs to Razorpay;
this project only indexes their publicly available documentation for
educational, non-commercial demonstration purposes.
