"""
app.py
Streamlit chat UI for the Razorpay Docs RAG bot.

On each question:
  1. Embed the question
  2. Retrieve the top-k most relevant chunks from Chroma
  3. Send those chunks + the question to Groq's LLM
  4. Show the answer AND which source pages it was grounded in

Run with:
    streamlit run app.py

Requires a free Groq API key: https://console.groq.com/keys
Set it as an environment variable GROQ_API_KEY, or in a .env file.
"""

import os
import streamlit as st
from dotenv import load_dotenv
from groq import Groq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

load_dotenv()

PERSIST_DIR = "chroma_db"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
GROQ_MODEL = "llama-3.1-8b-instant"
TOP_K = 4

SYSTEM_PROMPT = """You are a support assistant that answers questions about \
Razorpay's payment products using ONLY the context provided below.

Rules:
- If the answer is not in the context, say "I don't have that information \
in the docs I was given" — do not guess or use outside knowledge.
- Keep answers concise and accurate.
- When relevant, mention specific product/feature names exactly as they \
appear in the context.
"""


@st.cache_resource
def load_vector_store():
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    return Chroma(persist_directory=PERSIST_DIR, embedding_function=embeddings)


@st.cache_resource
def load_groq_client():
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        st.error("GROQ_API_KEY not set. Add it to a .env file or your environment.")
        st.stop()
    return Groq(api_key=api_key)


def retrieve_context(vectordb, query, k=TOP_K):
    results = vectordb.similarity_search(query, k=k)
    return results


def build_prompt(context_docs, question):
    context_text = "\n\n---\n\n".join(doc.page_content for doc in context_docs)
    return f"Context:\n{context_text}\n\nQuestion: {question}"


def generate_answer(client, context_docs, question):
    user_prompt = build_prompt(context_docs, question)
    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
        max_tokens=500,
    )
    return response.choices[0].message.content


# ---------------- Streamlit UI ----------------

st.set_page_config(page_title="Razorpay Docs Assistant", page_icon="💳")
st.title("💳 Razorpay Docs Assistant")
st.caption("A RAG chatbot answering questions using Razorpay's public documentation only.")

vectordb = load_vector_store()
groq_client = load_groq_client()

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

question = st.chat_input("Ask something about Razorpay payments, e.g. 'How do refunds work?'")

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching docs and generating answer..."):
            context_docs = retrieve_context(vectordb, question)
            answer = generate_answer(groq_client, context_docs, question)

            st.markdown(answer)

            with st.expander("📄 Sources used for this answer"):
                seen = set()
                for doc in context_docs:
                    url = doc.metadata.get("source_url", "unknown source")
                    if url not in seen:
                        st.markdown(f"- {url}")
                        seen.add(url)

    st.session_state.messages.append({"role": "assistant", "content": answer})
