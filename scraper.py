"""
scraper.py
Collects public Razorpay documentation pages and saves each as a plain
text file in ./data/raw/  — one file per page.

This is a polite, minimal scraper: it only follows links within the
docs domain, respects a small delay between requests, and stops after
MAX_PAGES so you don't hammer their site or ingest too much for a
learning project.

Usage:
    python scraper.py
"""

import os
import time
import re
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

START_URLS = [
    "https://razorpay.com/docs/payments/",
    "https://razorpay.com/docs/payments/payment-gateway/",
    "https://razorpay.com/docs/api/",
]
ALLOWED_DOMAIN = "razorpay.com"
ALLOWED_PATH_PREFIX = "/docs"
MAX_PAGES = 60          # keep the dataset small for a learning project
REQUEST_DELAY_SECONDS = 1.0
OUTPUT_DIR = "data/raw"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (educational-rag-project; contact: you@example.com)"
}


def is_allowed(url: str) -> bool:
    parsed = urlparse(url)
    return (
        parsed.netloc.endswith(ALLOWED_DOMAIN)
        and parsed.path.startswith(ALLOWED_PATH_PREFIX)
        and not parsed.path.endswith((".pdf", ".zip", ".png", ".jpg"))
    )


def clean_filename(url: str) -> str:
    path = urlparse(url).path.strip("/").replace("/", "_")
    return re.sub(r"[^a-zA-Z0-9_\-]", "", path) or "index"


def extract_text(html: str) -> str:
    soup = BeautifulSoup(html, "lxml")

    # Remove elements that add noise, not knowledge
    for tag in soup(["script", "style", "nav", "footer", "header", "svg"]):
        tag.decompose()

    main = soup.find("main") or soup.body
    if not main:
        return ""

    text = main.get_text(separator="\n")
    lines = [line.strip() for line in text.splitlines()]
    lines = [line for line in lines if line]
    return "\n".join(lines)


def find_links(html: str, base_url: str) -> list[str]:
    soup = BeautifulSoup(html, "lxml")
    links = set()
    for a in soup.find_all("a", href=True):
        full_url = urljoin(base_url, a["href"]).split("#")[0]
        if is_allowed(full_url):
            links.add(full_url)
    return list(links)


def scrape():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    visited = set()
    queue = list(START_URLS)
    saved = 0

    while queue and saved < MAX_PAGES:
        url = queue.pop(0)
        if url in visited:
            continue
        visited.add(url)

        try:
            resp = requests.get(url, headers=HEADERS, timeout=10)
            resp.raise_for_status()
        except requests.RequestException as e:
            print(f"  skip (error): {url} -> {e}")
            continue

        text = extract_text(resp.text)
        if len(text) < 200:
            print(f"  skip (too short): {url}")
            continue

        filename = clean_filename(url) + ".txt"
        filepath = os.path.join(OUTPUT_DIR, filename)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(f"SOURCE_URL: {url}\n\n{text}")

        saved += 1
        print(f"[{saved}/{MAX_PAGES}] saved: {url}")

        for link in find_links(resp.text, url):
            if link not in visited:
                queue.append(link)

        time.sleep(REQUEST_DELAY_SECONDS)

    print(f"\nDone. Saved {saved} pages to {OUTPUT_DIR}/")


if __name__ == "__main__":
    scrape()
