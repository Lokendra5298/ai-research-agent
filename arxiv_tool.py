# arxiv_tool.py

# Step1: Access arXiv using URL
import re
from urllib.parse import quote_plus

import requests


def search_arxiv_papers(topic: str, max_results: int = 5) -> dict:
    # The language model sometimes wraps phrases in quotes or parentheses
    # (e.g. '"medical image segmentation" transformer'). arXiv's query
    # parser treats those as operators, so strip them out and collapse the
    # remaining words into a single search string rather than failing.
    cleaned = re.sub(r'[()"\']+', " ", topic.lower()).strip()

    if not cleaned:
        raise ValueError(f"Empty search query after cleaning input: {topic!r}")

    # quote_plus turns spaces into '+' and percent-encodes anything else
    # that is unsafe in a URL.
    query = quote_plus(cleaned)
    url = (
            "http://export.arxiv.org/api/query"
            f"?search_query=all:{query}"
            f"&max_results={max_results}"
            "&sortBy=submittedDate"
            "&sortOrder=descending"
        )
    print(f"Making request to arXiv API: {url}")
    resp = requests.get(url, timeout=30)

    if not resp.ok:
        print(f"ArXiv API request failed: {resp.status_code} - {resp.text}")
        raise ValueError(f"Bad response from arXiv API: {resp}\n{resp.text}")
    
    data = parse_arxiv_xml(resp.text)
    return data


# Step2: Parse XML
import xml.etree.ElementTree as ET
def parse_arxiv_xml(xml_content: str) -> dict:
    """Parse the XML content from arXiv API response."""

    entries = []
    ns = {
        "atom": "http://www.w3.org/2005/Atom",
        "arxiv": "http://arxiv.org/schemas/atom"
    }
    root = ET.fromstring(xml_content)
    # Loop through each <entry> in Atom namespace
    for entry in root.findall("atom:entry", ns):
        # Extract authors
        authors = [
            author.findtext("atom:name", namespaces=ns)
            for author in entry.findall("atom:author", ns)
        ]
        
        # Extract categories (term attribute)
        categories = [
            cat.attrib.get("term")
            for cat in entry.findall("atom:category", ns)
        ]
        
        # Extract PDF link (rel="related" and type="application/pdf")
        pdf_link = None
        for link in entry.findall("atom:link", ns):
            if link.attrib.get("type") == "application/pdf":
                pdf_link = link.attrib.get("href")
                break

        entries.append({
            "title": entry.findtext("atom:title", namespaces=ns),
            "summary": entry.findtext("atom:summary", namespaces=ns).strip(),
            "authors": authors,
            "categories": categories,
            "pdf": pdf_link
        })

    return {"entries": entries}

# Step3: Convert the functionality into a tool
from langchain_core.tools import tool


@tool
def arxiv_search(topic: str) -> list[dict]:
    """Search for recently uploaded arXiv papers

    Args:
        topic: The topic to search for papers about

    Returns:
        List of papers with their metadata including title, authors, summary, etc.
    """
    print("ARXIV Agent called")
    print(f"Searching arXiv for papers about: {topic}")
    papers = search_arxiv_papers(topic)
    if len(papers["entries"]) == 0:
        print(f"No papers found for topic: {topic}")
        raise ValueError(f"No papers found for topic: {topic}")
    print(f"Found {len(papers['entries'])} papers about {topic}")
    return papers


# Manual test (only runs when executing this file directly).
if __name__ == "__main__":
    topic = "machine learning"
    papers = search_arxiv_papers(topic=topic, max_results=5)
    print(papers)