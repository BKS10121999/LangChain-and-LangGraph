"""Knowledge base retrieval tools using keyword-based relevance scoring (RAG)."""
import os
import re
import sys
from langchain_core.tools import tool
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# Database path
UDAHUB_DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data", "core", "udahub.db")


def _get_udahub_session():
    """Get a database session for UDA-Hub."""
    engine = create_engine(f"sqlite:///{UDAHUB_DB_PATH}", echo=False)
    Session = sessionmaker(bind=engine)
    return Session()


def _normalize_words(text: str) -> set[str]:
    words = set(re.findall(r"[a-z0-9']+", (text or "").lower()))
    normalized = set()
    for word in words:
        normalized.add(word)
        if word.endswith("ies") and len(word) > 4:
            normalized.add(word[:-3] + "y")
        elif word.endswith("es") and len(word) > 3:
            normalized.add(word[:-2])
        elif word.endswith("s") and len(word) > 3:
            normalized.add(word[:-1])
        if word.endswith("ation"):
            normalized.add(word[:-5])
    return normalized


def _compute_relevance_score(query: str, article_title: str, article_content: str, article_tags: str) -> float:
    """Compute a keyword-based relevance score between query and article.
    
    Scoring:
      - Title word matches: 3x weight
      - Tag word matches: 2x weight  
      - Content word matches: 1x weight
    Normalized to [0, 1].
    """
    query_words = _normalize_words(query)
    title_words = _normalize_words(article_title)
    content_words = _normalize_words(article_content)
    tag_words = _normalize_words(article_tags)

    title_overlap = len(query_words & title_words)
    content_overlap = len(query_words & content_words)
    tag_overlap = len(query_words & tag_words)

    score = (title_overlap * 3.0 + tag_overlap * 2.0 + content_overlap * 1.0)
    max_possible = len(query_words) * 3.0

    if max_possible == 0:
        return 0.0
    return min(score / max_possible, 1.0)


def search_knowledge_records(query: str, account_id: str = "cultpass", limit: int = 3) -> list[dict]:
    """Return structured article matches for internal workflow use."""
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from data.models.udahub import Knowledge

    session = _get_udahub_session()
    try:
        articles = session.query(Knowledge).filter_by(account_id=account_id).all()
        scored_articles = []
        for article in articles:
            score = _compute_relevance_score(query, article.title, article.content, article.tags)
            if score > 0.08:
                scored_articles.append({
                    "title": article.title,
                    "content": article.content,
                    "tags": article.tags,
                    "relevance_score": round(score, 3),
                })

        scored_articles.sort(key=lambda item: item["relevance_score"], reverse=True)
        return scored_articles[:limit]
    finally:
        session.close()


@tool
def search_knowledge_base(query: str, account_id: str = "cultpass") -> str:
    """Search the knowledge base for relevant support articles based on a query. Returns the most relevant articles with confidence scores. Use this to find answers to customer questions."""
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from data.models.udahub import Knowledge

    session = _get_udahub_session()
    try:
        if not session.query(Knowledge).filter_by(account_id=account_id).first():
            return "No knowledge base articles found. Please escalate to human support."
    finally:
        session.close()

    top_articles = search_knowledge_records(query=query, account_id=account_id, limit=3)
    if not top_articles:
        return "NO_RELEVANT_ARTICLES_FOUND: No relevant knowledge base articles match this query. Consider escalating to human support."

    result = f"Found {len(top_articles)} relevant article(s):\n\n"
    for i, article in enumerate(top_articles, 1):
        result += f"--- Article {i} (Relevance: {article['relevance_score']}) ---\n"
        result += f"Title: {article['title']}\n"
        result += f"Content: {article['content']}\n"
        result += f"Tags: {article['tags']}\n\n"

    return result
