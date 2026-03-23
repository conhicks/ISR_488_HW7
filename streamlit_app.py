import streamlit as st
from openai import OpenAI

import os
import re
import csv
import json
import math
import textwrap
from collections import defaultdict
from datetime import datetime
from typing import Optional

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ModuleNotFoundError:
    anthropic = None
    ANTHROPIC_AVAILABLE = False

import streamlit as st

NEWS_CSV = "news.csv"

@st.cache_resource(show_spinner="Building RAG database…")
def build_rag_db(csv_path: str) -> dict:
    articles = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            doc_text = row.get("Document", "")
            headline = doc_text.split(" Description:")[0].strip()
            description = ""
            if " Description:" in doc_text:
                rest = doc_text.split(" Description:", 1)[1]
                description = rest.split(" content:")[0].strip()
            content = ""
            if " content:" in doc_text:
                content = doc_text.split(" content:", 1)[1].strip()

            articles.append({
                "company": row.get("company_name", "").strip(),
                "date": row.get("Date", ""),
                "headline": headline,
                "description": description,
                "content": content,
                "url": row.get("URL", ""),
                "full_text": f"{headline} {description} {content}".lower(),
            })

    def tokenize(text: str) -> list[str]:
        return re.findall(r"\b[a-z]{3,}\b", text.lower())

    N = len(articles)
    df: dict[str, int] = defaultdict(int)
    token_lists = []
    for art in articles:
        tokens = set(tokenize(art["full_text"]))
        token_lists.append(tokenize(art["full_text"]))
        for t in tokens:
            df[t] += 1

    idf = {t: math.log((N + 1) / (v + 1)) + 1 for t, v in df.items()}

    tfidf_matrix = []
    for tokens in token_lists:
        tf: dict[str, float] = defaultdict(float)
        for t in tokens:
            tf[t] += 1
        total = max(len(tokens), 1)
        vec = {t: (cnt / total) * idf.get(t, 1) for t, cnt in tf.items()}
        norm = math.sqrt(sum(v * v for v in vec.values())) or 1
        vec = {t: v / norm for t, v in vec.items()}
        tfidf_matrix.append(vec)

    return {"articles": articles, "tfidf_matrix": tfidf_matrix, "idf": idf}


def cosine_sim(vec_a: dict, vec_b: dict) -> float:
    return sum(vec_a.get(t, 0) * v for t, v in vec_b.items())


def query_vector(query: str, idf: dict) -> dict:
    tokens = re.findall(r"\b[a-z]{3,}\b", query.lower())
    tf: dict[str, float] = defaultdict(float)
    for t in tokens:
        tf[t] += 1
    total = max(len(tokens), 1)
    vec = {t: (cnt / total) * idf.get(t, 1) for t, cnt in tf.items()}
    norm = math.sqrt(sum(v * v for v in vec.values())) or 1
    return {t: v / norm for t, v in vec.items()}


def search_news(
    db: dict,
    query: str,
    company_filter: Optional[str] = None,
    top_k: int = 5,
) -> list[dict]:
    articles = db["articles"]
    tfidf_matrix = db["tfidf_matrix"]
    idf = db["idf"]

    q_vec = query_vector(query, idf)
    scores = []
    for i, vec in enumerate(tfidf_matrix):
        art = articles[i]
        if company_filter and art["company"].lower() != company_filter.lower():
            continue
        score = cosine_sim(q_vec, vec)
        scores.append((score, i))

    scores.sort(reverse=True)
    results = []
    for score, idx in scores[:top_k]:
        art = articles[idx]
        results.append({
            "rank": len(results) + 1,
            "score": round(score, 4),
            "company": art["company"],
            "date": art["date"][:10] if art["date"] else "",
            "headline": art["headline"],
            "description": art["description"],
            "url": art["url"],
        })
    return results


def rank_interesting_news(
    db: dict,
    company_filter: Optional[str] = None,
    top_k: int = 10,
) -> list[dict]:
    HIGH_VALUE_TERMS = {
        "lawsuit", "litigation", "settlement", "fraud", "investigation",
        "cfpb", "sec", "doj", "ftc", "regulatory", "enforcement", "fine",
        "penalty", "criminal", "indictment", "court", "ruling", "verdict",
        "acquisition", "merger", "ipo", "bankruptcy", "layoffs", "billion",
        "revenue", "profit", "loss", "guidance", "forecast", "downgrade",
        "partnership", "contract", "deal", "expansion", "restructuring",
    }

    articles = db["articles"]
    results = []

    now = datetime.now()
    for art in articles:
        if company_filter and art["company"].lower() != company_filter.lower():
            continue

        text = art["full_text"]
        tokens = set(re.findall(r"\b[a-z]{3,}\b", text))

        keyword_hits = tokens & HIGH_VALUE_TERMS
        keyword_score = len(keyword_hits) * 2

        recency_score = 0
        try:
            pub = datetime.fromisoformat(art["date"].replace("Z", "+00:00").split("+")[0])
            days_old = (now - pub.replace(tzinfo=None)).days
            recency_score = max(0, 10 - days_old * 0.3)
        except Exception:
            pass

        content_score = min(5, len(art["content"].split()) / 100)

        total = keyword_score + recency_score + content_score

        results.append({
            "score": round(total, 2),
            "company": art["company"],
            "date": art["date"][:10] if art["date"] else "",
            "headline": art["headline"],
            "description": art["description"],
            "url": art["url"],
            "matched_keywords": sorted(keyword_hits),
        })

    results.sort(key=lambda x: x["score"], reverse=True)
    seen = set()
    deduped = []
    for r in results:
        key = r["headline"][:60].lower()
        if key not in seen:
            seen.add(key)
            deduped.append(r)
        if len(deduped) >= top_k:
            break

    for i, r in enumerate(deduped):
        r["rank"] = i + 1
    return deduped


TOOLS = [
    {
        "name": "search_news",
        "description": (
            "Search the news database for articles matching a topic, entity, or keyword. "
            "Use this for 'find news about X' queries. "
            "Optionally filter by company name. Returns ranked articles with headlines, "
            "descriptions, dates, and URLs."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query (topic, company, keyword, event).",
                },
                "company_filter": {
                    "type": "string",
                    "description": (
                        "Optional. Restrict results to a specific client company name "
                        "(must match exactly as in the database)."
                    ),
                },
                "top_k": {
                    "type": "integer",
                    "description": "Number of results to return (default 5, max 15).",
                    "default": 5,
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "rank_interesting_news",
        "description": (
            "Return a ranked list of the most interesting/important news articles. "
            "Use this when the user asks for 'most interesting news', 'top stories', "
            "'important updates', or similar broad ranking requests. "
            "Rankings are based on regulatory/legal relevance, financial materiality, "
            "and recency — factors most important to a law firm monitoring clients."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "company_filter": {
                    "type": "string",
                    "description": "Optional. Restrict ranking to a single client company.",
                },
                "top_k": {
                    "type": "integer",
                    "description": "Number of results to return (default 10, max 20).",
                    "default": 10,
                },
            },
        },
    },
]


def format_tool_output(tool_name: str, results: list[dict]) -> str:
    if not results:
        return "No matching news items found."

    lines = [f"### {tool_name} results"]
    for i, r in enumerate(results, start=1):
        lines.append(
            f"{i}. {r.get('company','Unknown')} - {r.get('headline','(no headline)')} "
            f"({r.get('date','')}), score {r.get('score','-')}\n" 
            f"{r.get('description','')}\n{r.get('url','')}"
        )
    return "\n\n".join(lines)


def run_agent_openai(
    client: OpenAI,
    model: str,
    db: dict,
    messages: list[dict],
    company_filter: Optional[str] = None,
) -> str:
    # Simple fallback if Anthropic is unavailable.
    query = ""
    for m in reversed(messages):
        if m.get("role") == "user" and isinstance(m.get("content"), str):
            query = m["content"].strip()
            break

    if not query:
        return "No user query provided."

    if re.search(r"\b(interesting|important|top|key|priority)\b", query, re.I):
        results = rank_interesting_news(db, company_filter=company_filter, top_k=10)
        return format_tool_output("rank_interesting_news", results)

    results = search_news(db, query=query, company_filter=company_filter, top_k=5)
    return format_tool_output("search_news", results)


def run_agent(
    client,
    model: str,
    db: dict,
    messages: list[dict],
    company_filter: Optional[str] = None,
) -> str:
    if not ANTHROPIC_AVAILABLE:
        return run_agent_openai(client, model, db, messages, company_filter=company_filter)

    while True:
        response = client.messages.create(
            model=model,
            max_tokens=2048,
            system=SYSTEM_PROMPT,
            tools=TOOLS,
            messages=messages,
        )

        text_blocks = [b.text for b in response.content if b.type == "text"]
        tool_use_blocks = [b for b in response.content if b.type == "tool_use"]

        if response.stop_reason == "end_turn" or not tool_use_blocks:
            return "\n".join(text_blocks)

        messages.append({"role": "assistant", "content": response.content})

        tool_results = []
        for block in tool_use_blocks:
            tool_input = block.input
            if company_filter and not tool_input.get("company_filter"):
                tool_input["company_filter"] = company_filter

            if block.name == "search_news":
                result = search_news(
                    db,
                    query=tool_input["query"],
                    company_filter=tool_input.get("company_filter"),
                    top_k=min(int(tool_input.get("top_k", 5)), 15),
                )
            elif block.name == "rank_interesting_news":
                result = rank_interesting_news(
                    db,
                    company_filter=tool_input.get("company_filter"),
                    top_k=min(int(tool_input.get("top_k", 10)), 20),
                )
            else:
                result = {"error": f"Unknown tool: {block.name}"}

            tool_results.append({
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": json.dumps(result),
            })

        messages.append({"role": "user", "content": tool_results})


def main():
    st.title("Client News Intelligence Bot")

    with st.sidebar:
        st.header("Configuration")

        api_key = st.text_input(
            "API Key (Anthropic or OpenAI)",
            type="password",
            value=os.getenv("ANTHROPIC_API_KEY", ""),
        )

        provider = st.radio(
            "Provider",
            options=["Anthropic", "OpenAI"],
            index=0 if ANTHROPIC_AVAILABLE else 1,
        )

        model_choice = st.text_input(
            "Model",
            value="claude-haiku-4-5-20251001" if provider == "Anthropic" else "gpt-4o-mini",
        )

        st.divider()
        st.subheader("Filter by Client")

        if "db" in st.session_state:
            companies = sorted(
                {a["company"] for a in st.session_state.db["articles"]}
            )
            company_filter = st.selectbox(
                "Company",
                options=["All clients"] + companies,
            )
            company_filter = None if company_filter == "All clients" else company_filter
        else:
            company_filter = None

        st.divider()
        if st.button("Clear conversation"):
            st.session_state.messages = []
            st.rerun()

    if not os.path.exists(NEWS_CSV):
        st.error(
            f"Could not find `{NEWS_CSV}`. "
            "Please place the news CSV file in the same directory as HW7.py."
        )
        st.stop()

    if "db" not in st.session_state:
        st.session_state.db = build_rag_db(NEWS_CSV)

    db = st.session_state.db
    article_count = len(db["articles"])
    company_count = len({a["company"] for a in db["articles"]})

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        if isinstance(msg["content"], str):
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

    if not api_key:
        st.info("Enter your API key in the sidebar to start.")
        st.stop()

    if provider == "Anthropic" and not ANTHROPIC_AVAILABLE:
        st.error(
            "Anthropic SDK is not installed. Add `anthropic` to your requirements and rerun, "
            "or switch to OpenAI provider."
        )
        st.stop()

    if prompt := st.chat_input("Ask about client news… e.g. 'Find the most interesting news' or 'Find news about AI'"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        if provider == "Anthropic":
            client = anthropic.Anthropic(api_key=api_key)
        else:
            client = OpenAI(api_key=api_key)

        api_messages = [
            {"role": m["role"], "content": m["content"]}
            for m in st.session_state.messages
            if isinstance(m["content"], str)
        ]

        with st.chat_message("assistant"):
            with st.spinner("Searching news database…"):
                response_text = run_agent(
                    client=client,
                    model=model_choice,
                    db=db,
                    messages=api_messages,
                    company_filter=company_filter,
                )
            st.markdown(response_text)

        st.session_state.messages.append({"role": "assistant", "content": response_text})


if __name__ == "__main__":
    main()