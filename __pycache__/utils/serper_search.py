"""
serper_search.py — Serper.dev Google Search Integration for TicketHub
Handles real-time search for: Bus, Flight, Concert, Hotel, Train
"""

import http.client
import json
import os

SERPER_API_KEY = os.getenv("SERPER_API_KEY", "")
SERPER_HOST    = "google.serper.dev"


def _serper_request(endpoint: str, query: str, extra: dict = None) -> dict:
    """Make a POST request to Serper API."""
    if not SERPER_API_KEY:
        return {}
    payload = {"q": query, "gl": "in", "hl": "en", "num": 10}
    if extra:
        payload.update(extra)

    conn = http.client.HTTPSConnection(SERPER_HOST)
    conn.request(
        "POST",
        f"/{endpoint}",
        json.dumps(payload),
        {
            "X-API-KEY": SERPER_API_KEY,
            "Content-Type": "application/json",
        },
    )
    res  = conn.getresponse()
    data = res.read().decode("utf-8")
    conn.close()
    return json.loads(data)


# ─────────────────────────────────────────────
# BUS SEARCH
# ─────────────────────────────────────────────
def search_bus(source: str, destination: str, date: str = "") -> list:
    """Search for bus tickets using Serper."""
    query = f"{source} to {destination} bus ticket booking online"
    if date:
        query += f" {date}"

    data    = _serper_request("search", query)
    results = []

    # Organic results
    for r in data.get("organic", [])[:6]:
        results.append({
            "type":        "BUS",
            "title":       r.get("title", ""),
            "snippet":     r.get("snippet", ""),
            "source":      source,
            "destination": destination,
            "icon":        "🚌",
            "color":       "linear-gradient(135deg,#065f46,#10b981)",
        })

    # Knowledge box / answerBox extra
    if data.get("answerBox"):
        ab = data["answerBox"]
        results.insert(0, {
            "type":        "BUS",
            "title":       ab.get("title", f"{source} → {destination} Bus"),
            "snippet":     ab.get("answer") or ab.get("snippet", ""),
            "source":      source,
            "destination": destination,
            "icon":        "🚌",
            "color":       "linear-gradient(135deg,#065f46,#10b981)",
            "featured":    True,
        })

    return results


# ─────────────────────────────────────────────
# FLIGHT SEARCH
# ─────────────────────────────────────────────
def search_flight(source: str, destination: str, date: str = "") -> list:
    """Search for flight tickets using Serper."""
    query = f"flights from {source} to {destination} cheap airfare book online India"
    if date:
        query += f" {date}"

    data    = _serper_request("search", query)
    results = []

    for r in data.get("organic", [])[:6]:
        results.append({
            "type":        "FLIGHT",
            "title":       r.get("title", ""),
            "snippet":     r.get("snippet", ""),
            "source":      source,
            "destination": destination,
            "icon":        "✈️",
            "color":       "linear-gradient(135deg,#135f6d,#1a7a8a)",
        })

    if data.get("answerBox"):
        ab = data["answerBox"]
        results.insert(0, {
            "type":        "FLIGHT",
            "title":       ab.get("title", f"{source} → {destination} Flight"),
            "snippet":     ab.get("answer") or ab.get("snippet", ""),
            "source":      source,
            "destination": destination,
            "icon":        "✈️",
            "color":       "linear-gradient(135deg,#135f6d,#1a7a8a)",
            "featured":    True,
        })

    return results


# ─────────────────────────────────────────────
# HOTEL SEARCH
# ─────────────────────────────────────────────
def search_hotel(city: str, checkin: str = "", checkout: str = "") -> list:
    """Search for hotels using Serper."""
    query = f"best hotels in {city} book online India"
    if checkin:
        query += f" checkin {checkin}"

    data    = _serper_request("search", query)
    results = []

    for r in data.get("organic", [])[:6]:
        results.append({
            "type":        "HOTEL",
            "title":       r.get("title", ""),
            "snippet":     r.get("snippet", ""),
            "source":      city,
            "destination": city,
            "icon":        "🏨",
            "color":       "linear-gradient(135deg,#92400e,#f59e0b)",
        })

    # Places results (richer for hotels)
    for p in data.get("places", [])[:4]:
        results.insert(0, {
            "type":        "HOTEL",
            "title":       p.get("title", ""),
            "snippet":     f"⭐ {p.get('rating','N/A')} · {p.get('address','')}" if p.get("rating") else p.get("address", ""),
            "source":      city,
            "destination": city,
            "icon":        "🏨",
            "color":       "linear-gradient(135deg,#92400e,#f59e0b)",
            "rating":      p.get("rating"),
            "featured":    True,
        })

    return results[:8]


# ─────────────────────────────────────────────
# CONCERT / EVENT SEARCH
# ─────────────────────────────────────────────
def search_concert(city: str, keyword: str = "") -> list:
    """Search for concerts and events using Serper."""
    q_term = keyword if keyword else "upcoming concerts events"
    query  = f"{q_term} in {city} India 2025 2026 tickets buy online"

    data    = _serper_request("search", query)
    results = []

    for r in data.get("organic", [])[:6]:
        results.append({
            "type":        "CONCERT",
            "title":       r.get("title", ""),
            "snippet":     r.get("snippet", ""),
            "source":      city,
            "destination": city,
            "icon":        "🎵",
            "color":       "linear-gradient(135deg,#9d174d,#ec4899)",
        })

    # Top stories for events
    for s in data.get("topStories", [])[:3]:
        results.insert(0, {
            "type":        "CONCERT",
            "title":       s.get("title", ""),
            "snippet":     s.get("source", "") + " · " + s.get("date", ""),
            "source":      city,
            "destination": city,
            "icon":        "🎵",
            "color":       "linear-gradient(135deg,#9d174d,#ec4899)",
            "featured":    True,
        })

    return results[:8]


# ─────────────────────────────────────────────
# TRAIN SEARCH
# ─────────────────────────────────────────────
def search_train(source: str, destination: str, date: str = "") -> list:
    """Search for train tickets using Serper."""
    query = f"{source} to {destination} train ticket IRCTC booking"
    if date:
        query += f" {date}"

    data    = _serper_request("search", query)
    results = []

    for r in data.get("organic", [])[:6]:
        results.append({
            "type":        "TRAIN",
            "title":       r.get("title", ""),
            "snippet":     r.get("snippet", ""),
            "source":      source,
            "destination": destination,
            "icon":        "🚂",
            "color":       "linear-gradient(135deg,#1e40af,#3b82f6)",
        })

    if data.get("answerBox"):
        ab = data["answerBox"]
        results.insert(0, {
            "type":        "TRAIN",
            "title":       ab.get("title", f"{source} → {destination} Train"),
            "snippet":     ab.get("answer") or ab.get("snippet", ""),
            "source":      source,
            "destination": destination,
            "icon":        "🚂",
            "color":       "linear-gradient(135deg,#1e40af,#3b82f6)",
            "featured":    True,
        })

    return results


# ─────────────────────────────────────────────
# UNIFIED SEARCH DISPATCHER
# ─────────────────────────────────────────────
def unified_search(ticket_type: str, source: str = "", destination: str = "",
                   date: str = "", keyword: str = "") -> list:
    """Route the search to the correct handler based on ticket type."""
    t = ticket_type.upper().strip()

    try:
        if t == "BUS":
            return search_bus(source, destination, date)
        elif t == "FLIGHT":
            return search_flight(source, destination, date)
        elif t == "HOTEL":
            return search_hotel(destination or source, date)
        elif t in ("CONCERT", "EVENT"):
            return search_concert(destination or source, keyword)
        elif t == "TRAIN":
            return search_train(source, destination, date)
        else:
            # Generic fallback
            query = f"{ticket_type} {source} {destination} {keyword} ticket India".strip()
            data  = _serper_request("search", query)
            return [
                {
                    "type":        ticket_type.upper(),
                    "title":       r.get("title", ""),
                    "snippet":     r.get("snippet", ""),
                    "source":      source,
                    "destination": destination,
                    "icon":        "🎟",
                    "color":       "linear-gradient(135deg,#4b5563,#9ca3af)",
                }
                for r in data.get("organic", [])[:6]
            ]
    except Exception as e:
        print(f"Serper search error ({t}): {e}")
        return []
