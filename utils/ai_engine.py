"""
ai_engine.py — TicketHub AI Recommendation Engine (Production)

Fixes applied:
  1. Multi-factor scoring: price, discount, type match, destination match,
     popularity, seat availability — weighted and normalised to 0-100.
  2. Serper API: structured result extraction with location, description,
     rating, popularity, best_time, travel_suitability.
  3. AI explanation includes concrete pros/cons and budget fit.
  4. RAG chatbot injects live ticket data for price/route queries.
  5. All Serper results validated before use; fallback on failure.
"""

import json
import math
import requests
import os
from datetime import datetime


# ── Groq AI config ────────────────────────────────────────────────────
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
AI_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

SERPER_API_KEY = os.getenv("SERPER_API_KEY", "")
SERPER_URL     = "https://google.serper.dev/search"


# ══════════════════════════════════════════════════════════════════════
# UTILITY
# ══════════════════════════════════════════════════════════════════════

def call_ai(messages: list, max_tokens: int = 512, temperature: float = 0.20, stream: bool = False) -> str:
    if not GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY is not configured.")

    payload = {
        "model":       AI_MODEL,
        "messages":    messages,
        "max_tokens":  max_tokens,
        "temperature": temperature,
        "top_p":       0.70,
        "stream":      stream,
    }
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }
    resp = requests.post(GROQ_API_URL, headers=headers, json=payload, timeout=45, stream=stream)
    resp.raise_for_status()
    if not stream:
        return resp.json()["choices"][0]["message"]["content"].strip()

    chunks = []
    for line in resp.iter_lines(decode_unicode=True):
        if not line or not line.startswith("data: "):
            continue
        data = line[6:].strip()
        if data == "[DONE]":
            break
        try:
            obj = json.loads(data)
            delta = obj.get("choices", [{}])[0].get("delta", {}).get("content", "")
            if delta:
                chunks.append(delta)
        except json.JSONDecodeError:
            continue
    return "".join(chunks).strip()


def _serper_search(query: str, num: int = 6) -> dict:
    """POST to Serper and return raw JSON. Raises on network failure."""
    if not SERPER_API_KEY:
        raise RuntimeError("SERPER_API_KEY is not configured.")
    headers = {
        "X-API-KEY":    SERPER_API_KEY,
        "Content-Type": "application/json",
    }
    payload = json.dumps({"q": query, "gl": "in", "hl": "en", "num": num})
    resp = requests.post(SERPER_URL, headers=headers, data=payload, timeout=15)
    resp.raise_for_status()
    return resp.json()


# ══════════════════════════════════════════════════════════════════════
# 1. TICKET SCORING ENGINE
# ══════════════════════════════════════════════════════════════════════

# Scoring weights (must sum to 100)
_WEIGHTS = {
    "price":        35,   # how well the price fits the budget
    "discount":     20,   # size of discount
    "type_match":   20,   # matches user's preferred transport type
    "dest_match":   10,   # matches preferred destination
    "availability": 10,   # seat availability (higher = better)
    "popularity":    5,   # bonus for well-known routes
}
assert sum(_WEIGHTS.values()) == 100


def _score_price(final_price: float, budget: float) -> float:
    """0-35: Best score at 60-80% of budget. Penalty beyond budget."""
    if budget <= 0:
        return 17.5  # neutral
    ratio = final_price / budget
    if ratio <= 0:
        return 0.0
    if ratio <= 0.6:
        # Cheap but not the cheapest — good value
        return _WEIGHTS["price"] * 0.85
    if ratio <= 0.8:
        # Sweet spot
        return _WEIGHTS["price"] * 1.0
    if ratio <= 1.0:
        # Slightly tight but within budget
        return _WEIGHTS["price"] * 0.70
    # Over budget — exponential penalty
    overshoot = ratio - 1.0
    return max(0.0, _WEIGHTS["price"] * (1.0 - min(overshoot * 2.5, 1.0)))


def _score_discount(discount: float) -> float:
    """0-20: Linear up to 25%, capped at 20 points."""
    return min(discount / 25.0 * _WEIGHTS["discount"], _WEIGHTS["discount"])


def _score_type_match(ticket_type: str, interests: list) -> float:
    """0-20: Full if type matches interest, half if not."""
    if ticket_type.upper() in [i.upper() for i in interests]:
        return float(_WEIGHTS["type_match"])
    return _WEIGHTS["type_match"] * 0.4


def _score_dest_match(ticket_dest: str, pref_dest: str) -> float:
    """0-10: Full match, partial match, or no match."""
    if not pref_dest:
        return _WEIGHTS["dest_match"] * 0.5   # neutral when no preference
    td = ticket_dest.upper()
    pd = pref_dest.upper()
    if pd == td:
        return float(_WEIGHTS["dest_match"])
    if pd in td or td in pd:
        return _WEIGHTS["dest_match"] * 0.6
    return 0.0


def _score_availability(total_seats: int) -> float:
    """0-10: More seats = more available = better for group booking."""
    if total_seats <= 0:
        return _WEIGHTS["availability"] * 0.5
    # Sigmoid-ish: 30 seats → ~6pts, 150 seats → ~10pts
    return min(_WEIGHTS["availability"] * math.log1p(total_seats) / math.log1p(300),
               float(_WEIGHTS["availability"]))


def _score_popularity(ticket: dict) -> float:
    """0-5: Bonus for popular routes and well-known operators."""
    popular_keywords = [
        "rajdhani", "shatabdi", "vande bharat", "indigo", "air india",
        "taj", "volvo", "arijit", "diljit", "sunburn", "goa", "mumbai",
        "delhi", "bangalore", "leela", "itc",
    ]
    title_lower = ticket.get("title", "").lower()
    matches = sum(1 for kw in popular_keywords if kw in title_lower)
    return min(matches * 1.5, float(_WEIGHTS["popularity"]))


def score_ticket(ticket: dict, prefs: dict) -> float:
    """
    Score a ticket 0-100 using weighted multi-factor algorithm.
    prefs: {budget: float, interests: [str], destination: str}
    """
    price    = float(ticket.get("price", 9999))
    discount = float(ticket.get("discount", 0))
    final    = round(price * (1 - discount / 100), 2)

    budget   = float(prefs.get("budget", 5000))
    interests = prefs.get("interests", [])
    pref_dest = prefs.get("destination", "")
    t_type    = ticket.get("type", "")
    total_seats = int(ticket.get("total_seats", 40))

    score = (
        _score_price(final, budget)
        + _score_discount(discount)
        + _score_type_match(t_type, interests)
        + _score_dest_match(ticket.get("destination", ""), pref_dest)
        + _score_availability(total_seats)
        + _score_popularity(ticket)
    )
    return round(min(score, 100.0), 2)


def get_best_plan(tickets: list, prefs: dict = None) -> dict:
    """
    Score & rank all tickets. Return best pick + AI explanation.
    prefs = {budget, interests, destination}
    """
    if not tickets:
        return {"best_plan": None, "ai_reason": "No tickets available.", "ranked": [], "prefs_used": prefs}

    if prefs is None:
        prefs = {"budget": 5000, "interests": ["BUS", "TRAIN", "FLIGHT"], "destination": ""}

    budget    = float(prefs.get("budget", 5000))
    interests = [i.upper() for i in prefs.get("interests", [])]
    pref_dest = prefs.get("destination", "").upper()

    # Score every ticket
    scored = []
    for t in tickets:
        price    = float(t.get("price", 0))
        discount = float(t.get("discount", 0))
        final    = round(price * (1 - discount / 100), 2)
        s        = score_ticket(t, prefs)
        scored.append({**t, "score": s, "final_price": final})

    scored.sort(key=lambda x: x["score"], reverse=True)
    best = scored[0]

    # Build ranked list with structured pros/cons
    ranked = []
    for rank, t in enumerate(scored[:5], 1):
        pros, cons = [], []

        # Price analysis
        if t["final_price"] <= budget:
            pct = round((1 - t["final_price"] / budget) * 100) if budget > 0 else 0
            pros.append(f"Within budget (saves ~{pct}% of budget)")
        else:
            over = round(t["final_price"] - budget, 2)
            cons.append(f"₹{over:,.0f} over budget")

        # Discount analysis
        if t.get("discount", 0) >= 15:
            pros.append(f"Great deal — {t['discount']}% discount")
        elif t.get("discount", 0) > 0:
            pros.append(f"{t['discount']}% discount applied")
        else:
            cons.append("No discount available")

        # Type match
        if t.get("type", "").upper() in interests:
            pros.append(f"Matches your preferred type ({t.get('type','')})")
        else:
            cons.append(f"Not your preferred type (you prefer {', '.join(interests)})")

        # Destination match
        if pref_dest and pref_dest in t.get("destination", "").upper():
            pros.append("Goes to your preferred destination")

        # Seat availability
        seats = t.get("total_seats", 0)
        if seats >= 100:
            pros.append(f"High availability ({seats} seats)")
        elif seats < 30:
            cons.append(f"Limited seats ({seats} remaining)")

        # Popularity
        pop_score = _score_popularity(t)
        if pop_score >= 3:
            pros.append("Popular & highly rated route")

        ranked.append({
            "rank":        rank,
            "id":          str(t.get("id", t.get("_id", ""))),
            "title":       t.get("title", ""),
            "type":        t.get("type", ""),
            "source":      t.get("source", ""),
            "destination": t.get("destination", ""),
            "price":       t.get("price", 0),
            "discount":    t.get("discount", 0),
            "final_price": t["final_price"],
            "total_seats": t.get("total_seats", 0),
            "score":       t["score"],
            "pros":        pros,
            "cons":        cons,
        })

    # AI explanation for the best pick
    ai_reason = ""
    if best:
        try:
            pros_text = "; ".join(ranked[0]["pros"]) if ranked else ""
            prompt = (
                f"You are a travel advisor. In 2-3 concise sentences explain why this "
                f"ticket is the best recommendation for the user.\n"
                f"Ticket: {best.get('title')}\n"
                f"Type: {best.get('type')}\n"
                f"Route: {best.get('source')} → {best.get('destination')}\n"
                f"Price: ₹{best['final_price']:,.0f} (after {best.get('discount', 0)}% discount)\n"
                f"User budget: ₹{budget:,.0f}\n"
                f"User interests: {', '.join(interests)}\n"
                f"Key strengths: {pros_text}\n"
                f"Be specific, positive, and mention the price saving if any."
            )
            ai_reason = call_ai([{"role": "user", "content": prompt}], max_tokens=180, temperature=0.3)
        except Exception as e:
            print(f"AI reason error: {e}")
            disc = best.get("discount", 0)
            saving = round(float(best.get("price", 0)) * disc / 100, 0) if disc else 0
            ai_reason = (
                f"This {best.get('type', 'ticket')} scores highest across price, discount, "
                f"and your preferences. "
                + (f"You save ₹{saving:,.0f} with the {disc}% discount. " if disc > 0 else "")
                + f"Final price ₹{best['final_price']:,.0f} is {'within' if best['final_price'] <= budget else 'close to'} your budget."
            )

    return {
        "best_plan":  ranked[0] if ranked else None,
        "ai_reason":  ai_reason,
        "ranked":     ranked,
        "prefs_used": prefs,
    }


# ══════════════════════════════════════════════════════════════════════
# 2. SERPER TRENDING LOCATIONS
# ══════════════════════════════════════════════════════════════════════

# Popularity signals used to boost scores
_HIGH_POPULARITY = [
    "goa", "jaipur", "kerala", "manali", "shimla", "varanasi", "agra",
    "rajasthan", "mumbai", "delhi", "ooty", "darjeeling", "andaman",
    "coorg", "rishikesh", "leh", "ladakh", "udaipur",
]

_SUITABILITY_MAP = {
    "beach":     "Beach & Coastal",
    "hill":      "Hill Station",
    "mountain":  "Adventure & Trekking",
    "fort":      "Heritage & Culture",
    "palace":    "Heritage & Culture",
    "temple":    "Spiritual & Religious",
    "wildlife":  "Nature & Wildlife",
    "backwater": "Nature & Backwaters",
    "adventure": "Adventure & Sports",
    "shopping":  "City & Shopping",
}


def _extract_rating_from_snippet(snippet: str) -> float:
    """Try to parse a numeric rating from a Serper snippet string."""
    import re
    m = re.search(r'\b([4-5]\.\d)\b', snippet)
    if m:
        return float(m.group(1))
    m = re.search(r'rated?\s*([4-5]\.\d)', snippet, re.IGNORECASE)
    if m:
        return float(m.group(1))
    return None


def _guess_travel_suitability(text: str) -> str:
    text_lower = text.lower()
    for kw, label in _SUITABILITY_MAP.items():
        if kw in text_lower:
            return label
    return "General Tourism"


def _guess_best_time(name: str) -> str:
    n = name.lower()
    if any(k in n for k in ["goa", "beach", "coastal", "andaman"]):
        return "Oct–Mar"
    if any(k in n for k in ["manali", "shimla", "hill", "mountain", "leh", "ladakh"]):
        return "Mar–Jun"
    if any(k in n for k in ["rajasthan", "jaipur", "udaipur", "desert"]):
        return "Nov–Feb"
    if any(k in n for k in ["kerala", "coorg", "ooty", "south"]):
        return "Sep–Mar"
    if any(k in n for k in ["rishikesh", "uttarakhand", "darjeeling"]):
        return "Mar–May"
    return "Oct–Mar"


def _popularity_score(name: str, snippet: str) -> str:
    text = (name + " " + snippet).lower()
    hits = sum(1 for kw in _HIGH_POPULARITY if kw in text)
    if hits >= 2:
        return "Very High"
    if hits == 1:
        return "High"
    return "Moderate"


def _parse_serper_location(r: dict, source_label: str = "Serper") -> dict:
    """
    Convert a single Serper organic result into a structured location dict.
    Extracts: name, description, rating, popularity, travel_suitability, best_time.
    """
    title   = r.get("title", "").strip()
    snippet = r.get("snippet", "").strip()

    # Clean up common SEO suffixes from titles
    for suffix in [" - Wikipedia", " | India Tourism", " Travel Guide",
                   " - Incredible India", " - Times of India", " | Holidify"]:
        title = title.replace(suffix, "")
    title = title.strip()

    # Extract or estimate rating
    rating = _extract_rating_from_snippet(snippet)
    if rating is None:
        # Deterministic pseudo-rating based on title hash (consistent across calls)
        hash_int = int(sum(ord(c) for c in title[:8]))
        rating = round(3.8 + (hash_int % 14) * 0.1, 1)
        rating = min(rating, 5.0)

    return {
        "name":              title[:60],
        "description":       snippet[:200] if snippet else f"A popular destination in India.",
        "rating":            rating,
        "popularity":        _popularity_score(title, snippet),
        "travel_suitability": _guess_travel_suitability(title + " " + snippet),
        "best_time":         _guess_best_time(title + " " + snippet),
        "image":             r.get("imageUrl", ""),
        "link":              r.get("link", ""),
        "source":            source_label,
    }


def fetch_trending_locations(db=None) -> list:
    """
    Fetch trending Indian tourist destinations via Serper API.
    Results are cached in MongoDB for 6 hours to respect API limits.
    Falls back to curated list on failure.
    """
    CACHE_TTL = 6 * 3600  # 6 hours

    # Check cache
    if db is not None:
        try:
            cached = db.trending_locations.find_one(
                {"cached_at": {"$gt": datetime.now().timestamp() - CACHE_TTL}}
            )
            if cached and cached.get("locations"):
                return cached["locations"]
        except Exception:
            pass

    queries = [
        "best tourist destinations India 2026",
        "top travel places India trending",
        "most visited cities India tourism",
    ]

    all_results = []
    seen_names  = set()

    try:
        for query in queries:
            data = _serper_search(query, num=6)

            # Knowledge graph — most authoritative result
            kg = data.get("knowledgeGraph", {})
            if kg.get("title") and kg["title"].lower() not in seen_names:
                loc = {
                    "name":              kg["title"][:60],
                    "description":       kg.get("description", "")[:200] or "Featured destination.",
                    "rating":            4.6,
                    "popularity":        "Very High",
                    "travel_suitability": _guess_travel_suitability(kg.get("title", "") + kg.get("description", "")),
                    "best_time":         _guess_best_time(kg.get("title", "")),
                    "image":             kg.get("imageUrl", ""),
                    "link":              kg.get("website", ""),
                    "source":            "Serper KG",
                    "featured":          True,
                }
                all_results.insert(0, loc)
                seen_names.add(kg["title"].lower()[:20])

            # Organic results
            for r in data.get("organic", [])[:5]:
                loc = _parse_serper_location(r)
                name_key = loc["name"].lower()[:20]
                if name_key not in seen_names and len(loc["name"]) > 3:
                    all_results.append(loc)
                    seen_names.add(name_key)

            # Answer box
            ab = data.get("answerBox", {})
            if ab.get("title") and ab["title"].lower()[:20] not in seen_names:
                loc = _parse_serper_location({
                    "title":   ab.get("title", ""),
                    "snippet": ab.get("answer") or ab.get("snippet", ""),
                    "link":    ab.get("link", ""),
                })
                if loc["name"] and len(loc["name"]) > 3:
                    all_results.append(loc)
                    seen_names.add(ab["title"].lower()[:20])

        # Sort: featured first, then by rating desc
        all_results.sort(key=lambda x: (not x.get("featured", False), -x.get("rating", 0)))
        final = all_results[:12]

        # Cache to DB
        if db is not None and final:
            try:
                db.trending_locations.delete_many({})
                db.trending_locations.insert_one({
                    "locations":  final,
                    "cached_at":  datetime.now().timestamp(),
                })
            except Exception as e:
                print(f"Trending cache write error: {e}")

        return final if final else _fallback_locations()

    except Exception as e:
        print(f"Serper trending fetch error: {e}")
        return _fallback_locations()


def _fallback_locations() -> list:
    return [
        {"name": "Goa", "description": "Beautiful beaches, vibrant nightlife and Portuguese heritage.", "rating": 4.6, "popularity": "Very High", "travel_suitability": "Beach & Coastal", "best_time": "Oct–Mar", "image": "", "source": "Fallback", "featured": True},
        {"name": "Jaipur", "description": "The Pink City — forts, palaces, bazaars and royal heritage.", "rating": 4.5, "popularity": "Very High", "travel_suitability": "Heritage & Culture", "best_time": "Nov–Feb", "image": "", "source": "Fallback"},
        {"name": "Kerala", "description": "God's Own Country — backwaters, spice gardens and hill stations.", "rating": 4.7, "popularity": "Very High", "travel_suitability": "Nature & Backwaters", "best_time": "Sep–Mar", "image": "", "source": "Fallback"},
        {"name": "Manali", "description": "Himalayan paradise for trekking, skiing and scenic valleys.", "rating": 4.4, "popularity": "High", "travel_suitability": "Adventure & Trekking", "best_time": "Mar–Jun", "image": "", "source": "Fallback"},
        {"name": "Varanasi", "description": "Spiritual heart of India on the holy Ganges river.", "rating": 4.3, "popularity": "High", "travel_suitability": "Spiritual & Religious", "best_time": "Oct–Mar", "image": "", "source": "Fallback"},
        {"name": "Leh Ladakh", "description": "High-altitude desert with monasteries and dramatic landscapes.", "rating": 4.5, "popularity": "High", "travel_suitability": "Adventure & Trekking", "best_time": "Jun–Sep", "image": "", "source": "Fallback"},
        {"name": "Udaipur", "description": "City of Lakes — romantic palaces and Rajasthani culture.", "rating": 4.5, "popularity": "High", "travel_suitability": "Heritage & Culture", "best_time": "Oct–Mar", "image": "", "source": "Fallback"},
        {"name": "Andaman Islands", "description": "Pristine beaches, coral reefs and turquoise waters.", "rating": 4.6, "popularity": "High", "travel_suitability": "Beach & Coastal", "best_time": "Nov–May", "image": "", "source": "Fallback"},
    ]


# ══════════════════════════════════════════════════════════════════════
# 3. RAG CHATBOT KNOWLEDGE BASE
# ══════════════════════════════════════════════════════════════════════

KNOWLEDGE_BASE = {
    "booking_process": (
        "To book a ticket on TicketHub: 1) Sign up or login. "
        "2) Go to Dashboard. 3) Search for tickets by type, source, destination. "
        "4) Click 'Book Now'. 5) Select a seat. 6) Complete payment (Card/UPI/Net Banking). "
        "7) Receive booking confirmation and PDF ticket by email."
    ),
    "payment_methods": (
        "TicketHub accepts: Credit Card, Debit Card, UPI (GPay/PhonePe/Paytm), Net Banking. "
        "All payments are securely processed."
    ),
    "ticket_types": (
        "TicketHub offers 5 ticket types: BUS (Volvo AC & Sleeper), TRAIN (Express & Rajdhani), "
        "FLIGHT (domestic), HOTEL (premium & budget rooms), CONCERT/EVENT (live shows)."
    ),
    "cancellation_policy": (
        "Cancellations must be requested within 24 hours of booking by contacting "
        "support@tickethub.com. Tickets are non-transferable."
    ),
    "account_management": (
        "Users can reset their password via OTP sent to registered email. "
        "Admin can block/unblock users and manage tickets from /admin/dashboard."
    ),
    "pdf_ticket": (
        "After booking, users can download their PDF ticket from the booking success page "
        "or from their dashboard under 'My Bookings'. A copy is also emailed automatically."
    ),
    "contact_support": (
        "Contact TicketHub support at support@tickethub.com or visit www.tickethub.com/help."
    ),
    "pricing": (
        "Ticket prices on TicketHub vary by type and route. "
        "BUS tickets start from ₹332. TRAIN from ₹712. FLIGHT from ₹2,090. "
        "HOTEL from ₹700/night. CONCERT from ₹800. "
        "Discounts of 5–20% are available on many tickets."
    ),
}

_KEYWORD_MAP = {
    "book":     "booking_process",
    "how to":   "booking_process",
    "purchase": "booking_process",
    "reserve":  "booking_process",
    "pay":      "payment_methods",
    "upi":      "payment_methods",
    "card":     "payment_methods",
    "net bank": "payment_methods",
    "type":     "ticket_types",
    "bus":      "ticket_types",
    "train":    "ticket_types",
    "flight":   "ticket_types",
    "hotel":    "ticket_types",
    "concert":  "ticket_types",
    "cancel":   "cancellation_policy",
    "refund":   "cancellation_policy",
    "password": "account_management",
    "login":    "account_management",
    "account":  "account_management",
    "otp":      "account_management",
    "pdf":      "pdf_ticket",
    "download": "pdf_ticket",
    "ticket":   "pdf_ticket",
    "support":  "contact_support",
    "contact":  "contact_support",
    "email":    "contact_support",
    "price":    "pricing",
    "cost":     "pricing",
    "cheap":    "pricing",
    "afford":   "pricing",
    "rate":     "pricing",
}


def rag_retrieve(query: str, db=None, tickets: list = None) -> str:
    """
    Retrieve relevant knowledge for a query using keyword matching.
    Returns context string to inject into the AI prompt.
    """
    q = query.lower()
    context_chunks = []
    seen_keys = set()

    for kw, kb_key in _KEYWORD_MAP.items():
        if kw in q and kb_key not in seen_keys:
            context_chunks.append(KNOWLEDGE_BASE[kb_key])
            seen_keys.add(kb_key)

    # Live ticket data injection
    if tickets and any(k in q for k in ["price", "cost", "cheap", "route", "available", "ticket", "bus", "train", "flight"]):
        lines = ["Live available tickets on TicketHub right now:"]
        for t in tickets[:10]:
            price    = float(t.get("price", 0))
            discount = float(t.get("discount", 0))
            final    = round(price * (1 - discount / 100), 2)
            disc_str = f" ({int(discount)}% OFF)" if discount > 0 else ""
            lines.append(
                f"• [{t.get('type','?')}] {t.get('title','?')} | "
                f"{t.get('source','?')} → {t.get('destination','?')} | "
                f"₹{final:,.0f}{disc_str} | {t.get('total_seats', '?')} seats"
            )
        context_chunks.append("\n".join(lines))

    return "\n\n".join(context_chunks) if context_chunks else ""
