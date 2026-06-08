"""
Movie Recommender System
========================
Uses TF-IDF + cosine similarity (KNN-style) to recommend movies
based on user preferences collected via terminal Q&A.
The Groq API (via OpenAI-compatible SDK) translates free-text answers
into a structured feature vector and generates per-movie explanations.
"""

import os
import sys
import json
import textwrap

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI
from dotenv import load_dotenv

# ─────────────────────────────────────────────
# 0. Configuration
# ─────────────────────────────────────────────

load_dotenv()  # reads GROQ_API_KEY from .env if present

GROQ_API_KEY  = os.getenv("GROQ_API_KEY")
GROQ_BASE_URL = "https://api.groq.com/openai/v1"
GROQ_MODEL    = "meta-llama/llama-4-scout-17b-16e-instruct"

DATA_PATH     = "movies.csv"
TOP_N         = 10          # final recommendations shown
CANDIDATE_N   = 200         # wider net before post-filtering


# ─────────────────────────────────────────────
# 1. Load & prepare the dataset
# ─────────────────────────────────────────────

def load_movies(path: str) -> pd.DataFrame:
    """Load movies.csv and build a combined text field for TF-IDF."""
    df = pd.read_csv(path)

    # Fill missing text fields so vectorisation doesn't choke
    df["genres"] = df["genres"].fillna("")
    df["tags"]   = df["tags"].fillna("")

    # Replace pipe separators with spaces so each token is independent
    df["genres_clean"] = df["genres"].str.replace("|", " ", regex=False)
    df["tags_clean"]   = df["tags"].str.replace("|", " ", regex=False)

    # Combined corpus field: genres get repeated twice to upweight them
    df["corpus"] = df["genres_clean"] + " " + df["genres_clean"] + " " + df["tags_clean"]

    # Ensure numeric columns are numeric (coerce bad values to NaN → 0)
    for col in ["year", "weighted_rating", "rating_count", "mean_rating"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    return df


# ─────────────────────────────────────────────
# 2. Terminal Q&A
# ─────────────────────────────────────────────

QUESTIONS = [
    ("genres",
     "1. What are your favourite genres?\n"
     "   (e.g. Action, Drama, Sci-Fi, Comedy, Horror, Romance — list as many as you like)\n"
     "   > "),

    ("era",
     "2. Which era or decade do you prefer?\n"
     "   (e.g. 80s, 90s, 2000s, 2010s, or 'any')\n"
     "   > "),

    ("mood",
     "3. What mood or vibe are you after?\n"
     "   (e.g. dark and psychological, feel-good, mind-blowing, action-packed, romantic, funny)\n"
     "   > "),

    ("quality",
     "4. How important is the rating quality?\n"
     "   (e.g. only highly rated, moderately rated is fine, doesn't matter)\n"
     "   > "),

    ("themes",
     "5. Any specific themes or keywords you enjoy?\n"
     "   (e.g. time travel, redemption, friendship, dystopia — free text)\n"
     "   > "),

    ("popularity",
     "6. Do you prefer popular blockbusters or hidden gems?\n"
     "   (type 'popular', 'hidden gem', or 'either')\n"
     "   > "),
]


def collect_answers() -> dict:
    """Print questions and collect raw user answers."""
    print("\n" + "=" * 60)
    print("  🎬  Welcome to the AI Movie Recommender")
    print("=" * 60)
    print("Answer a few quick questions and we'll find your perfect films.\n")

    answers = {}
    for key, prompt in QUESTIONS:
        answers[key] = input(prompt).strip()

    return answers


# ─────────────────────────────────────────────
# 3. Translate answers → structured feature vector (Groq)
# ─────────────────────────────────────────────

FEATURE_SYSTEM_PROMPT = """
You are a movie-preference parsing assistant.
The user has answered a short questionnaire about their film taste.
Your job is to output a single JSON object — nothing else, no markdown fences.

Schema:
{
  "genres": [<list of genre strings matching MovieLens genres>],
  "tags":   [<list of descriptive tag strings>],
  "year_range": [<start_year_int>, <end_year_int>],
  "min_weighted_rating": <float 0-5, or 0 if not important>,
  "popularity": "<high | low | any>"
}

Rules:
- genres must be from: Action, Adventure, Animation, Children, Comedy, Crime,
  Documentary, Drama, Fantasy, Film-Noir, Horror, IMAX, Musical, Mystery,
  Romance, Sci-Fi, Thriller, War, Western
- tags should be short descriptive phrases (2-4 words max each)
- year_range: convert decade descriptions (e.g. "90s" → [1990, 1999]);
  "any" → [1900, 2030]
- min_weighted_rating: "highly rated" → 3.8, "moderately" → 3.0, else 0
- popularity: "popular" → "high", "hidden gem" → "low", else "any"
Output ONLY the JSON object.
""".strip()


def parse_preferences(client: OpenAI, answers: dict) -> dict:
    """Send raw answers to Groq; receive a structured feature dict."""
    user_text = "\n".join(f"{k}: {v}" for k, v in answers.items())

    print("\n⏳ Analysing your preferences with AI...")

    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": FEATURE_SYSTEM_PROMPT},
            {"role": "user",   "content": user_text},
        ],
        temperature=0.2,
    )

    raw = response.choices[0].message.content.strip()

    # Strip accidental markdown fences if model adds them
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()

    try:
        prefs = json.loads(raw)
    except json.JSONDecodeError:
        print("⚠️  Could not parse AI response — using sensible defaults.")
        prefs = {
            "genres": [],
            "tags": [],
            "year_range": [1900, 2030],
            "min_weighted_rating": 0,
            "popularity": "any",
        }

    return prefs


# ─────────────────────────────────────────────
# 4. Build TF-IDF index & find candidates
# ─────────────────────────────────────────────

def build_tfidf_index(df: pd.DataFrame):
    """Fit a TF-IDF vectoriser on the movie corpus and return (vectoriser, matrix)."""
    vectoriser = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),   # unigrams + bigrams catch phrases like "twist ending"
        min_df=2,             # ignore tokens appearing in fewer than 2 movies
        sublinear_tf=True,    # apply log(1 + tf) to compress term frequency
    )
    tfidf_matrix = vectoriser.fit_transform(df["corpus"])
    return vectoriser, tfidf_matrix


def build_query_vector(vectoriser: TfidfVectorizer, prefs: dict):
    """Convert structured preferences into a TF-IDF query vector."""
    # Repeat genres twice to match how the corpus was built
    genre_text = " ".join(prefs.get("genres", []))
    tag_text   = " ".join(prefs.get("tags",   []))
    query_text = f"{genre_text} {genre_text} {tag_text}".strip()

    if not query_text:
        query_text = "movie"  # fallback so transform doesn't get an empty string

    return vectoriser.transform([query_text])


def find_candidates(df: pd.DataFrame, tfidf_matrix, query_vec, n: int) -> pd.DataFrame:
    """Return the top-n movies by cosine similarity to the query vector."""
    sims = cosine_similarity(query_vec, tfidf_matrix).flatten()
    top_idx = sims.argsort()[::-1][:n]
    candidates = df.iloc[top_idx].copy()
    candidates["similarity"] = sims[top_idx]
    return candidates


# ─────────────────────────────────────────────
# 5. Post-filter based on year, rating, popularity
# ─────────────────────────────────────────────

def post_filter(candidates: pd.DataFrame, prefs: dict, df: pd.DataFrame) -> pd.DataFrame:
    """Apply hard filters for year range, minimum rating and popularity."""
    year_min, year_max = prefs.get("year_range", [1900, 2030])
    min_rating         = prefs.get("min_weighted_rating", 0)
    popularity         = prefs.get("popularity", "any")

    # Year filter
    mask = (candidates["year"] >= year_min) & (candidates["year"] <= year_max)
    filtered = candidates[mask].copy()

    # Rating filter
    if min_rating > 0:
        filtered = filtered[filtered["weighted_rating"] >= min_rating]

    # Popularity filter — use median rating_count as the split point
    if popularity == "high":
        median_count = df["rating_count"].median()
        filtered = filtered[filtered["rating_count"] >= median_count]
    elif popularity == "low":
        median_count = df["rating_count"].median()
        filtered = filtered[filtered["rating_count"] < median_count]

    # If filtering was too aggressive, fall back to unfiltered candidates
    if len(filtered) < TOP_N:
        print("⚠️  Filters removed too many movies — relaxing constraints.")
        filtered = candidates.copy()

    # Final ranking: blend similarity score with weighted_rating
    filtered["score"] = (
        0.7 * filtered["similarity"] +
        0.3 * (filtered["weighted_rating"] / 5.0)  # normalise to [0,1]
    )
    return filtered.sort_values("score", ascending=False).head(TOP_N)


# ─────────────────────────────────────────────
# 6. Generate per-movie explanations (Groq)
# ─────────────────────────────────────────────

EXPLAIN_SYSTEM_PROMPT = """
You are a friendly film critic assistant.
Given a user's taste profile and a list of recommended movies,
write ONE short sentence (max 20 words) for each movie explaining
why it suits this particular viewer.
Output ONLY a JSON array of strings, one per movie, in the same order.
No markdown, no numbering — just the raw JSON array.
""".strip()


def generate_explanations(client: OpenAI, prefs: dict, recs: pd.DataFrame) -> list[str]:
    """Ask Groq to write a short 'why recommended' blurb per movie."""
    movie_list = "\n".join(
        f"{i+1}. {row['title']} — genres: {row['genres']}, tags: {row['tags'][:80]}"
        for i, (_, row) in enumerate(recs.iterrows())
    )
    taste_summary = (
        f"Genres: {', '.join(prefs.get('genres', []))}. "
        f"Tags/mood: {', '.join(prefs.get('tags', []))}. "
        f"Era: {prefs.get('year_range')}. "
        f"Popularity preference: {prefs.get('popularity')}."
    )

    print("⏳ Generating personalised explanations...")

    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": EXPLAIN_SYSTEM_PROMPT},
            {"role": "user",   "content": f"Viewer taste: {taste_summary}\n\nMovies:\n{movie_list}"},
        ],
        temperature=0.5,
    )

    raw = response.choices[0].message.content.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()

    try:
        explanations = json.loads(raw)
        if not isinstance(explanations, list):
            raise ValueError
    except (json.JSONDecodeError, ValueError):
        # Fallback: generic explanation for every movie
        explanations = ["Matches your stated preferences."] * len(recs)

    # Pad/trim to match the number of recommendations
    while len(explanations) < len(recs):
        explanations.append("Matches your stated preferences.")

    return explanations[:len(recs)]


# ─────────────────────────────────────────────
# 7. Pretty-print results
# ─────────────────────────────────────────────

def print_recommendations(recs: pd.DataFrame, explanations: list[str]) -> None:
    """Display final recommendations in a readable format."""
    print("\n" + "=" * 60)
    print("  🎬  Your Top 10 Movie Recommendations")
    print("=" * 60)

    for rank, ((_, row), reason) in enumerate(zip(recs.iterrows(), explanations), start=1):
        title    = row["title"]
        year     = int(row["year"]) if row["year"] else "?"
        genres   = row["genres"].replace("|", " · ")
        rating   = f"{row['weighted_rating']:.2f}" if row["weighted_rating"] else "N/A"
        reason   = textwrap.fill(reason, width=56, subsequent_indent="     ")

        print(f"\n{'─'*60}")
        print(f"  #{rank:02d}  {title}")
        print(f"        Year: {year}  |  Rating: {rating}/5")
        print(f"        Genres: {genres}")
        print(f"        💡 {reason}")

    print("\n" + "=" * 60)
    print("  Enjoy your movie night! 🍿")
    print("=" * 60 + "\n")


# ─────────────────────────────────────────────
# 8. Main entry-point
# ─────────────────────────────────────────────

def main():
    # ── Validate setup ──────────────────────────────────────
    if not GROQ_API_KEY:
        sys.exit(
            "❌  GROQ_API_KEY not found.\n"
            "    Set it in a .env file or as an environment variable and retry.\n"
            "    Get your key at: https://console.groq.com"
        )

    if not os.path.exists(DATA_PATH):
        sys.exit(
            f"❌  Dataset not found at '{DATA_PATH}'.\n"
            "    Place movies.csv in the same directory as this script."
        )

    # ── Initialise Groq client (OpenAI-compatible) ───────────
    client = OpenAI(api_key=GROQ_API_KEY, base_url=GROQ_BASE_URL)

    # ── Step 1: Load data ────────────────────────────────────
    print("\n📂 Loading movie dataset...")
    df = load_movies(DATA_PATH)
    print(f"   Loaded {len(df):,} movies.")

    # ── Step 2: Build TF-IDF index ───────────────────────────
    print("🔢 Building TF-IDF index...")
    vectoriser, tfidf_matrix = build_tfidf_index(df)
    print(f"   Vocabulary size: {len(vectoriser.vocabulary_):,} terms.")

    # ── Step 3: Collect user preferences ────────────────────
    answers = collect_answers()

    # ── Step 4: Parse answers into structured prefs ──────────
    prefs = parse_preferences(client, answers)
    print(f"\n✅ Structured preferences:\n   {json.dumps(prefs, indent=6)}")

    # ── Step 5: Find similar movies ──────────────────────────
    print(f"\n🔍 Finding best matches from {len(df):,} movies...")
    query_vec  = build_query_vector(vectoriser, prefs)
    candidates = find_candidates(df, tfidf_matrix, query_vec, n=CANDIDATE_N)

    # ── Step 6: Apply post-filters ───────────────────────────
    recs = post_filter(candidates, prefs, df)
    print(f"   Narrowed to {len(recs)} recommendations after filtering.")

    # ── Step 7: Generate explanations ────────────────────────
    explanations = generate_explanations(client, prefs, recs)

    # ── Step 8: Display results ──────────────────────────────
    print_recommendations(recs, explanations)


if __name__ == "__main__":
    main()
