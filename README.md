# movie-recommender
AI-powered movie recommender that learns your taste through a quick Q&amp;A and suggests the 3 best films for you, with real IMDb ratings.
🎬 AI Movie Recommender
A terminal-based movie recommendation system that combines machine learning with AI to find the perfect film for your mood.
Answer 6 quick questions about your favourite genres, era, mood, and themes — and the system will analyse your taste using the Groq AI API, match it against a dataset of 9,724 movies using TF-IDF and cosine similarity, and return the 3 best recommendations with real IMDb ratings and a personalised explanation for why each film suits you.
How it works

Your answers are parsed by an LLM into a structured preference profile
Each movie is encoded as a TF-IDF vector built from its genres and tags
Cosine similarity finds the closest matches to your taste
Results are filtered by year, rating, and popularity
Live IMDb ratings are fetched via the OMDB API

Tech stack
Python · pandas · scikit-learn · Groq API (Llama 4) · OMDB API
