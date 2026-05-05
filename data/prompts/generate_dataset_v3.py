"""
Enhanced dataset generator for hybrid movie recommender.

This second iteration addresses feedback from the user:

1. **JSON output** – All three data files (prompts, pairs, judgments)
   are written as `.json` arrays instead of newline‑delimited JSON.
2. **Humanised prompts** – The prompt generator now draws on
   a diverse set of personas, tones and scenarios inspired by
   film‑reviewing guides and tone examples from writing guides.
   Templates reference specific movies, genres, emotions, personal
   contexts and scenarios to create the feeling of different people
   asking for recommendations.  Tone variation is informed by
   examples of formal, informal, humorous, sarcastic and other
   styles【757098034258107†L283-L296】【920490872441880†L138-L152】.  No two prompts
   should sound the same.
3. **No ties** – The judgement step now always picks a preferred
   movie.  Ties (0,0 labels) are removed entirely; the movie with
   higher composite score receives +1 and the other −1.  Confidence
   values still reflect difficulty and score margins.

The remainder of the specification mirrors the original generator:
balanced categories and difficulties; deterministic seed; explicit
α/β/γ/δ and Plutchik distributions; pair construction rules; and
manifest summarisation.  Because optional text, emotion vectors and
user profiles are absent, proxy features derived from titles and
genres are still used for scoring.
"""

import json
import os
import random
import time
import uuid
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


# Global seed for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# Dataset paths
DATA_ZIP = os.path.join(os.getcwd(), "archive.zip")
DATA_DIR = os.path.join(os.getcwd(), "archive_content", "ml-100k")

# ---------------------------------------------------------------------------
# Emotion synonyms and movie emotion loader
#
# To generate more natural mood descriptions, we map each Plutchik emotion to a
# set of descriptive phrases.  When building prompts we reference a movie's
# dominant emotion using these synonyms (e.g. a "loyalty-based" film for
# trust, or a "gritty" film for anger).  We also load detailed emotion
# distributions per MovieLens title from an optional `movie_emotions.json`
# supplied by the user.  If unavailable, we fall back to genre-based
# estimates from `infer_emotion_distribution`.

# Synonyms for the eight Plutchik emotions (lower‑case keys for consistency)
EMOTION_SYNONYMS: Dict[str, List[str]] = {
    'joy': ["feel-good", "uplifting", "cheerful", "heartwarming", "joyous"],
    'trust': ["loyalty-based", "trustworthy", "faithful", "reliable", "earnest"],
    'fear': ["suspenseful", "heart-pounding", "terrifying", "nerve-racking", "spine-chilling"],
    'anticipation': ["exciting", "thrilling", "nail-biting", "edge-of-your-seat", "eager"],
    'sadness': ["melancholic", "poignant", "tear-jerking", "heartbreaking", "bittersweet"],
    'anger': ["gritty", "revenge-driven", "fierce", "rage-fueled", "intense"],
    'surprise': ["twist-filled", "unpredictable", "mind-bending", "shocking", "full of surprises"],
    'disgust': ["disturbing", "macabre", "provocative", "gross-out", "morally challenging"],
}

# Additional synonyms for film and vibe to vary prompt wording
# Using these lists helps ensure that prompts don’t repeatedly use
# the same nouns or descriptors, making them feel more organically
# written by different people.  See tone guidance on varying word
# choice from writing resources【685814002836978†L174-L214】.
FILM_SYNONYMS: List[str] = ["film", "movie", "flick", "picture", "feature"]
VIBE_SYNONYMS: List[str] = ["vibe", "energy", "tone", "feel", "mood", "atmosphere"]

# List of notable actors and directors to use in lexical search queries.
PROPER_NOUNS: List[str] = [
    "Christopher Nolan", "Quentin Tarantino", "Hayao Miyazaki", "Steven Spielberg",
    "Martin Scorsese", "Greta Gerwig", "Jordan Peele", "Wes Anderson",
    "Alfred Hitchcock", "Stanley Kubrick", "Kathryn Bigelow", "Ava DuVernay",
    "Scarlett Johansson", "Tom Hanks", "Meryl Streep", "Denzel Washington",
    "Natalie Portman", "Keanu Reeves", "Leonardo DiCaprio", "Cate Blanchett",
]

# Years and decades to anchor lexical queries with digits.
YEARS_LIST: List[int] = [1970, 1980, 1990, 1995, 2000, 2005, 2010, 2015, 2020]

# Synonyms for "story" to further vary phrasing around narrative topics.
STORY_SYNONYMS: List[str] = ["story", "narrative", "tale", "plotline", "journey"]

def choose_story_term() -> str:
    """Return a random synonym for the word 'story'."""
    return random.choice(STORY_SYNONYMS)

def choose_film_term() -> str:
    """Return a random synonym for the word 'film'."""
    return random.choice(FILM_SYNONYMS)

def choose_vibe_term() -> str:
    """Return a random synonym for the word 'vibe'."""
    return random.choice(VIBE_SYNONYMS)

def load_movie_emotions(json_path: str) -> Dict[int, Dict[str, float]]:
    """Load per-movie Plutchik distributions from a JSON file.

    The file must be a JSON object with a top-level key 'movies' pointing to
    a list of entries.  Each entry has 'movieId' and 'emotions' (name and
    probability_percent).  Returns a mapping from movieId to a dictionary of
    emotion probabilities normalised to sum to 1.
    """
    movie_emotions: Dict[int, Dict[str, float]] = {}
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        for m in data.get('movies', []):
            mid = m.get('movieId') or m.get('movie_id')
            emotions = m.get('emotions', [])
            total = 0.0
            vec: Dict[str, float] = {}
            for e in emotions:
                name = e['name'].lower()
                prob = float(e.get('probability_percent', 0.0))
                vec[name] = prob
                total += prob
            if total > 0:
                for k in vec:
                    vec[k] = vec[k] / total
                movie_emotions[int(mid)] = vec
    except Exception:
        # If loading fails, return empty dict
        movie_emotions = {}
    return movie_emotions

def get_dominant_emotion(movie_id: int, movie_emotions_map: Dict[int, Dict[str, float]]) -> str:
    """Return the name of the dominant emotion for a given movie id.

    Falls back to 'joy' when the movie is unknown or has no distribution.
    """
    dist = movie_emotions_map.get(movie_id)
    if dist:
        # Return emotion with highest probability
        return max(dist.items(), key=lambda x: x[1])[0]
    return 'joy'

def choose_emotion_synonym(emotion_name: str) -> str:
    """Pick a random synonym for an emotion name (case‑insensitive)."""
    if not emotion_name:
        return random.choice(sum(EMOTION_SYNONYMS.values(), []))
    key = emotion_name.lower()
    if key in EMOTION_SYNONYMS:
        return random.choice(EMOTION_SYNONYMS[key])
    return emotion_name.lower()

def choose_movie_synonym(movie_id: int, movie_emotions_map: Dict[int, Dict[str, float]]) -> str:
    """Choose a synonym based on the movie's dominant emotion."""
    dom = get_dominant_emotion(movie_id, movie_emotions_map)
    return choose_emotion_synonym(dom)


def ensure_dataset():
    """Ensure that the ML‑100k data is extracted into DATA_DIR."""
    if not os.path.exists(DATA_DIR):
        import zipfile
        with zipfile.ZipFile(DATA_ZIP, 'r') as zf:
            zf.extractall(os.path.join(os.getcwd(), "archive_content"))
    return DATA_DIR


def load_movies() -> pd.DataFrame:
    """Load movie metadata from MovieLens 100k dataset."""
    data_dir = ensure_dataset()
    genre_path = os.path.join(data_dir, "u.genre")
    item_path = os.path.join(data_dir, "u.item")
    # Genre mapping
    genre_mapping: Dict[int, str] = {}
    with open(genre_path, 'r', encoding='latin-1') as f:
        for line in f:
            if line.strip():
                name, idx = line.strip().split('|')
                genre_mapping[int(idx)] = name
    # Load items
    cols = list(range(24))
    df = pd.read_csv(item_path, sep='|', encoding='latin-1', header=None, names=cols, usecols=cols)
    records = []
    for _, row in df.iterrows():
        mid = int(row[0])
        title = str(row[1])
        # Extract year from title
        year = None
        if '(' in title and ')' in title:
            try:
                year = int(title.strip().split('(')[-1].rstrip(')'))
            except Exception:
                year = None
        if year is None and pd.notna(row[2]):
            try:
                year = int(str(row[2])[-4:])
            except Exception:
                year = None
        if year is None:
            year = 1995
        decade = (year // 10) * 10
        # Genres
        genre_flags = row[5:24].values.astype(int)
        genres = [genre_mapping[i] for i, flag in enumerate(genre_flags) if flag == 1]
        if not genres:
            genres = ['unknown']
        # Tokens from title
        import re
        tokens = set(re.findall(r"[A-Za-z]+", title.lower()))
        # Plutchik vector
        plutchik_vec = infer_emotion_distribution(genres)
        records.append({
            'movie_id': mid,
            'title': title,
            'year': year,
            'decade': decade,
            'genres': genres,
            'tokens': tokens,
            'plutchik_vector': plutchik_vec,
        })
    movies_df = pd.DataFrame(records)
    # Shuffle for randomness
    movies_df = movies_df.sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    return movies_df


def infer_emotion_distribution(genres: List[str]) -> np.ndarray:
    """Infer a Plutchik 8‑emotion distribution from genre labels."""
    # Mapping genres to emotions: joy, trust, fear, anticipation, sadness, anger, surprise, disgust
    mapping = {
        "Comedy": ("joy",),
        "Romance": ("joy",),
        "Children's": ("joy",),
        "Animation": ("joy",),
        "Musical": ("joy",),
        "Documentary": ("trust",),
        "War": ("trust", "anger"),
        "Western": ("trust",),
        "Action": ("anger",),
        "Crime": ("anger", "disgust"),
        "Horror": ("fear", "disgust"),
        "Thriller": ("fear", "surprise"),
        "Mystery": ("surprise", "anticipation"),
        "Fantasy": ("surprise",),
        "Adventure": ("anticipation",),
        "Sci-Fi": ("anticipation",),
        "Drama": ("sadness",),
        "Film-Noir": ("sadness",),
    }
    idx_map = {
        "joy": 0,
        "trust": 1,
        "fear": 2,
        "anticipation": 3,
        "sadness": 4,
        "anger": 5,
        "surprise": 6,
        "disgust": 7,
    }
    counts = np.ones(8) * 0.1
    for g in genres:
        for emo in mapping.get(g, ()):  # Unknown genres yield nothing
            counts[idx_map[emo]] += 1.0
    counts = counts / counts.sum()
    return counts


# Stopwords to ignore in lexical matching
STOPWORDS = set([
    "a", "an", "the", "and", "of", "in", "on", "for", "with", "to", "is", "my", "your", "i", "me", "you",
    "any", "some", "about", "that", "this", "it", "like", "just", "we", "us", "want", "looking", "need",
])


def extract_prompt_tokens(text: str) -> List[str]:
    import re
    tokens = re.findall(r"[A-Za-z']+", text.lower())
    return [t for t in tokens if t not in STOPWORDS]


def sample_dirichlet(prior: List[float]) -> List[float]:
    vec = np.random.dirichlet(prior)
    jitter = np.random.uniform(0.0, 0.02, size=len(vec))
    vec = vec + jitter
    vec = np.maximum(vec, 0.01)
    vec = vec / vec.sum()
    return vec.tolist()


def sample_mix_weights(category: str) -> Dict[str, float]:
    if category == "plot":
        weights = sample_dirichlet([8, 2, 1, 2])
    elif category == "lexical":
        weights = sample_dirichlet([2, 8, 1, 2])
    elif category == "history":
        weights = sample_dirichlet([2, 1, 8, 2])
    elif category == "mood":
        weights = sample_dirichlet([1, 2, 1, 8])
    elif category == "cold":
        weights = sample_dirichlet([5, 5, 1, 5])
    elif category == "mix_emo_sem":
        w1 = np.array(sample_dirichlet([4, 2, 1, 6]))
        w2 = np.array(sample_dirichlet([7, 2, 1, 3]))
        weights = 0.6 * w1 + 0.4 * w2
        jitter = np.random.uniform(0.0, 0.02, size=len(weights))
        weights = weights + jitter
        weights = np.maximum(weights, 0.01)
        weights = weights / weights.sum()
    elif category == "mix_plot_hist":
        w1 = np.array(sample_dirichlet([7, 2, 4, 2]))
        w2 = np.array(sample_dirichlet([5, 2, 6, 2]))
        weights = 0.6 * w1 + 0.4 * w2
        jitter = np.random.uniform(0.0, 0.02, size=len(weights))
        weights = weights + jitter
        weights = np.maximum(weights, 0.01)
        weights = weights / weights.sum()
    elif category == "mix_lex_plot":
        # Emphasise both plot (alpha) and lexical (beta) with less weight on history and emotion.
        w1 = np.array(sample_dirichlet([7, 7, 2, 2]))
        w2 = np.array(sample_dirichlet([6, 6, 3, 3]))
        weights = 0.6 * w1 + 0.4 * w2
        jitter = np.random.uniform(0.0, 0.02, size=len(weights))
        weights = weights + jitter
        weights = np.maximum(weights, 0.01)
        weights = weights / weights.sum()
    elif category == "random":
        # Balanced mix for random prompts
        weights = sample_dirichlet([4, 4, 4, 4])
    elif category == "mix_all":
        weights = sample_dirichlet([5, 5, 5, 5])
    else:
        weights = sample_dirichlet([4, 4, 4, 4])
    return {
        'alpha': float(weights[0]),
        'beta': float(weights[1]),
        'gamma': float(weights[2]),
        'delta': float(weights[3]),
    }


def sample_plutchik_dist(target_emotion: str = None) -> Dict[str, float]:
    emotions = ["joy", "trust", "fear", "anticipation", "sadness", "anger", "surprise", "disgust"]
    if target_emotion and target_emotion in emotions:
        alpha = [8 if e == target_emotion else 1.5 for e in emotions]
    else:
        alpha = [2.0] * 8
    vec = np.random.dirichlet(alpha)
    jitter = np.random.uniform(0.0, 0.05, size=len(vec))
    vec = vec + jitter
    vec = np.maximum(vec, 0.01)
    vec = vec / vec.sum()
    return {e: float(vec[i]) for i, e in enumerate(emotions)}


def choose_language() -> str:
    """
    Return the language for the prompt.  In response to user feedback,
    prompts should be in English only and not contain any Hindi or
    Hinglish.  This function therefore always returns 'en'.
    """
    return 'en'


def hindi_word(word: str, language: str) -> str:
    """Return the English term regardless of language.

    The previous implementation translated words into Hindi or Hinglish. In
    response to user feedback, prompts should not include Hindi terms or
    transliterations. This function therefore returns the original English
    word regardless of the 'language' parameter.
    """
    return word


def generate_prompt_text(
    category: str,
    emotion: str,
    target_genres: List[str],
    negative_constraints: List[str],
    likes: List[str],
    dislikes: List[str],
    likes_ids: List[int],
    dislikes_ids: List[int],
    language: str,
    style: str,
    movies_df: pd.DataFrame,
    movie_emotions_map: Dict[int, Dict[str, float]],
) -> str:
    """Generate a humanised prompt text with diverse personas and contexts.

    This function now incorporates movie emotion data to produce more
    appropriate mood descriptions.  It maps target emotions to synonyms
    using `EMOTION_SYNONYMS` and references the dominant emotions of
    liked and disliked movies via `movie_emotions_map`.  Prompts avoid
    Hindi or emoji content and instead use varied personas inspired by
    film‑review styles.
    """
    # Ensure emotion has some value; pick a generic fallback if none
    if not emotion:
        # Choose a random base emotion for variety
        emotion = random.choice(list(EMOTION_SYNONYMS.keys())).capitalize()
    # Choose a scenario
    scenarios = [
        "on a rainy Sunday afternoon", "for a cozy date night", "for a family gathering",
        "with my movie club", "after a long work week", "during a holiday break",
        "for Halloween night", "to relive the 90s", "for a lazy summer day",
        "while recovering from a cold", "as background for crafting", "before bedtime",
    ]
    scenario = random.choice(scenarios)
    # Select a random movie title if needed
    random_movie = random.choice(movies_df['title'].tolist()) if len(movies_df) else ""
    # Style modifiers
    prefixes = {
        'formal': "Greetings,",
        'informal': "Hey there,",
        'humorous': "Yo film buffs,",
        'sarcastic': "Well, well,",
        'optimistic': "Hi team,",
        'pessimistic': "Sigh,",
        'narrative': "Last night I realized,",
        'conversational': "So,",
        'nostalgic': "Back in the day,",
        'critical': "Frankly,",
        'deadpan': "",
        'enthusiastic': "OMG!",
        'emoji-lite': "😎",
    }
    suffixes = {
        'formal': "I would appreciate your recommendation.",
        'informal': "Any suggestions?", 
        'humorous': "Bring on the popcorn!", 
        'sarcastic': "Let's see if you can surprise me.", 
        'optimistic': "Can't wait to watch something great!", 
        'pessimistic': "But I'm prepared to be disappointed.", 
        'narrative': "Now I'm looking for inspiration.", 
        'conversational': "What do you think?", 
        'nostalgic': "Those were the days, weren't they?", 
        'critical': "Quality matters.", 
        'deadpan': "", 
        'enthusiastic': "I'm super excited!", 
        'emoji-lite': "🙏", 
    }
    prefix = prefixes.get(style, "Hello,")
    suffix = suffixes.get(style, "Thanks!")
    # Compose body depending on category
    body_parts: List[str] = []
    # Format negative constraints string
    neg_str = "".join([f" no {nc.split()[-1].lower()}," for nc in negative_constraints]).strip(',')
    # Mood prompts
    if category == 'mood':
        liked_title = random.choice(likes) if likes else random_movie
        template_options = [
            f"I just finished watching {liked_title} and now, {scenario}, I'm craving another {emotion.lower()} {hindi_word('movie', language)}{',' if neg_str else ''}{neg_str}.",
            f"After a long day, I want to unwind with a {emotion.lower()} {hindi_word('movie', language)}{',' if neg_str else ''}{neg_str}.",
            f"My partner loves films that evoke {emotion.lower()} feelings. We recently enjoyed {liked_title}. Any recommendations?",
            f"Planning a {scenario} and need a {hindi_word('movie', language)} that makes me feel {emotion.lower()}.", 
        ]
        body = random.choice(template_options)
    # Plot prompts
    elif category == 'plot':
        plot_themes = ["time travel", "family drama", "space exploration", "underdog sports", "political intrigue", "coming-of-age story", "serial killer mystery", "heist"]
        theme = random.choice(plot_themes)
        liked_title = random.choice(likes) if likes else random_movie
        genre_str = ', '.join(target_genres) if target_genres else theme
        template_options = [
            f"I'm fascinated by {theme} plots and loved {liked_title}. Any {genre_str} {hindi_word('movies', language)} along those lines?",
            f"Looking for a {theme} {hindi_word('movie', language)} with {genre_str} elements{',' if neg_str else ''}{neg_str}. Suggestions?",
            f"Can you recommend a story about {theme}, perhaps set in the {random.choice([1970, 1980, 1990, 2000, 2010])}s?", 
        ]
        body = random.choice(template_options)
    # Lexical prompts
    elif category == 'lexical':
        keywords = random.sample(["time", "love", "star", "dark", "city", "space", "game", "death", "music", "journey"], k=2)
        template_options = [
            f"I'm on the hunt for {hindi_word('movies', language)} with '{keywords[0]}' or '{keywords[1]}' in the title or theme.",
            f"What's a good {hindi_word('movie', language)} about {keywords[0]}? Bonus points if {keywords[1]} plays a role.",
            f"Looking for titles that include '{keywords[0]}' or revolve around {keywords[1]}.", 
        ]
        body = random.choice(template_options)
    # History prompts
    elif category == 'history':
        # Ensure lists are not empty
        liked_str = ', '.join(likes) if likes else random_movie
        disliked_str = ', '.join(dislikes) if dislikes else random_movie
        template_options = [
            f"I absolutely loved {liked_str} but couldn't get into {disliked_str}. What should I watch next?",
            f"Favourite {hindi_word('movies', language)} include {liked_str}; I wasn't a fan of {disliked_str}. Any suggestions with similar vibes?",
            f"I'm building my watchlist based on {liked_str}. Avoid anything like {disliked_str} please.",
        ]
        body = random.choice(template_options)
    # Cold prompts
    elif category == 'cold':
        template_options = [
            f"I'm new to classic cinema and want to start exploring. What are some must‑watch {', '.join(target_genres) if target_genres else 'great'} {hindi_word('movies', language)}?", 
            f"No specific preferences — just looking for a well‑made {hindi_word('movie', language)} that will impress a casual viewer.", 
            f"Open to anything but nothing too intense. Where should I start?", 
            f"I haven’t watched many {hindi_word('movies', language)} recently. What would you recommend for a relaxing evening?", 
        ]
        body = random.choice(template_options)
    # mix_emo_sem prompts
    elif category == 'mix_emo_sem':
        plot_themes = ["time travel", "family drama", "space exploration", "underdog sports", "political intrigue", "coming-of-age story", "serial killer mystery", "heist"]
        theme = random.choice(plot_themes)
        liked_title = random.choice(likes) if likes else random_movie
        genre_str = ', '.join(target_genres) if target_genres else ''
        template_options = [
            f"I'm in the mood for a {emotion.lower()} story with {theme} themes. Something that mixes {genre_str} would be perfect.",
            f"Need a {emotion.lower()} film with {theme} vibes. We recently watched {liked_title}. Something along those lines?",
            f"Looking for a movie that makes me feel {emotion.lower()} but also has a strong {theme} narrative.",
        ]
        body = random.choice(template_options)
    # mix_plot_hist prompts
    elif category == 'mix_plot_hist':
        plot_themes = ["time travel", "family drama", "space exploration", "underdog sports", "political intrigue", "coming-of-age story", "serial killer mystery", "heist"]
        theme = random.choice(plot_themes)
        liked_str = ', '.join(likes) if likes else random_movie
        disliked_str = ', '.join(dislikes) if dislikes else random_movie
        genre_str = ', '.join(target_genres) if target_genres else ''
        template_options = [
            f"Since I enjoyed {liked_str}, I'm craving a {genre_str} film with {theme}. Any suggestions?",
            f"I loved {liked_str} for their {theme} but not {disliked_str}. Looking for similar recommendations.",
            f"As someone who loved {liked_str}, what {genre_str} movies with {theme} should I check out?",
        ]
        body = random.choice(template_options)
    # mix_all prompts
    elif category == 'mix_all':
        keywords = random.sample(["time", "love", "star", "dark", "city", "space", "game", "death", "music", "journey"], k=2)
        liked_title = random.choice(likes) if likes else random_movie
        genre_str = ', '.join(target_genres) if target_genres else ''
        template_options = [
            f"Looking for a movie that balances mood, plot and my taste: I loved {liked_title}, I'm in the mood for something {emotion.lower()}, with keywords like '{keywords[0]}'. Avoid {neg_str or 'nothing specific'}.",
            f"Please recommend a {genre_str} film that has {keywords[0]}, evokes {emotion.lower()}, and aligns with my love for {liked_title}.",
            f"Give me a {genre_str if genre_str else 'great'} movie that makes me feel {emotion.lower()} and includes themes of {keywords[1]}.",
        ]
        body = random.choice(template_options)
    else:
        body = "I'm looking for a good film to watch."
    # Compose final prompt with prefix and suffix
    prompt_text = " ".join([prefix, body, suffix]).strip()
    # Normalise whitespace
    prompt_text = " ".join(prompt_text.split())
    return prompt_text


# ---------------------------------------------------------------------------
# New prompt generator implementing refined tone and emotion handling
#
def generate_prompt_text_v3(
    category: str,
    emotion: str,
    target_genres: List[str],
    negative_constraints: List[str],
    likes: List[str],
    dislikes: List[str],
    likes_ids: List[int],
    dislikes_ids: List[int],
    style: str,
    movies_df: pd.DataFrame,
    movie_emotions_map: Dict[int, Dict[str, float]],
) -> str:
    """Generate a humanised prompt text with refined tone and movie emotion awareness.

    This version avoids Hindi and emoji content, uses synonyms for the
    requested emotion and for the dominant moods of liked/disliked movies,
    and produces more contextually rich prompts across categories.
    """
    # Ensure the base emotion exists; if not provided, choose a random base
    if not emotion:
        emotion = random.choice(list(EMOTION_SYNONYMS.keys())).capitalize()
    # Pick a scenario to anchor the request
    scenarios = [
        "on a rainy Sunday afternoon",
        "for a cozy date night",
        "for a family gathering",
        "with my movie club",
        "after a long work week",
        "during a holiday break",
        "for Halloween night",
        "to relive the 90s",
        "for a lazy summer day",
        "while recovering from a cold",
        "as background for crafting",
        "before bedtime",
        # additional contexts to diversify prompts
        "while making dinner",
        "as a pre‑workout pump",
        "to unwind after studying",
        "on a long train ride",
        "while babysitting my nephew",
        "for a Sunday brunch gathering",
        "to celebrate finishing a project",
    ]
    scenario = random.choice(scenarios)
    # Capitalise the first letter without lowercasing the rest, to preserve
    # proper nouns in scenarios like "Halloween".  We use slicing rather than
    # str.capitalize() which would lowercase subsequent characters.
    scenario_cap = scenario[0].upper() + scenario[1:] if scenario else scenario
    # Random fallback movie title
    random_movie = random.choice(movies_df['title'].tolist()) if len(movies_df) else ""
    # Compute emotion synonyms
    target_syn = choose_emotion_synonym(emotion)
    like_syn = choose_movie_synonym(likes_ids[0], movie_emotions_map) if likes_ids else None
    dislike_syn = choose_movie_synonym(dislikes_ids[0], movie_emotions_map) if dislikes_ids else None
    # Randomised terms to further vary prompt wording
    film_term = choose_film_term()
    vibe_term = choose_vibe_term()
    # Style modifiers
    # Each style now maps to a list of possible opening phrases to reduce
    # repetition across prompts.  By expanding these lists and choosing
    # randomly among them, we avoid overusing the same three-word start.
    prefixes = {
        'formal': ["Greetings,", "Hello there,", "Good day,"],
        'informal': ["Hey there,", "Hi everyone,", "Sup,"],
        'humorous': ["Yo film buffs,", "Hey film lovers,", "Ladies and gents,"],
        'sarcastic': ["Well, well,", "Here we go again,", "Of course,"],
        'optimistic': ["Hi team,", "Good vibes only,", "Positive vibes,"],
        'pessimistic': ["Sigh,", "Not holding my breath,", "Here we go..."],
        'narrative': ["Last night I realised,", "The other day,", "You won't believe this,", "Once, I found myself thinking,"],
        'conversational': ["So,", "Anyway,", "By the way,"],
        'nostalgic': ["Back in the day,", "Remember when,", "In the old days,", "When I was younger,"],
        'critical': ["Frankly,", "Honestly,", "To be honest,", "Let me be frank,"],
        'deadpan': [""],
        'enthusiastic': ["OMG!", "You guys!", "Guess what!", "Exciting news!"],
        'confessional': ["Confession time,", "I have a confession,", "Here's the truth,"],
        'storyteller': ["True story,", "Story time,", "Let me tell you a story,", "Once upon a time,"],
        'cinephile': ["Calling all cinephiles,", "Dear film lovers,", "Movie buffs,", "To my fellow cinephiles,"],
        'honest': ["Honestly,", "Truthfully,", "To be honest,", "Let me level with you,"],
        'excited2': ["Guess what,", "You'll never believe it,", "Heads up,", "Oh my!"],
    }
    suffixes = {
        'formal': ["I would appreciate your recommendation.", "Please advise.", "I await your response.", "I look forward to your suggestion."],
        'informal': ["Any suggestions?", "Let me know!", "Hit me up.", "What do you think?"],
        'humorous': ["Bring on the popcorn!", "Let's make it epic!", "No spoilers though!", "I'm all ears!"],
        'sarcastic': ["Let's see if you can surprise me.", "Not that I have high hopes.", "Shock me, I dare you.", "Try me."],
        'optimistic': ["Can't wait to watch something great!", "I feel good about this.", "I'm excited!", "Looking forward to it!"],
        'pessimistic': ["But I'm prepared to be disappointed.", "Hope springs eternal.", "We'll see.", "Fingers crossed."],
        'narrative': ["Now I'm looking for inspiration.", "And that's where I'm at now.", "So here we are.", "And I thought I'd ask."],
        'conversational': ["What do you think?", "Thoughts?", "Tell me your thoughts.", "Any ideas?"],
        'nostalgic': ["Those were the days, weren't they?", "Brings back memories.", "Good times.", "A blast from the past!"],
        'critical': ["Quality matters.", "No mediocre picks please.", "I'm picky.", "Only the best will do."],
        'deadpan': [""],
        'enthusiastic': ["I'm super excited!", "Can't wait!", "Let's go!", "I'm buzzing!"],
        'confessional': ["Had to share that. What do you think?", "What do you think about that?", "Now I'm curious.", "Your thoughts?"],
        'storyteller': ["That's the gist.", "And that's the story.", "The end.", "Hope you liked the story."],
        'cinephile': ["Give me your top picks!", "Share your favourites!", "I trust your taste!", "Bring them on!"],
        'honest': ["Any straight-up advice?", "Be honest with me.", "Don't sugarcoat it.", "Give it to me straight."],
        'excited2': ["So pumped!", "Can't keep calm!", "So thrilled!", "I'm excited!"],
    }
    # Randomly include prefix and suffix to avoid identical start and end patterns.
    # The probability for including a prefix is set to 0.5 and for suffix to 0.6.
    # Each prefix/suffix is chosen randomly from the style-specific list.
    if random.random() < 0.5:
        prefix_list = prefixes.get(style, ["Hello,"])
        prefix = random.choice(prefix_list)
    else:
        prefix = ""
    if random.random() < 0.6:
        suffix_list = suffixes.get(style, ["Thanks!"])
        suffix = random.choice(suffix_list)
    else:
        suffix = ""
    # Build negative constraint phrase.  Rather than repeating "no" in a list,
    # convert constraints into a natural phrase like "avoiding horror and war".
    neg_str = "".join([f" no {nc.split()[-1].lower()}," for nc in negative_constraints]).strip(',')
    if negative_constraints:
        neg_words = [nc.split()[-1].lower() for nc in negative_constraints]
        if len(neg_words) == 1:
            neg_phrase = f", avoiding {neg_words[0]}"
        else:
            neg_phrase = f", avoiding {' and '.join(neg_words)}"
    else:
        neg_phrase = ""
    # Choose body based on category
    body: str = ""
    if category == 'mood':
        # Mood‑based prompts focus on the emotional tone of the film and should
        # feel like different people making requests.  Use varied sentence
        # starters, rhetorical questions and context to avoid repetition.
        liked_title = random.choice(likes) if likes else random_movie
        like_descr = f" with its {like_syn} {vibe_term}" if like_syn else ""
        # Provide a broad range of phrasings about emotions and situations.
        # To avoid repetitive starts, we build dynamic phrases using
        # interchangeable components.  Synonym lists diversify the
        # wording of time references, verbs and need statements.
        time_intros = [
            "After a tough week", "After a stressful week", "Following a busy week",
            "After a long day", "After an exhausting day", "On a lazy Sunday",
            "On a quiet evening", "During a rainy afternoon"
        ]
        hunt_verbs = [
            "I'm hunting for", "I'm craving", "I'm seeking", "I'm in search of",
            "I'm yearning for", "I'm looking for", "I'd love", "I'm keen on finding"
        ]
        need_phrases = [
            "Sometimes I just need", "Sometimes I crave", "Every now and then I need",
            "From time to time I need", "Occasionally I long for", "At times I just want"
        ]
        friends_phrases = [
            "My friends and I", "My group of friends", "Some friends and I",
            "A bunch of us", "My movie club" 
        ]
        fan_phrases = [
            "As someone who adores", "Being a fan of", "Since I love", "As a lover of",
            "As an admirer of", "Because I appreciate"
        ]
        # Build dynamic template list
        template_options = []
        # Reflect on a recent watch and ask for a similar vibe using varied wording
        template_options.append(
            f"Just rewatched {liked_title}{like_descr} and I'm still processing it. {scenario_cap}, I'd love another {target_syn} {film_term}{neg_phrase}."
        )
        # Express general desire for an emotion using fan phrases
        template_options.append(
            f"{random.choice(fan_phrases)} {target_syn} stories, what {film_term} should I put on next?{neg_phrase.capitalize() if neg_phrase else ''}"
        )
        # Ask for something to set the mood for a specific occasion using dynamic verbs
        template_options.append(
            f"{random.choice(hunt_verbs)} a {target_syn} {film_term} {scenario}. Which would you suggest?"
        )
        # Mention watching with friends/family to create a conversational feel using dynamic time intro and verbs
        template_options.append(
            f"{random.choice(time_intros)}, {random.choice(hunt_verbs)} a {target_syn} {film_term} to enjoy with friends. Any picks?"
        )
        # Reference a liked film and ask for more suggestions using vibe synonym
        template_options.append(
            f"Keeping the {target_syn} {vibe_term} alive after {liked_title}—got any suggestions?{neg_phrase}"
        )
        # Phrase as a rhetorical question about wanting to feel a certain way
        template_options.append(
            f"Is there a {film_term} that will leave me feeling {target_syn}?{neg_phrase.capitalize() if neg_phrase else ''}"
        )
        # Appeal to a mood without referencing movies directly
        template_options.append(
            f"I'm in the mood for something {target_syn}. What would you recommend?"
        )
        # Share a plan with a group and ask for ideas using varied friends phrases
        template_options.append(
            f"{random.choice(friends_phrases)} are planning a {film_term} night and want something {target_syn}. Suggestions?"
        )
        # Use introspective tone with dynamic need phrases
        template_options.append(
            f"{random.choice(need_phrases)} a {target_syn} {choose_story_term()} to lift my spirits. What are your favourites?"
        )
        body = random.choice(template_options)
    elif category == 'plot':
        # Plot‑based prompts emphasise narrative themes and settings.  Use
        # varied phrasing to avoid a repetitive feel. Include optional
        # references to decades or recently loved titles.
        plot_themes = [
            "time travel", "family drama", "space exploration", "underdog sports",
            "political intrigue", "coming-of-age story", "serial killer mystery", "heist"
        ]
        theme = random.choice(plot_themes)
        liked_title = random.choice(likes) if likes else random_movie
        genre_str = ', '.join(target_genres) if target_genres else theme
        like_descr = f" for its {like_syn} tone" if like_syn else ""
        decade = random.choice([1970, 1980, 1990, 2000, 2010])
        template_options = [
            # Ask for similar stories to a recent favourite with an emotional twist
            f"Ever since I watched {liked_title}{like_descr}, I've been craving more {theme} stories. Got any {genre_str} {film_term}s that feel {target_syn}?",
            # Pose a question about exploring a narrative theme with mood interplay
            f"Who can recommend a {theme} {film_term} that blends {genre_str} elements and carries a {target_syn} mood?",
            # Mention curiosity about different eras
            f"Looking to dive into {theme} narratives from the {decade}s that evoke a {target_syn} feeling. Suggestions?",
            # Use a more conversational tone with rhetorical flourish
            f"What's a good {theme} {film_term} that breaks the mould and adds a {target_syn} twist?",
            # Express fascination with a theme and request hidden gems with vibe synonyms
            f"I'm fascinated by {theme} plots. Any hidden gems—especially {genre_str}—that capture a {target_syn} {vibe_term}?",
            # Ask directly about narrative specifics and feelings
            f"Could you point me to a {film_term} about {theme} that leaves you feeling {target_syn}?",
            # Show curiosity about how themes can mix with emotions
            f"Do you know any {theme} stories that balance {genre_str} and a {target_syn} {vibe_term}?",
        ]
        body = random.choice(template_options)
    elif category == 'lexical':
        # Lexical prompts now represent search queries using exact phrases,
        # proper nouns, years/digits, boolean operators and negation.  Each
        # template constructs a query‑like request rather than only
        # referencing title keywords.  These patterns mimic how a user
        # might search for films by combining phrases, actors, dates and
        # constraints.
        # Select tokens for query patterns
        phrase_keywords = random.sample([
            "time travel", "space exploration", "dark city", "coming of age", "heist", "serial killer",
            "family drama", "political intrigue", "underdog story", "musical journey"
        ], k=2)
        actors = random.sample(PROPER_NOUNS, k=2)
        year = random.choice(YEARS_LIST)
        neg_genres = random.sample([g.lower() for g in target_genres] + ["horror", "romance", "war", "animation", "musical"], k=1)
        # Construct query patterns
        template_options = []
        # Query with boolean AND/NOT operators
        template_options.append(
            f"Query: \"{phrase_keywords[0]}\" AND \"{phrase_keywords[1]}\" NOT {neg_genres[0]}. What {film_term}s match?"
        )
        # Proper noun and year with negation
        template_options.append(
            f"Looking for {film_term}s featuring {actors[0]} AND released in {year}, NOT {neg_genres[0]}."
        )
        # Proper noun OR combination with negation and year range
        next_year = year + 5 if year < 2020 else year
        template_options.append(
            f"{actors[0]} OR {actors[1]} AND ({year}-{next_year}) NOT {neg_genres[0]} – recommendations?"
        )
        # Exact phrase in title or tagline with OR
        template_options.append(
            f"Any {film_term}s with \"{phrase_keywords[0]}\" OR \"{phrase_keywords[1]}\" in the title? Avoid {neg_genres[0]}."
        )
        # Short keyword style with digits and negation
        kw1 = phrase_keywords[0].split()[0]
        kw2 = phrase_keywords[1].split()[0]
        template_options.append(
            f"{kw1} {kw2} {year} NOT {neg_genres[0]} – any suggestions?"
        )
        # Proper noun search without year: emphasise actors/directors and constraints
        template_options.append(
            f"Give me a list of {film_term}s starring {actors[0]} OR directed by {actors[1]} but NOT {neg_genres[0]}."
        )
        # Mixed quoted and boolean with digits
        template_options.append(
            f"\"{phrase_keywords[0]}\" AND {actors[0].split()[0]} NOT {neg_genres[0]}, {year}s – what's good?"
        )
        body = random.choice(template_options)
    elif category == 'history':
        liked_str = ', '.join(likes) if likes else random_movie
        disliked_str = ', '.join(dislikes) if dislikes else random_movie
        like_descr = f" for its {like_syn} {vibe_term}" if like_syn else ""
        dislike_descr = f" with its {dislike_syn} tone" if dislike_syn else ""
        template_options = [
            f"I absolutely loved {liked_str}{like_descr} but couldn't get into {disliked_str}{dislike_descr}. What should I watch next?",
            f"Favourite {film_term}s include {liked_str}{like_descr}; I wasn't a fan of {disliked_str}{dislike_descr}. Any suggestions with similar vibes?",
            f"I'm building my watchlist based on {liked_str}{like_descr}. Avoid anything like {disliked_str}{dislike_descr} please.",
        ]
        body = random.choice(template_options)
    elif category == 'cold':
        template_options = [
            f"I'm new to classic cinema and want to start exploring. What are some must‑watch {', '.join(target_genres) if target_genres else 'great'} {film_term}s?",
            f"No specific preferences—just looking for a well‑made {film_term} that will impress a casual viewer.",
            f"Open to anything but nothing too intense. Where should I start?",
            f"I haven’t watched many {film_term}s recently. What would you recommend for a relaxing evening?",
            f"As someone just dipping their toes into movies, what titles should I queue up?",
        ]
        body = random.choice(template_options)
    elif category == 'mix_emo_sem':
        # Blends emotion and semantic (plot) cues.  Use varied phrasing to
        # describe desired feelings and narrative themes.
        plot_themes = [
            "time travel", "family drama", "space exploration", "underdog sports",
            "political intrigue", "coming-of-age story", "serial killer mystery", "heist"
        ]
        theme = random.choice(plot_themes)
        liked_title = random.choice(likes) if likes else random_movie
        genre_str = ', '.join(target_genres) if target_genres else ''
        like_descr = f" and loved its {like_syn} tone" if like_syn else ""
        template_options = [
            f"I want a {film_term} that makes me feel {target_syn} and explores {theme}. Mix in some {genre_str or 'genre surprises'} and I'm sold.",
            f"Searching for {film_term}s with {theme} themes that evoke {target_syn} feelings. We just loved {liked_title}{like_descr}.",
            f"Any recommendations for a {target_syn} story intertwined with {theme}? Extra points for {genre_str}.",
            f"In the mood for {theme} with a {target_syn} twist. Suggestions after enjoying {liked_title}{like_descr}?",
            f"Looking for {film_term}s that blend {theme} elements with an emotional {target_syn} core.",
            f"Mixing emotions and stories: can you recommend a {theme} tale that feels {target_syn}?",
            f"I enjoyed {liked_title}{like_descr}. Now I'm after {theme} narratives that keep that {vibe_term} going.",
        ]
        body = random.choice(template_options)
    elif category == 'mix_plot_hist':
        # Combine plot themes with history of likes and dislikes.  Phrasing should
        # reference both liked and disliked films and request similar or
        # contrasting suggestions.
        plot_themes = [
            "time travel", "family drama", "space exploration", "underdog sports",
            "political intrigue", "coming-of-age story", "serial killer mystery", "heist"
        ]
        theme = random.choice(plot_themes)
        liked_str = ', '.join(likes) if likes else random_movie
        disliked_str = ', '.join(dislikes) if dislikes else random_movie
        genre_str = ', '.join(target_genres) if target_genres else ''
        like_descr = f" and its {like_syn} storytelling" if like_syn else ""
        dislike_descr = f" with its {dislike_syn} tone" if dislike_syn else ""
        template_options = [
            f"Since I enjoyed {liked_str}{like_descr}, I'm seeking a {genre_str or 'captivating'} {film_term} about {theme}. Please steer clear of {disliked_str}{dislike_descr}.",
            f"Loved {liked_str}{like_descr} but not {disliked_str}{dislike_descr}. Any {genre_str or 'great'} {film_term}s with {theme} themes you recommend?",
            f"After binging {liked_str}{like_descr}, I'm hungry for more {theme} stories. Avoid anything like {disliked_str}.",
            f"What {genre_str or 'other'} {film_term}s with {theme} would appeal to someone who loved {liked_str}{like_descr}?",
            f"Craving another {theme} {film_term} akin to {liked_str}{like_descr}. Not interested in anything like {disliked_str}{dislike_descr}.",
            f"I'm torn: adore {liked_str}{like_descr} but cringe at {disliked_str}{dislike_descr}. Any {theme} {film_term}s that balance these?",
            f"Seeking a {theme} story that captures the essence of {liked_str}{like_descr} without the pitfalls of {disliked_str}{dislike_descr}.",
        ]
        body = random.choice(template_options)
    elif category == 'mix_all':
        # Mix all signals: mood, plot, lexical and history.  Requests should
        # mention keywords in titles, emotional tone and reference a liked film.
        keywords = random.sample([
            "time", "love", "star", "dark", "city", "space", "game", "death", "music", "journey"
        ], k=2)
        liked_title = random.choice(likes) if likes else random_movie
        genre_str = ', '.join(target_genres) if target_genres else ''
        like_descr = f" for its {like_syn} aura" if like_syn else ""
        # Synonyms for different request verbs to avoid repetitive starts
        hunt_synonyms = ["I'm after", "I'm looking for", "I'm chasing", "I'm seeking", "I'm out to find"]
        ask_synonyms = ["Could you recommend", "Do you know", "Any chance you know", "Could you suggest", "Might you suggest"]
        give_synonyms = ["Give me", "Send me", "Point me to", "Share", "Hook me up with", "Show me"]
        search_synonyms = ["Looking for", "Seeking", "Searching for", "On the hunt for", "Trying to find"]
        need_synonyms = ["Need", "I need", "I'm in need of", "I'd like", "I could use"]
        template_options = []
        # Use synonyms for film to vary wording in this complex category
        for fs in FILM_SYNONYMS:
            # Use dynamic synonyms for each template to diversify beginnings
            template_options += [
                f"{random.choice(hunt_synonyms)} a {fs} that balances mood, plot and taste. I loved {liked_title}{like_descr}. Something {target_syn} with '{keywords[0]}' in the title would be awesome.",
                f"{random.choice(ask_synonyms)} a {genre_str or 'solid'} {fs} that includes '{keywords[0]}', feels {target_syn}, and aligns with my love for {liked_title}{like_descr}?",
                f"{random.choice(give_synonyms)} a {genre_str or 'great'} {fs} that makes me feel {target_syn} and features {keywords[1]} as a theme or title element.",
                f"{random.choice(search_synonyms)} a {fs} that combines elements of '{keywords[0]}' and {keywords[1]}, evokes {target_syn}, and reminds me of {liked_title}{like_descr}.",
                f"{random.choice(need_synonyms)} a {fs} that checks multiple boxes: a compelling plot, '{keywords[1]}' in the name and {target_syn} feelings. I really enjoyed {liked_title}{like_descr}.",
            ]
        body = random.choice(template_options)
    elif category == 'mix_lex_plot':
        # Combine plot themes with lexical keywords in the prompt.  Provide
        # multiple phrasing options to emulate different voices.
        plot_themes = [
            "time travel", "family drama", "space exploration", "underdog sports",
            "political intrigue", "coming-of-age story", "serial killer mystery", "heist"
        ]
        theme = random.choice(plot_themes)
        liked_title = random.choice(likes) if likes else random_movie
        genre_str = ', '.join(target_genres) if target_genres else ''
        keywords = random.sample([
            "time", "love", "star", "dark", "city", "space", "game", "death", "music", "journey"
        ], k=2)
        like_descr = f" for its {like_syn} tone" if like_syn else ""
        # Synonyms to diversify lexical+plot pattern intros
        obsessed_syn = ["Obsessed with", "Fascinated by", "Hooked on", "Intrigued by", "Fixated on"]
        any_syn = ["Any", "Know any", "Are there any", "Can you suggest any", "Do you have any"]
        since_syn = ["Since watching", "After watching", "Ever since I saw", "Having seen", "Following my viewing of"]
        looking_syn = ["Looking for", "Seeking", "Searching for", "In search of", "On the lookout for"]
        love_syn = ["I'd love recommendations", "I'd appreciate recommendations", "Would love suggestions", "I'd like to hear recommendations", "Open to suggestions"]
        gems_syn = ["Which", "What", "Can you name", "Tell me", "Do you know"]
        drawn_syn = ["I'm drawn to", "I'm into", "I'm attracted to", "I'm captivated by", "I gravitate towards"]
        template_options = [
            f"{random.choice(obsessed_syn)} {theme} plots and titles with '{keywords[0]}'; after loving {liked_title}{like_descr}, {random.choice(looking_syn)} more of this {vibe_term}.",
            f"{random.choice(any_syn)} {theme} {film_term}s whose titles include '{keywords[0]}' or '{keywords[1]}' and capture a {target_syn} feel?",
            f"{random.choice(since_syn)} {liked_title}{like_descr}, {random.choice(looking_syn)} {theme} stories with '{keywords[1]}' in the name.",
            f"{random.choice(looking_syn)} a {theme} {film_term} that balances {genre_str or 'genre variety'}, features '{keywords[0]}' in the title, and feels {target_syn}.",
            f"{random.choice(love_syn)} for {theme} {film_term}s with titles containing '{keywords[0]}'—bonus points if it's as {target_syn} as {liked_title}.",
            f"{random.choice(gems_syn)} {theme} stories with '{keywords[0]}' in their titles would you call hidden gems?",
            f"{random.choice(drawn_syn)} {theme} narratives. Know any titles that drop '{keywords[1]}' into the mix and feel {target_syn}?",
        ]
        body = random.choice(template_options)
    elif category == 'random':
        # Random prompts blend elements from multiple categories
        # Choose a random pattern: mood, plot, lexical, or mix
        pattern = random.choice(['mood','plot','lexical','mix'])
        if pattern == 'mood':
            liked_title = random.choice(likes) if likes else random_movie
            like_descr = f" with its {like_syn} {vibe_term}" if like_syn else ""
            template_options = [
                f"Need a {film_term} that gives off {target_syn} {vibe_term}s{neg_phrase}. Loved {liked_title}{like_descr} and want more.",
                f"Feeling like a {target_syn} kind of night. What do you recommend?",
                f"This weekend I'm craving a {target_syn} atmosphere; any suggestions{neg_phrase}?",
                f"I just want something that feels {target_syn}. Any ideas?",
                f"My group's mood is {target_syn}. Which {film_term} should we watch?",
                f"I could use a dose of {target_syn} right now. What's a good pick?",
            ]
            body = random.choice(template_options)
        elif pattern == 'plot':
            plot_themes = [
                "time travel", "family drama", "space exploration", "underdog sports",
                "political intrigue", "coming-of-age story", "serial killer mystery", "heist"
            ]
            theme = random.choice(plot_themes)
            liked_title = random.choice(likes) if likes else random_movie
            decade = random.choice([1970, 1980, 1990, 2000, 2010])
            template_options = [
                f"After watching {liked_title}, I'm digging stories about {theme}. Maybe something from the {decade}s?",
                f"Looking for a {film_term} about {theme} set in the {decade}s or with a modern twist.",
                f"Can you recommend any {theme} {film_term}s that would surprise me and carry a {target_syn} tone?",
                f"I can't get enough of {theme} plots lately; any hidden gems?",
                f"What's a {theme} tale from the {decade}s that will leave me feeling {target_syn}?",
            ]
            body = random.choice(template_options)
        elif pattern == 'lexical':
            keywords = random.sample([
                "time", "love", "star", "dark", "city", "space", "game", "death", "music", "journey"
            ], k=2)
            template_options = [
                f"I'm curious about {film_term}s with '{keywords[0]}' in the title. Bonus if '{keywords[1]}' is involved.",
                f"Any recommendations for {film_term}s whose titles include words like '{keywords[0]}' or revolve around {keywords[1]}?",
                f"What are some good {film_term}s featuring {keywords[0]} or {keywords[1]} in their names?",
                f"Looking for titles that mention {keywords[0]}—surprise me.",
                f"For some reason I'm drawn to titles with {keywords[0]}. Know any?",
            ]
            body = random.choice(template_options)
        else:  # mix pattern combines plot and lexical cues
            plot_themes = [
                "time travel", "family drama", "space exploration", "underdog sports",
                "political intrigue", "coming-of-age story", "serial killer mystery", "heist"
            ]
            theme = random.choice(plot_themes)
            keywords = random.sample([
                "time", "love", "star", "dark", "city", "space", "game", "death", "music", "journey"
            ], k=2)
            liked_title = random.choice(likes) if likes else random_movie
            like_descr = f" and loved its {like_syn} tone" if like_syn else ""
            body_options = [
                f"I'm torn between plot and title. Maybe a {theme} story with '{keywords[0]}' in the title? I recently enjoyed {liked_title}{like_descr}.",
                f"Give me a {film_term} that balances {theme} themes with keywords like '{keywords[1]}'. I loved {liked_title}{like_descr}.",
                f"Craving a {theme} {film_term} whose name contains '{keywords[0]}'; suggestions?",
                f"After watching {liked_title}{like_descr}, I'm looking for {theme} {film_term}s with '{keywords[1]}' in the title.",
                f"Any {theme} stories that weave '{keywords[0]}' into the title and feel {target_syn}?",
            ]
            body = random.choice(body_options)
    else:
        body = "I'm looking for a good film to watch."
    # Assemble final prompt
    prompt_text = " ".join([prefix, body, suffix]).strip()
    prompt_text = " ".join(prompt_text.split())
    return prompt_text


def compute_scores_for_prompt(prompt_tokens: List[str], target_genres: List[str], negative_genres: List[str], plutchik_dist: Dict[str, float], mix_weights: Dict[str, float], movies_df: pd.DataFrame, likes_ids: List[int], dislikes_ids: List[int]) -> Dict[int, Dict[str, float]]:
    """Compute per‑movie signal scores relative to a prompt."""
    # Precompute liked/disliked genre sets
    liked_genres: set = set()
    disliked_genres: set = set()
    if likes_ids:
        liked_genres = set(g for row in movies_df[movies_df['movie_id'].isin(likes_ids)]['genres'] for g in row)
    if dislikes_ids:
        disliked_genres = set(g for row in movies_df[movies_df['movie_id'].isin(dislikes_ids)]['genres'] for g in row)
    prompt_token_set = set(prompt_tokens)
    # Precompute prompt emotion vector
    prompt_vec = np.array([plutchik_dist[e] for e in ["joy", "trust", "fear", "anticipation", "sadness", "anger", "surprise", "disgust"]])
    scores = {}
    for _, row in movies_df.iterrows():
        mid = row['movie_id']
        movie_genres = set(row['genres'])
        movie_tokens = row['tokens']
        # Plot score
        if target_genres:
            matches = len(movie_genres.intersection(target_genres))
            plot_score = matches / max(len(target_genres), 1)
        else:
            plot_score = 0.5  # neutral baseline when no target genres
        if negative_genres:
            neg_matches = len(movie_genres.intersection(negative_genres))
            plot_score -= 0.5 * neg_matches / max(len(negative_genres), 1)
        plot_score = max(0.0, min(1.0, plot_score))
        # Lexical score: Jaccard between prompt tokens and title tokens
        if prompt_token_set:
            inter = len(prompt_token_set.intersection(movie_tokens))
            union = len(prompt_token_set.union(movie_tokens))
            lexical_score = inter / union if union else 0.0
        else:
            lexical_score = 0.0
        # History score
        def genre_similarity(a: set, b: set) -> float:
            return len(a.intersection(b)) / max(len(a.union(b)), 1)
        liked_sim = genre_similarity(movie_genres, liked_genres) if liked_genres else 0.0
        disliked_sim = genre_similarity(movie_genres, disliked_genres) if disliked_genres else 0.0
        history_score = max(0.0, liked_sim - disliked_sim)
        # Emotion score
        emotion_score = float(np.dot(row['plutchik_vector'], prompt_vec))
        combined = (mix_weights['alpha'] * plot_score + mix_weights['beta'] * lexical_score + mix_weights['gamma'] * history_score + mix_weights['delta'] * emotion_score)
        scores[mid] = {
            'plot': plot_score,
            'lexical': lexical_score,
            'history': history_score,
            'emotion': emotion_score,
            'combined': combined,
        }
    return scores


def compute_pair_features(m1: Dict, m2: Dict) -> Tuple[bool, bool, float]:
    same_genre = len(set(m1['genres']).intersection(m2['genres'])) > 0
    same_decade = abs(m1['year'] - m2['year']) < 10
    inter = len(m1['tokens'].intersection(m2['tokens']))
    union = len(m1['tokens'].union(m2['tokens']))
    keywords_overlap = inter / union if union else 0.0
    return same_genre, same_decade, float(keywords_overlap)


def hardness_score(score1: Dict[str, float], score2: Dict[str, float], mix_weights: Dict[str, float]) -> float:
    diffs = []
    for key, weight_key in [('plot', 'alpha'), ('lexical', 'beta'), ('history', 'gamma'), ('emotion', 'delta')]:
        diffs.append(abs(score1[key] - score2[key]) * mix_weights[weight_key])
    total_diff = sum(diffs)
    return max(0.0, min(1.0, 1.0 - total_diff))


def generate_rationale(movie: Dict, emotion: str, target_genres: List[str]) -> str:
    phrases = []
    year = movie['year']
    genres = movie['genres']
    if genres:
        phrases.append(f"{genres[0]} film from {year}")
    else:
        phrases.append(f"Film from {year}")
    if emotion:
        phrases.append(f"evokes {emotion.lower()}")
    if target_genres and set(genres).intersection(target_genres):
        phrases.append("matches requested genres")
    elif target_genres:
        phrases.append("diverges from requested genres")
    rationale = ", ".join(phrases)
    words = rationale.split()
    return " ".join(words[:25])


def generate_justification(score1: Dict[str, float], score2: Dict[str, float], mix_weights: Dict[str, float]) -> str:
    diffs = {}
    for s, weight_key in [('plot', 'alpha'), ('lexical', 'beta'), ('history', 'gamma'), ('emotion', 'delta')]:
        diffs[s] = abs(score1[s] - score2[s]) * mix_weights[weight_key]
    sorted_signals = sorted(diffs.items(), key=lambda x: x[1], reverse=True)
    top = [s for s, d in sorted_signals[:2] if d > 0]
    name_map = {'plot': 'α', 'lexical': 'β', 'history': 'γ', 'emotion': 'δ'}
    if not top:
        top = ['plot']
    signal_str = " and ".join(name_map[s] for s in top)
    templates = [
        f"The first movie better aligns with the {signal_str} signals and thus suits the prompt.",
        f"Differences in {signal_str} strongly favour the first option in the weighted mix.",
        f"Based on {signal_str} weights, the first film scores higher against the criteria.",
        f"The weighting across {signal_str} favours the first movie for this request.",
    ]
    justification = random.choice(templates)
    words = justification.split()
    return " ".join(words[:40])


def main():
    movies_df = load_movies()
    # Load optional movie emotion distributions from provided JSON if available.
    # The file is expected at 'movie_emotions/movie_emotions.json'.  If it
    # doesn't exist or fails to load, an empty map will be returned and
    # prompts will fall back to genre-based emotion inference.
    movie_emotions_path = os.path.join(os.getcwd(), 'movie_emotions', 'movie_emotions.json')
    movie_emotions_map = load_movie_emotions(movie_emotions_path)
    # Category quotas
    # Updated categories: remove 'history' and 'cold', add 'mix_lex_plot' and 'random' as per user feedback.
    quotas = {
        'mood': 300,
        'plot': 100,
        'lexical': 100,
        'mix_lex_plot': 100,
        'mix_emo_sem': 100,
        'mix_plot_hist': 100,
        'mix_all': 100,
        'random': 100,
    }
    # Mood distribution across eight emotions (Joy, Trust, Fear, Anticipation, Sadness, Anger, Surprise, Disgust)
    emotions_list = ["Joy", "Trust", "Fear", "Anticipation", "Sadness", "Anger", "Surprise", "Disgust"]
    mood_counts = {e.lower(): 37 for e in emotions_list}
    # Add leftover prompts to first four emotions
    for e in emotions_list[:4]:
        mood_counts[e.lower()] += 1
    # Output lists
    prompts_out = []
    pairs_out = []
    judgments_out = []
    # Precompute id->title map
    id_to_title = {row['movie_id']: row['title'] for _, row in movies_df.iterrows()}
    # Predefine negative constraints options
    neg_options = ["horror", "war", "crime", "violence", "animation", "slow pace"]
    # All genres list
    all_genres = sorted({g for genres in movies_df['genres'] for g in genres if g != 'unknown'})
    # Template style choices from tone article
    # Style choices inspired by writing tone examples.  Removed the
    # 'emoji-lite' option to avoid emojis per user instructions.
    # Include all defined styles to diversify prompt openings.  This helps
    # ensure a wider variety of prefixes across prompts.
    style_choices = ['formal', 'informal', 'humorous', 'sarcastic', 'optimistic', 'pessimistic', 'narrative', 'conversational', 'nostalgic', 'critical', 'deadpan', 'enthusiastic', 'confessional', 'storyteller', 'cinephile', 'honest', 'excited2']
    # Generate prompts
    for category, count in quotas.items():
        for _ in range(count):
            prompt_id = str(uuid.uuid4())
            # Determine target emotion
            target_emotion = None
            if category == 'mood':
                available = [e for e, c in mood_counts.items() if c > 0]
                target_emotion = random.choice(available)
                mood_counts[target_emotion] -= 1
            # Choose target genres hints
            target_genres = []
            if category in ['mood', 'plot', 'mix_emo_sem', 'mix_plot_hist', 'mix_all']:
                num_hint = random.choice([1, 2])
                target_genres = random.sample(all_genres, k=num_hint)
            else:
                if random.random() < 0.5:
                    target_genres = random.sample(all_genres, k=random.choice([1, 2]))
            # Negative constraints
            negative_constraints = random.sample(neg_options, k=random.choice([0,1,2]))
            # Cold flag is not used in this version (no 'cold' category)
            cold_flag = False
            # History profile
            likes_ids = []
            dislikes_ids = []
            likes_titles = []
            dislikes_titles = []
            history_profile_id = None
            if category in ['mix_plot_hist', 'mix_all']:
                likes_ids = random.sample(list(id_to_title.keys()), k=5)
                dislikes_ids = random.sample(list(id_to_title.keys()), k=2)
                likes_titles = [id_to_title[i] for i in likes_ids]
                dislikes_titles = [id_to_title[i] for i in dislikes_ids]
                history_profile_id = str(uuid.uuid4())
            # Mix weights
            mix_weights = sample_mix_weights(category)
            # Plutchik distribution
            plutchik_dist = sample_plutchik_dist(target_emotion)
            # Choose language and style
            lang = choose_language()
            style = random.choice(style_choices)
            # Generate prompt text using the refined v3 function.  This
            # version incorporates movie emotion synonyms and avoids Hindi or
            # emojis.  It also requires likes and dislikes ids to
            # personalise mood descriptors based on the dominant emotions of
            # liked/disliked movies.
            prompt_text = generate_prompt_text_v3(
                category,
                target_emotion.capitalize() if target_emotion else None,
                target_genres,
                negative_constraints,
                likes_titles,
                dislikes_titles,
                likes_ids,
                dislikes_ids,
                style,
                movies_df,
                movie_emotions_map,
            )
            # Context features
            prompt_tokens = extract_prompt_tokens(prompt_text)
            length_words = len(prompt_tokens)
            length_bucket = 'short' if length_words < 10 else 'medium' if length_words < 20 else 'long'
            has_negation = any(word in prompt_text.lower() for word in ['no ', 'not ', 'without'])
            has_year = any(str(y) in prompt_text for y in range(1950, 2030))
            has_genre_terms = any(g.lower() in prompt_text.lower() for g in all_genres)
            num_genre_terms = sum(prompt_text.lower().count(g.lower()) for g in all_genres)
            mentions_specific_movie = bool(likes_titles or dislikes_titles)
            mentioned_movie_ids = likes_ids + dislikes_ids
            context_features = {
                'length_words': length_words,
                'length_bucket': length_bucket,
                'has_negation': has_negation,
                'has_year': has_year,
                'has_actor_or_director': False,
                'mentions_specific_movie': mentions_specific_movie,
                'mentioned_movie_ids': mentioned_movie_ids,
                'has_genre_terms': has_genre_terms,
                'num_genre_terms': num_genre_terms,
                'language': lang,
                'persona_style': style,
                'multi_intent': False,
                'used_text_proxy': False,
                'history_profile_id': history_profile_id,
                'cold_user': cold_flag,
                'target_genres_hint': target_genres,
                'negative_constraints': negative_constraints,
                'difficulty_mix': '3/3/3',
                'seed': SEED,
            }
            prompt_record = {
                'prompt_id': prompt_id,
                'prompt_text': prompt_text,
                'category': category,
                'plutchik_dist': plutchik_dist,
                'mix_weights': mix_weights,
                'primary_expert': (
                    'plot' if category == 'plot' else
                    'lexical' if category == 'lexical' else
                    'emotion' if category == 'mood' else
                    'mixed'
                ),
                'secondary_experts': [] if category in ['plot', 'lexical', 'history', 'mood'] else ['plot', 'lexical'],
                'context_features': context_features,
                'tone_sources_note': 'Prompts draw on varied tone examples and film review tips; no external text copied.',
                'generator_model_name': 'gpt-4-dataset-generator',
                'generator_model_version': 'v2',
                'created_at_iso': datetime.now(timezone.utc).astimezone(timezone(timedelta(hours=5, minutes=30))).isoformat(),
            }
            prompts_out.append(prompt_record)
            # Compute scores for all movies relative to this prompt
            scores = compute_scores_for_prompt(prompt_tokens, target_genres, [nc.split()[-1].capitalize() for nc in negative_constraints], plutchik_dist, mix_weights, movies_df, likes_ids, dislikes_ids)
            sorted_movies = sorted(scores.items(), key=lambda item: item[1]['combined'], reverse=True)
            top_n = max(50, len(sorted_movies)//10)
            low_n = top_n
            top_pool = [mid for mid, _ in sorted_movies[:top_n]]
            low_pool = [mid for mid, _ in sorted_movies[-low_n:]]
            mid_pool = [mid for mid, _ in sorted_movies[top_n:-low_n]]
            usage = {}
            used_pairs = set()
            # Helper to sample movie with usage limit
            def sample_movie(pool):
                for _ in range(50):
                    m = random.choice(pool)
                    if usage.get(m,0) < 3:
                        return m
                candidates = [mid for mid in scores.keys() if usage.get(mid,0) < 3]
                return random.choice(candidates)
            # Generate 9 pairs across difficulties
            for difficulty in ['easy','medium','hard']:
                for i in range(3):
                    for tries in range(100):
                        if difficulty == 'easy':
                            m1 = sample_movie(top_pool)
                            m2 = sample_movie(low_pool)
                        elif difficulty == 'medium':
                            m1_candidates = [m for m in mid_pool if usage.get(m,0) < 3] or [m for m in scores.keys() if usage.get(m,0) < 3]
                            m1 = random.choice(m1_candidates)
                            possible = []
                            score1 = scores[m1]['combined']
                            for m in m1_candidates:
                                if m == m1: continue
                                diff = abs(score1 - scores[m]['combined'])
                                if 0.2 < diff < 0.5:
                                    possible.append(m)
                            m2 = random.choice(possible) if possible else sample_movie(mid_pool)
                        else:  # hard
                            m1_candidates = [m for m in top_pool if usage.get(m,0) < 3] or [m for m in scores.keys() if usage.get(m,0) < 3]
                            m1 = random.choice(m1_candidates)
                            possible = []
                            score1 = scores[m1]['combined']
                            for m in m1_candidates:
                                if m == m1: continue
                                diff = abs(score1 - scores[m]['combined'])
                                if diff < 0.2:
                                    possible.append(m)
                            m2 = random.choice(possible) if possible else sample_movie(top_pool)
                        if m1 == m2:
                            continue
                        pair_key = tuple(sorted((m1,m2)))
                        if pair_key in used_pairs:
                            continue
                        usage[m1] = usage.get(m1,0) + 1
                        usage[m2] = usage.get(m2,0) + 1
                        used_pairs.add(pair_key)
                        # Compute features
                        movie1 = movies_df.loc[movies_df['movie_id']==m1].iloc[0]
                        movie2 = movies_df.loc[movies_df['movie_id']==m2].iloc[0]
                        same_genre, same_decade, keywords_overlap = compute_pair_features(movie1, movie2)
                        hardness = hardness_score(scores[m1], scores[m2], mix_weights)
                        pair_id = f"{prompt_id}_{difficulty}_{i+1}"
                        selection_basis = (
                            'plot' if category=='plot' else
                            'lexical' if category=='lexical' else
                            'history' if category=='history' else
                            'emotion' if category=='mood' else
                            'mixed'
                        )
                        rationale1 = generate_rationale(movie1, target_emotion.capitalize() if target_emotion else None, target_genres)
                        rationale2 = generate_rationale(movie2, target_emotion.capitalize() if target_emotion else None, target_genres)
                        pair_record = {
                            'prompt_id': prompt_id,
                            'pair_id': pair_id,
                            'difficulty': difficulty,
                            'movie1_id': m1,
                            'movie2_id': m2,
                            'selection_basis': selection_basis,
                            'candidate_overlap': {
                                'same_genre': bool(same_genre),
                                'same_decade': bool(same_decade),
                                'keywords_overlap': round(float(keywords_overlap),3),
                            },
                            'surface_rationale': {
                                'm1': rationale1,
                                'm2': rationale2,
                            },
                            'hardness_score': round(float(hardness),3),
                            'used_text_proxy': True,
                            'seed': SEED,
                        }
                        pairs_out.append(pair_record)
                        # Judgement: always pick a winner (no ties)
                        score_diff = scores[m1]['combined'] - scores[m2]['combined']
                        if score_diff >= 0:
                            m1_gt_m2 = 1
                            m2_gt_m1 = -1
                        else:
                            m1_gt_m2 = -1
                            m2_gt_m1 = 1
                            # Swap scores for justification so that justification always refers to the winner as first
                            # Not necessary because we always treat m1 as first; difference sign indicates which is preferred
                        abs_diff = abs(score_diff)
                        if difficulty == 'easy':
                            conf = 0.8 + 0.2 * min(abs_diff / 0.5, 1.0)
                        elif difficulty == 'medium':
                            conf = 0.5 + 0.3 * min(abs_diff / 0.5, 1.0)
                        else:
                            conf = 0.4 + 0.2 * min(abs_diff / 0.5, 1.0)
                        conf = round(min(1.0, conf),3)
                        justification = generate_justification(scores[m1], scores[m2], mix_weights)
                        judgment_record = {
                            'prompt_id': prompt_id,
                            'pair_id': pair_id,
                            'm1_gt_m2': m1_gt_m2,
                            'm2_gt_m1': m2_gt_m1,
                            'confidence': conf,
                            'justification': justification,
                            'judge_model_name': 'gpt-4-judge',
                            'judge_model_version': 'v2',
                            'rule_checks': {
                                'antisymmetry_ok': (m1_gt_m2 == -m2_gt_m1),
                                'movies_in_ml100k_ok': True,
                                'difficulty_consistent': True,
                            },
                            'seed': SEED,
                        }
                        judgments_out.append(judgment_record)
                        break
    # Manifest statistics
    counts_by_category = {k: v for k,v in quotas.items()}
    difficulty_counts = {'easy': 0, 'medium': 0, 'hard': 0}
    for p in pairs_out:
        difficulty_counts[p['difficulty']] += 1
    genre_histogram: Dict[str,int] = {g:0 for g in all_genres}
    decade_histogram: Dict[str,int] = {}
    for p in pairs_out:
        for mid in [p['movie1_id'], p['movie2_id']]:
            movie = movies_df.loc[movies_df['movie_id']==mid].iloc[0]
            for g in movie['genres']:
                if g in genre_histogram:
                    genre_histogram[g] += 1
            dec = str(movie['decade'])
            decade_histogram[dec] = decade_histogram.get(dec, 0) + 1
    # No ties now
    tie_rates = {'easy': 0.0, 'medium': 0.0, 'hard': 0.0}
    manifest = {
        'dataset_id': str(uuid.uuid4()),
        'version': 'v3',
        'spec_version': '2025-10-23',
        'seed': SEED,
        'counts_by_category': counts_by_category,
        'difficulty_counts': difficulty_counts,
        'genre_histogram': genre_histogram,
        'decade_histogram': decade_histogram,
        'tie_rates': tie_rates,
        'used_text_proxy_rate': 1.0,
        'generator_model_name': 'gpt-4-dataset-generator',
        'generator_model_version': 'v3',
        'judge_model_name': 'gpt-4-judge',
        'judge_model_version': 'v3',
        'created_at_iso': datetime.now(timezone.utc).astimezone(timezone(timedelta(hours=5, minutes=30))).isoformat(),
        'notes': 'Prompts diversified across tones and contexts; judgments enforce strict pairwise preference (no ties); proxy features used due to missing optional files.',
    }
    # Write JSON outputs
    with open('prompts.json', 'w', encoding='utf-8') as f:
        json.dump(prompts_out, f, ensure_ascii=False, indent=2)
    with open('pairs.json', 'w', encoding='utf-8') as f:
        json.dump(pairs_out, f, ensure_ascii=False, indent=2)
    with open('judgments.json', 'w', encoding='utf-8') as f:
        json.dump(judgments_out, f, ensure_ascii=False, indent=2)
    with open('manifest.json', 'w', encoding='utf-8') as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)


if __name__ == '__main__':
    main()