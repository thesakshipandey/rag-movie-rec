"""
Load movies and User 1's rating history
"""
import pandas as pd
from typing import Dict, List, Any
from config import MOVIES_CSV, RATINGS_FILE, USER_ID


def load_user_history() -> Dict[int, int]:
    """
    Load User 1's rating history from u.data

    Returns:
        Dict mapping movieId → rating (1-5)
    """
    df = pd.read_csv(
        RATINGS_FILE,
        sep='\t',
        names=['user_id', 'movie_id', 'rating', 'timestamp'],
        engine='python'
    )

    user_df = df[df['user_id'] == USER_ID]
    return dict(zip(user_df['movie_id'], user_df['rating']))


def year_from_date(date_str: Any) -> str:
    """Extract year from release_date string"""
    if isinstance(date_str, str) and len(date_str) >= 4 and date_str[:4].isdigit():
        return date_str[:4]
    return ""


def load_movies(user_history: Dict[int, int]) -> List[Dict[str, Any]]:
    """
    Load all movies from CSV with plot_sum_160 and user ratings

    Args:
        user_history: User's rating dictionary

    Returns:
        List of movie dicts with fields:
        - movieId: int
        - title: str
        - year: str
        - plot: str (from plot_sum_160)
        - user_rating: int or None
    """
    df = pd.read_csv(MOVIES_CSV)

    candidates = []
    for _, row in df.iterrows():
        movie_id = int(row['movieId'])

        # Use plot_sum_160, fallback to overview (truncated), fallback to empty
        plot_text = str(row.get('plot_sum_160') or '')
        if not plot_text or plot_text == 'nan':
            overview = str(row.get('overview') or '')
            plot_text = overview[:160] if overview else ""

        # Clean up plot text
        plot_text = plot_text.replace('\n', ' ').strip()

        candidate = {
            'movieId': movie_id,
            'title': str(row['title']),
            'year': year_from_date(row.get('release_date', '')),
            'plot': plot_text,
            'user_rating': user_history.get(movie_id)  # None if not rated
        }

        candidates.append(candidate)

    return candidates


def format_candidate_for_llm(candidate: Dict[str, Any]) -> str:
    """
    Format a single candidate for LLM consumption

    Example:
      [1] Toy Story (1995) [User Rating: 5★]: A group of sentient toys...
      [2] GoldenEye (1995): James Bond fights to prevent...
    """
    movie_id = candidate['movieId']
    title = candidate['title']
    year = candidate['year']
    plot = candidate['plot']
    user_rating = candidate['user_rating']

    # Build title line
    title_line = f"[{movie_id}] {title}"
    if year:
        title_line += f" ({year})"

    # Add user rating if present
    if user_rating is not None:
        stars = '★' * user_rating
        title_line += f" [User Rating: {stars}]"

    # Add plot
    if plot:
        title_line += f": {plot}"

    return title_line
