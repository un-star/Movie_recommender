import streamlit as st
import pickle
import pandas as pd
import requests
from requests.adapters import HTTPAdapter, Retry
import os
import requests

def download_file(url, filename):
    if not os.path.exists(filename):
        print(f"Downloading {filename}...")
        response = requests.get(url)
        with open(filename, "wb") as f:
            f.write(response.content)
        print("Download complete.")

download_file(
    "https://huggingface.co/un01huggingface/movie-recommender-similarity/resolve/main/similarity.pkl",
    "similarity.pkl"
)


# ---------------------------------------------------------
# 1. PAGE CONFIG — must be the very first Streamlit command.
#    Sets browser tab title/icon and makes the app use full width.
# ---------------------------------------------------------
st.set_page_config(
    page_title="Movie Recommender",
    page_icon="🎬",
    layout="wide"
)

# ---------------------------------------------------------
# 2. CUSTOM CSS — makes poster images look nicer (rounded
#    corners + a little zoom effect when you hover over them).
# ---------------------------------------------------------
st.markdown("""
    <style>
    div[data-testid="stImage"] img {
        border-radius: 12px;
        transition: transform 0.2s ease-in-out;
    }
    div[data-testid="stImage"] img:hover {
        transform: scale(1.05);
        box-shadow: 0 4px 15px rgba(0,0,0,0.4);
    }
    </style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# 3. TITLE + SUBTITLE
# ---------------------------------------------------------
st.title("🎬 Movie Recommender System")
st.caption("Pick a movie you like, and we'll suggest 5 similar ones!")

# ---------------------------------------------------------
# 4. LOAD DATA — movie list and similarity matrix were
#    precomputed earlier and saved using pickle.
# ---------------------------------------------------------
movies = pd.DataFrame(pickle.load(open('movie_dict.pkl', 'rb')))
similarity = pickle.load(open('similarity.pkl', 'rb'))

# ---------------------------------------------------------
# 5. HTTP SESSION WITH RETRIES — if the poster API (TMDB)
#    is briefly slow or fails, this retries automatically
#    instead of giving up right away.
# ---------------------------------------------------------
session = requests.Session()
session.mount('https://', HTTPAdapter(max_retries=Retry(total=3, backoff_factor=0.5)))


# ---------------------------------------------------------
# 6. FETCH MOVIE DETAILS FUNCTION
#    @st.cache_data means: if we already fetched this movie's
#    details before, reuse the saved result instead of calling
#    the API again. Makes the app faster and more reliable.
# ---------------------------------------------------------
@st.cache_data(show_spinner=False)
def fetch_movie_details(movie_id):
    url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key=fd4e544cbfeeddda51b9aeffc9547c46"
    try:
        data = session.get(url, timeout=10).json()

        poster_path = data.get('poster_path')
        poster_url = f"https://image.tmdb.org/t/p/w500/{poster_path}" if poster_path else None

        rating = data.get('vote_average', 0)          # e.g. 7.1
        overview = data.get('overview', 'No description available.')

        return poster_url, rating, overview

    except requests.exceptions.RequestException:
        # If the API call fails even after retries, return safe defaults
        return None, 0, "No description available."


# ---------------------------------------------------------
# 7. RECOMMEND FUNCTION
#    Finds the 5 most "similar" movies using the precomputed
#    similarity matrix.
# ---------------------------------------------------------
def recommend(movie):
    # Find the row index of the selected movie
    movie_index = movies[movies['original_title'] == movie].index[0]

    # Similarity scores of this movie with every other movie
    distances = similarity[movie_index]

    # Sort by similarity (highest first), skip index 0 (itself),
    # take the next 5 most similar movies
    movies_list = sorted(
        list(enumerate(distances)),
        reverse=True,
        key=lambda x: x[1]
    )[1:6]

    names, posters, ratings, overviews = [], [], [], []

    for i, _ in movies_list:
        movie_id = movies.iloc[i].movie_id_y
        poster, rating, overview = fetch_movie_details(movie_id)

        names.append(movies.iloc[i].original_title)
        posters.append(poster)
        ratings.append(rating)
        overviews.append(overview)

    return names, posters, ratings, overviews


# ---------------------------------------------------------
# 8. MOVIE SELECTION DROPDOWN
# ---------------------------------------------------------
selected_movie_name = st.selectbox(
    "🔍 Select a movie you like:",
    movies['original_title'].values
)

# ---------------------------------------------------------
# 9. SHOW THE SELECTED MOVIE'S POSTER
#    This runs as soon as a movie is picked, even before the
#    "Recommend" button is clicked — so the user gets instant
#    visual feedback on their choice.
# ---------------------------------------------------------
selected_movie_id = movies[
    movies['original_title'] == selected_movie_name
].iloc[0].movie_id_y

selected_poster, selected_rating, selected_overview = fetch_movie_details(selected_movie_id)

st.subheader("You selected:")
col1, col2 = st.columns([1, 3])   # narrow column for poster, wide column for text

with col1:
    if selected_poster:
        st.image(selected_poster, use_container_width=True)
    else:
        st.write("🚫 Poster not available")

with col2:
    st.markdown(f"### {selected_movie_name}")
    star_count = round(selected_rating / 2)   # TMDB rates out of 10 -> convert to 5 stars
    st.markdown("⭐" * star_count + "☆" * (5 - star_count))
    st.write(selected_overview)

st.divider()

# ---------------------------------------------------------
# 10. RECOMMEND BUTTON + RESULTS DISPLAY
# ---------------------------------------------------------
if st.button("✨ Recommend"):

    # Show a spinner while fetching data, so the app doesn't
    # look frozen while waiting on the API
    with st.spinner("Finding movies you'll love..."):
        names, posters, ratings, overviews = recommend(selected_movie_name)

    st.success(f"Because you liked **{selected_movie_name}**, you might enjoy:")

    columns = st.columns(5)  # 5 equal-width columns, one per recommended movie

    for col, name, poster, rating, overview in zip(columns, names, posters, ratings, overviews):
        with col:
            st.markdown(f"**{name}**")

            if poster:
                st.image(poster, use_container_width=True)
            else:
                st.write("🚫 Poster not available")

            star_count = round(rating / 2)
            st.markdown("⭐" * star_count + "☆" * (5 - star_count))

            with st.expander("More info"):
                st.write(overview)