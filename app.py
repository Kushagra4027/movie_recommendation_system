import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import difflib

st.set_page_config(
    page_title="CineMatch - Movie Recommender",
    page_icon="ðŸŽ¬",
    layout="wide",
)

st.markdown("""
<style>
    html, body, [class*="css"] { font-family: 'Segoe UI', sans-serif; }
    .stApp { background: #0D1117; color: #E6EDF3; }
    .hero {
        background: linear-gradient(135deg, #161B22 0%, #1C2330 100%);
        border: 1px solid #30363D;
        border-radius: 16px;
        padding: 2.5rem 3rem;
        margin-bottom: 2rem;
    }
    .hero h1 { font-size: 2.4rem; font-weight: 700; color: #E6EDF3; margin: 0; }
    .hero span { color: #E8B04B; }
    .hero p { color: #8B949E; margin-top: 0.5rem; font-size: 1rem; }
    .movie-card {
        background: #161B22;
        border: 1px solid #21262D;
        border-radius: 12px;
        padding: 1.2rem 1.4rem;
        margin-bottom: 0.8rem;
    }
    .rank-badge {
        display: inline-block;
        background: #E8B04B;
        color: #0D1117;
        font-weight: 700;
        font-size: 0.8rem;
        border-radius: 6px;
        padding: 2px 8px;
        margin-right: 10px;
    }
    .movie-title { font-size: 1.05rem; font-weight: 600; color: #E6EDF3; }
    .tag {
        display: inline-block;
        background: #1C2330;
        border: 1px solid #30363D;
        border-radius: 20px;
        padding: 2px 10px;
        font-size: 0.75rem;
        color: #58A6FF;
        margin-right: 4px;
    }
    .score-bar-bg {
        background: #21262D;
        border-radius: 4px;
        height: 6px;
        width: 100%;
        margin-top: 6px;
    }
    .stat-box {
        background: #161B22;
        border: 1px solid #21262D;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
    }
    .stat-num { font-size: 1.8rem; font-weight: 700; color: #E8B04B; }
    .stat-label { font-size: 0.8rem; color: #8B949E; }
    div[data-testid="stTextInput"] input {
        background: #161B22 !important;
        border: 1px solid #30363D !important;
        border-radius: 10px !important;
        color: #E6EDF3 !important;
        font-size: 1rem !important;
    }
    .stButton > button {
        background: linear-gradient(135deg, #E8B04B, #D4941A) !important;
        color: #0D1117 !important;
        font-weight: 700 !important;
        border: none !important;
        border-radius: 10px !important;
    }
    footer { visibility: hidden; }
    #MainMenu { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_and_process():
    df = pd.read_csv("MRS/movies.csv")
    df["genres_clean"]   = df["genres"].fillna("").apply(lambda x: x.lower().replace(" ", "_"))
    df["keywords_clean"] = df["keywords"].fillna("").apply(lambda x: x.lower().replace(" ", "_"))
    df["cast_clean"]     = df["cast"].fillna("").apply(lambda x: " ".join(x.split()[:4]).lower())
    df["director_clean"] = df["director"].fillna("").apply(lambda x: x.lower().replace(" ", "_"))
    df["overview_clean"] = df["overview"].fillna("").str.lower()
    df["features"] = (
        df["genres_clean"] + " " + df["genres_clean"] + " " +
        df["keywords_clean"] + " " +
        df["director_clean"] + " " + df["director_clean"] + " " +
        df["cast_clean"] + " " +
        df["overview_clean"]
    )
    df["year"]   = pd.to_datetime(df["release_date"], errors="coerce").dt.year.fillna(0).astype(int)
    df["rating"] = df["vote_average"].fillna(0).round(1)
    return df


@st.cache_resource
def build_model(df):
    tfidf = TfidfVectorizer(max_features=15000, ngram_range=(1, 2), stop_words="english")
    matrix = tfidf.fit_transform(df["features"].fillna(""))
    return cosine_similarity(matrix)


def recommend(movie_name, df, similarity, n=10, genre_filter=None, year_from=None, year_to=None):
    titles = df["title"].tolist()
    matches = difflib.get_close_matches(movie_name, titles, n=5, cutoff=0.4)
    if not matches:
        return None, None
    matched_title = matches[0]
    idx = df[df["title"] == matched_title].index[0]
    scores = sorted(enumerate(similarity[idx]), key=lambda x: x[1], reverse=True)
    results = []
    for i, score in scores[1:]:
        row = df.iloc[i]
        if genre_filter and genre_filter != "All":
            if genre_filter.lower() not in row["genres"].lower():
                continue
        if year_from and row["year"] < year_from:
            continue
        if year_to and row["year"] > year_to:
            continue
        results.append({
            "title":    row["title"],
            "genres":   row["genres"],
            "director": row["director"],
            "cast":     row["cast"],
            "year":     int(row["year"]) if row["year"] else "N/A",
            "rating":   row["rating"],
            "score":    round(score * 100, 1),
            "overview": row["overview"],
        })
        if len(results) >= n:
            break
    return matched_title, results


df = load_and_process()
similarity = build_model(df)

all_genres = sorted(set(
    g.strip() for genres in df["genres"].dropna()
    for g in genres.split() if len(g) > 2
))

st.markdown("""
<div class="hero">
    <h1>CineMatch</h1>
    <p>Content-based Movie Recommendation Engine &nbsp;|&nbsp; 4800+ Movies &nbsp;|&nbsp; No login required</p>
</div>
""", unsafe_allow_html=True)

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown(f'<div class="stat-box"><div class="stat-num">{len(df):,}</div><div class="stat-label">Movies in Database</div></div>', unsafe_allow_html=True)
with c2:
    st.markdown(f'<div class="stat-box"><div class="stat-num">{df["director"].nunique():,}</div><div class="stat-label">Unique Directors</div></div>', unsafe_allow_html=True)
with c3:
    st.markdown(f'<div class="stat-box"><div class="stat-num">{int(df["year"].min())} - {int(df["year"].max())}</div><div class="stat-label">Year Range</div></div>', unsafe_allow_html=True)
with c4:
    st.markdown('<div class="stat-box"><div class="stat-num">TF-IDF</div><div class="stat-label">+ Cosine Similarity</div></div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)
st.markdown("### Find Movies")

col_inp, col_btn = st.columns([5, 1])
with col_inp:
    movie_input = st.text_input("Enter a movie name", placeholder="e.g. The Dark Knight, Inception, Avatar...", label_visibility="collapsed")
with col_btn:
    search_clicked = st.button("Search", use_container_width=True)

with st.expander("Filters & Settings", expanded=False):
    fc1, fc2, fc3 = st.columns(3)
    with fc1:
        genre_filter = st.selectbox("Genre", ["All"] + all_genres)
    with fc2:
        year_range = st.slider("Release Year", 1900, 2020, (1990, 2020))
    with fc3:
        top_n = st.slider("Number of Recommendations", 5, 20, 10)

if search_clicked and movie_input.strip():
    with st.spinner("Finding similar movies..."):
        matched, results = recommend(
            movie_input, df, similarity, n=top_n,
            genre_filter=genre_filter if genre_filter != "All" else None,
            year_from=year_range[0], year_to=year_range[1],
        )

    if not results:
        st.error(f"No match found for '{movie_input}'. Try a different title.")
    else:
        st.success(f"Showing recommendations based on: {matched}")
        st.markdown("<br>", unsafe_allow_html=True)

        left_col, right_col = st.columns([2, 1])

        with left_col:
            st.markdown("#### Top Picks For You")
            for i, m in enumerate(results):
                genres_html = " ".join(
                    f'<span class="tag">{g.strip()}</span>'
                    for g in str(m["genres"]).split()[:4] if len(g) > 2
                )
                cast_preview = ", ".join(str(m["cast"]).split()[:3]) if m["cast"] else "N/A"
                st.markdown(f"""
                <div class="movie-card">
                    <span class="rank-badge">#{i+1}</span>
                    <span class="movie-title">{m['title']}</span>
                    &nbsp;&nbsp;<span style="color:#8B949E;font-size:0.85rem">({m['year']})</span>
                    <div style="font-size:0.85rem;color:#8B949E;margin-top:4px">
                        Rating: {m['rating']}/10 &nbsp;|&nbsp; Director: {m['director']} &nbsp;|&nbsp; Cast: {cast_preview}
                    </div>
                    <div style="margin-top:8px">{genres_html}</div>
                    <div style="margin-top:8px;font-size:0.82rem;color:#6E7681">
                        Similarity: <b style="color:#E8B04B">{m['score']}%</b>
                        <div class="score-bar-bg"><div style="background:#E8B04B;height:6px;border-radius:4px;width:{min(m['score'],100)}%"></div></div>
                    </div>
                    <div style="margin-top:8px;font-size:0.82rem;color:#8B949E;font-style:italic">
                        {str(m['overview'])[:180]}{'...' if len(str(m['overview'])) > 180 else ''}
                    </div>
                </div>
                """, unsafe_allow_html=True)

        with right_col:
            st.markdown("#### Recommendation Breakdown")
            genre_counts = {}
            for m in results:
                for g in str(m["genres"]).split()[:3]:
                    if len(g) > 2:
                        genre_counts[g] = genre_counts.get(g, 0) + 1

            st.markdown("**Genre Mix**")
            for g, count in sorted(genre_counts.items(), key=lambda x: -x[1])[:6]:
                pct = int(count / len(results) * 100)
                st.markdown(f"""
                <div style="margin-bottom:6px">
                    <span style="font-size:0.85rem;color:#E6EDF3">{g}</span>
                    <span style="float:right;font-size:0.8rem;color:#8B949E">{count} movies</span>
                    <div class="score-bar-bg"><div style="background:#58A6FF;height:5px;border-radius:3px;width:{pct}%"></div></div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("<br>**Directors**", unsafe_allow_html=True)
            dir_counts = {}
            for m in results:
                d = str(m["director"]).strip()
                if d and d != "nan":
                    dir_counts[d] = dir_counts.get(d, 0) + 1
            for d, c in sorted(dir_counts.items(), key=lambda x: -x[1])[:5]:
                st.markdown(f'<div style="font-size:0.85rem;color:#E6EDF3;margin-bottom:4px">- {d} ({c})</div>', unsafe_allow_html=True)

            st.markdown("<br>**Avg Rating**", unsafe_allow_html=True)
            avg_rating = np.mean([m["rating"] for m in results if m["rating"] > 0])
            st.markdown(f'<div class="stat-box"><div class="stat-num">{avg_rating:.1f}</div><div class="stat-label">across recommendations</div></div>', unsafe_allow_html=True)

elif not movie_input and search_clicked:
    st.warning("Please enter a movie name first.")

st.markdown("---")
st.markdown(
    '<div style="text-align:center;color:#8B949E;font-size:0.8rem">'
    'Built by <b style="color:#E8B04B">Kushagra Singh</b> | '
    'Microsoft x AICTE x Edunet Foundation Internship | '
    'TMDB 5000 Dataset | TF-IDF + Cosine Similarity'
    '</div>',
    unsafe_allow_html=True
)
