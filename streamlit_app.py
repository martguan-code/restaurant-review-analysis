import streamlit as st
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.cluster import KMeans
import plotly.express as px
import plotly.graph_objects as go

# ─────────────────────────────────────────────
# Core Pipeline Functions
# ─────────────────────────────────────────────

def redact_pii(text: str) -> str:
    """Remove emails, phone numbers, and URLs from review text."""
    if not isinstance(text, str):
        return text
    text = re.sub(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}", "[EMAIL]", text)
    text = re.sub(r"\+?\d[\d \-]{7,}\d", "[PHONE]", text)
    text = re.sub(r"https?://\S+|www\.\S+", "[URL]", text)
    return text


def clean_text(text: str) -> str:
    """Lowercase, strip non-alpha characters, and remove stop words."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    return " ".join(w for w in text.split() if w not in ENGLISH_STOP_WORDS)


def map_strategy(avg_rating: float) -> str:
    """Return an AI-driven recommendation based on average cluster rating."""
    if avg_rating >= 4.5:
        return (
            "Marketing Highlight: Promote as a premium experience. "
            "Leverage in social media campaigns, press releases, and loyalty programs."
        )
    elif avg_rating >= 3.5:
        return (
            "Operational Maintenance: Monitor service consistency. "
            "Encourage satisfied guests to leave reviews and identify emerging friction points."
        )
    else:
        return (
            "Operational Audit: Urgent review required. "
            "Address service speed, food temperature, or pricing transparency immediately."
        )


def run_pipeline(df: pd.DataFrame, n_clusters: int = 6):
    """
    Execute the full 4-stage pipeline on a DataFrame with 'text' and 'rating' columns.

    Returns
    -------
    df : pd.DataFrame  — enriched with theme_cluster column
    summary : pd.DataFrame — one row per cluster with stats and strategy
    """
    # Stage 1 – PII Redaction
    df = df.copy()
    df["text"] = df["text"].apply(redact_pii)
    if "author_name" in df.columns:
        df["author_name"] = [f"User_{i}" for i in range(len(df))]

    # Stage 2 – Text Cleaning
    df["cleaned_text"] = df["text"].apply(clean_text)

    # Stage 3 – TF-IDF Vectorization (1,000 features)
    vectorizer = TfidfVectorizer(max_features=1000, min_df=2)
    tfidf_matrix = vectorizer.fit_transform(df["cleaned_text"])
    feature_names = vectorizer.get_feature_names_out()

    # Stage 4 – KMeans Clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df["theme_cluster"] = kmeans.fit_predict(tfidf_matrix)

    # Build summary table
    centroids = kmeans.cluster_centers_
    rows = []
    for i in range(n_clusters):
        top_idx = centroids[i].argsort()[::-1][:10]
        top_kw = ", ".join(feature_names[j] for j in top_idx)
        cluster_df = df[df["theme_cluster"] == i]
        avg_r = cluster_df["rating"].mean() if "rating" in df.columns else 0.0
        rows.append(
            {
                "Theme": f"Theme {i}",
                "Size": len(cluster_df),
                "Avg_Rating": round(avg_r, 2),
                "Top_Keywords": top_kw,
                "AI_Strategy": map_strategy(avg_r),
            }
        )

    summary = pd.DataFrame(rows)
    return df, summary


# ─────────────────────────────────────────────
# Streamlit Layout
# ─────────────────────────────────────────────

st.set_page_config(
    page_title="Restaurant Insight-to-Action Engine",
    page_icon="🍽️",
    layout="wide",
)

# ── Header ──────────────────────────────────
st.title("🍽️ Restaurant Insight-to-Action Engine")
st.caption(
    "Upload a restaurant review CSV to automatically redact PII, cluster themes, "
    "and generate AI-driven marketing & operational strategies."
)

# ── Sidebar ─────────────────────────────────
with st.sidebar:
    st.header("⚙️ Configuration")
    st.markdown(
        """
**Required CSV columns**
| Column   | Type       | Description            |
|----------|------------|------------------------|
| `text`   | string     | Review body text       |
| `rating` | int/float  | Star rating (1 – 5)    |

Optional: `business_name`, `author_name`, `rating_category`
"""
    )
    uploaded_file = st.file_uploader(
        "Upload reviews CSV",
        type=["csv"],
        help="Accepts any UTF-8 encoded CSV with at least 'text' and 'rating' columns.",
    )
    n_clusters = st.slider("Number of theme clusters (K)", min_value=3, max_value=10, value=6)
    st.divider()
    st.caption("Built for MSIS 522 · Restaurant Review Analysis")

# ── Main Content ─────────────────────────────
if uploaded_file is None:
    st.info("👈 Upload a CSV file in the sidebar to start the analysis.")
    st.markdown(
        """
### How it works
| Stage | Step | What happens |
|-------|------|--------------|
| 1 | **PII Redaction** | Emails, phones, and URLs are replaced with tokens |
| 2 | **Text Cleaning** | Lowercasing, punctuation removal, stop-word filtering |
| 3 | **TF-IDF Vectorization** | Reviews encoded into 1,000-feature numeric vectors |
| 4 | **KMeans Clustering + Strategy** | Reviews grouped by theme; AI strategy assigned per cluster |
"""
    )
    st.stop()

# ── File Processing ──────────────────────────
try:
    raw_df = pd.read_csv(uploaded_file)
except Exception as e:
    st.error(f"Could not parse the uploaded file: {e}")
    st.stop()

missing = [c for c in ("text", "rating") if c not in raw_df.columns]
if missing:
    st.error(
        f"Missing required column(s): **{', '.join(missing)}**. "
        "Please check your CSV and re-upload."
    )
    st.stop()

if len(raw_df) < n_clusters:
    st.error(
        f"Your dataset has only {len(raw_df)} rows but K={n_clusters} clusters were requested. "
        "Reduce K or upload more data."
    )
    st.stop()

with st.spinner("Running 4-stage pipeline…"):
    try:
        processed_df, summary_df = run_pipeline(raw_df, n_clusters=n_clusters)
    except Exception as e:
        st.error(f"Pipeline error: {e}")
        st.stop()

# ── KPIs ────────────────────────────────────
st.subheader("📊 Dataset Overview")
kpi1, kpi2, kpi3, kpi4 = st.columns(4)
kpi1.metric("Total Reviews", len(processed_df))
kpi2.metric("Avg Rating", f"{processed_df['rating'].mean():.2f} ⭐")
kpi3.metric("Clusters (K)", n_clusters)
kpi4.metric(
    "Top Theme",
    summary_df.loc[summary_df["Avg_Rating"].idxmax(), "Theme"],
)

st.divider()

# ── Charts ───────────────────────────────────
st.subheader("📈 Theme Analysis")
col1, col2 = st.columns(2)

with col1:
    fig_pie = px.pie(
        summary_df,
        values="Size",
        names="Theme",
        title="Theme Distribution (Review Volume)",
        hover_data=["Top_Keywords"],
        color_discrete_sequence=px.colors.qualitative.Set2,
    )
    fig_pie.update_traces(textposition="inside", textinfo="percent+label")
    st.plotly_chart(fig_pie, use_container_width=True)

with col2:
    fig_bar = px.bar(
        summary_df.sort_values("Avg_Rating"),
        x="Theme",
        y="Avg_Rating",
        title="Average Rating per Theme",
        color="Avg_Rating",
        color_continuous_scale="RdYlGn",
        range_color=[1, 5],
        text="Avg_Rating",
    )
    global_avg = processed_df["rating"].mean()
    fig_bar.add_hline(
        y=global_avg,
        line_dash="dash",
        line_color="gray",
        annotation_text=f"Global avg: {global_avg:.2f}",
    )
    fig_bar.update_traces(texttemplate="%{text:.2f}", textposition="outside")
    fig_bar.update_layout(yaxis_range=[0, 5.5], coloraxis_showscale=False)
    st.plotly_chart(fig_bar, use_container_width=True)

# ── Rating distribution per cluster (box plot) ──
fig_box = px.box(
    processed_df,
    x="theme_cluster",
    y="rating",
    title="Rating Distribution per Cluster",
    color="theme_cluster",
    color_discrete_sequence=px.colors.qualitative.Set2,
    labels={"theme_cluster": "Cluster", "rating": "Rating"},
)
st.plotly_chart(fig_box, use_container_width=True)

st.divider()

# ── AI Strategy Roadmap ──────────────────────
st.subheader("🤖 AI-Driven Strategic Roadmap")
for _, row in summary_df.iterrows():
    rating_color = (
        "🟢" if row["Avg_Rating"] >= 4.5 else "🟡" if row["Avg_Rating"] >= 3.5 else "🔴"
    )
    with st.expander(
        f"{rating_color} {row['Theme']} · {row['Size']} reviews · Avg Rating: {row['Avg_Rating']}"
    ):
        st.markdown(f"**Top Keywords:** `{row['Top_Keywords']}`")
        if row["Avg_Rating"] >= 4.5:
            st.success(row["AI_Strategy"])
        elif row["Avg_Rating"] >= 3.5:
            st.warning(row["AI_Strategy"])
        else:
            st.error(row["AI_Strategy"])

st.divider()

# ── Data Preview ─────────────────────────────
with st.expander("🔍 View Processed Reviews (PII Redacted)"):
    display_cols = [c for c in ("business_name", "text", "rating", "theme_cluster") if c in processed_df.columns]
    st.dataframe(processed_df[display_cols].head(100), use_container_width=True)

with st.expander("📋 View Cluster Summary Table"):
    st.dataframe(summary_df, use_container_width=True)

# ── Download ─────────────────────────────────
st.subheader("⬇️ Download Results")
dl1, dl2 = st.columns(2)
with dl1:
    st.download_button(
        "Download Processed Reviews CSV",
        data=processed_df.to_csv(index=False).encode("utf-8"),
        file_name="processed_reviews.csv",
        mime="text/csv",
    )
with dl2:
    st.download_button(
        "Download Theme Summary CSV",
        data=summary_df.to_csv(index=False).encode("utf-8"),
        file_name="theme_summary.csv",
        mime="text/csv",
    )
