# Restaurant Insight-to-Action Engine

A privacy-safe, AI-driven Streamlit dashboard that transforms raw restaurant reviews into structured marketing and operational strategies through automated clustering and NLP.

---

## What It Does

Restaurant owners upload a CSV of customer reviews. The engine automatically:

1. **Redacts PII** from every review (emails, phone numbers, URLs)
2. **Cleans and vectorizes** the text using TF-IDF (1,000 features)
3. **Clusters reviews** into themed groups using KMeans
4. **Generates strategy recommendations** for each cluster based on average ratings

The result is an interactive dashboard showing theme distributions, rating trends, and concrete next steps — no data science expertise required.

---

## The 4-Stage Pipeline

```
┌──────────────────────────────────────────────────────────┐
│  Stage 1 │  PII Redaction                                │
│          │  Regex removes emails → [EMAIL]               │
│          │               phones  → [PHONE]               │
│          │               URLs    → [URL]                  │
│          │  Author names replaced with User_0, User_1…   │
├──────────────────────────────────────────────────────────┤
│  Stage 2 │  Text Cleaning                                │
│          │  Lowercase · remove punctuation               │
│          │  Drop English stop words (scikit-learn list)  │
├──────────────────────────────────────────────────────────┤
│  Stage 3 │  TF-IDF Vectorization                         │
│          │  max_features=1,000 · min_df=2                │
│          │  Converts each cleaned review to a            │
│          │  1,000-dimensional numeric vector             │
├──────────────────────────────────────────────────────────┤
│  Stage 4 │  KMeans Clustering + Strategy Mapping         │
│          │  K configurable (default 6, range 3-10)       │
│          │  Top-10 keywords extracted per cluster        │
│          │  map_strategy() assigns one of three tiers:  │
│          │    ≥ 4.5 → Marketing Highlight                │
│          │    ≥ 3.5 → Operational Maintenance            │
│          │    < 3.5 → Operational Audit                  │
└──────────────────────────────────────────────────────────┘
```

---

## Quickstart

### 1. Clone and install

```bash
git clone https://github.com/martguan-code/restaurant-review-analysis
cd restaurant-review-analysis
pip install -r requirements.txt
```

### 2. Run the app

```bash
streamlit run streamlit_app.py
```

### 3. Upload your data

Open `http://localhost:8501` in your browser. Use the sidebar to upload a CSV file.

---

## Required CSV Schema

| Column   | Type      | Description                          |
|----------|-----------|--------------------------------------|
| `text`   | string    | Full review body text                |
| `rating` | int/float | Star rating, typically 1–5           |

Optional columns recognized: `business_name`, `author_name`, `rating_category`, `photo`

---

## Strategy Mapping Logic

| Avg Cluster Rating | Label                   | Recommended Action |
|--------------------|-------------------------|--------------------|
| ≥ 4.5              | Marketing Highlight     | Promote on social media; use in PR campaigns and loyalty programs |
| 3.5 – 4.4          | Operational Maintenance | Monitor consistency; prompt satisfied guests to leave reviews |
| < 3.5              | Operational Audit       | Urgent investigation of pricing, food quality, or service speed |

---

## Dashboard Features

- **KPI row** — total reviews, average rating, cluster count, top-rated theme
- **Pie chart** — review volume distribution across clusters
- **Bar chart** — average rating per cluster with global average reference line
- **Box plot** — rating spread within each cluster
- **Expandable strategy cards** — color-coded (green / yellow / red) per cluster
- **Data preview** — first 100 PII-redacted rows
- **CSV downloads** — processed reviews and theme summary

---

## Business Impact

| Area | Benefit |
|------|---------|
| Privacy & Compliance | PII removed before any analysis; author identities anonymized |
| Marketing | Highest-rated clusters identified for targeted campaigns |
| Operations | Low-rated clusters flagged with root-cause keywords for quick triage |
| Scalability | Pipeline runs entirely in-memory; no external API calls required |
| Flexibility | K is user-configurable; new CSVs can be analyzed without code changes |

---

## Project Structure

```
restaurant-review-analysis/
├── streamlit_app.py           # Main Streamlit application
├── requirements.txt           # Python dependencies
├── README.md                  # This file
└── MSIS_522_Team_Assignment.ipynb  # Original research notebook
```

---

## Dependencies

| Package       | Purpose                          |
|---------------|----------------------------------|
| streamlit     | Web dashboard framework          |
| pandas        | Data loading and manipulation    |
| scikit-learn  | TF-IDF vectorization and KMeans  |
| plotly        | Interactive charts               |
