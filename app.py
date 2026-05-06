"""Amazon Recommendation — single-file Streamlit app (dashboards + shared data helpers)."""

from __future__ import annotations

import os
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import pyarrow.dataset as ds
import requests
import streamlit as st

st.set_page_config(
    page_title="Amazon Recommendation Project",
    layout="wide",
    page_icon="📦",
    initial_sidebar_state="expanded",
)

PROJECT_ROOT = Path(__file__).resolve().parent
CLEAN_PARQUET = PROJECT_ROOT / "data/amazon_clean.parquet/clean_data.parquet"
RAW_INGESTED_PARQUET = PROJECT_ROOT / "data/amazon_reviews.parquet"

# Raw JSONL headline (nominal corpus size for demos / missing local files)
RAW_JSONL_HEADLINE_GB = "12.02 GB"
RAW_JSONL_NOMINAL_GB = 12.02  # used when no local `data/*.jsonl` for storage chart scale
# Soft palette with a slight brightness bump (~1–2 steps vs prior muted set)
STAR_BAR_COLORS = ["#bf7070", "#d09070", "#c0aa78", "#82a890", "#8899bf"]

POSITIVE_WORDS = [
    "amazing",
    "awesome",
    "best",
    "excellent",
    "good",
    "great",
    "love",
    "nice",
    "perfect",
    "recommend",
    "satisfied",
    "thanks",
    "wonderful",
]

NEGATIVE_WORDS = [
    "awful",
    "bad",
    "broken",
    "disappoint",
    "garbage",
    "hate",
    "horrible",
    "junk",
    "mediocre",
    "pathetic",
    "poor",
    "refund",
    "rubbish",
    "terrible",
    "trash",
    "useless",
    "waste",
    "worst",
]


def _confidence_from_positive_strength(n: int) -> float:
    """Strong positive cues → ~0.82–0.99."""
    if n <= 0:
        return 0.50
    conf = 0.82 + 0.17 * min(1.0, max(0, n - 1) / 2.0)
    return float(round(min(0.99, conf), 4))


def _confidence_from_negative_strength(n: int) -> float:
    """Negative cues use a lower baseline than positives (one word ≠ same certainty as positive path)."""
    if n <= 0:
        return 0.50
    conf = 0.62 + 0.28 * min(1.0, max(0, n - 1) / 2.0)
    return float(round(min(0.93, conf), 4))


def classify_review_sentiment(text_lower: str) -> tuple[str, float]:
    """Keyword balance: net positive vs negative hits; ambiguous → Neutral with low score."""
    hp = sum(1 for w in POSITIVE_WORDS if w in text_lower)
    hn = sum(1 for w in NEGATIVE_WORDS if w in text_lower)
    net = hp - hn
    if net > 0:
        return "Positive", _confidence_from_positive_strength(net)
    if net < 0:
        return "Negative", _confidence_from_negative_strength(-net)
    if hp == 0 and hn == 0:
        return "Neutral", 0.48
    return "Neutral", 0.52


def format_bytes(num_bytes: int) -> str:
    if num_bytes >= 1024**4:
        return f"{num_bytes / (1024**4):.3f} TB"
    if num_bytes >= 1024**3:
        return f"{num_bytes / (1024**3):.3f} GB"
    if num_bytes >= 1024**2:
        return f"{num_bytes / (1024**2):.2f} MB"
    return f"{num_bytes:,} bytes"


def format_raw_jsonl_display(_num_bytes: int = 0) -> str:
    return RAW_JSONL_HEADLINE_GB


def _clean_dataset_path() -> Path | None:
    if CLEAN_PARQUET.is_dir() or CLEAN_PARQUET.is_file():
        return CLEAN_PARQUET
    return None


def _folder_size_bytes(root: Path) -> int:
    total = 0
    if root.is_file():
        return root.stat().st_size
    if root.is_dir():
        for dirpath, _, files in os.walk(root):
            for f in files:
                total += os.path.getsize(Path(dirpath) / f)
    return total


def jsonl_file_inventory():
    files = sorted(PROJECT_ROOT.glob("data/*.jsonl"))
    if not files:
        return pd.DataFrame(), 0
    rows = []
    total = 0
    for f in files:
        sz = f.stat().st_size
        total += sz
        rows.append(
            {
                "File": f.name,
                "Size": format_bytes(sz),
                "GB": round(sz / (1024**3), 4),
            }
        )
    return pd.DataFrame(rows), total


@st.cache_data(ttl=3600, show_spinner="Loading rating stats…")
def load_rating_series():
    p = _clean_dataset_path()
    if not p:
        return None
    dataset = ds.dataset(str(p), format="parquet")
    tbl = dataset.scanner(columns=["rating"]).to_table()
    return tbl.column(0).to_numpy(zero_copy_only=False)


@st.cache_data(ttl=3600, show_spinner="Loading row count…")
def load_review_count():
    p = _clean_dataset_path()
    if not p:
        return None
    return ds.dataset(str(p), format="parquet").count_rows()


@st.cache_data(ttl=3600, show_spinner="Sampling example user_id…")
def sample_example_user_id():
    p = _clean_dataset_path()
    if not p:
        return ""
    dataset = ds.dataset(str(p), format="parquet")
    for batch in dataset.scanner(columns=["user_id"], batch_size=4096).to_batches():
        if batch.num_rows > 0:
            return str(batch.column(0)[0].as_py())
    return ""


@st.cache_data(ttl=600, show_spinner="Loading user reviews…")
def user_review_frames(user_id: str):
    p = _clean_dataset_path()
    if not p or not user_id.strip():
        return None, None
    uid = user_id.strip()
    cols = ["user_id", "parent_asin", "rating", "timestamp"]
    try:
        schema_cols = set(ds.dataset(str(p), format="parquet").schema.names)
        use_cols = [c for c in cols if c in schema_cols]
        df = pd.read_parquet(
            str(p),
            columns=use_cols,
            filters=[("user_id", "==", uid)],
            engine="pyarrow",
        )
    except Exception:
        return None, None
    if df.empty:
        return df, None
    dist = df["rating"].value_counts().sort_index()
    timeline = None
    if "timestamp" in df.columns and df["timestamp"].notna().any():
        t = df.copy()
        ts = pd.to_numeric(t["timestamp"], errors="coerce")
        mx = ts.max()
        unit = "ms" if pd.notna(mx) and float(mx) > 1e12 else "s"
        t["_dt"] = pd.to_datetime(ts, unit=unit, utc=True, errors="coerce")
        timeline = t.dropna(subset=["_dt"]).sort_values("_dt")
    return df, {"distribution": dist, "timeline": timeline}


@st.cache_data(ttl=600, show_spinner="Looking up top products…")
def top_rated_products_for_user(user_id: str, k: int = 10):
    p = _clean_dataset_path()
    if not p or not user_id.strip():
        return None
    try:
        df = pd.read_parquet(
            str(p),
            columns=["user_id", "parent_asin", "rating"],
            filters=[("user_id", "==", user_id.strip())],
            engine="pyarrow",
        )
    except Exception:
        return None
    if df.empty:
        return None
    df = df.sort_values("rating", ascending=False).drop_duplicates(subset=["parent_asin"]).head(k)
    return df.reset_index(drop=True)


def storage_comparison_df() -> tuple[pd.DataFrame, list[str]]:
    """Three pipeline layers: raw JSONL → ingested Parquet → clean Parquet.

    Raw GB is measured from ``data/*.jsonl`` when present; otherwise **nominal** 12.02 GB
    so the chart still compares raw vs ingested vs clean on one axis.
    """
    rows = []
    inv, jtotal = jsonl_file_inventory()
    raw_gb = jtotal / (1024**3) if jtotal > 0 else RAW_JSONL_NOMINAL_GB
    rows.append({"Layer": "Raw JSONL (`data/*.jsonl`)", "GB": raw_gb})

    if RAW_INGESTED_PARQUET.exists():
        b = _folder_size_bytes(RAW_INGESTED_PARQUET)
        rows.append(
            {"Layer": "Ingested Parquet (`amazon_reviews.parquet/`)", "GB": b / (1024**3)}
        )

    if CLEAN_PARQUET.parent.exists():
        b = _folder_size_bytes(CLEAN_PARQUET.parent)
        rows.append({"Layer": "Clean Parquet (`amazon_clean/`)", "GB": b / (1024**3)})

    df = pd.DataFrame(rows)
    order = [r["Layer"] for r in rows]
    return df, order


def _chart_star_colors(domain: list[str]) -> alt.Scale:
    n = len(domain)
    if n <= len(STAR_BAR_COLORS):
        return alt.Scale(domain=list(domain), range=STAR_BAR_COLORS[:n])
    return alt.Scale(domain=list(domain), scheme="set3")


def chart_colored_vertical_bars(
    df: pd.DataFrame,
    x: str,
    y: str,
    *,
    height: int = 340,
    use_star_palette: bool = False,
) -> alt.Chart:
    dom = df[x].tolist()
    if use_star_palette:
        cscale = _chart_star_colors([str(d) for d in dom])
    else:
        cscale = alt.Scale(scheme="set3")
    return (
        alt.Chart(df)
        .mark_bar()
        .encode(
            x=alt.X(f"{x}:N", sort=None, title=None),
            y=alt.Y(f"{y}:Q"),
            color=alt.Color(f"{x}:N", legend=None, scale=cscale),
            tooltip=[alt.Tooltip(f"{x}:N", title="category"), alt.Tooltip(f"{y}:Q", format=",")],
        )
        .properties(height=height)
        .interactive()
    )


def chart_colored_horizontal_bars(
    df: pd.DataFrame,
    y: str,
    x: str,
    *,
    height: int | None = None,
    y_order: list[str] | None = None,
) -> alt.Chart:
    h = height if height is not None else max(220, 36 * len(df))
    y_sort: list[str] | alt.EncodingSortField = (
        y_order if y_order is not None else alt.EncodingSortField(field=x, order="descending")
    )
    return (
        alt.Chart(df)
        .mark_bar()
        .encode(
            x=alt.X(f"{x}:Q", title=x, scale=alt.Scale(domainMin=0)),
            y=alt.Y(f"{y}:N", sort=y_sort, title=None),
            color=alt.Color(f"{y}:N", legend=None, scale=alt.Scale(scheme="set3")),
            tooltip=[y, alt.Tooltip(f"{x}:Q", format=".4f")],
        )
        .properties(height=h)
        .interactive()
    )


def chart_colored_area(df: pd.DataFrame, x: str, y: str, *, height: int = 340) -> alt.Chart:
    line_c = "#968eb8"
    return (
        alt.Chart(df)
        .mark_area(
            interpolate="monotone",
            line={"color": line_c, "strokeWidth": 1.5},
            color=alt.Gradient(
                gradient="linear",
                stops=[
                    alt.GradientStop(color="#f4f2fb", offset=0),
                    alt.GradientStop(color="#d4cce6", offset=1),
                ],
                x1=1,
                x2=1,
                y1=1,
                y2=0,
            ),
        )
        .encode(
            x=alt.X(f"{x}:N", sort=None, title=None),
            y=alt.Y(f"{y}:Q"),
            tooltip=[alt.Tooltip(f"{x}:N"), alt.Tooltip(f"{y}:Q", format=",")],
        )
        .properties(height=height)
        .interactive()
    )


def render_home():
    st.title("Amazon Product Recommendation & Analytics")
    st.markdown(
        "Pipeline over the [McAuley-Lab/Amazon-Reviews-2023]"
        "(https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023) dataset."
    )
    st.info(
        "Use the **sidebar** to switch sections:\n\n"
        "- **Business Dashboard** — storage, raw JSONL breakdown, rating mix\n"
        "- **Customer personalization** — per-user charts from `clean_data.parquet`\n"
        "- **Review intelligence** — sentiment vs dataset prior\n"
    )
    st.markdown("---")
    st.caption(
        "Processed paths: `data/amazon_clean.parquet/clean_data.parquet` · Raw JSONL: `data/*.jsonl`"
    )


def render_business_dashboard():
    st.header("Business Dashboard")
    st.markdown(
        "Metrics and charts from **`data/amazon_clean.parquet`** plus **raw** **`data/*.jsonl`** "
        "and Notebook 1 **`amazon_reviews.parquet/`**."
    )

    if not _clean_dataset_path():
        st.error(f"Processed clean data not found at `{CLEAN_PARQUET}`. Run notebooks 01–02.")
        return

    n_reviews = load_review_count()
    size_gb = _folder_size_bytes(CLEAN_PARQUET.parent) / (1024**3)
    ratings = load_rating_series()

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Clean reviews (rows)", f"{n_reviews:,}" if n_reviews else "—")
    with c2:
        st.metric("Clean folder (`amazon_clean/`) GB", f"{size_gb:.3f}")
    with c3:
        if ratings is not None and len(ratings):
            st.metric("Mean rating (clean)", f"{float(np.mean(ratings)):.2f}")
        else:
            st.metric("Mean rating (clean)", "—")

    st.subheader("Raw data footprint")
    rc1, rc2 = st.columns(2)
    inv_df = pd.DataFrame()
    with rc1:
        if RAW_INGESTED_PARQUET.exists():
            rb = _folder_size_bytes(RAW_INGESTED_PARQUET)
            st.metric(
                "Ingested Parquet (`amazon_reviews.parquet/`)",
                format_bytes(rb),
                help="Notebook 1 output from JSONL.",
            )
        else:
            st.metric("Ingested Parquet (`amazon_reviews.parquet/`)", "—")
    with rc2:
        inv_df, jtotal = jsonl_file_inventory()
        st.metric(
            "Raw JSONL (`data/*.jsonl`)",
            format_raw_jsonl_display(jtotal),
            help=f"Headline **{RAW_JSONL_HEADLINE_GB}** (nominal raw corpus). Caption shows measured size when files exist.",
        )
        if jtotal > 0:
            st.caption(f"**{len(inv_df)} file(s)** · Exact total: **{format_bytes(jtotal)}** ({jtotal:,} bytes)")
        else:
            st.caption(
                f"No local `data/*.jsonl` detected · headline **{RAW_JSONL_HEADLINE_GB}** is the nominal corpus size."
            )

    st_df, storage_layer_order = storage_comparison_df()
    if not st_df.empty:
        st.markdown("**Storage layers (GB)** — raw JSONL vs ingested vs clean")
        st.altair_chart(
            chart_colored_horizontal_bars(
                st_df, "Layer", "GB", y_order=storage_layer_order
            ),
            use_container_width=True,
        )

    if inv_df is not None and not inv_df.empty:
        st.subheader("Per-file JSONL sizes")
        st.dataframe(inv_df, use_container_width=True, hide_index=True)

    st.markdown("### Charts")

    if ratings is None:
        st.warning("Could not load ratings.")
    else:
        vc = pd.Series(ratings).value_counts().sort_index()
        dist_df = pd.DataFrame({"reviews": vc.values}, index=[f"{int(i)} ★" for i in vc.index])
        mix_plot = dist_df.reset_index().rename(columns={"index": "rating"})

        a1, a2 = st.columns(2)
        with a1:
            st.markdown("**1. Rating mix (bar)** — `clean_data.parquet`")
            st.altair_chart(
                chart_colored_vertical_bars(mix_plot, "rating", "reviews", use_star_palette=True),
                use_container_width=True,
            )
        with a2:
            st.markdown("**2. Rating mix (area)** — same data")
            st.altair_chart(
                chart_colored_area(mix_plot, "rating", "reviews"),
                use_container_width=True,
            )


def render_customer_personalization():
    st.header("Customer personalization")
    st.markdown(
        "Charts use **your rows** in **`clean_data.parquet`** (observed ratings, not live ALS inference)."
    )

    if not _clean_dataset_path():
        st.error(f"Missing `{CLEAN_PARQUET}`.")
        return

    example_uid = sample_example_user_id()
    user_id = st.text_input(
        "User ID (exact string from dataset)",
        value=example_uid,
        help="Example pre-filled from Parquet.",
    )
    go = st.button("Load user charts", type="primary")

    if not go:
        st.info("Click **Load user charts** to render visualizations.")
        return

    uid = user_id.strip()
    if not uid:
        st.warning("Enter a user_id.")
        return

    df, meta = user_review_frames(uid)
    if df is None:
        st.error("Could not read Parquet.")
        return
    if df.empty:
        st.warning("No rows for this user_id.")
        return

    st.success(f"**{len(df):,}** review rows for this user.")

    top = top_rated_products_for_user(uid, k=12)
    if top is not None and not top.empty:
        st.markdown("### 1. Top products by star rating (unique `parent_asin`)")
        bar_df = top.set_index("parent_asin")[["rating"]].rename(columns={"rating": "stars"})
        top_plot = bar_df.reset_index().rename(columns={"parent_asin": "product"})
        st.altair_chart(
            chart_colored_vertical_bars(top_plot, "product", "stars", height=360),
            use_container_width=True,
        )

    if meta and meta.get("distribution") is not None:
        dist = meta["distribution"]
        ddf = pd.DataFrame({"count": dist.values}, index=[f"{int(i)} ★" for i in dist.index])
        st.markdown("### 2. This user’s rating counts (1–5)")
        udist = ddf.reset_index().rename(columns={"index": "rating"})
        st.altair_chart(
            chart_colored_vertical_bars(udist, "rating", "count", use_star_palette=True),
            use_container_width=True,
        )

    with st.expander("Raw rows (sample)"):
        st.dataframe(df.head(50), use_container_width=True)


def render_review_intelligence():
    st.header("Review intelligence")
    st.markdown(
        "Sentiment uses **FastAPI** when `http://127.0.0.1:8000` is up; otherwise the same "
        "keyword heuristic as `fastapi_app`."
    )

    ratings = None
    if _clean_dataset_path():
        ratings = load_rating_series()

    review_text = st.text_area(
        "Paste a review to analyze",
        placeholder="This product is amazing! Five stars.",
        height=120,
    )

    analyze = st.button("Analyze & show charts", type="primary")

    if analyze:
        if not review_text.strip():
            st.warning("Enter some text.")
        else:
            lower = review_text.lower()
            sentiment = None
            confidence = None
            try:
                r = requests.post(
                    "http://127.0.0.1:8000/predict_sentiment",
                    json={"text": review_text},
                    timeout=5,
                )
                r.raise_for_status()
                payload = r.json()
                sentiment = payload.get("sentiment")
                confidence = float(payload.get("confidence", 0))
            except Exception:
                pass
            if sentiment is None:
                sentiment, confidence = classify_review_sentiment(lower)

            if sentiment == "Positive":
                pos_weight = confidence
            elif sentiment == "Negative":
                pos_weight = 1.0 - confidence
            else:
                pos_weight = 0.5

            st.markdown("### Your text vs dataset prior (share with rating ≥ 4)")
            if ratings is not None and len(ratings):
                dataset_high = float(np.mean(ratings >= 4.0))
                cmp = pd.DataFrame(
                    {"value": [dataset_high, pos_weight]},
                    index=["Dataset P(rating ≥ 4)", "This text (positive lean)"],
                )
                cmp_plot = cmp.reset_index().rename(columns={"index": "series"})
                st.altair_chart(
                    chart_colored_vertical_bars(cmp_plot, "series", "value", height=300),
                    use_container_width=True,
                )
            else:
                st.info(
                    "Dataset prior needs clean Parquet at `data/amazon_clean.parquet/clean_data.parquet`."
                )

            outcome = f"**{sentiment}** · confidence-style score **{confidence:.4f}**"
            if sentiment == "Positive":
                st.success(outcome)
            elif sentiment == "Negative":
                st.error(outcome)
            else:
                st.warning(outcome)


def main():
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Go to",
        ["Home", "Business Dashboard", "Customer personalization", "Review intelligence"],
    )

    if page == "Home":
        render_home()
    elif page == "Business Dashboard":
        render_business_dashboard()
    elif page == "Customer personalization":
        render_customer_personalization()
    else:
        render_review_intelligence()


main()
