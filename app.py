"""Amazon Recommendation — single-file Streamlit app (dashboards + shared data helpers)."""

from __future__ import annotations

import os
import re
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

# ── Custom CSS for professional look ──
st.markdown("""
<style>
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1e1b4b 0%, #312e81 100%);
    }
    [data-testid="stSidebar"] * {
        color: #e0e7ff !important;
    }
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h1 {
        color: #c7d2fe !important;
        font-weight: 700;
    }
    /* Metric cards */
    [data-testid="stMetric"] {
        background: linear-gradient(135deg, #f8fafc 0%, #eef2ff 100%);
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 16px 20px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.06);
    }
    [data-testid="stMetric"] label {
        color: #64748b !important;
        font-size: 0.85rem !important;
        font-weight: 500 !important;
        text-transform: uppercase;
        letter-spacing: 0.03em;
    }
    [data-testid="stMetric"] [data-testid="stMetricValue"] {
        color: #1e293b !important;
        font-weight: 700 !important;
    }
    /* Section dividers */
    hr { border-color: #e2e8f0 !important; }
    /* Headers */
    h1 { color: #1e293b !important; }
    h2 { color: #334155 !important; border-bottom: 2px solid #6366f1; padding-bottom: 8px; }
    h3 { color: #475569 !important; }
    /* Buttons */
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #6366f1 0%, #818cf8 100%) !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        padding: 8px 24px !important;
    }
    /* Expanders */
    [data-testid="stExpander"] {
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        overflow: hidden;
    }
</style>
""", unsafe_allow_html=True)

# ── Altair theme for dark/light mode compatibility ──
alt.theme.enable("dark")

PROJECT_ROOT = Path(__file__).resolve().parent
CLEAN_PARQUET = PROJECT_ROOT / "data/amazon_clean.parquet/clean_data.parquet"
RAW_INGESTED_PARQUET = PROJECT_ROOT / "data/amazon_reviews.parquet"

# Raw JSONL headline (nominal corpus size for demos / missing local files)
RAW_JSONL_HEADLINE_GB = "12.02 GB"
RAW_JSONL_NOMINAL_GB = 12.02  # used when no local `data/*.jsonl` for storage chart scale
# Soft palette with a slight brightness bump (~1–2 steps vs prior muted set)
STAR_BAR_COLORS = ["#ef4444", "#f97316", "#eab308", "#22c55e", "#3b82f6"]

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

NEGATION_CUES = {"not", "no", "never", "none", "hardly", "barely", "scarcely"}


def _token_polarity_hits(text_lower: str) -> tuple[int, int]:
    """Count positive/negative hits with simple negation handling."""
    tokens = re.findall(r"[a-z]+(?:n't)?|[.!?]", text_lower)
    hp, hn = 0, 0
    neg_window = 0
    for tok in tokens:
        if tok in {".", "!", "?"}:
            neg_window = 0
            continue
        if tok in NEGATION_CUES or tok.endswith("n't"):
            neg_window = 3
            continue

        pos_hit = any(tok.startswith(w) for w in POSITIVE_WORDS)
        neg_hit = any(tok.startswith(w) for w in NEGATIVE_WORDS)
        if pos_hit:
            if neg_window > 0:
                hn += 1
            else:
                hp += 1
        if neg_hit:
            if neg_window > 0:
                hp += 1
            else:
                hn += 1

        if neg_window > 0:
            neg_window -= 1
    return hp, hn


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
    hp, hn = _token_polarity_hits(text_lower)
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


@st.cache_data(ttl=3600, show_spinner="Loading dashboard data…")
def load_dashboard_df():
    """Load key columns for dashboard charts."""
    p = _clean_dataset_path()
    if not p:
        return None
    cols = ["rating", "timestamp", "helpful_vote", "verified_purchase",
            "text", "title", "user_review_count", "product_review_count", "parent_asin"]
    try:
        schema_cols = set(ds.dataset(str(p), format="parquet").schema.names)
        use_cols = [c for c in cols if c in schema_cols]
        return pd.read_parquet(str(p), columns=use_cols, engine="pyarrow")
    except Exception:
        return None


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
    rows.append({"Layer": "① Raw JSONL (Source)", "GB": raw_gb})

    if RAW_INGESTED_PARQUET.exists():
        b = _folder_size_bytes(RAW_INGESTED_PARQUET)
        rows.append(
            {"Layer": "② Ingested Parquet", "GB": b / (1024**3)}
        )

    if CLEAN_PARQUET.parent.exists():
        b = _folder_size_bytes(CLEAN_PARQUET.parent)
        rows.append({"Layer": "③ Clean Parquet (Final)", "GB": b / (1024**3)})

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
            tooltip=[alt.Tooltip(f"{x}:N", title=x), alt.Tooltip(f"{y}:Q", format=",", title=y)],
        )
        .properties(height=height)
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
    )



def render_business_dashboard():
    st.header("📊 Business Dashboard")
    st.markdown(
        "Key performance indicators, storage metrics, and interactive analytics "
        "across the full data pipeline."
    )

    if not _clean_dataset_path():
        st.error(f"Processed clean data not found at `{CLEAN_PARQUET}`. Run notebooks 01–02.")
        return

    n_reviews = load_review_count()
    size_gb = _folder_size_bytes(CLEAN_PARQUET.parent) / (1024**3)
    ratings = load_rating_series()
    dash_df = load_dashboard_df()

    # ── KPI row ──
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Total Clean Reviews", f"{n_reviews:,}" if n_reviews else "—")
    with c2:
        st.metric("Clean Dataset Size", f"{size_gb:.3f} GB")
    with c3:
        if ratings is not None and len(ratings):
            st.metric("Average Rating", f"{float(np.mean(ratings)):.2f} ★")
        else:
            st.metric("Average Rating", "—")
    with c4:
        if dash_df is not None and "parent_asin" in dash_df.columns:
            st.metric("Unique Products", f"{dash_df['parent_asin'].nunique():,}")
        else:
            st.metric("Unique Products", "—")

    st.markdown("---")

    # ── Storage comparison ──
    st.subheader("💾 Storage Footprint")
    rc1, rc2 = st.columns(2)
    with rc1:
        if RAW_INGESTED_PARQUET.exists():
            rb = _folder_size_bytes(RAW_INGESTED_PARQUET)
            st.metric("Ingested Parquet", format_bytes(rb), help="Notebook 1 output from JSONL.")
        else:
            st.metric("Ingested Parquet", "—")
    with rc2:
        inv_df, jtotal = jsonl_file_inventory()
        st.metric("Raw JSONL Corpus", format_raw_jsonl_display(jtotal))
        if jtotal > 0:
            st.caption(f"**{len(inv_df)} file(s)** · Exact total: **{format_bytes(jtotal)}**")
        else:
            st.caption(f"No local `data/*.jsonl` · headline **{RAW_JSONL_HEADLINE_GB}** is nominal.")

    st_df, storage_layer_order = storage_comparison_df()
    if not st_df.empty:
        st.markdown("**Pipeline Storage Layers (GB)** — Raw → Ingested → Clean")
        st.altair_chart(
            chart_colored_horizontal_bars(st_df, "Layer", "GB", y_order=storage_layer_order),
            width="stretch",
        )

    if inv_df is not None and not inv_df.empty:
        with st.expander("📄 Per-file JSONL breakdown"):
            st.dataframe(inv_df, use_container_width=True, hide_index=True)

    st.markdown("---")

    # ── Rating charts ──
    if ratings is not None and len(ratings):
        st.subheader("⭐ Rating Distribution")
        vc = pd.Series(ratings).value_counts().sort_index()
        total = vc.sum()
        dist_df = pd.DataFrame({
            "Star Rating": [f"{int(i)} Star" for i in vc.index],
            "Number of Reviews": vc.values.tolist(),
            "Percentage": [round(v / total * 100, 1) for v in vc.values],
        })

        a1, a2 = st.columns(2)
        with a1:
            st.markdown("**Review Count by Star Rating**")
            chart = (
                alt.Chart(dist_df).mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
                .encode(
                    x=alt.X("Star Rating:N", sort=None, title="Star Rating"),
                    y=alt.Y("Number of Reviews:Q", title="Number of Reviews"),
                    color=alt.Color("Star Rating:N", legend=None,
                                    scale=_chart_star_colors(dist_df["Star Rating"].tolist())),
                    tooltip=["Star Rating", alt.Tooltip("Number of Reviews:Q", format=","),
                             alt.Tooltip("Percentage:Q", format=".1f", title="Share (%)")]
                ).properties(height=340)
            )
            st.altair_chart(chart, width="stretch")
        with a2:
            st.markdown("**Rating Share (Donut)**")
            donut = (
                alt.Chart(dist_df).mark_arc(innerRadius=60, outerRadius=120)
                .encode(
                    theta=alt.Theta("Number of Reviews:Q"),
                    color=alt.Color("Star Rating:N", legend=alt.Legend(title="Rating"),
                                    scale=_chart_star_colors(dist_df["Star Rating"].tolist())),
                    tooltip=["Star Rating", alt.Tooltip("Number of Reviews:Q", format=","),
                             alt.Tooltip("Percentage:Q", format=".1f", title="Share (%)")]
                ).properties(height=340, width=340)
            )
            st.altair_chart(donut, width="stretch")

    # ── Charts from full dashboard df ──
    if dash_df is not None and not dash_df.empty:
        st.markdown("---")

        # Review volume over time
        if "timestamp" in dash_df.columns:
            st.subheader("📅 Review Volume Over Time")
            time_granularity = st.selectbox(
                "Time granularity", ["Yearly", "Quarterly", "Monthly"], index=0,
                key="time_gran")
            tdf = dash_df.copy()
            ts = pd.to_numeric(tdf["timestamp"], errors="coerce")
            mx = ts.max()
            unit = "ms" if pd.notna(mx) and float(mx) > 1e12 else "s"
            tdf["Review Date"] = pd.to_datetime(ts, unit=unit, utc=True, errors="coerce")
            tdf = tdf.dropna(subset=["Review Date"])
            if time_granularity == "Yearly":
                tdf["Period"] = tdf["Review Date"].dt.year.astype(str)
            elif time_granularity == "Quarterly":
                tdf["Period"] = tdf["Review Date"].dt.to_period("Q").astype(str)
            else:
                tdf["Period"] = tdf["Review Date"].dt.to_period("M").astype(str)
            vol = tdf.groupby("Period").size().reset_index(name="Review Count")
            vol = vol.sort_values("Period")
            # Filter to meaningful range
            vol = vol[vol["Period"] >= "2005"]
            area_chart = (
                alt.Chart(vol).mark_area(
                    interpolate="monotone",
                    line={"color": "#6366f1", "strokeWidth": 2},
                    color=alt.Gradient(gradient="linear",
                        stops=[alt.GradientStop(color="#e0e7ff", offset=0),
                               alt.GradientStop(color="#818cf8", offset=1)],
                        x1=1, x2=1, y1=1, y2=0)
                ).encode(
                    x=alt.X("Period:N", sort=None, title="Time Period"),
                    y=alt.Y("Review Count:Q", title="Number of Reviews"),
                    tooltip=["Period:N", alt.Tooltip("Review Count:Q", format=",")]
                ).properties(height=350).interactive()
            )
            st.altair_chart(area_chart, width="stretch")

        st.markdown("---")
        b1, b2 = st.columns(2)

        # Verified purchase breakdown
        with b1:
            if "verified_purchase" in dash_df.columns:
                st.subheader("✅ Verified vs Unverified Reviews")
                vp = dash_df["verified_purchase"].value_counts()
                vp_df = pd.DataFrame({
                    "Purchase Type": ["Verified Purchase" if k else "Unverified Purchase" for k in vp.index],
                    "Count": vp.values.tolist()
                })
                vp_chart = (
                    alt.Chart(vp_df).mark_arc(innerRadius=50, outerRadius=110)
                    .encode(
                        theta="Count:Q",
                        color=alt.Color("Purchase Type:N", legend=alt.Legend(title="Type"),
                                        scale=alt.Scale(domain=vp_df["Purchase Type"].tolist(),
                                                        range=["#34d399", "#f87171"])),
                        tooltip=["Purchase Type", alt.Tooltip("Count:Q", format=",")]
                    ).properties(height=300, width=300)
                )
                st.altair_chart(vp_chart, width="stretch")

        # Helpful votes distribution
        with b2:
            if "helpful_vote" in dash_df.columns:
                st.subheader("👍 Helpful Vote Distribution")
                hv = dash_df["helpful_vote"]
                bins = [0, 1, 2, 5, 10, 50, hv.max() + 1]
                labels = ["0 votes", "1 vote", "2–4 votes", "5–9 votes", "10–49 votes", "50+ votes"]
                hv_binned = pd.cut(hv, bins=bins, labels=labels, right=False)
                hv_df = hv_binned.value_counts().reset_index()
                hv_df.columns = ["Helpful Votes", "Number of Reviews"]
                hv_df = hv_df.sort_index()
                hv_chart = (
                    alt.Chart(hv_df).mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
                    .encode(
                        x=alt.X("Helpful Votes:N", sort=labels, title="Helpful Vote Range"),
                        y=alt.Y("Number of Reviews:Q", title="Number of Reviews"),
                        color=alt.Color("Helpful Votes:N", legend=None,
                                        scale=alt.Scale(scheme="tealblues")),
                        tooltip=["Helpful Votes", alt.Tooltip("Number of Reviews:Q", format=",")]
                    ).properties(height=300)
                )
                st.altair_chart(hv_chart, width="stretch")

        st.markdown("---")

        # Review length analysis
        if "text" in dash_df.columns:
            st.subheader("📝 Review Length Analysis")
            dash_df["Review Length (chars)"] = dash_df["text"].str.len()
            len_col1, len_col2 = st.columns(2)
            with len_col1:
                st.markdown("**Review Length by Star Rating**")
                if "rating" in dash_df.columns:
                    len_by_star = dash_df.groupby("rating")["Review Length (chars)"].mean().reset_index()
                    len_by_star["Star Rating"] = len_by_star["rating"].apply(lambda x: f"{int(x)} Star")
                    len_by_star["Average Review Length"] = len_by_star["Review Length (chars)"].round(0)
                    len_chart = (
                        alt.Chart(len_by_star)
                        .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
                        .encode(
                            x=alt.X("Star Rating:N", sort=None, title="Star Rating"),
                            y=alt.Y("Average Review Length:Q", title="Avg Characters"),
                            color=alt.Color("Star Rating:N", legend=None,
                                            scale=_chart_star_colors(len_by_star["Star Rating"].tolist())),
                            tooltip=["Star Rating", alt.Tooltip("Average Review Length:Q", format=",.0f")]
                        ).properties(height=300)
                    )
                    st.altair_chart(len_chart, width="stretch")
            with len_col2:
                st.markdown("**Review Length Distribution (histogram)**")
                len_hist_df = dash_df[["Review Length (chars)"]].copy()
                len_hist_df = len_hist_df[len_hist_df["Review Length (chars)"] <= 3000]
                hist = (
                    alt.Chart(len_hist_df).mark_bar(color="#8b5cf6", opacity=0.7)
                    .encode(
                        x=alt.X("Review Length (chars):Q", bin=alt.Bin(maxbins=40),
                                title="Review Length (characters)"),
                        y=alt.Y("count()", title="Number of Reviews"),
                        tooltip=[alt.Tooltip("Review Length (chars):Q", bin=alt.Bin(maxbins=40),
                                             title="Length Range"),
                                 alt.Tooltip("count()", title="Count")]
                    ).properties(height=300).interactive()
                )
                st.altair_chart(hist, width="stretch")

        st.markdown("---")

        # Top products
        if "parent_asin" in dash_df.columns:
            st.subheader("🏆 Most Reviewed Products")
            top_n = st.slider("Number of products to show", 5, 25, 10, key="top_prod_n")
            prod_counts = dash_df["parent_asin"].value_counts().head(top_n).reset_index()
            prod_counts.columns = ["Product ID", "Total Reviews"]
            prod_counts["Product Rank"] = [f"#{i+1}: {pid}" for i, pid in enumerate(prod_counts["Product ID"])]
            top_chart = (
                alt.Chart(prod_counts)
                .mark_bar(cornerRadiusEnd=4)
                .encode(
                    x=alt.X("Total Reviews:Q", title="Number of Reviews"),
                    y=alt.Y("Product Rank:N",
                             sort=prod_counts["Product Rank"].tolist(),
                             title=None),
                    color=alt.Color("Total Reviews:Q", scale=alt.Scale(scheme="viridis"), legend=None),
                    tooltip=[alt.Tooltip("Product ID:N", title="Product"),
                             alt.Tooltip("Total Reviews:Q", format=",")]
                ).properties(height=max(250, top_n * 28))
            )
            st.altair_chart(top_chart, width="stretch")

        # Average rating by verified purchase
        if "verified_purchase" in dash_df.columns and "rating" in dash_df.columns:
            st.markdown("---")
            st.subheader("📈 Rating Patterns")
            rp1, rp2 = st.columns(2)
            with rp1:
                st.markdown("**Average Rating: Verified vs Unverified**")
                avg_by_vp = dash_df.groupby("verified_purchase")["rating"].mean().reset_index()
                avg_by_vp["Purchase Type"] = avg_by_vp["verified_purchase"].apply(
                    lambda x: "Verified Purchase" if x else "Unverified Purchase")
                avg_by_vp["Average Rating"] = avg_by_vp["rating"].round(2)
                vp_bar = (
                    alt.Chart(avg_by_vp)
                    .mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6)
                    .encode(
                        x=alt.X("Purchase Type:N", title=None),
                        y=alt.Y("Average Rating:Q", title="Average Rating",
                                scale=alt.Scale(domain=[0, 5])),
                        color=alt.Color("Purchase Type:N", legend=None,
                                        scale=alt.Scale(range=["#34d399", "#f87171"])),
                        tooltip=["Purchase Type", alt.Tooltip("Average Rating:Q", format=".2f")]
                    ).properties(height=300)
                )
                st.altair_chart(vp_bar, width="stretch")
            with rp2:
                st.markdown("**Rating Breakdown: Verified vs Unverified**")
                cross = dash_df.groupby(["rating", "verified_purchase"]).size().reset_index(name="Count")
                cross["Star Rating"] = cross["rating"].apply(lambda x: f"{int(x)} Star")
                cross["Purchase Type"] = cross["verified_purchase"].apply(
                    lambda x: "Verified" if x else "Unverified")
                stacked = (
                    alt.Chart(cross)
                    .mark_bar()
                    .encode(
                        x=alt.X("Star Rating:N", sort=None, title="Star Rating"),
                        y=alt.Y("Count:Q", title="Number of Reviews"),
                        color=alt.Color("Purchase Type:N", legend=alt.Legend(title="Type"),
                                        scale=alt.Scale(range=["#34d399", "#f87171"])),
                        tooltip=["Star Rating", "Purchase Type",
                                 alt.Tooltip("Count:Q", format=",")]
                    ).properties(height=300)
                )
                st.altair_chart(stacked, width="stretch")


def render_customer_personalization():
    st.header("👤 Customer Personalization")
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

    # KPI row
    kc1, kc2, kc3 = st.columns(3)
    with kc1:
        st.metric("Total Reviews by User", f"{len(df):,}")
    with kc2:
        st.metric("Average Rating Given", f"{df['rating'].mean():.2f} ★")
    with kc3:
        unique_prods = df["parent_asin"].nunique() if "parent_asin" in df.columns else len(df)
        st.metric("Unique Products Reviewed", f"{unique_prods:,}")

    st.markdown("---")

    top = top_rated_products_for_user(uid, k=12)
    if top is not None and not top.empty:
        st.subheader("🏅 Top Rated Products by This User")
        bar_df = top.copy()
        bar_df["Product"] = bar_df["parent_asin"].apply(lambda x: x[:16] + "…" if len(x) > 16 else x)
        bar_df["Stars Given"] = bar_df["rating"]
        top_chart = (
            alt.Chart(bar_df)
            .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
            .encode(
                x=alt.X("Product:N", sort=None, title="Product ID"),
                y=alt.Y("Stars Given:Q", title="Star Rating", scale=alt.Scale(domain=[0, 5])),
                color=alt.Color("Stars Given:Q", scale=alt.Scale(scheme="goldorange"), legend=None),
                tooltip=[alt.Tooltip("parent_asin:N", title="Full Product ID"),
                         alt.Tooltip("Stars Given:Q", title="Rating")]
            ).properties(height=360)
        )
        st.altair_chart(top_chart, width="stretch")


    if meta and meta.get("distribution") is not None:
        dist = meta["distribution"]
        ddf = pd.DataFrame({
            "Star Rating": [f"{int(i)} Star" for i in dist.index],
            "Number of Reviews": dist.values.tolist(),
        })

        d1, d2 = st.columns(2)
        with d1:
            st.subheader("⭐ User's Rating Distribution")
            udist_chart = (
                alt.Chart(ddf)
                .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
                .encode(
                    x=alt.X("Star Rating:N", sort=None, title="Star Rating"),
                    y=alt.Y("Number of Reviews:Q", title="Number of Reviews"),
                    color=alt.Color("Star Rating:N", legend=None,
                                    scale=_chart_star_colors(ddf["Star Rating"].tolist())),
                    tooltip=["Star Rating", alt.Tooltip("Number of Reviews:Q", format=",")]
                ).properties(height=320)
            )
            st.altair_chart(udist_chart, width="stretch")
        with d2:
            st.subheader("🍩 Rating Proportions")
            donut = (
                alt.Chart(ddf).mark_arc(innerRadius=50, outerRadius=110)
                .encode(
                    theta="Number of Reviews:Q",
                    color=alt.Color("Star Rating:N", legend=alt.Legend(title="Rating"),
                                    scale=_chart_star_colors(ddf["Star Rating"].tolist())),
                    tooltip=["Star Rating", alt.Tooltip("Number of Reviews:Q", format=",")]
                ).properties(height=320, width=320)
            )
            st.altair_chart(donut, width="stretch")

    # Timeline
    if meta and meta.get("timeline") is not None and not meta["timeline"].empty:
        tl = meta["timeline"]
        st.markdown("---")
        st.subheader("📅 Review Activity Timeline")
        tl["Month"] = tl["_dt"].dt.to_period("M").astype(str)
        tl_agg = tl.groupby("Month").agg(
            **{"Reviews Submitted": ("rating", "size"),
               "Average Rating": ("rating", "mean")}
        ).reset_index()
        tl_agg["Average Rating"] = tl_agg["Average Rating"].round(2)

        base = alt.Chart(tl_agg).encode(x=alt.X("Month:N", sort=None, title="Month"))
        bars = base.mark_bar(color="#818cf8", opacity=0.6).encode(
            y=alt.Y("Reviews Submitted:Q", title="Reviews Submitted"),
            tooltip=["Month", alt.Tooltip("Reviews Submitted:Q", format=","),
                      alt.Tooltip("Average Rating:Q", format=".2f")]
        )
        line = base.mark_line(color="#f59e0b", strokeWidth=2, point=True).encode(
            y=alt.Y("Average Rating:Q", title="Avg Rating", scale=alt.Scale(domain=[0, 5]))
        )
        combo = alt.layer(bars, line).resolve_scale(y="independent").properties(height=350)
        st.altair_chart(combo, width="stretch")

    with st.expander("📋 Raw Review Data (sample)"):
        st.dataframe(df.head(50), use_container_width=True)


def render_review_intelligence():
    st.header("🧠 Review Intelligence")
    st.markdown(
        "Sentiment uses **FastAPI** when `http://127.0.0.1:8000` is up; otherwise the same "
        "keyword heuristic as `fastapi_app`."
    )

    ratings = None
    dash_df = None
    if _clean_dataset_path():
        ratings = load_rating_series()
        dash_df = load_dashboard_df()

    # Example reviews
    example_reviews = {
        "Custom": "",
        "Positive example": "This product is amazing! Absolutely love it. Best purchase I've ever made. Highly recommend to everyone!",
        "Negative example": "Terrible quality, broke after one week. Complete waste of money. I want a refund. Worst product ever.",
        "Mixed example": "The product looks nice and arrived quickly, but the quality is disappointing. Not what I expected for the price.",
    }
    example_choice = st.selectbox("Choose an example or write your own", list(example_reviews.keys()))
    review_text = st.text_area(
        "Paste a review to analyze",
        value=example_reviews[example_choice],
        placeholder="This product is amazing! Five stars.",
        height=120,
    )

    analyze = st.button("🔍 Analyze Sentiment", type="primary")

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

            # Result display
            st.markdown("---")
            rc1, rc2, rc3 = st.columns(3)
            with rc1:
                emoji = "✅" if sentiment == "Positive" else ("❌" if sentiment == "Negative" else "⚖️")
                st.metric("Sentiment", f"{emoji} {sentiment}")
            with rc2:
                st.metric("Confidence Score", f"{confidence:.2%}")
            with rc3:
                st.metric("Positive Lean", f"{pos_weight:.2%}")

            # Confidence gauge
            st.markdown("**Sentiment Confidence Gauge**")
            gauge_df = pd.DataFrame({
                "Metric": ["Confidence"],
                "Score": [confidence],
            })
            gauge = (
                alt.Chart(gauge_df).mark_bar(
                    cornerRadiusEnd=8, height=30,
                    color="#34d399" if sentiment == "Positive" else (
                        "#f87171" if sentiment == "Negative" else "#fbbf24")
                ).encode(
                    x=alt.X("Score:Q", scale=alt.Scale(domain=[0, 1]),
                            title="Confidence (0 = uncertain, 1 = certain)"),
                    tooltip=[alt.Tooltip("Score:Q", format=".2%", title="Confidence")]
                ).properties(height=60)
            )
            st.altair_chart(gauge, width="stretch")

            # Word analysis
            hp, hn = _token_polarity_hits(lower)
            word_df = pd.DataFrame({
                "Category": ["Positive Signals", "Negative Signals"],
                "Word Count": [hp, hn],
            })
            w1, w2 = st.columns(2)
            with w1:
                st.markdown("**Sentiment Word Analysis**")
                word_chart = (
                    alt.Chart(word_df)
                    .mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6)
                    .encode(
                        x=alt.X("Category:N", title=None),
                        y=alt.Y("Word Count:Q", title="Keyword Hits"),
                        color=alt.Color("Category:N", legend=None,
                                        scale=alt.Scale(domain=["Positive Signals", "Negative Signals"],
                                                        range=["#34d399", "#f87171"])),
                        tooltip=["Category", "Word Count"]
                    ).properties(height=250)
                )
                st.altair_chart(word_chart, width="stretch")

            with w2:
                st.markdown("**Your Text vs Dataset Prior**")
                if ratings is not None and len(ratings):
                    dataset_high = float(np.mean(ratings >= 4.0))
                    cmp = pd.DataFrame({
                        "Comparison": ["Dataset (Rating ≥ 4 Share)", "Your Text (Positive Lean)"],
                        "Score": [dataset_high, pos_weight],
                    })
                    cmp_chart = (
                        alt.Chart(cmp)
                        .mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6)
                        .encode(
                            x=alt.X("Comparison:N", title=None),
                            y=alt.Y("Score:Q", title="Probability / Score",
                                    scale=alt.Scale(domain=[0, 1])),
                            color=alt.Color("Comparison:N", legend=None,
                                            scale=alt.Scale(range=["#818cf8", "#f59e0b"])),
                            tooltip=["Comparison", alt.Tooltip("Score:Q", format=".2%")]
                        ).properties(height=250)
                    )
                    st.altair_chart(cmp_chart, width="stretch")
                else:
                    st.info("Dataset prior needs `clean_data.parquet`.")

    # Dataset-wide sentiment overview
    if dash_df is not None and "text" in dash_df.columns and "rating" in dash_df.columns:
        st.markdown("---")
        st.subheader("📊 Dataset Sentiment Overview")
        st.markdown("Sampled dataset-wide sentiment analysis based on the local heuristic model.")
        sample_size = st.slider("Sample size for analysis", 500, 5000, 1000, step=500, key="sent_sample")
        sample = dash_df.sample(n=min(sample_size, len(dash_df)), random_state=42)
        results = sample["text"].str.lower().apply(
            lambda t: classify_review_sentiment(t) if isinstance(t, str) and t.strip() else ("Neutral", 0.48)
        )
        sample["Predicted Sentiment"] = [r[0] for r in results]
        sample["Star Rating"] = sample["rating"].apply(lambda x: f"{int(x)} Star")

        s1, s2 = st.columns(2)
        with s1:
            st.markdown("**Predicted Sentiment Distribution**")
            sent_counts = sample["Predicted Sentiment"].value_counts().reset_index()
            sent_counts.columns = ["Sentiment", "Count"]
            sc = (
                alt.Chart(sent_counts).mark_arc(innerRadius=50, outerRadius=110)
                .encode(
                    theta="Count:Q",
                    color=alt.Color("Sentiment:N", legend=alt.Legend(title="Sentiment"),
                                    scale=alt.Scale(domain=["Positive", "Neutral", "Negative"],
                                                    range=["#34d399", "#fbbf24", "#f87171"])),
                    tooltip=["Sentiment", alt.Tooltip("Count:Q", format=",")]
                ).properties(height=300, width=300)
            )
            st.altair_chart(sc, width="stretch")
        with s2:
            st.markdown("**Sentiment by Star Rating**")
            cross = sample.groupby(["Star Rating", "Predicted Sentiment"]).size().reset_index(name="Count")
            heat = (
                alt.Chart(cross).mark_bar()
                .encode(
                    x=alt.X("Star Rating:N", sort=None, title="Star Rating"),
                    y=alt.Y("Count:Q", title="Number of Reviews"),
                    color=alt.Color("Predicted Sentiment:N",
                                    legend=alt.Legend(title="Sentiment"),
                                    scale=alt.Scale(domain=["Positive", "Neutral", "Negative"],
                                                    range=["#34d399", "#fbbf24", "#f87171"])),
                    tooltip=["Star Rating", "Predicted Sentiment",
                             alt.Tooltip("Count:Q", format=",")]
                ).properties(height=300)
            )
            st.altair_chart(heat, width="stretch")


def check_spark_connection(bundle):
    try:
        # Check if the Java backend is still responsive
        bundle[0].sparkContext.version
        return True
    except Exception:
        return False

@st.cache_resource(show_spinner="Initializing ALS Collaborative Filtering Engine...", validate=check_spark_connection)
def load_spark_and_als():
    import os
    import sys
    import subprocess
    
    # Ensure PySpark workers use the exact same Python version as the driver
    os.environ['PYSPARK_PYTHON'] = sys.executable
    os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable

    if "JAVA_HOME" not in os.environ:
        try:
            os.environ["JAVA_HOME"] = subprocess.check_output(
                ["/usr/libexec/java_home", "-v", "17"]
            ).decode().strip()
        except Exception:
            pass

    from pyspark.sql import SparkSession
    from pyspark.ml import PipelineModel
    from pyspark.ml.recommendation import ALSModel

    spark = SparkSession.builder \
        .appName('Streamlit_ALS') \
        .config('spark.driver.memory', '4g') \
        .getOrCreate()
        
    pipe = PipelineModel.load("models/feature_pipeline")
    als = ALSModel.load("models/als_recommendation_model")
    
    # Extract indexers
    user_indexer = pipe.stages[0]
    product_labels = pipe.stages[1].labels
    
    return spark, user_indexer, als, product_labels

def render_als_recommendations():
    st.header("🛒 ALS Product Recommendations")
    st.markdown(
        "Powered by our **Alternating Least Squares (ALS)** Matrix Factorization engine. "
        "This model analyzes latent user-item interaction patterns across the dataset to surface "
        "highly personalized product suggestions in real-time."
    )
    
    try:
        spark, user_indexer, als, product_labels = load_spark_and_als()
    except Exception as e:
        st.error(f"Failed to start PySpark or load models. Make sure Notebooks 3 & 4 are completed. Error: {e}")
        return

    # Grab an example user id to pre-fill
    try:
        example_user = user_indexer.labels[100]
    except Exception:
        example_user = sample_example_user_id()

    user_id = st.text_input(
        "Enter a User ID to get recommendations",
        value=example_user,
        help="Must be an exact string from the processed dataset."
    )
    
    num_recs = st.slider("Number of recommendations to generate", 5, 20, 10)
    
    go = st.button("✨ Generate Recommendations", type="primary")

    if go and user_id.strip():
        uid = user_id.strip()
        
        with st.spinner("Running ALS Matrix Factorization inference..."):
            df = spark.createDataFrame([(uid,)], ["user_id"])
            try:
                df_idx = user_indexer.transform(df)
                uid_idx = df_idx.collect()[0]["user_id_index"]
            except Exception:
                st.warning(f"User ID '{uid}' not found in the trained StringIndexer vocabulary. Please try another one.")
                return

            # Request predictions
            subset = spark.createDataFrame([(int(uid_idx),)], ["user_id_index"])
            try:
                recs = als.recommendForUserSubset(subset, num_recs)
                rows = recs.collect()
            except Exception as e:
                st.error(f"ALS inference failed: {e}")
                return

            if not rows or not rows[0].recommendations:
                st.warning("No recommendations could be generated for this user.")
                return

            results = []
            for r in rows[0].recommendations:
                asin = product_labels[r.product_id_index]
                score = r.rating
                results.append({"Product ASIN": asin, "Predicted Rating": round(score, 2)})

            res_df = pd.DataFrame(results)
            
            st.success(f"Generated {len(res_df)} personalized suggestions for {uid}")
            
            c1, c2 = st.columns([1, 2])
            with c1:
                st.dataframe(res_df, hide_index=True, use_container_width=True)
            with c2:
                rec_chart = (
                    alt.Chart(res_df)
                    .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4, color="#f59e0b")
                    .encode(
                        x=alt.X("Product ASIN:N", sort="-y", title="Recommended Product"),
                        y=alt.Y("Predicted Rating:Q", title="ALS Score (Predicted Stars)", scale=alt.Scale(zero=False)),
                        tooltip=["Product ASIN", "Predicted Rating"]
                    ).properties(height=300)
                )
                st.altair_chart(rec_chart, width="stretch")
                st.caption(
                    "💡 **How to interpret this:** The ALS Score represents the algorithm's confidence "
                    "in the recommendation. Scores above 5.0 indicate an extremely high mathematical "
                    "affinity between the user's past behaviors and the item."
                )


def render_home():
    st.title("📦 Amazon Product Recommendation & Analytics")
    st.markdown(
        "End-to-end recommendation system built on the "
        "[McAuley-Lab/Amazon-Reviews-2023](https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023) "
        "dataset — from raw data ingestion to model serving."
    )

    # Quick stats
    n = load_review_count()
    ratings = load_rating_series()
    if n and ratings is not None:
        h1, h2, h3, h4 = st.columns(4)
        with h1:
            st.metric("Total Reviews", f"{n:,}")
        with h2:
            st.metric("Average Rating", f"{float(np.mean(ratings)):.2f} ★")
        with h3:
            pct_high = float(np.mean(ratings >= 4.0)) * 100
            st.metric("4+ Star Reviews", f"{pct_high:.1f}%")
        with h4:
            median_r = float(np.median(ratings))
            st.metric("Median Rating", f"{median_r:.1f} ★")

    st.markdown("---")

    # Pipeline architecture
    st.subheader("🔄 Pipeline Architecture")
    p1, p2, p3, p4 = st.columns(4)
    with p1:
        st.markdown("""
        **① Data Ingestion**
        - Raw JSONL → Parquet
        - Schema validation
        - Exploratory analysis
        - *Notebook 01*
        """)
    with p2:
        st.markdown("""
        **② Preprocessing**
        - Null handling & dedup
        - Type casting
        - AQE-optimized Spark
        - *Notebook 02*
        """)
    with p3:
        st.markdown("""
        **③ Feature Engineering**
        - User/product indexing
        - TF-IDF text features
        - Review count features
        - *Notebook 03*
        """)
    with p4:
        st.markdown("""
        **④ Model Tournament**
        - ALS, LR, RF, NB, KMeans
        - MLflow experiment tracking
        - Model comparison
        - *Notebook 04*
        """)

    st.markdown("---")

    # Tech stack
    st.subheader("🛠️ Technology Stack")
    t1, t2, t3 = st.columns(3)
    with t1:
        st.markdown("""
        **Data Processing**
        - Apache Spark (PySpark)
        - Adaptive Query Execution
        - PyArrow / Parquet
        """)
    with t2:
        st.markdown("""
        **ML / Analytics**
        - Spark MLlib (ALS, KMeans)
        - Scikit-learn pipelines
        - MLflow experiment tracking
        """)
    with t3:
        st.markdown("""
        **Serving & UI**
        - FastAPI (sentiment API)
        - Streamlit (this dashboard)
        - Altair visualizations
        """)

    st.markdown("---")
    st.info(
        "Use the **sidebar** to explore:\n\n"
        "- 📊 **Business Dashboard** — KPIs, rating analysis, review trends, verified purchase insights\n"
        "- 👤 **Customer Personalization** — per-user review patterns and product preferences\n"
        "- 🛒 **ALS Recommendations** — Live collaborative filtering product suggestions\n"
        "- 🧠 **Review Intelligence** — real-time sentiment analysis with dataset comparison\n"
    )


def main():
    st.sidebar.title("📦 Navigation")
    page = st.sidebar.radio(
        "Go to",
        ["🏠 Home", "📊 Business Dashboard", "👤 Customer Personalization", "🛒 ALS Recommendations", "🧠 Review Intelligence"],
    )

    st.sidebar.markdown("---")
    st.sidebar.caption(
        "**Amazon Recommendation System**\n\n"
        "DATA 228 · Spring 2026\n\n"
        "5 Models · 994K Reviews · 55K Products"
    )

    if page == "🏠 Home":
        render_home()
    elif page == "📊 Business Dashboard":
        render_business_dashboard()
    elif page == "👤 Customer Personalization":
        render_customer_personalization()
    elif page == "🛒 ALS Recommendations":
        render_als_recommendations()
    else:
        render_review_intelligence()


main()
