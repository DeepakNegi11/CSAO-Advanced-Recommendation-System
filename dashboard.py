# =============================================================================
#  CSAO Recommendation System — Streamlit Dashboard
#  Run with:  streamlit run dashboard.py
# =============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import time
import random

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CSAO Dashboard — Zomathon 2026",
    page_icon="🍕",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Zomato brand colours ───────────────────────────────────────────────────────
RED   = "#E23744"
DRED  = "#B02030"
DARK  = "#1A1A2E"
LGRAY = "#F5F5F5"

# ── Global CSS ────────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
  /* Main background */
  .stApp {{ background-color: #F9F9F9; }}

  /* Sidebar */
  section[data-testid="stSidebar"] {{
      background: {DARK};
  }}
  section[data-testid="stSidebar"] * {{
      color: #FFFFFF !important;
  }}

  /* Metric cards */
  div[data-testid="metric-container"] {{
      background: #FFFFFF;
      border: 1px solid #EEEEEE;
      border-left: 5px solid {RED};
      border-radius: 8px;
      padding: 16px 20px;
      box-shadow: 0 2px 6px rgba(0,0,0,0.06);
  }}

  /* Section headers */
  h2 {{ color: {DARK} !important; border-bottom: 3px solid {RED}; padding-bottom: 6px; }}
  h3 {{ color: {DRED} !important; }}

  /* Recommendation cards */
  .rec-card {{
      background: #FFFFFF;
      border-radius: 10px;
      border: 1px solid #EEE;
      padding: 14px 18px;
      margin-bottom: 10px;
      box-shadow: 0 2px 8px rgba(226,55,68,0.08);
      border-left: 5px solid {RED};
  }}
  .rec-rank {{ font-size: 22px; font-weight: bold; color: {RED}; }}
  .rec-item {{ font-size: 17px; font-weight: 600; color: {DARK}; }}
  .rec-meta {{ font-size: 13px; color: #888; margin-top: 4px; }}
  .rec-score {{ font-size: 13px; color: {DRED}; font-weight: 600; }}

  /* Top banner */
  .banner {{
      background: linear-gradient(135deg, {DARK} 0%, {DRED} 100%);
      color: white;
      padding: 28px 36px;
      border-radius: 12px;
      margin-bottom: 28px;
  }}
  .banner h1 {{ color: white !important; font-size: 28px; margin: 0; }}
  .banner p  {{ color: #FFD0D4; margin: 4px 0 0 0; font-size: 15px; }}
</style>
""", unsafe_allow_html=True)


# =============================================================================
#  MOCK DATA  (replace with real model/data calls when integrating)
# =============================================================================

ITEM_CATALOGUE = [
    {"item_id": "BEV001", "name": "Masala Chai",        "category": "Beverage", "price": 49,  "margin": 0.42, "popularity": 0.91},
    {"item_id": "BEV002", "name": "Cold Coffee",         "category": "Beverage", "price": 89,  "margin": 0.38, "popularity": 0.85},
    {"item_id": "BEV003", "name": "Fresh Lime Soda",     "category": "Beverage", "price": 59,  "margin": 0.40, "popularity": 0.78},
    {"item_id": "DES001", "name": "Gulab Jamun (2 pcs)", "category": "Dessert",  "price": 69,  "margin": 0.45, "popularity": 0.88},
    {"item_id": "DES002", "name": "Chocolate Brownie",   "category": "Dessert",  "price": 99,  "margin": 0.50, "popularity": 0.82},
    {"item_id": "DES003", "name": "Rasmalai",             "category": "Dessert",  "price": 79,  "margin": 0.47, "popularity": 0.76},
    {"item_id": "SID001", "name": "Garlic Naan",          "category": "Side",     "price": 39,  "margin": 0.30, "popularity": 0.80},
    {"item_id": "SID002", "name": "Papad Basket",         "category": "Side",     "price": 29,  "margin": 0.35, "popularity": 0.72},
    {"item_id": "SID003", "name": "Raita",                "category": "Side",     "price": 49,  "margin": 0.28, "popularity": 0.68},
    {"item_id": "SID004", "name": "Masala Fries",         "category": "Side",     "price": 79,  "margin": 0.33, "popularity": 0.74},
]

FEATURE_IMPORTANCE = {
    "price_ratio":                0.187,
    "user_beverage_interaction":  0.154,
    "item_popularity_score":      0.132,
    "cart_value":                 0.118,
    "user_dessert_interaction":   0.101,
    "item_margin":                0.089,
    "hour_of_day":                0.072,
    "beverage_preference_score":  0.061,
    "dessert_preference_score":   0.048,
    "high_margin_flag":           0.038,
}

MODEL_METRICS = {
    "NDCG@5":         {"model": 0.74, "baseline": 0.51},
    "Precision@5":    {"model": 0.61, "baseline": 0.38},
    "AUC-ROC":        {"model": 0.81, "baseline": 0.61},
    "MRR@5":          {"model": 0.68, "baseline": 0.44},
    "Cold-Start P@5": {"model": 0.43, "baseline": 0.29},
}

AB_DAYS = list(range(1, 15))
np.random.seed(42)
AB_DATA = pd.DataFrame({
    "Day": AB_DAYS,
    "Control_CTR":    [7.8 + np.random.uniform(-0.3, 0.3) for _ in AB_DAYS],
    "Treatment_CTR":  [9.4 + np.random.uniform(-0.2, 0.4) + d * 0.03 for d, _ in enumerate(AB_DAYS)],
    "Control_AOV":    [340 + np.random.uniform(-8, 8)  for _ in AB_DAYS],
    "Treatment_AOV":  [361 + np.random.uniform(-6, 6)  + d * 0.4 for d, _ in enumerate(AB_DAYS)],
})

LATENCY_COMPONENTS = {
    "API Gateway + Auth":    5,
    "Redis Feature Fetch":   2,
    "Candidate Assembly":    1,
    "Feature Engineering":   5,
    "LightGBM Inference":   15,
    "Re-rank + Serialise":   3,
}

KPI_SCENARIOS = pd.DataFrame({
    "Scenario":      ["Conservative", "Base Case", "Optimistic"],
    "AOV_Lift_INR":  [12, 20, 28],
    "CTR_Lift_pct":  [8,  15,  22],
    "CTO_Lift_pct":  [3,   5,   8],
    "Daily_Rev_INR": [420_000, 700_000, 980_000],
})


# =============================================================================
#  HELPER — Mock recommender
# =============================================================================

def mock_recommend(user_features: dict) -> pd.DataFrame:
    """Simulate model scoring with user context."""
    items = pd.DataFrame(ITEM_CATALOGUE)
    bev_pref  = user_features.get("beverage_preference_score", 0.5)
    des_pref  = user_features.get("dessert_preference_score",  0.5)
    cart_val  = user_features.get("cart_value", 300)
    hour      = user_features.get("hour_of_day", 13)
    freq      = user_features.get("order_frequency_30d", 5)

    is_cold   = freq < 2

    scores = []
    for _, row in items.iterrows():
        if is_cold:
            score = row["popularity"] * 0.8 + np.random.uniform(0, 0.2)
        else:
            price_ratio  = row["price"] / max(cart_val, 1)
            bev_interact = bev_pref * (1 if row["category"] == "Beverage" else 0)
            des_interact = des_pref * (1 if row["category"] == "Dessert"  else 0)
            dinner_boost = 0.1 if (hour >= 19 and row["category"] == "Dessert") else 0
            score = (
                0.25 * row["popularity"]
                + 0.20 * row["margin"]
                + 0.18 * bev_interact
                + 0.16 * des_interact
                + 0.12 * (1 - min(price_ratio, 1))
                + 0.09 * dinner_boost
                + np.random.uniform(0, 0.05)
            )
        scores.append(round(score, 3))

    items["score"] = scores
    items = items.sort_values("score", ascending=False)

    # Diversity: max 2 per category
    selected, cat_count = [], {}
    for _, row in items.iterrows():
        cat = row["category"]
        if cat_count.get(cat, 0) < 2:
            selected.append(row)
            cat_count[cat] = cat_count.get(cat, 0) + 1
        if len(selected) == 5:
            break

    return pd.DataFrame(selected).reset_index(drop=True)


# =============================================================================
#  SIDEBAR
# =============================================================================

with st.sidebar:
    st.markdown("## 🍕 CSAO Dashboard")
    st.markdown("**Zomathon 2026**")
    st.markdown("---")
    page = st.radio("Navigate to", [
        "📊 Model Metrics",
        "🎯 Live Demo",
        "❄️ Cold Start Analysis",
        "📈 Feature Importance",
        "🆚 Baseline Comparison",
        "🧪 A/B Test Simulation",
        "💰 Business KPIs",
    ])
    st.markdown("---")
    st.caption("Model: LightGBM + LLM Hybrid")
    st.caption("Candidates: Top-30 Hybrid Score")
    st.caption("P95 Latency: ~30ms")


# ── Banner ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="banner">
  <h1>🍕 Cart Super Add-On (CSAO) Recommendation System</h1>
  <p>Hybrid LightGBM + LLM engine · Real-time inference · Zomathon 2026</p>
</div>
""", unsafe_allow_html=True)


# =============================================================================
#  PAGE 1 — MODEL METRICS
# =============================================================================

if page == "📊 Model Metrics":
    st.markdown("## 📊 Model Performance Metrics")

    # KPI row
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("NDCG@5",      "0.74", "+45% vs baseline")
    col2.metric("Precision@5", "0.61", "+61% vs baseline")
    col3.metric("AUC-ROC",     "0.81", "+33% vs baseline")
    col4.metric("MRR@5",       "0.68", "+55% vs baseline")

    st.markdown("---")
    st.markdown("### Metric Deep Dive")

    # Radar chart
    metrics     = list(MODEL_METRICS.keys())
    model_vals  = [MODEL_METRICS[m]["model"]    for m in metrics]
    base_vals   = [MODEL_METRICS[m]["baseline"] for m in metrics]

    fig_radar = go.Figure()
    fig_radar.add_trace(go.Scatterpolar(
        r=model_vals + [model_vals[0]], theta=metrics + [metrics[0]],
        fill="toself", name="CSAO Model",
        line=dict(color=RED, width=2), fillcolor="rgba(226,55,68,0.15)"
    ))
    fig_radar.add_trace(go.Scatterpolar(
        r=base_vals + [base_vals[0]], theta=metrics + [metrics[0]],
        fill="toself", name="Baseline",
        line=dict(color="#AAAAAA", width=2, dash="dash"),
        fillcolor="rgba(170,170,170,0.10)"
    ))
    fig_radar.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        showlegend=True, height=420,
        paper_bgcolor="white", plot_bgcolor="white",
        legend=dict(orientation="h", y=-0.15)
    )
    st.plotly_chart(fig_radar, use_container_width=True)

    # Bar chart per metric
    st.markdown("### Model vs Baseline — All Metrics")
    fig_bar = go.Figure()
    fig_bar.add_trace(go.Bar(
        name="CSAO Model", x=metrics, y=model_vals,
        marker_color=RED, text=[f"{v:.2f}" for v in model_vals],
        textposition="outside"
    ))
    fig_bar.add_trace(go.Bar(
        name="Baseline", x=metrics, y=base_vals,
        marker_color="#CCCCCC", text=[f"{v:.2f}" for v in base_vals],
        textposition="outside"
    ))
    fig_bar.update_layout(
        barmode="group", height=380, yaxis=dict(range=[0, 1.0], title="Score"),
        paper_bgcolor="white", plot_bgcolor="white",
        legend=dict(orientation="h", y=1.12)
    )
    st.plotly_chart(fig_bar, use_container_width=True)

    # Latency donut
    st.markdown("### ⏱️ Latency Budget Breakdown (P95 ~30ms)")
    lat_labels = list(LATENCY_COMPONENTS.keys())
    lat_vals   = list(LATENCY_COMPONENTS.values())
    colors     = [RED, DRED, "#FF6B7A", "#FF9EA6", "#FFB3BA", "#FFCDD2"]
    fig_lat = go.Figure(go.Pie(
        labels=lat_labels, values=lat_vals, hole=0.55,
        marker=dict(colors=colors),
        textinfo="label+value", texttemplate="%{label}<br>%{value}ms"
    ))
    fig_lat.update_layout(
        height=380, paper_bgcolor="white",
        annotations=[dict(text="~30ms<br>Total", x=0.5, y=0.5,
                          font_size=18, showarrow=False, font_color=DARK)]
    )
    st.plotly_chart(fig_lat, use_container_width=True)


# =============================================================================
#  PAGE 2 — LIVE DEMO
# =============================================================================

elif page == "🎯 Live Demo":
    st.markdown("## 🎯 Live Recommendation Demo")
    st.markdown("Adjust the cart and user profile below to see real-time recommendations.")

    col_in, col_out = st.columns([1, 1.4], gap="large")

    with col_in:
        st.markdown("### 🛒 Cart & User Profile")
        cart_value     = st.slider("Cart Value (₹)",        100, 1200, 350, 50)
        num_items      = st.slider("Items in Cart",          1, 8, 2)
        hour_of_day    = st.slider("Hour of Day",            6, 23, 13)
        order_freq     = st.slider("Orders in Last 30 Days", 0, 20, 5)
        bev_pref       = st.slider("Beverage Preference",    0.0, 1.0, 0.6, 0.05)
        des_pref       = st.slider("Dessert Preference",     0.0, 1.0, 0.4, 0.05)
        avg_order_val  = st.slider("Avg Order Value (₹)",   100, 800, 300, 25)

        is_cold = order_freq < 2
        if is_cold:
            st.warning("⚠️ Cold-start user detected — using heuristic fallback")
        else:
            st.success("✅ Warm user — LightGBM model active")

        run = st.button("🔍 Get Recommendations", type="primary", use_container_width=True)

    with col_out:
        st.markdown("### 🏆 Top-5 Add-On Recommendations")
        if run:
            with st.spinner("Scoring candidates..."):
                time.sleep(0.4)  # simulate latency
                recs = mock_recommend({
                    "cart_value":                cart_value,
                    "num_items_in_cart":         num_items,
                    "hour_of_day":               hour_of_day,
                    "order_frequency_30d":       order_freq,
                    "beverage_preference_score": bev_pref,
                    "dessert_preference_score":  des_pref,
                    "avg_order_value":           avg_order_val,
                })

            emoji_map = {"Beverage": "🥤", "Dessert": "🍮", "Side": "🍟"}
            for i, row in recs.iterrows():
                emoji = emoji_map.get(row["category"], "🍽️")
                score_bar = "█" * int(row["score"] * 10) + "░" * (10 - int(row["score"] * 10))
                st.markdown(f"""
                <div class="rec-card">
                  <span class="rec-rank">#{i+1}</span>&nbsp;&nbsp;
                  <span class="rec-item">{emoji} {row['name']}</span><br>
                  <span class="rec-meta">
                    {row['category']} &nbsp;·&nbsp; ₹{int(row['price'])}
                    &nbsp;·&nbsp; Margin {int(row['margin']*100)}%
                  </span><br>
                  <span class="rec-score">Score: {row['score']:.3f} &nbsp; {score_bar}</span>
                </div>
                """, unsafe_allow_html=True)

            st.markdown(f"**Simulated latency:** `{random.randint(22, 35)}ms`")
        else:
            st.info("👈 Set the profile and click **Get Recommendations**")


# =============================================================================
#  PAGE 3 — COLD START ANALYSIS
# =============================================================================

elif page == "❄️ Cold Start Analysis":
    st.markdown("## ❄️ Cold Start vs Warm User Analysis")

    col1, col2, col3 = st.columns(3)
    col1.metric("Cold-Start Users",      "22% of base",  "order_freq < 2")
    col2.metric("Cold-Start P@5",        "0.43",         "-18pp vs warm")
    col3.metric("Incomplete Meal Profs", "18% of users", "e.g. lunch only")

    st.markdown("---")

    # Tier cascade diagram
    st.markdown("### 🔀 Cold-Start Routing Cascade")
    tiers = ["New User\n+ Cart Category Known", "New User\nNo Cart Signal", "Incomplete\nMeal-Time Profile"]
    strategies = [
        "Category-Aware Heuristic\n(Top-3 in category + popularity pad)",
        "Global Popularity-Margin Hybrid\n(60% purchase rate + 40% margin)",
        "LLM Preference Inference\n(Claude Haiku → pseudo feature vector)"
    ]
    precisions = [0.43, 0.36, 0.40]
    colors_tier = [RED, DRED, "#FF6B7A"]

    fig_tier = go.Figure()
    for i, (tier, strat, prec, col) in enumerate(zip(tiers, strategies, precisions, colors_tier)):
        fig_tier.add_trace(go.Bar(
            x=[tier], y=[prec], name=f"Tier {i+1}",
            marker_color=col,
            text=[f"P@5 = {prec}"], textposition="outside",
            hovertext=strat,
        ))
    fig_tier.add_hline(y=0.61, line_dash="dash", line_color="#333",
                       annotation_text="Warm user P@5 = 0.61", annotation_position="top right")
    fig_tier.update_layout(
        height=380, yaxis=dict(range=[0, 0.75], title="Precision@5"),
        paper_bgcolor="white", plot_bgcolor=LGRAY,
        showlegend=False, bargap=0.4
    )
    st.plotly_chart(fig_tier, use_container_width=True)

    # Comparison table
    st.markdown("### 📋 Strategy Comparison")
    df_cs = pd.DataFrame({
        "User Type":          ["Warm User (freq ≥ 2)", "Cold — Category Known", "Cold — No Signal", "Cold — Incomplete Meal"],
        "Routing":            ["LightGBM Ranker", "Category Heuristic", "Popularity-Margin Hybrid", "LLM + Warm-Start"],
        "Precision@5":        [0.61, 0.43, 0.36, 0.40],
        "Est. AOV Lift (₹)":  [20, 11, 7, 9],
        "Coverage":           ["60%", "18%", "14%", "8%"],
    })
    st.dataframe(df_cs.style.highlight_max(subset=["Precision@5", "Est. AOV Lift (₹)"],
                 color="#FFE0E3"), use_container_width=True)


# =============================================================================
#  PAGE 4 — FEATURE IMPORTANCE
# =============================================================================

elif page == "📈 Feature Importance":
    st.markdown("## 📈 Feature Importance (LightGBM SHAP-based)")

    features = list(FEATURE_IMPORTANCE.keys())
    importances = list(FEATURE_IMPORTANCE.values())
    df_fi = pd.DataFrame({"Feature": features, "Importance": importances})
    df_fi = df_fi.sort_values("Importance", ascending=True)

    fig_fi = go.Figure(go.Bar(
        x=df_fi["Importance"], y=df_fi["Feature"],
        orientation="h",
        marker=dict(
            color=df_fi["Importance"],
            colorscale=[[0, "#FFCDD2"], [0.5, DRED], [1.0, RED]],
            showscale=False,
        ),
        text=[f"{v:.3f}" for v in df_fi["Importance"]],
        textposition="outside",
    ))
    fig_fi.update_layout(
        height=480, xaxis=dict(title="Relative Importance", range=[0, 0.25]),
        paper_bgcolor="white", plot_bgcolor=LGRAY,
        margin=dict(l=220)
    )
    st.plotly_chart(fig_fi, use_container_width=True)

    # Feature groups
    st.markdown("### Feature Group Contribution")
    groups = {
        "User-Item Interaction": ["price_ratio", "user_beverage_interaction", "user_dessert_interaction"],
        "Item Quality":          ["item_popularity_score", "item_margin", "high_margin_flag"],
        "Temporal":              ["hour_of_day"],
        "User History":          ["beverage_preference_score", "dessert_preference_score"],
        "Cart Context":          ["cart_value"],
    }
    group_totals = {g: sum(FEATURE_IMPORTANCE[f] for f in feats) for g, feats in groups.items()}
    fig_grp = go.Figure(go.Pie(
        labels=list(group_totals.keys()),
        values=list(group_totals.values()),
        hole=0.4,
        marker=dict(colors=[RED, DRED, "#FF6B7A", "#FF9EA6", "#FFCDD2"]),
        textinfo="label+percent",
    ))
    fig_grp.update_layout(height=360, paper_bgcolor="white")
    st.plotly_chart(fig_grp, use_container_width=True)

    st.info("💡 **price_ratio** (item price ÷ avg order value) is the single strongest signal — "
            "users are most likely to add items that feel affordable relative to their cart.")


# =============================================================================
#  PAGE 5 — BASELINE COMPARISON
# =============================================================================

elif page == "🆚 Baseline Comparison":
    st.markdown("## 🆚 CSAO Model vs Popularity-Margin Baseline")

    metrics     = list(MODEL_METRICS.keys())
    model_vals  = [MODEL_METRICS[m]["model"]    for m in metrics]
    base_vals   = [MODEL_METRICS[m]["baseline"] for m in metrics]
    lifts       = [round((m - b) / b * 100, 1)  for m, b in zip(model_vals, base_vals)]

    # Lift summary
    cols = st.columns(len(metrics))
    for col, metric, lift in zip(cols, metrics, lifts):
        col.metric(metric, f"+{lift}%", "vs baseline")

    st.markdown("---")

    # Side-by-side bar
    fig_cmp = go.Figure()
    fig_cmp.add_trace(go.Bar(
        name="Baseline (Popularity-Margin)",
        x=metrics, y=base_vals,
        marker_color="#CCCCCC",
        text=[f"{v:.2f}" for v in base_vals], textposition="inside",
    ))
    fig_cmp.add_trace(go.Bar(
        name="CSAO Model (LightGBM + LLM)",
        x=metrics, y=model_vals,
        marker_color=RED,
        text=[f"{v:.2f}" for v in model_vals], textposition="inside",
    ))
    fig_cmp.update_layout(
        barmode="group", height=400,
        yaxis=dict(range=[0, 1.0], title="Score"),
        paper_bgcolor="white", plot_bgcolor=LGRAY,
        legend=dict(orientation="h", y=1.12),
    )
    st.plotly_chart(fig_cmp, use_container_width=True)

    # Lift waterfall
    st.markdown("### 📐 Relative Lift Over Baseline (%)")
    fig_lift = go.Figure(go.Bar(
        x=metrics, y=lifts,
        marker_color=[RED if l > 40 else DRED for l in lifts],
        text=[f"+{l}%" for l in lifts], textposition="outside",
    ))
    fig_lift.update_layout(
        height=340, yaxis=dict(title="Lift (%)", range=[0, 80]),
        paper_bgcolor="white", plot_bgcolor=LGRAY,
    )
    st.plotly_chart(fig_lift, use_container_width=True)

    st.markdown("### 📋 Full Comparison Table")
    df_cmp = pd.DataFrame({
        "Metric":    metrics,
        "Baseline":  base_vals,
        "Model":     model_vals,
        "Lift (%)":  [f"+{l}%" for l in lifts],
    })
    st.dataframe(df_cmp.style.applymap(
        lambda v: f"color: {RED}; font-weight: bold" if isinstance(v, str) and "+" in v else "",
    ), use_container_width=True, hide_index=True)


# =============================================================================
#  PAGE 6 — A/B TEST SIMULATION
# =============================================================================

elif page == "🧪 A/B Test Simulation":
    st.markdown("## 🧪 A/B Test Results Simulation (14-Day Run)")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Experiment Duration",   "14 days")
    col2.metric("Users per Arm",         "~400K")
    col3.metric("Avg CTR Lift",          "+1.6pp",  "8.0% → 9.6%")
    col4.metric("Avg AOV Lift",          "+₹21",    "₹340 → ₹361")

    st.markdown("---")

    tab1, tab2 = st.tabs(["📊 CTR Over Time", "💵 AOV Over Time"])

    with tab1:
        fig_ctr = go.Figure()
        fig_ctr.add_trace(go.Scatter(
            x=AB_DATA["Day"], y=AB_DATA["Control_CTR"],
            name="Control (Baseline)", line=dict(color="#AAAAAA", width=2, dash="dash"),
            mode="lines+markers"
        ))
        fig_ctr.add_trace(go.Scatter(
            x=AB_DATA["Day"], y=AB_DATA["Treatment_CTR"],
            name="Treatment (CSAO Model)", line=dict(color=RED, width=2),
            mode="lines+markers", fill="tonexty",
            fillcolor="rgba(226,55,68,0.08)"
        ))
        fig_ctr.add_vline(x=7, line_dash="dot", line_color=DRED,
                          annotation_text="Day 7 checkpoint", annotation_position="top right")
        fig_ctr.update_layout(
            height=380, yaxis=dict(title="CSAO Rail CTR (%)"),
            xaxis=dict(title="Day"), paper_bgcolor="white", plot_bgcolor=LGRAY,
            legend=dict(orientation="h", y=1.12)
        )
        st.plotly_chart(fig_ctr, use_container_width=True)

    with tab2:
        fig_aov = go.Figure()
        fig_aov.add_trace(go.Scatter(
            x=AB_DATA["Day"], y=AB_DATA["Control_AOV"],
            name="Control AOV", line=dict(color="#AAAAAA", width=2, dash="dash"),
            mode="lines+markers"
        ))
        fig_aov.add_trace(go.Scatter(
            x=AB_DATA["Day"], y=AB_DATA["Treatment_AOV"],
            name="Treatment AOV", line=dict(color=RED, width=2),
            mode="lines+markers", fill="tonexty",
            fillcolor="rgba(226,55,68,0.08)"
        ))
        fig_aov.update_layout(
            height=380, yaxis=dict(title="Average Order Value (₹)"),
            xaxis=dict(title="Day"), paper_bgcolor="white", plot_bgcolor=LGRAY,
            legend=dict(orientation="h", y=1.12)
        )
        st.plotly_chart(fig_aov, use_container_width=True)

    # Guardrail table
    st.markdown("### 🛡️ Guardrail Metrics Status")
    df_guard = pd.DataFrame({
        "Guardrail":           ["User Complaint Rate", "API Error Rate", "P95 Latency", "Cold-Start AOV"],
        "Threshold":           ["≤ +0.5pp", "≤ 0.1%", "≤ 250ms", "No regression"],
        "Control":             ["2.1%", "0.04%", "145ms", "₹312"],
        "Treatment":           ["2.2%", "0.05%", "162ms", "₹319"],
        "Status":              ["✅ Pass", "✅ Pass", "✅ Pass", "✅ Pass"],
    })
    st.dataframe(df_guard, use_container_width=True, hide_index=True)

    st.success("✅ All guardrail metrics within acceptable bounds. Experiment cleared for full rollout.")


# =============================================================================
#  PAGE 7 — BUSINESS KPIs
# =============================================================================

elif page == "💰 Business KPIs":
    st.markdown("## 💰 Business KPI Projections")

    st.markdown("### Scenario Selector")
    scenario = st.select_slider(
        "Projected Impact Scenario",
        options=["Conservative", "Base Case", "Optimistic"],
        value="Base Case"
    )
    row = KPI_SCENARIOS[KPI_SCENARIOS["Scenario"] == scenario].iloc[0]

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("AOV Lift",          f"₹{row['AOV_Lift_INR']}",   "per CSAO-influenced order")
    col2.metric("CSAO CTR Lift",     f"+{row['CTR_Lift_pct']}%",  "vs baseline")
    col3.metric("Cart-to-Order Lift",f"+{row['CTO_Lift_pct']}%",  "vs baseline")
    col4.metric("Daily Rev Impact",  f"₹{row['Daily_Rev_INR']:,.0f}", "estimated")

    st.markdown("---")

    # Scenario comparison bar
    st.markdown("### 📊 All Scenarios — AOV Lift")
    fig_scen = go.Figure(go.Bar(
        x=KPI_SCENARIOS["Scenario"],
        y=KPI_SCENARIOS["AOV_Lift_INR"],
        marker_color=[LGRAY, RED, DRED],
        text=[f"₹{v}" for v in KPI_SCENARIOS["AOV_Lift_INR"]],
        textposition="outside",
    ))
    fig_scen.update_layout(
        height=320, yaxis=dict(title="AOV Lift (₹)", range=[0, 36]),
        paper_bgcolor="white", plot_bgcolor=LGRAY, showlegend=False,
    )
    st.plotly_chart(fig_scen, use_container_width=True)

    # Offline → KPI translation table
    st.markdown("### 🔗 Offline Metric → Business KPI Translation")
    df_kpi = pd.DataFrame({
        "Offline Metric":   ["NDCG@5 = 0.74", "Precision@5 = 0.61", "AUC-ROC = 0.81",
                             "MRR@5 = 0.68",   "Cold-Start P@5 = 0.43"],
        "Business KPI":     ["CSAO Rail CTR",  "Add-to-Cart Rate",   "AOV Lift",
                             "Cart-to-Order",  "New User Retention"],
        "Projected Impact": ["+1.5–2pp CTR",   "+7pp add rate",      "₹18–22 AOV lift",
                             "+5% C-to-O",     "+15% Day-7 retention"],
        "Confidence":       ["High", "High", "Medium", "Medium", "Medium"],
    })
    def color_conf(val):
        if val == "High":   return f"background-color: #FFE0E3; color: {DRED}; font-weight: bold"
        if val == "Medium": return "background-color: #FFF3CD; color: #856404"
        return ""
    st.dataframe(
        df_kpi.style.applymap(color_conf, subset=["Confidence"]),
        use_container_width=True, hide_index=True
    )

    # Revenue projection line
    st.markdown("### 📈 Daily Revenue Impact (Optimistic Ramp-Up)")
    ramp_days = list(range(1, 31))
    ramp_rev  = [980_000 * (1 - np.exp(-d / 10)) for d in ramp_days]
    fig_rev = go.Figure(go.Scatter(
        x=ramp_days, y=ramp_rev,
        mode="lines", fill="tozeroy",
        line=dict(color=RED, width=2),
        fillcolor="rgba(226,55,68,0.10)"
    ))
    fig_rev.update_layout(
        height=320,
        xaxis=dict(title="Day Post-Launch"),
        yaxis=dict(title="Daily Revenue Impact (₹)", tickformat=",.0f"),
        paper_bgcolor="white", plot_bgcolor=LGRAY,
    )
    st.plotly_chart(fig_rev, use_container_width=True)