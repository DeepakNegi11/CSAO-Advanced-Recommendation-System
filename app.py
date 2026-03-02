from fastapi import FastAPI
import lightgbm as lgb
import pandas as pd
from src.feature_engineering import create_features
import numpy as np

app = FastAPI()

model = lgb.Booster(model_file="data/csao_ranker.txt")
full_data = pd.read_csv("data/csao_data.csv")
items = full_data.drop_duplicates("item_id").reset_index(drop=True)

# ----------------------------
# HYBRID CANDIDATE GENERATION
# ----------------------------

# Average purchase rate per item
item_purchase_rate = full_data.groupby("item_id")["purchased"].mean()

# Average margin per item
item_margin_avg = full_data.groupby("item_id")["item_margin"].mean()

# Normalize margin
normalized_margin = item_margin_avg / item_margin_avg.max()

# Hybrid score (Popularity + Margin)
item_scores = (
    0.6 * item_purchase_rate +
    0.4 * normalized_margin
)

# Select Top 30 candidates
top_30_items = item_scores.sort_values(
    ascending=False
).head(30).index.tolist()

# Pre-store candidate items
candidate_items = items[
    items["item_id"].isin(top_30_items)
].reset_index(drop=True)

# ----------------------------
# COLD START: SEGMENT PROFILES
# ----------------------------
# Pre-compute per-category popularity for targeted cold-start fallback
category_top_items = (
    full_data.groupby(["item_category", "item_id"])["purchased"]
    .mean()
    .reset_index()
    .sort_values("purchased", ascending=False)
)


def cold_start_recommend(user_features: dict) -> list:
    """
    Improved cold start: uses cart contents to infer a preferred category,
    then returns the most popular items from that category first,
    padding with overall popular items if needed.
    """
    # Try to infer user preference from cart category hint if provided
    preferred_category = user_features.get("cart_dominant_category", None)

    if preferred_category:
        # Return top items from preferred category first
        cat_items = category_top_items[
            category_top_items["item_category"] == preferred_category
        ].head(3)["item_id"].tolist()

        # Merge with item metadata
        cat_recs = items[items["item_id"].isin(cat_items)].copy()
        cat_recs["final_score"] = cat_recs["item_id"].map(
            category_top_items.set_index("item_id")["purchased"]
        )
    else:
        cat_recs = pd.DataFrame()

    # Fill remaining slots from overall popular items (excluding already selected)
    already_selected = cat_recs["item_id"].tolist() if not cat_recs.empty else []
    slots_needed = 5 - len(already_selected)

    popular_items = (
        items[~items["item_id"].isin(already_selected)]
        .sort_values("item_popularity_score", ascending=False)
        .head(slots_needed)
        .copy()
    )
    popular_items["final_score"] = popular_items["item_popularity_score"]

    result = pd.concat([cat_recs, popular_items], ignore_index=True)
    return result[["item_id", "item_price", "item_category", "final_score"]].to_dict(orient="records")


@app.post("/recommend")
def recommend(user_features: dict):

    # ---- FIX 1: Better cold start handling ----
    # Previously: returned flat popularity list with no category awareness
    # Now: category-aware cold start with hybrid fallback
    if user_features.get("order_frequency_30d", 0) < 2:
        return cold_start_recommend(user_features)

    # ---- FIX 2: Candidate items now used in scoring ----
    # Previously: candidate_items was computed at startup but never used —
    #             the model was scoring an empty cart_df with no item rows.
    # Now: we cross-join user features with candidate items so the model
    #      scores each (user, item) pair as intended.

    user_df = pd.DataFrame([user_features])

    # Ensure required base columns exist
    required_base_cols = [
        "avg_order_value",
        "beverage_preference_score",
        "dessert_preference_score",
        "order_frequency_30d",
        "cart_value",
        "num_items_in_cart",
        "hour_of_day"
    ]
    for col in required_base_cols:
        if col not in user_df.columns:
            user_df[col] = 0

    # Build one row per candidate item, each paired with user features
    user_df["_key"] = 1
    candidates = candidate_items.copy()
    candidates["_key"] = 1
    cart_df = candidates.merge(user_df, on="_key").drop(columns=["_key"])

    cart_df = create_features(cart_df)

    feature_cols = [
        "avg_order_value",
        "beverage_preference_score",
        "dessert_preference_score",
        "order_frequency_30d",
        "cart_value",
        "num_items_in_cart",
        "hour_of_day",
        "item_price",
        "item_margin",
        "item_popularity_score",
        "item_is_beverage",
        "item_is_dessert",
        "cart_value_bucket",
        "price_ratio",
        "user_beverage_interaction",
        "item_price_bucket",
        "price_bucket_match",
        "high_margin_flag",
        "is_dinner_time",
        "user_dessert_interaction"
    ]

    # ---- FIX 3: Remove duplicate model.predict call ----
    # Previously: model.predict was called twice — first result was thrown away,
    #             second applied a manual sigmoid on raw scores redundantly.
    # Now: single predict call with raw_score=False (LightGBM applies sigmoid
    #      internally for binary classifiers), giving clean [0, 1] probabilities.
    cart_df["final_score"] = model.predict(cart_df[feature_cols], raw_score=False)

    # ---- FIX 4: Improved epsilon-greedy exploration ----
    # Previously: epsilon exploration replaced final_score entirely with random
    #             values, meaning good recommendations could be completely lost.
    # Now: exploration is applied as a soft score perturbation — random noise
    #      is blended in rather than substituted, preserving signal while
    #      still allowing lower-ranked items a chance to surface.
    epsilon = 0.1
    if np.random.rand() < epsilon:
        noise = np.random.rand(len(cart_df)) * 0.3   # max 30% perturbation
        cart_df["final_score"] = 0.7 * cart_df["final_score"] + 0.3 * noise

    cart_df = cart_df.sort_values("final_score", ascending=False)

    # Diversity enforcement: max 2 items per category
    selected = []
    category_count = {}

    for _, row in cart_df.iterrows():
        cat = row["item_category"]
        if category_count.get(cat, 0) < 2:
            selected.append(row)
            category_count[cat] = category_count.get(cat, 0) + 1
        if len(selected) == 5:
            break

    top_items = pd.DataFrame(selected)
    return top_items[["item_id", "item_price", "item_category", "final_score"]].to_dict(orient="records")