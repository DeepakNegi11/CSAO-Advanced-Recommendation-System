import pandas as pd

def create_features(df):

    # Cart bucket
    df["cart_value_bucket"] = pd.cut(
        df["cart_value"],
        bins=[0, 300, 700, 2000],
        labels=[0, 1, 2]
    ).fillna(0).astype(int)

    # Price ratio
    df["price_ratio"] = df["item_price"] / df["cart_value"]

    # User × Item interaction
    df["user_beverage_interaction"] = (
        df["beverage_preference_score"] * df["item_is_beverage"]
    )

    df["user_dessert_interaction"] = (
        df["dessert_preference_score"] * df["item_is_dessert"]
    )

    # Revenue feature
    df["expected_revenue"] = df["item_margin"] * df["purchase_prob"]

    # Item price bucket
    df["item_price_bucket"] = pd.cut(
        df["item_price"],
        bins=[0, 100, 250, 1000],
        labels=[0, 1, 2]
    ).fillna(0).astype(int)

    # Bucket match feature
    df["price_bucket_match"] = (
        df["cart_value_bucket"] == df["item_price_bucket"]
    ).fillna(0).astype(int)

    df["high_margin_flag"] = (
    df["item_margin"] > df["item_margin"].median()
    ).fillna(0).astype(int)

    df["is_dinner_time"] = df["hour_of_day"].apply(
    lambda x: 1 if 18 <= x <= 22 else 0
    )

    # Final safety: remove all NaNs
    import numpy as np

    # Replace infinite values with 0
    df = df.replace([np.inf, -np.inf], 0)

    # Replace NaN values with 0
    df = df.fillna(0)
    return df