import pandas as pd
import lightgbm as lgb
from feature_engineering import create_features

df = pd.read_csv("data/csao_data.csv")
df = create_features(df)

model = lgb.Booster(model_file="data/csao_ranker.txt")

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

df["model_score"] = model.predict(df[feature_cols])

# Group into fake carts
df["cart_id"] = df.index // 10

# A: Pure model
df_A = df.copy()
df_A["rank"] = df_A.groupby("cart_id")["model_score"].rank(ascending=False)
top_A = df_A[df_A["rank"] <= 5]

revenue_A = (top_A["purchased"] * top_A["item_margin"]).sum()

# B: Revenue-aware
df_B = df.copy()
df_B["revenue_score"] = df_B["model_score"] * df_B["item_margin"]
df_B["rank"] = df_B.groupby("cart_id")["revenue_score"].rank(ascending=False)
top_B = df_B[df_B["rank"] <= 5]

revenue_B = (top_B["purchased"] * top_B["item_margin"]).sum()

print("Revenue - Pure Model:", revenue_A)
print("Revenue - Revenue Aware:", revenue_B)

lift = (revenue_B - revenue_A) / revenue_A * 100
print("Revenue Lift %:", round(lift, 2))