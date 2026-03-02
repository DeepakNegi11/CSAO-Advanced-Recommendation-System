import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from feature_engineering import create_features

# Load data
df = pd.read_csv("data/csao_data.csv")
df = create_features(df)

# Create fake cart_id groups
df["cart_id"] = df.index // 10  # every 10 rows = 1 cart

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
print(df.columns)
X = df[feature_cols]
y = df["purchased"]
groups = df.groupby("cart_id").size().to_list()

train_data = lgb.Dataset(X, label=y, group=groups)

params = {
    "objective": "lambdarank",
    "metric": "ndcg",
    "ndcg_eval_at": [5],
    "learning_rate": 0.03,
    "num_leaves": 63,
    "max_depth": -1,
    "min_data_in_leaf": 50,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "lambda_l1": 0.5,
    "lambda_l2": 0.5,
    "verbosity": -1
}

model = lgb.train(params, train_data, num_boost_round=300)

model.save_model("data/csao_ranker.txt")

import json

# -----------------------------------
# Save feature order for inference
# -----------------------------------
with open("data/feature_cols.json", "w") as f:
    json.dump(feature_cols, f)

print("Feature list saved successfully.")

print("Ranking model trained successfully.")