import pandas as pd
import numpy as np

np.random.seed(42)

# -------------------------
# USERS
# -------------------------
n_users = 1000

users = pd.DataFrame({
    "user_id": np.arange(n_users),
    "avg_order_value": np.random.randint(150, 800, n_users),
    "beverage_preference_score": np.random.rand(n_users),
    "dessert_preference_score": np.random.rand(n_users),
    "order_frequency_30d": np.random.randint(1, 20, n_users),
})

# -------------------------
# ITEMS
# -------------------------
n_items = 100

categories = ["main", "beverage", "dessert", "side"]

items = pd.DataFrame({
    "item_id": np.arange(n_items),
    "item_price": np.random.randint(50, 400, n_items),
    "item_margin": np.random.randint(10, 150, n_items),
    "item_category": np.random.choice(categories, n_items),
    "item_popularity_score": np.random.rand(n_items)
})

items["item_is_beverage"] = (items["item_category"] == "beverage").astype(int)
items["item_is_dessert"] = (items["item_category"] == "dessert").astype(int)

# -------------------------
# CART INTERACTIONS
# -------------------------
n_samples = 20000

data = pd.DataFrame({
    "user_id": np.random.choice(users["user_id"], n_samples),
    "item_id": np.random.choice(items["item_id"], n_samples),
    "cart_value": np.random.randint(100, 1200, n_samples),
    "num_items_in_cart": np.random.randint(1, 6, n_samples),
    "hour_of_day": np.random.randint(0, 24, n_samples),
})

# Merge user + item features
data = data.merge(users, on="user_id")
data = data.merge(items, on="item_id")

# Simulate purchase probability
data["purchase_prob"] = (
    0.3 * data["beverage_preference_score"] * data["item_is_beverage"] +
    0.3 * data["dessert_preference_score"] * data["item_is_dessert"] +
    0.2 * (data["item_popularity_score"]) +
    0.2 * (1 - data["item_price"] / 400)
)

data["purchased"] = (np.random.rand(n_samples) < data["purchase_prob"]).astype(int)

data.to_csv("data/csao_data.csv", index=False)

print("Dataset generated successfully.")