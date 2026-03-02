import pandas as pd
import shap
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
    "user_dessert_interaction"
]

sample = df[feature_cols].iloc[:200]

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(sample)

shap.summary_plot(shap_values, sample)