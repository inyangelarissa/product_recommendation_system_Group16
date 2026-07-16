import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, log_loss, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

warnings.filterwarnings("ignore")

CONFIG = {
    "customer_key_aliases": ["customer_id", "cust_id", "user_id", "id",
                              "customer_id_new", "customer_id_legacy"],

    # customer_transactions.csv
    "txn_date_aliases": ["transaction_date", "purchase_date", "date", "order_date"],
    "txn_amount_aliases": ["amount", "purchase_amount", "transaction_amount", "price", "total"],
    "txn_product_aliases": ["product", "product_category", "product_name", "item_purchased", "category"],

    # customer_social_profiles.csv
    "social_likes_aliases": ["likes", "num_likes", "total_likes"],
    "social_comments_aliases": ["comments", "num_comments", "total_comments"],
    "social_shares_aliases": ["shares", "num_shares", "total_shares"],
    "social_followers_aliases": ["followers", "follower_count", "num_followers"],
    "social_posts_aliases": ["posts", "num_posts", "post_count"],
    "social_engagement_score_aliases": ["engagement_score"],
    "social_platform_aliases": ["social_media_platform", "platform"],
    "social_purchase_interest_aliases": ["purchase_interest_score"],
    "social_sentiment_aliases": ["review_sentiment", "sentiment"],
    "txn_rating_aliases": ["customer_rating", "rating"],
}


def normalize_customer_key(series: pd.Series) -> pd.Series:
    """Normalize customer IDs to a common format so sources with different key
    schemes can be joined — e.g. social profiles use 'A178' while transactions
    use the plain integer '178'. Strips any non-digit prefix and leading zeros
    so both resolve to the same canonical numeric string."""
    return (
        series.astype(str)
        .str.extract(r"(\d+)", expand=False)
        .astype(float)
        .astype("Int64")
        .astype(str)
    )

TARGET_COL = "product"  # standardized name we create after merge


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def find_col(df: pd.DataFrame, aliases: list, required: bool = True, label: str = ""):
    """Find the first matching column (case-insensitive) from a list of aliases."""
    lower_map = {c.lower(): c for c in df.columns}
    for alias in aliases:
        if alias.lower() in lower_map:
            return lower_map[alias.lower()]
    if required:
        raise KeyError(
            f"Could not find a column for '{label}'. "
            f"Tried aliases {aliases}. Available columns: {list(df.columns)}"
        )
    return None


def load_and_clean(path: str, name: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    print(f"[{name}] loaded shape: {df.shape}")

    # Standardize column names: strip whitespace
    df.columns = [c.strip() for c in df.columns]

    # Drop exact duplicate rows
    before = len(df)
    df = df.drop_duplicates()
    print(f"[{name}] dropped {before - len(df)} duplicate rows")

    # Report nulls
    null_counts = df.isnull().sum()
    if null_counts.sum() > 0:
        print(f"[{name}] null counts:\n{null_counts[null_counts > 0]}")

    return df


def clean_transactions(df: pd.DataFrame) -> pd.DataFrame:
    date_col = find_col(df, CONFIG["txn_date_aliases"], label="transaction date")
    amt_col = find_col(df, CONFIG["txn_amount_aliases"], label="transaction amount")
    prod_col = find_col(df, CONFIG["txn_product_aliases"], label="product purchased")

    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df[amt_col] = pd.to_numeric(df[amt_col], errors="coerce")

    # Drop rows with unusable key fields
    df = df.dropna(subset=[date_col, amt_col, prod_col])

    # Guard against negative/zero amounts (data entry errors)
    df = df[df[amt_col] > 0]

    df = df.rename(columns={date_col: "_txn_date", amt_col: "_txn_amount", prod_col: TARGET_COL})
    return df


def clean_social(df: pd.DataFrame) -> pd.DataFrame:
    # Numeric engagement fields — coerce, fill missing with 0 (no activity)
    for key in ["social_likes_aliases", "social_comments_aliases", "social_shares_aliases",
                "social_followers_aliases", "social_posts_aliases",
                "social_engagement_score_aliases", "social_purchase_interest_aliases"]:
        col = find_col(df, CONFIG[key], required=False, label=key)
        if col:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    return df


def engineer_transaction_features(df: pd.DataFrame, key_col: str) -> pd.DataFrame:
    """Recency / Frequency / Monetary features, plus the most recent product
    purchased per customer (used as the prediction target)."""
    snapshot_date = df["_txn_date"].max() + pd.Timedelta(days=1)

    rating_col = find_col(df, CONFIG["txn_rating_aliases"], required=False, label="customer rating")
    if rating_col:
        df[rating_col] = pd.to_numeric(df[rating_col], errors="coerce")

    agg_dict = {
        "recency_days": ("_txn_date", lambda x: (snapshot_date - x.max()).days),
        "frequency": ("_txn_date", "count"),
        "monetary_total": ("_txn_amount", "sum"),
        "monetary_avg": ("_txn_amount", "mean"),
    }
    grouped = df.groupby(key_col).agg(**agg_dict).reset_index()

    if rating_col:
        avg_rating = df.groupby(key_col)[rating_col].mean().reset_index().rename(
            columns={rating_col: "avg_customer_rating"}
        )
        avg_rating["avg_customer_rating"] = avg_rating["avg_customer_rating"].fillna(
            avg_rating["avg_customer_rating"].mean()
        )
        grouped = grouped.merge(avg_rating, on=key_col, how="left")

    # Target: most recent product each customer purchased (what we try to predict
    # they'd purchase next, based on RFM + engagement history)
    last_purchase = (
        df.sort_values("_txn_date")
        .groupby(key_col)
        .tail(1)[[key_col, TARGET_COL]]
        .rename(columns={TARGET_COL: TARGET_COL})
    )

    features = grouped.merge(last_purchase, on=key_col, how="left")
    return features


def engineer_social_features(df: pd.DataFrame, key_col: str) -> pd.DataFrame:
    """Aggregates social profile activity per customer. Customers can appear on
    multiple platforms (one row each), so this groups by customer and rolls
    each signal up into a single feature row per customer."""
    likes_col = find_col(df, CONFIG["social_likes_aliases"], required=False, label="likes")
    comments_col = find_col(df, CONFIG["social_comments_aliases"], required=False, label="comments")
    shares_col = find_col(df, CONFIG["social_shares_aliases"], required=False, label="shares")
    followers_col = find_col(df, CONFIG["social_followers_aliases"], required=False, label="followers")
    posts_col = find_col(df, CONFIG["social_posts_aliases"], required=False, label="posts")
    engagement_col = find_col(df, CONFIG["social_engagement_score_aliases"], required=False, label="engagement_score")
    platform_col = find_col(df, CONFIG["social_platform_aliases"], required=False, label="platform")
    interest_col = find_col(df, CONFIG["social_purchase_interest_aliases"], required=False, label="purchase_interest_score")
    sentiment_col = find_col(df, CONFIG["social_sentiment_aliases"], required=False, label="review_sentiment")

    agg_map = {}
    for raw_col, out_name, agg_fn in [
        (likes_col, "likes", "sum"),
        (comments_col, "comments", "sum"),
        (shares_col, "shares", "sum"),
        (followers_col, "followers", "max"),
        (posts_col, "posts", "sum"),
        (engagement_col, "avg_engagement_score", "mean"),
        (interest_col, "avg_purchase_interest_score", "mean"),
    ]:
        if raw_col:
            agg_map[out_name] = (raw_col, agg_fn)

    if agg_map:
        out = df.groupby(key_col).agg(**agg_map).reset_index()
    else:
        out = df[[key_col]].drop_duplicates().reset_index(drop=True)

    # Number of social profile records per customer (proxy for cross-platform presence)
    profile_count = df.groupby(key_col).size().reset_index(name="num_social_platforms")
    out = out.merge(profile_count, on=key_col, how="left")

    # Dominant platform per customer -> one-hot encoded
    if platform_col:
        dominant_platform = (
            df.groupby(key_col)[platform_col]
            .agg(lambda x: x.mode().iloc[0])
            .reset_index()
            .rename(columns={platform_col: "dominant_platform"})
        )
        out = out.merge(dominant_platform, on=key_col, how="left")
        out = pd.get_dummies(out, columns=["dominant_platform"], prefix="platform")

    # Sentiment -> share of positive / negative mentions per customer
    if sentiment_col:
        sentiment_counts = (
            df.groupby([key_col, sentiment_col]).size().unstack(fill_value=0)
        )
        sentiment_share = sentiment_counts.div(sentiment_counts.sum(axis=1), axis=0)
        sentiment_share.columns = [f"sentiment_share_{c.lower()}" for c in sentiment_share.columns]
        out = out.merge(sentiment_share.reset_index(), on=key_col, how="left")

    # Engagement rate: interactions per follower (avoid div by zero), only if
    # follower counts are present in the source data
    if "followers" in out.columns:
        interactions = out.get("likes", 0) + out.get("comments", 0) + out.get("shares", 0)
        out["engagement_rate"] = interactions / out["followers"].replace(0, np.nan)
        out["engagement_rate"] = out["engagement_rate"].fillna(0)

    return out


def merge_datasets(social_df, txn_features_df, key_col):
    merged = txn_features_df.merge(social_df, on=key_col, how="inner")

    print(f"[merge] transactions customers: {txn_features_df[key_col].nunique()}")
    print(f"[merge] social customers: {social_df[key_col].nunique()}")
    print(f"[merge] merged customers: {merged[key_col].nunique()}")
    print(f"[merge] merged shape: {merged.shape}")

    # Post-merge validation
    assert merged[key_col].is_unique, "Merge produced duplicate customer keys — check join logic"
    null_after = merged.isnull().sum()
    if null_after.sum() > 0:
        print(f"[merge] nulls after merge (filling with 0/mode):\n{null_after[null_after > 0]}")
        for c in merged.columns:
            if merged[c].isnull().any():
                if merged[c].dtype in [np.float64, np.int64]:
                    merged[c] = merged[c].fillna(0)
                else:
                    merged[c] = merged[c].fillna(merged[c].mode().iloc[0])

    return merged


def train_and_evaluate(merged: pd.DataFrame, key_col: str, outdir: Path):
    feature_cols = [c for c in merged.columns if c not in [key_col, TARGET_COL]]
    X = merged[feature_cols].copy()
    y = merged[TARGET_COL].copy()

    # Drop classes with too few samples for a stratified split
    class_counts = y.value_counts()
    valid_classes = class_counts[class_counts >= 2].index
    mask = y.isin(valid_classes)
    X, y = X[mask], y[mask]

    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=0.2, random_state=42, stratify=y_enc
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    models = {
        "RandomForest": RandomForestClassifier(n_estimators=300, max_depth=None, random_state=42),
        "LogisticRegression": LogisticRegression(max_iter=1000),
    }
    if HAS_XGB:
        models["XGBoost"] = XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.1,
            eval_metric="mlogloss", random_state=42, use_label_encoder=False
        )

    results = {}
    fitted = {}
    for name, model in models.items():
        model.fit(X_train_s, y_train)
        fitted[name] = model

        preds = model.predict(X_test_s)
        probs = model.predict_proba(X_test_s)

        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds, average="weighted")
        try:
            loss = log_loss(y_test, probs, labels=np.arange(len(le.classes_)))
        except ValueError:
            loss = np.nan

        results[name] = {"accuracy": acc, "f1_score": f1, "log_loss": loss}
        print(f"\n[{name}] accuracy={acc:.4f}  f1={f1:.4f}  log_loss={loss:.4f}")
        print(classification_report(y_test, preds, target_names=le.classes_.astype(str), zero_division=0))

    best_name = max(results, key=lambda k: results[k]["f1_score"])
    best_model = fitted[best_name]
    print(f"\nBest model by weighted F1: {best_name}")

    outdir.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {"model": best_model, "scaler": scaler, "label_encoder": le, "feature_cols": feature_cols},
        outdir / "product_model.pkl",
    )
    with open(outdir / "model_evaluation.json", "w") as f:
        json.dump(results, f, indent=2)

    return results, best_name


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--social", required=True, help="Path to customer_social_profiles.csv")
    parser.add_argument("--transactions", required=True, help="Path to customer_transactions.csv")
    parser.add_argument("--outdir", default="outputs", help="Output directory")
    args = parser.parse_args()

    outdir = Path(args.outdir)

    social_raw = load_and_clean(args.social, "social_profiles")
    txn_raw = load_and_clean(args.transactions, "transactions")

    key_col_social = find_col(social_raw, CONFIG["customer_key_aliases"], label="customer key (social)")
    key_col_txn = find_col(txn_raw, CONFIG["customer_key_aliases"], label="customer key (transactions)")

    social_clean = clean_social(social_raw)
    txn_clean = clean_transactions(txn_raw)

    # Normalize keys BEFORE feature engineering so groupby aggregation keys match
    # across sources even when ID formats differ (e.g. 'A178' vs '178')
    social_clean[key_col_social] = normalize_customer_key(social_clean[key_col_social])
    txn_clean[key_col_txn] = normalize_customer_key(txn_clean[key_col_txn])

    txn_features = engineer_transaction_features(txn_clean, key_col_txn)
    social_features = engineer_social_features(social_clean, key_col_social)

    # Standardize key column name across both frames before merge
    txn_features = txn_features.rename(columns={key_col_txn: "customer_id"})
    social_features = social_features.rename(columns={key_col_social: "customer_id"})

    merged = merge_datasets(social_features, txn_features, "customer_id")

    outdir.mkdir(parents=True, exist_ok=True)
    merged_path = outdir / "merged_customer_dataset.csv"
    merged.to_csv(merged_path, index=False)
    print(f"\nSaved merged dataset -> {merged_path}")

    results, best_name = train_and_evaluate(merged, "customer_id", outdir)
    print(f"\nSaved model + evaluation report -> {outdir}/product_model.pkl, {outdir}/model_evaluation.json")


if __name__ == "__main__":
    main()
