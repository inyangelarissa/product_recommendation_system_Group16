#!/usr/bin/env python
"""
Task 3: Audio Pipeline & Voiceprint Verification Model
Formative 2 -- Multimodal Data Preprocessing Assignment

Given each member's recorded phrases ("Yes, approve" / "Confirm transaction"),
this script loads the audio, displays waveforms/spectrograms, applies
augmentations, extracts MFCC/spectral/energy features into audio_features.csv,
then trains and evaluates a speaker-identification (voiceprint) model.

Expected input layout (mirrors the image task's <member>/<expression> convention):

    <audio_dir>/
        larissa/
            yes_approve.wav
            confirm_transaction.wav
        rachel/
            yes_approve.wav
            confirm_transaction.wav
        ...

Member folder names become the class labels. Phrase filenames are matched
via aliases below, so minor naming differences ("approve.wav", "yes.wav",
"confirm.wav", "transaction.wav", etc.) still resolve correctly -- no need
to rename files to match exactly.

Install deps first (not preinstalled -- Colab needs these too):
    pip install librosa soundfile scikit-learn xgboost joblib matplotlib

Usage:
    python audio_pipeline.py --audio_dir data/audio --outdir outputs/audio
"""

import argparse
import glob
import json
import os
import warnings
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import librosa
import librosa.display
import soundfile as sf

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                              confusion_matrix, f1_score, log_loss)
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

warnings.filterwarnings("ignore")

SR = 22050  # resample rate -- keeps feature vectors a consistent length

PHRASE_ALIASES = {
    "yes_approve": ["yes_approve", "approve", "yes", "yesapprove"],
    "confirm_transaction": ["confirm_transaction", "confirm", "transaction", "confirmtransaction"],
}


# ---------------------------------------------------------------------------
# 1. Build manifest from the folder tree
# ---------------------------------------------------------------------------
def match_phrase(filename: str):
    stem = os.path.splitext(os.path.basename(filename))[0].lower().replace(" ", "_").replace("-", "_")
    for canonical, aliases in PHRASE_ALIASES.items():
        if any(alias in stem for alias in aliases):
            return canonical
    return None


def build_manifest(audio_dir: str) -> pd.DataFrame:
    records = []
    for member_dir in sorted(glob.glob(os.path.join(audio_dir, "*"))):
        if not os.path.isdir(member_dir):
            continue
        member = os.path.basename(member_dir)
        for path in glob.glob(os.path.join(member_dir, "*")):
            if not path.lower().endswith((".wav", ".mp3", ".m4a", ".flac", ".ogg")):
                continue
            phrase = match_phrase(path)
            if phrase is None:
                print(f"[warn] could not match phrase for {path}, skipping")
                continue
            records.append({"member": member, "phrase": phrase, "path": path})

    manifest = pd.DataFrame(records)
    if manifest.empty:
        raise RuntimeError(f"No usable audio files found under {audio_dir}")

    # Validation: every member should have both phrases
    expected = set(PHRASE_ALIASES.keys())
    for member, group in manifest.groupby("member"):
        missing = expected - set(group["phrase"])
        if missing:
            print(f"[warn] {member} is missing phrase(s): {missing}")

    print(f"[manifest] {len(manifest)} recordings across {manifest['member'].nunique()} members")
    return manifest


# ---------------------------------------------------------------------------
# 2. Display waveform + spectrogram for each recording
# ---------------------------------------------------------------------------
def display_audio(manifest: pd.DataFrame, outdir: Path):
    plot_dir = outdir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    for _, row in manifest.iterrows():
        y, sr = librosa.load(row["path"], sr=SR)

        fig, axes = plt.subplots(2, 1, figsize=(8, 5))
        librosa.display.waveshow(y, sr=sr, ax=axes[0])
        axes[0].set_title(f'{row["member"]} - {row["phrase"]} (waveform)')

        S = librosa.feature.melspectrogram(y=y, sr=sr)
        S_db = librosa.power_to_db(S, ref=np.max)
        img = librosa.display.specshow(S_db, sr=sr, x_axis="time", y_axis="mel", ax=axes[1])
        axes[1].set_title("mel-spectrogram")
        fig.colorbar(img, ax=axes[1], format="%+2.0f dB")

        fig.tight_layout()
        out_path = plot_dir / f'{row["member"]}_{row["phrase"]}.png'
        fig.savefig(out_path)
        plt.close(fig)

    print(f"[display] saved waveform/spectrogram plots -> {plot_dir}")


# ---------------------------------------------------------------------------
# 3. Augmentations: pitch shift, time stretch, add noise
# ---------------------------------------------------------------------------
def augment_and_save(y: np.ndarray, sr: int, member: str, phrase: str, aug_dir: Path) -> list:
    augmentations = {
        "pitch_shift": librosa.effects.pitch_shift(y, sr=sr, n_steps=3),
        "time_stretch": librosa.effects.time_stretch(y, rate=1.2),
        "noise": y + 0.005 * np.random.randn(len(y)),
    }
    saved = []
    for aug_name, aug_y in augmentations.items():
        out_path = aug_dir / f"{member}_{phrase}_{aug_name}.wav"
        sf.write(out_path, aug_y, sr)
        saved.append({"member": member, "phrase": phrase, "augmentation": aug_name, "path": str(out_path)})
    return saved


def build_augmented_manifest(manifest: pd.DataFrame, outdir: Path) -> pd.DataFrame:
    aug_dir = outdir / "augmented"
    aug_dir.mkdir(parents=True, exist_ok=True)

    all_records = []
    for _, row in manifest.iterrows():
        y, sr = librosa.load(row["path"], sr=SR)
        all_records.append({"member": row["member"], "phrase": row["phrase"],
                             "augmentation": "original", "path": row["path"]})
        all_records.extend(augment_and_save(y, sr, row["member"], row["phrase"], aug_dir))

    full_manifest = pd.DataFrame(all_records)
    print(f"[augment] {len(manifest)} originals -> {len(full_manifest)} total (originals + augmentations)")
    return full_manifest


# ---------------------------------------------------------------------------
# 4. Feature extraction: MFCCs, spectral rolloff, RMS energy, + extras
# ---------------------------------------------------------------------------
def extract_features(path: str) -> dict:
    y, sr = librosa.load(path, sr=SR)

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    rms = librosa.feature.rms(y=y)
    zcr = librosa.feature.zero_crossing_rate(y=y)
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)

    feats = {}
    for i in range(mfcc.shape[0]):
        feats[f"mfcc{i+1}_mean"] = float(np.mean(mfcc[i]))
        feats[f"mfcc{i+1}_std"] = float(np.std(mfcc[i]))
    feats["spectral_rolloff_mean"] = float(np.mean(rolloff))
    feats["spectral_rolloff_std"] = float(np.std(rolloff))
    feats["rms_energy_mean"] = float(np.mean(rms))
    feats["rms_energy_std"] = float(np.std(rms))
    feats["zero_crossing_rate_mean"] = float(np.mean(zcr))
    feats["spectral_centroid_mean"] = float(np.mean(centroid))
    return feats


def build_feature_table(full_manifest: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in full_manifest.iterrows():
        feats = extract_features(row["path"])
        feats.update({
            "member": row["member"],
            "phrase": row["phrase"],
            "augmentation": row["augmentation"],
            "filename": os.path.basename(row["path"]),
        })
        rows.append(feats)

    df = pd.DataFrame(rows)
    meta_cols = ["member", "phrase", "augmentation", "filename"]
    feature_cols = [c for c in df.columns if c not in meta_cols]
    df = df[meta_cols + feature_cols]
    return df


# ---------------------------------------------------------------------------
# 5. Train + evaluate speaker-ID model (leakage-safe split, like the image task)
# ---------------------------------------------------------------------------
def train_and_evaluate(df: pd.DataFrame, outdir: Path):
    feature_cols = [c for c in df.columns if c not in ["member", "phrase", "augmentation", "filename"]]

    # Hold out one entire phrase (+ its augmentations) for testing so the
    # model is judged on genuinely unseen content, not near-duplicate
    # augmented copies of a phrase it already trained on.
    TEST_PHRASE = "confirm_transaction"
    train_df = df[df["phrase"] != TEST_PHRASE].reset_index(drop=True)
    test_df = df[df["phrase"] == TEST_PHRASE].reset_index(drop=True)
    print(f"[split] train: {len(train_df)} rows ({sorted(train_df['phrase'].unique())}) | "
          f"test: {len(test_df)} rows ({sorted(test_df['phrase'].unique())})")

    X_train, y_train = train_df[feature_cols].values, train_df["member"].values
    X_test, y_test = test_df[feature_cols].values, test_df["member"].values

    le = LabelEncoder()
    le.fit(df["member"])
    y_train_enc = le.transform(y_train)
    y_test_enc = le.transform(y_test)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    candidates = {
        "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
        "SVM (linear)": SVC(kernel="linear", probability=True, random_state=42),
        "RandomForest": RandomForestClassifier(n_estimators=200, random_state=42),
    }
    if HAS_XGB:
        candidates["XGBoost"] = XGBClassifier(eval_metric="mlogloss", random_state=42)

    results = {}
    fitted = {}
    for name, model in candidates.items():
        model.fit(X_train_s, y_train_enc)
        preds = model.predict(X_test_s)
        probs = model.predict_proba(X_test_s)

        acc = accuracy_score(y_test_enc, preds)
        f1 = f1_score(y_test_enc, preds, average="macro")
        try:
            loss = log_loss(y_test_enc, probs, labels=np.arange(len(le.classes_)))
        except ValueError:
            loss = float("nan")

        results[name] = {"accuracy": acc, "f1_score": f1, "log_loss": loss}
        fitted[name] = model
        print(f"\n[{name}] accuracy={acc:.4f} f1={f1:.4f} log_loss={loss:.4f}")
        print(classification_report(y_test_enc, preds, target_names=le.classes_, zero_division=0))

    best_name = max(results, key=lambda k: results[k]["f1_score"])
    best_model = fitted[best_name]
    print(f"\nBest model by macro F1: {best_name}")

    cm = confusion_matrix(y_test_enc, best_model.predict(X_test_s))
    print(f"\nConfusion matrix ({best_name}):\n{cm}")

    outdir.mkdir(parents=True, exist_ok=True)

    # Bundle everything inference needs into ONE file -- learned from the
    # image task, where the scaler/feature-columns/class-names were saved
    # as separate files and never made it into the repo alongside the model.
    joblib.dump(
        {"model": best_model, "scaler": scaler, "label_encoder": le, "feature_cols": feature_cols},
        outdir / "voice_model.pkl",
    )
    with open(outdir / "model_evaluation.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved -> {outdir}/voice_model.pkl, {outdir}/model_evaluation.json")
    return results, best_name


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio_dir", required=True, help="Root folder: <audio_dir>/<member>/<phrase>.wav")
    parser.add_argument("--outdir", default="outputs/audio", help="Output directory")
    parser.add_argument("--skip_plots", action="store_true", help="Skip waveform/spectrogram generation")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    manifest = build_manifest(args.audio_dir)

    if not args.skip_plots:
        display_audio(manifest, outdir)

    full_manifest = build_augmented_manifest(manifest, outdir)
    features_df = build_feature_table(full_manifest)

    features_path = outdir / "audio_features.csv"
    features_df.to_csv(features_path, index=False)
    print(f"\nSaved {len(features_df)} rows x {features_df.shape[1]} columns -> {features_path}")

    train_and_evaluate(features_df, outdir)


if __name__ == "__main__":
    main()
