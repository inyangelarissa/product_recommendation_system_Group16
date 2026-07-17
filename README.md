# Product Recommendation System — Group 16

Multimodal authentication + recommendation pipeline built for the Formative 2
assignment. Three independent pipelines (structured data, face, voice) feed
into a CLI demo that chains face check → voice check → product prediction.

## Team & task breakdown

| Task | Owner | Deliverables |
|---|---|---|
| 1. Data merge & product recommendation model | Larissa Inyange | `product_model.py`, `outputs/merged_customer_dataset.csv`, `outputs/product_model.pkl` |
| 2. Image pipeline & facial recognition model | Rachel Toronga | `image/image_pipeline.py`, `image/image_features.csv`, `image/face_model.pkl` |
| 3. Audio pipeline & voiceprint verification model | Alliane Umutoniwase | `audio/audio_pipeline.py`, `audio/audio_features.csv`, `audio/voice_model.pkl` |
| 4. System integration & CLI demo | Merveille Munana | `cli_app.py`, final notebook, report — **TODO, see below** |

Identity labels are kept consistent across the face and voice models so Task 4
can cross-check both: `Alliane`, `larissa`, `meme`, `rachel`.

## Repo structure

```
.
├── customer_social_profiles - customer_social_profiles.csv   # raw data
├── customer_transactions - customer_transactions.csv          # raw data
├── product_model.py                                           # Task 1
├── outputs/
│   ├── merged_customer_dataset.csv
│   ├── product_model.pkl
│   └── model_evaluation.json
├── image/                                                      # Task 2
│   ├── Image Data Collection and Processing.ipynb              # extraction + augmentation + feature CSV
│   ├── formative2.zip                                          # raw photo submissions
│   ├── image_features.csv
│   ├── image_pipeline.py                                       # trains face_model.pkl
│   ├── face_model.pkl
│   ├── face_scaler.pkl
│   ├── face_feature_columns.pkl
│   └── face_class_names.pkl
└── audio/                                                      # Task 3
    ├── audio_pipeline.py                                       # extraction + augmentation + feature CSV + training, all-in-one
    ├── audio_features.csv
    ├── voice_model.pkl                                         # bundles model + scaler + label encoder + feature order
    ├── model_evaluation.json
    ├── raw_recordings/<member>/<phrase>.ogg                    # raw voice submissions
    ├── plots/                                                  # waveform + mel-spectrogram per recording
    └── augmented/                                              # pitch-shift / time-stretch / noise .wav files
```

## Setup

```
pip install pandas numpy scikit-learn xgboost joblib
pip install matplotlib seaborn opencv-python scikit-image      # image task
pip install librosa soundfile                                  # audio task
```

## Re-running each pipeline

**Task 1 — product recommendation:**
```
python product_model.py --social "customer_social_profiles - customer_social_profiles.csv" --transactions "customer_transactions - customer_transactions.csv" --outdir outputs
```

**Task 2 — face model** (feature extraction happens in the notebook; this retrains the model from the existing CSV):
```
cd image
python image_pipeline.py
```

**Task 3 — voice model** (single script does extraction + augmentation + training):
```
python audio/audio_pipeline.py --audio_dir audio/raw_recordings --outdir audio
```

## Current results (small-sample caveat)

All three models are trained on very small datasets (tens of rows), so treat
accuracy numbers as directional, not production-grade:

- **Product model:** Random Forest best, accuracy 0.38 / F1 0.37 — 61 merged
  customers across 6 product classes.
- **Face model:** Random Forest best, accuracy 0.44 / macro-F1 0.39 — 12 real
  photos (4 people × 3 expressions) + augmentations, evaluated on a held-out
  expression never seen in training.
- **Voice model:** Random Forest best, accuracy 1.00 / macro-F1 1.00 — 8 real
  recordings (4 people × 2 phrases) + augmentations, evaluated on a held-out
  phrase never seen in training. Perfect score reflects a genuinely tiny
  eval set (4 people, 4 test clips each after augmentation), not a claim the
  model is production-ready.

## TODO — Task 4 (Merveille)

- [ ] `cli_app.py`: chain face check → voice check → product prediction, with
      a denial branch at each step (load `face_model.pkl` + its scaler/feature
      columns/class names, `voice_model.pkl`, `outputs/product_model.pkl`)
- [ ] Simulate one authorized full flow (correct face + correct voice → product
      prediction runs) and one unauthorized attempt (bad face or bad voice →
      denied before reaching product prediction)
- [ ] Assemble the final Jupyter notebook pulling together all three tasks
- [ ] Compile the group report (data description, methodology, results,
      limitations per task, CLI demo walkthrough)
