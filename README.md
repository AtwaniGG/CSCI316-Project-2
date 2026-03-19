


# CSCI 316 Assignment 2 — Sentiment Analysis in Arabizi via Transfer Learning

[Presentation Video](https://youtu.be/14a2IQUC8IY?si=Zap-9FeVZP-Xg6gW)

## Overview

This project investigates **cross-lingual transfer learning** for sentiment analysis on Lebanese Arabizi (Arabic written in Latin script) tweets. We fine-tune **mBERT** (`bert-base-multilingual-cased`) on Modern Standard Arabic (MSA) tweets, then transfer to the Arabizi target domain using multiple strategies.

The key research question: *Can knowledge from Arabic-script sentiment data transfer effectively to Latin-script Arabizi text, despite the orthographic gap?*

## Datasets

| Dataset | Language/Script | Samples | Classes |
|---------|----------------|---------|---------|
| **mksaad MSA** (source) | Arabic script | 56,795 | Positive, Negative |
| **Raïdy Arabizi** (target) | Latin script (Arabizi) | 1,799 (augmented to 2,398) | Positive, Negative, Neutral |

- Back-translation augmentation was applied to balance the neutral class (599 → 1,198 samples)
- Arabizi cleaning: digit-letter substitution (3→ع, 7→ح, 2→ء), lowercase normalization

## Transfer Learning Strategies

| Strategy | Framework | Best Val F1 | Test Macro F1 | Train Time | Trainable Params |
|----------|-----------|-------------|---------------|------------|-----------------|
| No Transfer (Baseline) | PyTorch | 0.6200 | — | 137s | ~178M (100%) |
| Full Fine-Tuning | PyTorch | 0.6132 | — | 138s | ~178M (100%) |
| LoRA (PEFT) | PyTorch | — | — | — | ~590K (0.33%) |
| Full Fine-Tuning | TF/Keras | — | 0.6574 | 205s | ~178M (100%) |
| LoRA (PEFT) | TF/Keras | — | — | — | Custom LoRA layers |

## Key Findings

- **Cross-script transfer gap**: Transfer from MSA Arabic to Arabizi showed minimal improvement over the no-transfer baseline, likely due to orthographic divergence (Arabic script vs Latin Arabizi)
- **LoRA robustness**: LoRA achieved lower DTGS variance (0.001341) vs Full FT (0.004239), suggesting frozen base weights better preserve cross-lingual representations
- **DTGS metric**: The custom Dialectal Transfer Gap Score measures F1 variance across Arabizi intensity quartiles — lower variance indicates more robust cross-dialect transfer

## Project Structure

```
├── CSCI_316_assignment2 PyTorch.ipynb   # Main notebook (PyTorch): preprocessing,
│                                         # source training, Full FT, LoRA, baseline,
│                                         # evaluation, DTGS, Gradio demo
├── NEW__CSCI316_TF_Keras.ipynb          # TF/Keras notebook: Full FT, LoRA,
│                                         # evaluation, DTGS (must run in separate
│                                         # Colab session — no PyTorch imports)
├── NLP/
│   ├── Arabic Tweets - sentiment/       # Source domain MSA dataset
│   ├── Arabizi Tweets/
│   │   ├── models/                      # Saved tokenizer and model checkpoints
│   │   └── target/                      # Target domain train/val/test splits
│   └── data/
│       ├── source/                      # Source domain splits
│       └── target/                      # Target domain splits
└── README.md
```

## Environment

- **Platform**: Google Colab (T4 GPU)
- **PyTorch**: 2.10.0+cu128
- **TensorFlow**: 2.19.0
- **Key packages**: `transformers==4.40.0`, `peft`, `accelerate`, `tf-keras`

> **Note**: TensorFlow and PyTorch cannot coexist in the same Colab session due to circular import conflicts. Run the TF notebook in a fresh session.

## How to Run

1. Upload the `NLP/` folder to Google Drive root (`My Drive/NLP/`)
2. **PyTorch notebook**: Open `CSCI_316_assignment2 PyTorch.ipynb` in Colab, select GPU runtime, run all cells
3. **TF/Keras notebook**: Open `NEW__CSCI316_TF_Keras.ipynb` in a **fresh** Colab session, select GPU runtime, run all cells

## Evaluation Metrics

- **Classification Report**: Per-class precision, recall, F1-score
- **Confusion Matrix**: Visual heatmap of predictions vs ground truth
- **Dialectal Transfer Gap Score (DTGS)**: Custom metric measuring F1 variance across Arabizi intensity quartiles — captures whether a model degrades on heavily code-switched text
