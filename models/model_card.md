# Model Card for RoBERTa-Toxicity-Detector

## Model Details
- **Model Name:** RoBERTa Toxicity Detector
- **Model Type:** Text Classification (Binary)
- **Base Model:** `roberta-base` (125M parameters)
- **Developers:** Hind KHAYATI, Sagnol Boutal KAMDEM DJOKO, Pape Mamadou DIAGNE, Sarra HERELLI
- **License:** CC-SA-3.0
- **Model Date:** 01/01/2026

## Intended Use
- **Primary Use Case:** Detection of toxic comments in online discussions.
- **Intended Users:** Moderators, community managers, or automated filtering systems.
- **Out-of-Scope Use Cases:** - Should not be used for "Hate Speech" detection without further finetuning (toxicity != hate speech).
    - Not suitable for languages other than English.

## Training Data
- **Dataset:** Google Jigsaw Toxic Comment Classification Challenge.
- **Source:** Wikipedia Talk Pages.
- **Size:** 159,571 comments.
- **Preprocessing:** - Text cleaned (newlines removed).
    - Tokenized using `RoBERTa` tokenizer (Max length: 128).
    - Labels binarized: Any sub-label (toxic, severe_toxic, obscene, threat, insult, identity_hate) -> `1` (Toxic).

## Training Procedure
- **Hyperparameters:**
    - Batch Size: 16
    - Learning Rate: 2e-5
    - Epochs: 2
    - Optimizer: AdamW
- **Hardware:** Google Colab T4 GPU (free version)
- **Evaluation Strategy:** Validated on a held-out 10% split using F1-Score as the primary metric.

## Evaluation Results
| Metric | Score (Test Set) |
| :--- | :--- |
| **ROC-AUC** | 0.99 |
| **F1-Score** | 0.90 |
| **Avg Latency** | 17.92 ms |

## Limitations & Bias
- **Imbalanced Data:** The training data is heavily skewed towards non-toxic comments (90%).
- **Identity Bias:** The model may disproportionately flag comments containing identity terms (e.g., "gay", "black") as toxic due to dataset bias.
- **Context:** The model analyzes individual sentences and may miss toxicity that depends on long conversation history.

## Ethical Considerations
This model automates moderation but is not perfect. It should be used as a "flagging" tool for human review rather than an autonomous banning system, to avoid censoring innocent comments (False Positives).