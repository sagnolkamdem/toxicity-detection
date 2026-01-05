# 🛡️ RoBERTa Toxicity Detector

A high-performance, low-latency AI system designed to detect toxic comments in online discussions. This project finetunes a **RoBERTa** model to achieve **~97% accuracy** while being **10x-50x faster** than Large Language Models (LLMs) like Mistral 7B.

## 📌 Project Overview
The goal of this project is to build a production-ready toxicity detection service. It compares two approaches:
1.  **Zero-Shot LLM:** Using Mistral 7B (via API) to classify comments.
2.  **Finetuned Encoder:** Training `roberta-base` specifically for this task.
3.  **Developers:** Hind KHAYATI, Sagnol Boutal KAMDEM DJOKO, Pape Mamadou DIAGNE, Sarra HERELLI

**Key Findings:**
* **RoBERTa** achieved comparable accuracy to the LLM.
* **Latency:** RoBERTa runs in **~15-20ms** (GPU) / **~200ms** (CPU), compared to **2-5 seconds** for the LLM.
* **Cost:** Significantly cheaper to deploy and maintain.

---

## 📂 Project Structure

```text
├── api/                            # Research & Training Notebooks
│   ├── app.py
│
├── notebooks/                      # Research & Training Notebooks
│   ├── 00_main_notebook.ipynb      # Recap of all steps
│   ├── 01_eda.ipynb                # Exploratory & Data Analysis
│   ├── 02_llm.ipynb                # LLM baseline
│   ├── 03_RoBERTa_training.ipynb
│   ├── 04_Latency_Comparison.ipynb
│   └── 05_Explainability_SHAP.ipynb
│   └── 06_fairness_bias.ipynb
│
├── Data/                           # Dataset folder (Ignored by Git)
│   ├── README.md                   # Dataset documentation
│   ├── test.csv
│   ├── test_labels.csv
│   └── train.csv
│
├── models/                         # Model folder (some files was ignored by Git)
│   ├── model_card.md               # Model documentation
│   └── model.safetensors           # Fake model file
│                                   
├── requirements.txt                # Global project dependencies
└── README.md                       # Project documentation
```

## ⚙️ Installation & Setup

### 1. Set Up Python Environment
```bash
    git clone https://gitlab.esiea.fr/kamdemdjoko/nlp-final-project.git
    cd nlp-final-project
```

### 2. Set Up Python Environment
It is recommended to use a virtual environment.

```bash
    # Create virtual env
    python -m venv venv

    # Activate (Windows)
    .\venv\Scripts\activate
    # Activate (Mac/Linux)
    source venv/bin/activate
    
    # Install dependencies
    pip install -r api/requirements.txt
```

### 3. ⚠️ Download Model Weights
- Due to file size limits, the trained model weights (pytorch_model.bin) are not hosted on GitHub.
    - Option A (Pre-trained): Download the model folder from https://drive.google.com/drive/folders/1PEyyBh02zZG-b8hYsWFuPYVq_1szZA36?usp=sharing. 
    - Option B (Train Yourself): Run the 4_RoBERTa_Training.ipynb notebook to generate the model.

- Place the files: Unzip the content into the ./model/ folder.

- Your structure should look like: ./model/config.json, ./model/model.safetensors, etc.

### 🚀 Usage: Running the API
We use FastAPI to serve the model. The model is loaded into memory once at startup to ensure low latency.

*Start the Server*

Navigate to the api folder and run Uvicorn:

```bash
    cd api
    uvicorn app:app --reload
```
- The API will start at http://127.0.0.1:8000.

*Test the API*

- Option 1: Swagger UI (Browser)
    - Go to: http://127.0.0.1:8000/docs
    - Click POST /predict -> Try it out. 
    - Enter JSON: {"text": "You are amazing!"} or {"text": "You are an idiot."} or any other JSON
    - Click Execute.

- Option 2: cURL (Terminal)
```bash
    curl -X 'POST' \
  '[http://127.0.0.1:8000/predict](http://127.0.0.1:8000/predict)' \
  -H 'Content-Type: application/json' \
  -d '{"text": "This is a toxic comment test."}'
```

### 🧠 Model & Training Details

*Dataset*
- Source: Jigsaw Toxic Comment Classification Challenge.

- Size: ~160k training samples.

- Labels: toxic, severe_toxic, obscene, threat, insult, identity_hate.

- Handling: We treat this as a Binary Classification task (Toxic vs. Non-Toxic).

*Training Configuration*
- Base Model: roberta-base

- Batch Size: 16

- Learning Rate: 2e-5

- Epochs: 2

- Loss Function: CrossEntropyLoss

- Hardware: Trained on NVIDIA T4 GPU (Google Colab).

*Evaluation Metrics (Test Set)*

| Metric | Score (Test Set) |
| :--- |:-----------------|
| **ROC-AUC** | 0.97             |
| **F1-Score** | 0.67             |
| **Avg Latency** | ±20 ms           |

### 📊 Latency Comparison
One of the main goals was to prove RoBERTa's efficiency over LLMs.

| Model           | Avg Latency GPU | Avg Latency CPU |
|:----------------|:----------------|----------------|
| **Miatral 7B**  | ±400 ms         | ±2-5s          |
| **RoBERTa**     | ±20 ms          | ±200ms         |

Note: API latency is measured using server-side headers (X-Process-Time) to exclude network overhead.

### 🔍 Explainability
We use SHAP (SHapley Additive exPlanations) to ensure the model isn't just guessing.

- Positive Contributors (Red): Words like "idiot", "hate", "stupid" push the score toward Toxic.

- Negative Contributors (Blue): Words like "thanks", "agree", "support" push the score toward Non-Toxic.

(See ./notebooks/05_explainability_shap.ipynb for visualizations)

### 📝 License
This project uses the Jigsaw dataset (CC0) and the RoBERTa model (MIT). Project created by:
- Hind KHAYATI 
- Pape Mamadou DIAGNE
- Sarra HERELLI
- Sagnol Boutal KAMDEM DJOKO