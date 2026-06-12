# 🏦 Home Credit Default Risk

<div align="center">

![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=for-the-badge&logo=python&logoColor=white)
![DVC](https://img.shields.io/badge/DVC-Tracked-945DD6?style=for-the-badge&logo=dvc&logoColor=white)
![MLflow](https://img.shields.io/badge/MLflow-DagsHub-0194E2?style=for-the-badge&logo=mlflow&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white)
![HuggingFace](https://img.shields.io/badge/🤗%20Hugging%20Face-Space-FFD21E?style=for-the-badge)
![CI/CD](https://img.shields.io/github/actions/workflow/status/ahmedsamir42003/Home_Credit_Default_Risk/deploy.yml?style=for-the-badge&label=CI%2FCD&logo=githubactions&logoColor=white)

<br/>

**End-to-end ML pipeline for predicting loan default probability.**  
Feature engineering · Ensemble modelling · Experiment tracking · Automated deployment.

<br/>

[![🚀 Live Demo](https://img.shields.io/badge/🚀%20Live%20Demo-Hugging%20Face%20Space-FFD21E?style=for-the-badge)](https://huggingface.co/spaces/Ahmed-Samir-Abdel-fattah/home-credit-risk)
[![📊 MLflow](https://img.shields.io/badge/📊%20Experiments-DagsHub%20MLflow-945DD6?style=for-the-badge)](https://dagshub.com/ahmedsamir42003/Home_Credit_Default_Risk.mlflow)
[![📁 DagsHub](https://img.shields.io/badge/📁%20Data%20&%20Models-DagsHub-945DD6?style=for-the-badge)](https://dagshub.com/ahmedsamir42003/Home_Credit_Default_Risk)

</div>

---

## 📌 Overview

This project tackles the [Home Credit Default Risk](https://www.kaggle.com/c/home-credit-default-risk) challenge — predicting whether a loan applicant will default, using a rich set of financial, behavioural, and demographic features.

The goal is to ensure that people who are capable of repayment are not rejected, while minimising exposure to clients who are likely to default.

---

## 🏗️ Architecture

```
Raw Data (7 CSV tables)
        │
        ▼
┌─────────────────────────────────┐
│  Stage 1 — prepare              │  data_prep.py + features.py
│  Feature engineering            │  300+ features, KNN meta-features,
│  Aggregations, target encoding  │  target encoding, correlation pruning
└────────────────┬────────────────┘
                 │  Parquet files (DVC tracked)
                 ▼
┌─────────────────────────────────┐
│  Stage 2 — train                │  train.py
│  5-model ensemble               │  LGB-A · LGB-B · LGB-seed · XGB · CatBoost
│  Level-2 stacking (LR)          │  + rank-blend optimisation
│  MLflow logging                 │  metrics, params, submission.csv artifact
└────────────────┬────────────────┘
                 │
                 ▼
        submission.csv  ←──── pulled by CI/CD at deploy time
                 │
                 ▼
┌─────────────────────────────────┐
│  FastAPI + Dashboard            │  Docker · port 7860
│  /predict/{SK_ID_CURR}          │  default probability lookup
│  /stats  /health  /docs         │  portfolio dashboard
└────────────────┬────────────────┘
                 │  GitHub Actions
                 ▼
      Hugging Face Space (Docker)
```

---

## 🤖 Models

| Model | Description |
|---|---|
| **LGB-A** | LightGBM primary — all features, aggressive regularisation |
| **LGB-B** | LightGBM top-400 features by importance |
| **LGB-seed** | LightGBM averaged over 3 seeds for stability |
| **XGBoost** | XGBoost with early stopping |
| **CatBoost** | CatBoost with symmetric tree growth |
| **Stack-LR** | Logistic Regression meta-model on OOF predictions + raw features |
| **Blend** | Rank-normalised weighted blend, weights optimised via SLSQP/Powell |

---

## 📊 Features engineered

- **Base application** — ratios, age buckets, employment flags, annuity/income/credit ratios
- **Bureau & balance** — credit history aggregations, overdue counts, utilisation rates
- **Previous applications** — approval rates, refused amounts, contract type distributions
- **POS / installments / credit card** — payment behaviour, lateness patterns, balance trends
- **KNN meta-features** — neighbourhood default rates at k=200 and k=500
- **Target encoding** — 13 categorical columns with smoothing
- **Groupby statistics** — mean/std/median of numeric cols per categorical group
- **Feature selection** — null importance pruning + correlation threshold (0.985)

---

## 🔁 CI/CD Pipeline

Every push to `main` triggers:

```
Push to main
     │
     ▼
GitHub Actions
     │
     ├─ Connect to MLflow on DagsHub
     ├─ Find last FINISHED run in "home-credit-default-risk" experiment
     ├─ Download submission.csv artifact
     ├─ Inject into Docker build context
     └─ Push deploy/app/ → Hugging Face Space
              │
              ▼
     HF rebuilds Docker image → live in ~2 min
```

`submission.csv` is **never committed to git** — always pulled fresh from the latest MLflow run.

---

## 🚀 API Endpoints

| Endpoint | Description |
|---|---|
| `GET /` | Interactive risk dashboard |
| `GET /predict/{SK_ID_CURR}` | Default probability for a single client |
| `GET /stats` | Portfolio-level summary statistics |
| `GET /health` | Health check |
| `GET /docs` | Auto-generated Swagger UI |

**Example:**
```bash
curl https://ahmed-samir-abdel-fattah-home-credit-risk.hf.space/predict/100001
```
```json
{
  "SK_ID_CURR": 100001,
  "default_probability": 0.568957,
  "risk_label": "🟠 High Risk",
  "will_default": true
}
```

---

## 🗂️ Project Structure

```
Home_Credit_Default_Risk/
├── src/
│   ├── config.py          # all hyperparameters and paths
│   ├── data_prep.py        # Stage 1 — feature engineering
│   ├── features.py         # feature builders
│   ├── train.py            # Stage 2 — ensemble training
│   └── utils.py
├── deploy/
│   └── app/
│       ├── main.py         # FastAPI app + dashboard
│       ├── Dockerfile
│       ├── requirements.txt
│       └── README.md       # HF Space card
├── .github/
│   └── workflows/
│       └── deploy.yml      # CI/CD pipeline
├── main.py                 # pipeline entry point
├── params.yaml             # DVC params
├── dvc.yaml                # DVC pipeline definition
└── data/
    └── inputs.dvc          # DVC pointer to raw data
```

---

## ⚙️ Run locally

```bash
# 1. Clone
git clone https://github.com/ahmedsamir42003/Home_Credit_Default_Risk
cd Home_Credit_Default_Risk

# 2. Install
pip install uv
uv sync

# 3. Set env vars
cp .env.example .env
# fill in MLFLOW_TRACKING_URI, DAGSHUB credentials

# 4. Pull data
dvc pull

# 5. Run full pipeline
uv run main.py --stage all

# 6. Run API locally
cd deploy/app
cp ~/data/outputs/submission.csv .
uvicorn main:app --reload --port 8000
# open http://localhost:8000
```

---

## 🔗 Links

| | |
|---|---|
| 🚀 **Live App** | https://huggingface.co/spaces/Ahmed-Samir-Abdel-fattah/home-credit-risk |
| 📊 **MLflow Experiments** | https://dagshub.com/ahmedsamir42003/Home_Credit_Default_Risk.mlflow |
| 📁 **DagsHub Repo** | https://dagshub.com/ahmedsamir42003/Home_Credit_Default_Risk |
| 🐙 **GitHub** | https://github.com/ahmedsamir42003/Home_Credit_Default_Risk |

---

## 🔐 Secrets required (GitHub Actions)

| Secret | Description |
|---|---|
| `DAGSHUB_USERNAME` | DagsHub username |
| `DAGSHUB_TOKEN` | DagsHub access token |
| `MLFLOW_TRACKING_URI` | `https://dagshub.com/ahmedsamir42003/Home_Credit_Default_Risk.mlflow` |
| `HF_TOKEN` | Hugging Face write token |
| `HF_SPACE_ID` | `Ahmed-Samir-Abdel-fattah/home-credit-risk` |

---

<div align="center">
Made with ❤️ · <a href="https://huggingface.co/spaces/Ahmed-Samir-Abdel-fattah/home-credit-risk">Try the live demo</a>
</div>
