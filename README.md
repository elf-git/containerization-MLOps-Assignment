# ⚽ FIFA World Cup 2026 Winner Prediction & Tournament Simulator
### Machine Learning, Dynamic ELO Ratings, Docker Containerization & Scalable MLOps

---

### 👥 Project Group & Team Members

| # | Team Member | Registration / Student ID |
|:---:|---|:---:|
| 1 | **Anurag Chowdhury** | `2548512` |
| 2 | **Elvis Vidit Fernandiz** | `2548520` |
| 3 | **Abhinash K Prabhu** | `2548528` |
| 4 | **Sriram K** | `2548557` |

---

## 📌 Project Overview

This repository contains the complete end-to-end Machine Learning, Tournament Simulation, and Containerized Web Application system for predicting match outcomes and tournament champions in the **FIFA World Cup 2026** (48-team format).

- **Complete Technical Report & Screenshots:** See [**`REPORT.md`**](REPORT.md)
- **Interactive Web Dashboard:** Built with **Streamlit** (Live at port `8501`)
- **Containerization:** Production Docker image (`python:3.10-slim`, non-root user `appuser`, healthchecks)
- **ML Algorithm:** **XGBoost Classifier (`multi:softprob`)** trained on 20,700+ historical international matches (2000–2021) with **63.4%–64.1% test accuracy** on the 2022 World Cup benchmark.

---

## 🚀 Quick Start with Docker (Recommended)

### 1. Start the Containerized Application
```bash
docker compose up -d
```
Open your browser at:
👉 **http://localhost:8501**

### 2. Run the CLI Simulation Pipeline Inside Docker
```bash
docker compose run --rm fifa-predictor python run_project.py
```

### 3. Stop the Application
```bash
docker compose down
```

---

## 💻 Native Python Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run end-to-end ML training and tournament simulation
python run_project.py

# 3. Launch interactive Streamlit web dashboard
streamlit run app.py
```

---

## 📂 Repository Structure

```text
├── Dockerfile                  # Multi-layer Docker build with healthcheck & non-root security
├── docker-compose.yml          # Container orchestration with volume mounts & port mapping
├── .dockerignore               # Optimized Docker build context exclusions
├── requirements.txt            # Python dependencies (XGBoost, Scikit-Learn, Streamlit, etc.)
├── app.py                      # Interactive Streamlit Web Dashboard
├── run_project.py              # CLI batch training & knockout bracket tournament simulator
├── REPORT.md                   # Full Academic Project Report with Screenshots & K8s Architecture
├── README_DOCKER.md            # Detailed Docker setup and troubleshooting guide
├── screenshots/                # Application UI screenshots used in the report
│   ├── 01_tournament_bracket.png
│   ├── 02_monte_carlo_simulation.png
│   ├── 03_group_stage_standings.png
│   ├── 04_head_to_head_predictor.png
│   └── 05_model_performance_elo.png
└── FIFA-2026-Winner-Prediction-main/
    └── FIFA Prediction/
        ├── data/               # Historical match data, ELO ratings, and generated predictions
        └── notebooks/          # Exploratory analysis & data collection Jupyter notebooks
```

---

## 🧠 Key Features in the Dashboard

1. **🏆 Tournament Simulation & Winner:** 32-team knockout bracket visualization from Round of 32 to Final.
2. **🎲 Monte Carlo Simulator:** Run 100 to 2,000 stochastic tournament simulations with win rate bar charts.
3. **📊 Group Stage Explorer:** Browse Groups A through L with qualification badges and individual match odds.
4. **⚔️ Custom Head-to-Head Predictor:** Select any two international teams for live ML win/loss/draw forecasts.
5. **📈 Model Evaluation & ELO Leaderboard:** View 2022 World Cup metrics and global 48-nation ELO ratings.
