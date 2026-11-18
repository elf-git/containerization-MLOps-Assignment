# 🐳 Docker Guide: FIFA 2026 Winner Prediction

This project is fully containerized using **Docker** and **Docker Compose**. It packages Python 3.10, XGBoost, Scikit-Learn, Streamlit, and all tournament data into an isolated, reproducible container.

---

## 🚀 Quick Start (Streamlit Web Dashboard)

### Using Docker Compose (Recommended)

1. **Build and start the container in the background:**
   ```bash
   docker compose up -d
   ```

2. **Open the application in your browser:**
   👉 [http://localhost:8501](http://localhost:8501)

3. **Check container status and health:**
   ```bash
   docker compose ps
   ```

4. **View live container logs:**
   ```bash
   docker compose logs -f
   ```

5. **Stop the container:**
   ```bash
   docker compose down
   ```

---

## ⚙️ Running via Docker CLI (Without Compose)

### 1. Build the Docker Image
```bash
docker build -t fifa-2026-winner-prediction:latest .
```

### 2. Run the Web Dashboard Container
```bash
docker run -d \
  --name fifa-2026-predictor \
  -p 8501:8501 \
  fifa-2026-winner-prediction:latest
```
Access at: [http://localhost:8501](http://localhost:8501)

### 3. Run the CLI Simulation Pipeline Inside Docker
If you want to run the full machine learning training & bracket tournament simulation script directly in the container:

```bash
docker run --rm fifa-2026-winner-prediction:latest python run_project.py
```

Or with Docker Compose:
```bash
docker compose run --rm fifa-predictor python run_project.py
```

---

## 📂 Container Architecture & Highlights

- **Base Image:** `python:3.10-slim`
- **Security:** Runs under an unprivileged non-root user (`appuser`, UID 1000).
- **Healthcheck:** Probes Streamlit's internal health endpoint (`/_stcore/health`) every 30s.
- **Volume Mount:** `./FIFA-2026-Winner-Prediction-main/FIFA Prediction/data` is mounted to persist generated simulation CSVs (`2026_predictions.csv` and `2026_tournament_simulation.csv`) directly to the host machine.
- **Port:** Exposes `8501`.
