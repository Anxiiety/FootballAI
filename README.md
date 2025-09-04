![Python](https://img.shields.io/badge/Python-3.11+-blue)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![Status](https://img.shields.io/badge/status-alpha-orange)

# FootballAI
Modello di machine learning per predire risultati 1X2 della Serie A. Include feature engineering (Elo, rolling stats, quote), training Logistic Regression, valutazione e predizioni su partite future tramite scraping o CSV.
# football-ai — Serie A match predictor (1X2)

Modello semplice ma solido per predire risultati **1X2** (casa/pareggio/trasferta) in Serie A.
Pipeline:
1) **Scraping** (o import) partite/quote → `data/raw/`
2) **Feature engineering** (Elo + rolling medie) → `data/processed/`
3) **Training** Logistic Regression → `data/models/model.joblib`
4) **Evaluation** + **Prediction** su prossima giornata

> ✅ Pronto per partire **anche senza scraping**: puoi importare un CSV tuo.

## Install
```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt


## Shortcut
# crea segnaposto cartelle
mkdir -p data/processed data/models data/raw
type nul > data/processed/.gitkeep
type nul > data/models/.gitkeep

# file python init
echo __version__ = "0.1.0" > src/__init__.py
echo # scraping providers live here > src/scrape/__init__.py

# csv demo
notepad data/raw/matches.csv  # incolla il CSV sopra e salva
git add .
git commit -m "Add package inits, .gitkeep and demo matches.csv"
git push
