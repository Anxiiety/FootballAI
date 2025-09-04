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


##Shortcut
```bash
date,season,matchweek,home_team,away_team,home_goals,away_goals,home_odds,draw_odds,away_odds
2024-08-24,24-25,1,Inter,Genoa,2,0,1.55,4.10,7.20
2024-08-25,24-25,1,Milan,Torino,1,1,1.90,3.50,4.20
2024-08-31,24-25,2,Juventus,Monza,3,1,1.65,3.90,6.00
2024-09-01,24-25,2,Napoli,Fiorentina,2,2,2.00,3.30,3.70
2024-09-14,24-25,3,Roma,Lazio,0,1,2.30,3.20,3.10
