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
