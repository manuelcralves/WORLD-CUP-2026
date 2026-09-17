# World Cup 2026 ML Predictor

A solo project that forecast the 2026 FIFA World Cup and ran live at [worldcup2026ml.pt](https://worldcup2026ml.pt) during the tournament. Elo ratings built from 49,520 international matches since 1872 feed a Dixon-Coles Poisson model of goals, and a Monte Carlo simulation plays the tournament a million times to get title odds, brackets and Golden Boot projections. During the World Cup, a GitHub Actions job pulled new results every 2 hours, re-ran the model and redeployed the site, which also hosted "Beat the Machine", a prediction game with Google sign-in and Supabase. A blind version, trained and tuned only on matches played before 11 June 2026, called the result of 71 of the 104 matches, and its four favourites were exactly the four semi-finalists (the champion, Spain, was its second pick). Extra complexity added little: on 4,567 held-out matches from 2022 to June 2026, a model using only Elo scored the same RPS as the full model (0.17, against 0.23 for a naive guess), and XGBoost only tied it.

## Results

| Test | Model | Naive baseline |
| --- | --- | --- |
| Holdout: 4,567 matches from 2022 to June 2026 | RPS 0.17, 60% of results called | RPS 0.23, 48% |
| Walk-forward: 67 tournaments since 2002, retrained before each one (2,757 matches) | 56% of results called | 44% |
| World Cup 2026, blind (trained before 11 June) | 71 of 104 results called, RPS 0.15 | RPS 0.23 |

RPS is the ranked probability score: lower is better. The naive baseline predicts the average rates of home wins, draws and away wins.

## Run it

```bash
pip install -r requirements.txt
python run_pipeline.py 1000000 both   # live and blind versions, a million simulations each
```

Results data comes from [martj42/international_results](https://github.com/martj42/international_results). Match details during the tournament came from the Highlightly API. MIT License.
