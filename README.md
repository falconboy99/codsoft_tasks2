# Spam SMS Detection

Classifies SMS messages as **spam** or **ham** (legitimate) using TF-IDF features
(word 1–2-grams + character 2–5-grams) and three classic classifiers.

**Data:** `data/spam.csv`, the SMS Spam Collection dataset uploaded to the project (5,572 messages: 4,825 ham / 747 spam).
Split: stratified 80/20 (4,457 train / 1,115 test). Hyperparameters tuned with 5-fold CV on the training set (F1 on spam).

## Results (held-out test set, spam class)

| Model | Accuracy | Precision | Recall | F1 | CV F1 |
|---|---|---|---|---|---|
| Multinomial Naive Bayes (alpha=0.01) | 0.9874 | 0.9655 | 0.9396 | 0.9524 | 0.9661 |
| Logistic Regression (C=10) | 0.9919 | 0.9930 | 0.9463 | 0.9691 | 0.9711 |
| **Linear SVM (C=0.3)** | 0.9919 | 0.9930 | 0.9463 | 0.9691 | **0.9718** |

The saved model is chosen by cross-validated F1 (so the test set stays unbiased): the linear SVM,
wrapped in probability calibration so it outputs p(spam). Logistic Regression ties it on the test set;
both beat Naive Bayes. Only 1 legitimate message in the test set was flagged as spam.

## Files
- `train.py` — training, evaluation and prediction (writes `model/spam_classifier.joblib`, not committed; run it to regenerate)
- `export_web.py` — exports the TF-IDF + Logistic Regression model to `web/model.json`
- `web/` — static live demo (HTML + JS, runs the model in the browser, no server needed)
- `results/comparison.csv`, `results/metrics.json` — metrics, best params, confusion matrices
- `results/misclassified.csv` — test messages the chosen model got wrong
- `results/train_log.txt` — full classification reports

## Usage
```bash
pip install -r requirements.txt
python train.py                                   # retrain + evaluate
python train.py --predict "WIN a FREE prize! Text CLAIM to 80082"
```

## Web demo
`web/` is a static site: open `web/index.html` through any local server
(`python -m http.server -d web`) or deploy it to Netlify by dragging the `web` folder onto
https://app.netlify.com/drop (or connect this repo with base directory `web`).
The JavaScript port reproduces the Python model's probabilities on all 5,572 messages (max difference 2e-6).

## Limitation
The corpus is from ~2012 (mostly UK prize/ringtone spam). Modern phishing styles
("your account is suspended, verify at bit.ly/...") score lower (p=0.11 in a quick test);
retrain with recent examples for production use.
