"""Export a browser-runnable model (TF-IDF + Logistic Regression) to web/model.json."""
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train as T

df = T.load()
Xtr, Xte, ytr, yte = train_test_split(df["text"], df["y"], test_size=0.2, stratify=df["y"], random_state=T.SEED)
pipe = Pipeline([("tfidf", T.features()),
                 ("clf", LogisticRegression(C=10, max_iter=2000, class_weight="balanced", solver="liblinear"))])
pipe.fit(Xtr, ytr)
print("test metrics:", {k: round(v, 4) for k, v in T.metrics(yte, pipe.predict(Xte)).items()})

coef = pipe.named_steps["clf"].coef_[0]
out, off = {"intercept": float(pipe.named_steps["clf"].intercept_[0])}, 0
for name, vec in pipe.named_steps["tfidf"].transformer_list:
    vocab = vec.vocabulary_
    terms = sorted(vocab, key=vocab.get)
    out[name] = {"terms": terms,
                 "idf": [round(float(x), 5) for x in vec.idf_],
                 "coef": [round(float(x), 5) for x in coef[off:off + len(terms)]]}
    off += len(terms)
web = T.ROOT / "web"
json.dump(out, open(web / "model.json", "w"), ensure_ascii=False, separators=(",", ":"))
# reference probabilities for checking the JS port
np.savetxt(T.ROOT / "results" / "web_reference_probs.txt", pipe.predict_proba(df["text"])[:, 1])
