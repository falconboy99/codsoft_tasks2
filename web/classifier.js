// Browser port of the scikit-learn pipeline: word + char_wb TF-IDF (sublinear tf, l2) -> logistic regression.
(function (root) {
  function prep(model) {
    for (const k of ["word", "char"]) {
      const m = model[k], index = new Map();
      m.terms.forEach((t, i) => index.set(t, i));
      m.index = index;
    }
    return model;
  }
  function wordGrams(text) {
    const toks = text.toLowerCase().match(/[\p{L}\p{N}\p{M}_]{2,}/gu) || [];
    const out = toks.slice();
    for (let i = 0; i + 1 < toks.length; i++) out.push(toks[i] + " " + toks[i + 1]);
    return out;
  }
  function charGrams(text) {
    const out = [];
    for (const raw of text.toLowerCase().replace(/\s\s+/g, " ").split(/\s+/).filter(Boolean)) {
      const w = Array.from(" " + raw + " ");
      for (let n = 2; n <= 5; n++) {
        let off = 0;
        out.push(w.slice(off, off + n).join(""));
        while (off + n < w.length) { off++; out.push(w.slice(off, off + n).join("")); }
        if (off === 0) break;
      }
    }
    return out;
  }
  // returns {score, contributions:[[term, weight]]}
  function block(m, grams) {
    const tf = new Map();
    for (const g of grams) { const i = m.index.get(g); if (i !== undefined) tf.set(i, (tf.get(i) || 0) + 1); }
    let norm = 0; const vals = [];
    for (const [i, c] of tf) { const v = (1 + Math.log(c)) * m.idf[i]; vals.push([i, v]); norm += v * v; }
    norm = Math.sqrt(norm) || 1;
    let score = 0; const contrib = [];
    for (const [i, v] of vals) { const s = (v / norm) * m.coef[i]; score += s; contrib.push([m.terms[i], s]); }
    return { score, contrib };
  }
  function classify(model, text) {
    const w = block(model.word, wordGrams(text)), c = block(model.char, charGrams(text));
    const z = model.intercept + w.score + c.score;
    return { prob: 1 / (1 + Math.exp(-z)), words: w.contrib.sort((a, b) => Math.abs(b[1]) - Math.abs(a[1])) };
  }
  const api = { prep, classify };
  if (typeof module !== "undefined") module.exports = api; else root.SpamClassifier = api;
})(this);
