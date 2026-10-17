"""The PhishGuard model and its explanations.

A two-level stacked model:

  1. Text model: logistic regression over TF-IDF character 3-5-grams and word
     tokens of the normalised URL. It learns patterns no one wrote a rule for,
     such as "wp-content/plugins" phishing kits or "-verify-" fragments.
  2. Meta model: gradient-boosted trees over the 30 hand-crafted security
     features in features.py *plus* the text model's score. The trees learn
     when to trust the text model and when a red flag such as a raw IP,
     typosquat or brand-in-subdomain should override it.

The text score the meta model trains on is produced out-of-fold, grouped by
domain, so it is as noisy as the score it will see on brand-new sites.
Training on in-sample text scores would teach the trees to over-trust them.

On top of the ML score, explain() applies a reputation check, as browsers do:
a URL on one of the 10,000 most popular registered domains is treated as safe,
unless the host is a shortener or a platform that serves user content (free
hosting, Google Docs/Forms, cloud storage), where phishers can upload pages.
predict_proba() is the pure ML score and is what the evaluation measures.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import scipy.sparse as sp
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold

from . import __version__
from .features import (FEATURE_DESCRIPTIONS, FEATURE_NAMES, FREE_HOSTS, SHORTENERS,
                       brand_impersonation, extract_features, feature_matrix,
                       typosquat_target)
from .urltools import normalize_url, parse_url

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "phishguard.joblib"
TOP_DOMAINS_PATH = ROOT / "data" / "raw" / "top10k_domains.txt"

# Features where a large value is itself a red flag; compared against the
# 95th percentile of legitimate URLs seen in training.
_NUMERIC_FLAGS = ["url_length", "host_length", "path_length", "query_length",
                  "num_subdomains", "num_hyphens_host", "num_digits_host",
                  "host_entropy", "longest_token", "path_depth", "num_special_chars"]


@dataclass
class Prediction:
    url: str
    probability: float
    verdict: str          # "safe" | "suspicious" | "phishing"
    reasons: list[str] = field(default_factory=list)
    suspicious_fragments: list[str] = field(default_factory=list)
    model_scores: dict[str, float] = field(default_factory=dict)
    trusted_domain: bool = False

    def to_dict(self) -> dict:
        return self.__dict__.copy()


class TextModel:
    """TF-IDF character n-grams + word tokens -> logistic regression."""

    def __init__(self, C: float = 4.0, min_df: int = 3):
        self.char = TfidfVectorizer(analyzer="char", ngram_range=(3, 5), min_df=min_df,
                                    max_features=300_000, sublinear_tf=True, dtype=np.float32)
        self.word = TfidfVectorizer(analyzer="word", token_pattern=r"[A-Za-z0-9]{2,}",
                                    min_df=max(min_df - 1, 1), sublinear_tf=True, dtype=np.float32)
        self.clf = LogisticRegression(C=C, max_iter=3000, class_weight="balanced")

    def _transform(self, norm_urls, fit: bool = False):
        if fit:
            return sp.hstack([self.char.fit_transform(norm_urls),
                              self.word.fit_transform(norm_urls)]).tocsr()
        return sp.hstack([self.char.transform(norm_urls), self.word.transform(norm_urls)]).tocsr()

    def fit(self, norm_urls, y) -> "TextModel":
        self.clf.fit(self._transform(norm_urls, fit=True), y)
        return self

    def predict_proba(self, norm_urls) -> np.ndarray:
        return self.clf.predict_proba(self._transform(norm_urls))[:, 1]

    def top_fragments(self, norm_url: str, top_k: int = 5) -> list[str]:
        """Character n-grams and words in this URL that pushed the score up most."""
        row = self._transform([norm_url])
        contrib = row.multiply(self.clf.coef_[0]).tocsr()
        vocab = np.concatenate([self.char.get_feature_names_out(),
                                self.word.get_feature_names_out()])
        picked: list[str] = []
        for i in np.argsort(-contrib.data):
            if contrib.data[i] <= 0 or len(picked) == top_k:
                break
            gram = vocab[contrib.indices[i]]
            # Skip fragments that just overlap one already reported.
            if not any(gram in p or p in gram for p in picked):
                picked.append(gram)
        return picked


class PhishGuard:
    def __init__(self, seed: int = 42, oof_folds: int = 5):
        self.seed = seed
        self.oof_folds = oof_folds
        self.threshold = 0.5
        self.version = __version__
        self.text = TextModel()
        self.meta = HistGradientBoostingClassifier(
            max_iter=400, learning_rate=0.08, max_leaf_nodes=48,
            l2_regularization=1.0, class_weight="balanced", random_state=seed)
        self.benign_p95: dict[str, float] = {}
        self.trusted_domains: set[str] = set()

    # ---- training -----------------------------------------------------------
    def fit(self, urls, labels) -> "PhishGuard":
        urls, y = list(urls), np.asarray(labels)
        norm = np.array([normalize_url(u) for u in urls], dtype=object)
        groups = [parse_url(u).registered_domain for u in urls]

        # Out-of-fold text scores for the meta model (see module docstring).
        oof = np.zeros(len(urls))
        n_splits = min(self.oof_folds, len(set(groups)))
        for fit_idx, pred_idx in GroupKFold(n_splits).split(norm, y, groups):
            fold_model = TextModel(min_df=self.text.char.min_df).fit(norm[fit_idx], y[fit_idx])
            oof[pred_idx] = fold_model.predict_proba(norm[pred_idx])
        self.text.fit(norm, y)

        X_lex = feature_matrix(urls)
        self.meta.fit(np.c_[X_lex, oof], y)
        benign = X_lex[y == 0]
        self.benign_p95 = {name: float(np.percentile(benign[:, FEATURE_NAMES.index(name)], 95))
                           for name in _NUMERIC_FLAGS}
        return self

    def load_reputation(self, path: Path = TOP_DOMAINS_PATH) -> "PhishGuard":
        """Load the popular-domain list. Shorteners and user-content hosts are
        excluded per host in is_trusted(), so docs.google.com stays untrusted
        while accounts.google.com is trusted."""
        if Path(path).exists():
            with open(path, encoding="utf-8") as fh:
                self.trusted_domains = {line.strip().lower() for line in fh if line.strip()}
        return self

    # ---- scoring ------------------------------------------------------------
    def component_proba(self, urls) -> tuple[np.ndarray, np.ndarray]:
        """Return (final score, text-model score) for each URL."""
        urls = list(urls)
        p_text = self.text.predict_proba([normalize_url(u) for u in urls])
        p_final = self.meta.predict_proba(np.c_[feature_matrix(urls), p_text])[:, 1]
        return p_final, p_text

    def predict_proba(self, urls) -> np.ndarray:
        return self.component_proba(urls)[0]

    def predict(self, urls) -> np.ndarray:
        return (self.predict_proba(urls) >= self.threshold).astype(int)

    def verdict(self, probability: float) -> str:
        if probability >= self.threshold:
            return "phishing"
        if probability >= self.threshold / 2:
            return "suspicious"
        return "safe"

    def is_trusted(self, url: str) -> bool:
        p = parse_url(url)
        if p.host in SHORTENERS or any(p.host == h or p.host.endswith("." + h) for h in FREE_HOSTS):
            return False
        return p.registered_domain in self.trusted_domains

    # ---- explanations -------------------------------------------------------
    def _reasons(self, url: str) -> list[str]:
        feats = extract_features(url)
        parsed = parse_url(url)
        reasons = []
        brand = brand_impersonation(parsed)
        if brand:
            reasons.append(f"Mentions '{brand}', but the real domain is "
                           f"{parsed.registered_domain}")
        squat = typosquat_target(parsed)
        if squat:
            reasons.append(f"'{parsed.registered_domain}' looks like a misspelling of '{squat}'")
        for name in FEATURE_NAMES:
            if name in ("brand_impersonation", "typosquat"):
                continue
            value = feats[name]
            if name in self.benign_p95:
                if value > self.benign_p95[name] and value > 0:
                    reasons.append(f"{FEATURE_DESCRIPTIONS[name]} ({name}={value:.3g}, "
                                   f"above 95% of legitimate URLs)")
            elif name == "num_sensitive_words":
                if value >= 2:
                    reasons.append(f"{FEATURE_DESCRIPTIONS[name]} ({int(value)} found)")
            elif name.startswith(("has_", "is_", "suspicious_", "risky_", "tld_")) and value:
                reasons.append(FEATURE_DESCRIPTIONS[name])
        return reasons

    def explain(self, url: str) -> Prediction:
        p_final, p_text = (float(a[0]) for a in self.component_proba([url]))
        scores = {"final": round(p_final, 4), "text_model": round(p_text, 4)}
        if self.is_trusted(url):
            domain = parse_url(url).registered_domain
            return Prediction(
                url=url, probability=round(p_final, 4), verdict="safe",
                reasons=[f"{domain} is one of the 10,000 most popular websites "
                         f"(ML score alone: {p_final:.0%})"],
                model_scores=scores, trusted_domain=True)
        verdict = self.verdict(p_final)
        return Prediction(
            url=url, probability=round(p_final, 4), verdict=verdict,
            reasons=self._reasons(url),
            suspicious_fragments=(self.text.top_fragments(normalize_url(url))
                                  if verdict != "safe" else []),
            model_scores=scores)

    # ---- persistence --------------------------------------------------------
    def save(self, path: Path = MODEL_PATH) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path, compress=3)
        return path

    @staticmethod
    def load(path: Path = MODEL_PATH) -> "PhishGuard":
        # joblib uses pickle: only load model files you created yourself.
        return joblib.load(path)
