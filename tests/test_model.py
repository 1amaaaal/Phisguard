import numpy as np
import pytest

from phishguard.model import PhishGuard, TextModel

BENIGN = ["google.com", "wikipedia.org/wiki/Python", "github.com/python/cpython",
          "bbc.co.uk/news/world", "stackoverflow.com/questions/123/how-to-sort",
          "amazon.com/gp/cart", "nytimes.com/section/technology", "reddit.com/r/python"] * 4
PHISH = ["paypal.com.verify-account.xyz/login", "secure-appleid-unlock.top/signin.php",
         "192.168.4.2/bank/login.php?cmd=_update", "micros0ft-support.online/verify",
         "amaz0n-billing.club/account/update", "netflix-payment-confirm.site/auth",
         "foo.000webhostapp.com/office365/login.php", "chase.com-secure.ml/verify"] * 4


@pytest.fixture(scope="module")
def tiny_model():
    model = PhishGuard(oof_folds=2)
    model.text = TextModel(min_df=1)
    model.meta.set_params(max_iter=50, min_samples_leaf=2)
    return model.fit(BENIGN + PHISH, [0] * len(BENIGN) + [1] * len(PHISH))


def test_fit_predict(tiny_model):
    prob = tiny_model.predict_proba(BENIGN[:8] + PHISH[:8])
    assert prob.shape == (16,)
    assert ((prob >= 0) & (prob <= 1)).all()
    assert prob[8:].mean() > prob[:8].mean()


def test_explain(tiny_model):
    pred = tiny_model.explain("http://paypal.com.verify-account.xyz/login")
    assert pred.verdict in {"safe", "suspicious", "phishing"}
    assert any("paypal" in r for r in pred.reasons)
    assert set(pred.model_scores) == {"final", "text_model"}


def test_reputation_layer(tiny_model):
    tiny_model.trusted_domains = {"paypal.com", "google.com"}
    try:
        assert tiny_model.explain("https://www.paypal.com/signin").verdict == "safe"
        # User-content platforms on a trusted domain must not be allowlisted.
        assert not tiny_model.is_trusted("https://docs.google.com/forms/d/e/abc/viewform")
        assert not tiny_model.is_trusted("http://paypal.com.evil.xyz/")
        assert tiny_model.is_trusted("https://accounts.google.com/ServiceLogin")
    finally:
        tiny_model.trusted_domains = set()


def test_save_load_roundtrip(tiny_model, tmp_path):
    path = tiny_model.save(tmp_path / "m.joblib")
    loaded = PhishGuard.load(path)
    urls = PHISH[:3]
    np.testing.assert_allclose(loaded.predict_proba(urls), tiny_model.predict_proba(urls))
