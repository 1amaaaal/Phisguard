import pytest
from fastapi.testclient import TestClient

from phishguard import api
from phishguard.model import MODEL_PATH

pytestmark = pytest.mark.skipif(not MODEL_PATH.exists(), reason="run `python -m phishguard.train` first")
client = TestClient(api.app)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"


def test_predict_phishing():
    r = client.post("/predict", json={"url": "http://paypal.com.account-verify.xyz/webscr/login.php"})
    assert r.status_code == 200
    body = r.json()
    assert body["verdict"] == "phishing"
    assert body["reasons"]


def test_predict_safe():
    assert client.post("/predict", json={"url": "https://www.wikipedia.org"}).json()["verdict"] == "safe"


def test_batch_and_validation():
    r = client.post("/predict/batch", json={"urls": ["google.com", "http://1.2.3.4/login.php"]})
    assert r.status_code == 200 and len(r.json()) == 2
    assert client.post("/predict", json={"url": ""}).status_code == 422


def test_index_page():
    r = client.get("/")
    assert r.status_code == 200 and "PhishGuard" in r.text
