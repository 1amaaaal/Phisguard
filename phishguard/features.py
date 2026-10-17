"""Hand-crafted lexical features, each with a human-readable explanation.

Every feature is computed from the URL string alone. FEATURES lists them in a
fixed order so the same vector layout is used in training and inference.
"""

from __future__ import annotations

import math
import re
from collections import Counter

import numpy as np

from .urltools import ParsedURL, normalize_url, parse_url

BRANDS = [
    "paypal", "apple", "icloud", "microsoft", "office365", "outlook",
    "hotmail", "google", "gmail", "facebook", "instagram", "whatsapp",
    "amazon", "netflix", "ebay", "chase", "wellsfargo", "bankofamerica",
    "citibank", "hsbc", "barclays", "santander", "dropbox", "docusign",
    "linkedin", "twitter", "yahoo", "steam", "steamcommunity", "roblox",
    "runescape", "coinbase", "binance", "metamask", "blockchain", "dhl",
    "fedex", "usps", "irs", "adobe", "wetransfer", "spotify", "absa",
    "itau", "bradesco", "nubank", "correios", "americanexpress", "visa",
    "mastercard", "alibaba", "aliexpress", "telegram", "discord",
]

SENSITIVE_WORDS = [
    "login", "log-in", "signin", "sign-in", "logon", "verify", "verification",
    "account", "update", "secure", "security", "banking", "confirm",
    "password", "passwd", "credential", "webscr", "cmd=_", "suspend",
    "unlock", "restore", "recover", "billing", "invoice", "payment",
    "wallet", "authenticate", "auth", "validate", "support", "service",
    "client", "customer", "ebayisapi", "limited", "alert", "token",
]

# TLDs with a high share of abuse in public phishing feeds (Spamhaus, Interisle).
SUSPICIOUS_TLDS = {
    "xyz", "top", "tk", "ml", "ga", "cf", "gq", "buzz", "club", "work",
    "live", "click", "link", "rest", "fit", "cyou", "icu", "monster", "sbs",
    "cfd", "online", "site", "website", "space", "info", "ru", "cn", "su",
    "pw", "cc", "ws", "bid", "loan", "win", "date", "zip", "mov", "support",
    "help", "shop", "store", "vip", "app", "page", "best", "bar", "quest",
}

SHORTENERS = {
    "bit.ly", "tinyurl.com", "goo.gl", "t.co", "ow.ly", "is.gd", "buff.ly",
    "rebrand.ly", "cutt.ly", "shorturl.at", "rb.gy", "t.ly", "bl.ink",
    "tiny.cc", "s.id", "v.gd", "qrco.de", "lnkd.in", "shorte.st", "adf.ly",
}

# Free hosting and file-sharing platforms that phishers abuse because the
# parent domain has a good reputation.
FREE_HOSTS = {
    "000webhostapp.com", "weebly.com", "wixsite.com", "wix.com", "firebaseapp.com",
    "web.app", "github.io", "glitch.me", "netlify.app", "vercel.app",
    "herokuapp.com", "blogspot.com", "wordpress.com", "sites.google.com",
    "docs.google.com", "forms.gle", "firebasestorage.googleapis.com",
    "storage.googleapis.com", "sharepoint.com", "onedrive.live.com",
    "dropboxusercontent.com", "ipfs.io", "pages.dev", "workers.dev",
    "r2.dev", "squarespace.com", "webflow.io", "jimdosite.com",
    "yolasite.com", "godaddysites.com", "mystrikingly.com", "ngrok.io",
    "repl.co", "azurewebsites.net", "appspot.com", "translate.goog",
}

RISKY_EXTENSIONS = (".php", ".asp", ".aspx", ".exe", ".zip", ".scr", ".js", ".cgi")


def shannon_entropy(text: str) -> float:
    if not text:
        return 0.0
    counts = Counter(text)
    n = len(text)
    return -sum(c / n * math.log2(c / n) for c in counts.values())


def levenshtein(a: str, b: str) -> int:
    if len(a) < len(b):
        a, b = b, a
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        current = [i]
        for j, cb in enumerate(b, 1):
            current.append(min(previous[j] + 1, current[j - 1] + 1,
                               previous[j - 1] + (ca != cb)))
        previous = current
    return previous[-1]


_HOMOGLYPHS = str.maketrans({"0": "o", "1": "l", "3": "e", "4": "a", "5": "s",
                             "7": "t", "8": "b", "@": "a", "$": "s"})


def _domain_label(p: ParsedURL) -> str:
    return p.registered_domain.split(".")[0] if p.registered_domain else ""


def brand_impersonation(p: ParsedURL) -> str | None:
    """A brand name appears in the URL but the site is not that brand's domain."""
    label = _domain_label(p)
    if label in BRANDS:
        return None
    text = f"{p.subdomain} {label} {p.path} {p.query}".lower()
    for brand in BRANDS:
        if len(brand) >= 5 and brand in text:
            return brand
    return None


def typosquat_target(p: ParsedURL) -> str | None:
    """The domain is a near-miss spelling of a brand, e.g. paypa1 or amaz0n."""
    label = _domain_label(p)
    if not label or label in BRANDS or p.is_ip:
        return None
    deglyphed = label.translate(_HOMOGLYPHS)
    for brand in BRANDS:
        if len(brand) < 5:
            continue
        if deglyphed == brand:
            return brand
        if abs(len(label) - len(brand)) <= 2 and levenshtein(label, brand) <= (1 if len(brand) < 7 else 2):
            return brand
    return None


def _is_free_host(host: str) -> bool:
    return any(host == h or host.endswith("." + h) for h in FREE_HOSTS)


# (name, description shown to users when the feature pushes towards phishing)
FEATURES: list[tuple[str, str]] = [
    ("url_length", "The URL is unusually long, which can hide the real destination"),
    ("host_length", "The hostname is unusually long"),
    ("path_length", "The path is unusually long"),
    ("query_length", "The query string is unusually long"),
    ("num_subdomains", "Many subdomains, a common trick to put a trusted name in front of the real domain"),
    ("num_hyphens_host", "Hyphens in the hostname, typical of look-alike domains like secure-paypal-login.com"),
    ("num_digits_host", "Digits in the hostname"),
    ("digit_ratio", "A high share of digits in the URL"),
    ("host_entropy", "The hostname looks random or machine-generated"),
    ("path_entropy", "The path looks random or machine-generated"),
    ("longest_token", "Contains a very long unbroken token, often an encoded ID or tracking blob"),
    ("path_depth", "The page is nested deep inside many folders"),
    ("num_query_params", "Many query parameters"),
    ("num_special_chars", "Many special characters such as % ~ _ = &"),
    ("has_ip_host", "Uses a raw IP address instead of a domain name"),
    ("has_port", "Specifies a non-standard port"),
    ("has_at_symbol", "Contains '@', and browsers ignore everything before it"),
    ("has_double_slash_path", "Contains '//' inside the path, a redirect trick"),
    ("has_punycode", "Uses punycode (xn--), which can disguise look-alike Unicode characters"),
    ("has_non_ascii", "Contains non-ASCII characters"),
    ("suspicious_tld", "Uses a top-level domain that is heavily abused for phishing"),
    ("is_shortener", "Uses a URL shortener that hides the final destination"),
    ("is_free_host", "Hosted on a free hosting or file-sharing platform phishers often abuse"),
    ("num_sensitive_words", "Contains words phishing pages use, like 'login', 'verify' or 'account'"),
    ("brand_impersonation", "Mentions a well-known brand, but the site is not that brand's domain"),
    ("typosquat", "The domain is a misspelling of a well-known brand"),
    ("risky_extension", "Points to a server script or executable (.php, .exe, .zip, ...)"),
    ("has_email_in_url", "Contains an email address, often used to pre-fill a fake login form"),
    ("tld_in_path", "Has a domain-like '.com' or '.net' inside the path"),
    ("vowel_ratio_host", "The hostname has an unnatural mix of letters"),
]
FEATURE_NAMES = [name for name, _ in FEATURES]
FEATURE_DESCRIPTIONS = dict(FEATURES)

_EMAIL_RE = re.compile(r"[\w.+-]+(@|%40)[\w-]+\.[\w.]+")
_TLD_IN_PATH_RE = re.compile(r"\.(com|net|org|info|co|io)(?=[/\-_.?]|$)")
_TOKEN_SPLIT_RE = re.compile(r"[/?&=.\-_:~%+#]")


def extract_features(url: str) -> dict[str, float]:
    p = parse_url(url)
    body = normalize_url(p.raw)
    lower = body.lower()
    host_letters = [c for c in p.host if c.isalpha()]

    return {
        "url_length": len(body),
        "host_length": len(p.host),
        "path_length": len(p.path),
        "query_length": len(p.query),
        "num_subdomains": len([s for s in p.subdomain.split(".") if s]),
        "num_hyphens_host": p.host.count("-"),
        "num_digits_host": sum(c.isdigit() for c in p.host),
        "digit_ratio": sum(c.isdigit() for c in body) / max(len(body), 1),
        "host_entropy": shannon_entropy(p.host),
        "path_entropy": shannon_entropy(p.path),
        "longest_token": max((len(t) for t in _TOKEN_SPLIT_RE.split(body)), default=0),
        "path_depth": len([s for s in p.path.split("/") if s]),
        "num_query_params": len([q for q in p.query.split("&") if q]),
        "num_special_chars": sum(body.count(c) for c in "%~_=&;!*$,"),
        "has_ip_host": float(p.is_ip),
        "has_port": float(p.port not in (None, 80, 443)),
        "has_at_symbol": float("@" in body.split("?")[0]),
        "has_double_slash_path": float("//" in p.path),
        "has_punycode": float("xn--" in p.host),
        "has_non_ascii": float(any(ord(c) > 127 for c in body)),
        "suspicious_tld": float(p.tld.split(".")[-1] in SUSPICIOUS_TLDS),
        "is_shortener": float(p.host in SHORTENERS),
        "is_free_host": float(_is_free_host(p.host)),
        "num_sensitive_words": sum(w in lower for w in SENSITIVE_WORDS),
        "brand_impersonation": float(brand_impersonation(p) is not None),
        "typosquat": float(typosquat_target(p) is not None),
        "risky_extension": float(p.path.lower().endswith(RISKY_EXTENSIONS)),
        "has_email_in_url": float(bool(_EMAIL_RE.search(body))),
        "tld_in_path": float(bool(_TLD_IN_PATH_RE.search(p.path.lower()))),
        "vowel_ratio_host": (sum(c in "aeiou" for c in host_letters) / len(host_letters)
                             if host_letters else 0.0),
    }


def feature_matrix(urls) -> np.ndarray:
    return np.array([[f[name] for name in FEATURE_NAMES]
                     for f in map(extract_features, urls)], dtype=np.float64)
