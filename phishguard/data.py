"""Build a clean, de-biased training set from the raw downloads.

Sources (see scripts/download_data.sh):
  * benign_iscx.csv            35k benign URLs crawled for ISCX-URL2016
  * phishtank_online_valid.csv 14.8k human-verified PhishTank phishing URLs
  * top10k_domains.txt         popular domains, used as benign homepages
  * faizann_urls.csv           345k benign URLs from 108k different sites (the
                               "bad" rows mix in malware, so they are not used)

Three problems in the raw data would let a model score near 100% without
learning anything about phishing. Each one is fixed here:

  1. Scheme and "www." leakage: 94% of benign URLs are http:// and almost none
     have www., while phishing is roughly half https. Fixed by normalize_url().
  2. Domain memorisation: the ISCX benign set holds only ~310 sites, a few with
     over 1,000 URLs each, and almost none have subdomains. Fixed by adding the
     much more varied faizann24 benign URLs, capping URLs per domain, and
     splitting train/test by registered domain, so the test set holds only
     unseen sites.
  3. "Phishing = homepage": almost every benign URL has a long path. Fixed by
     adding homepages of popular domains as benign examples.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .urltools import normalize_url, registered_domain

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
MAX_PER_DOMAIN = 40
EXTRA_BENIGN = 25_000   # sample size drawn from faizann_urls.csv


def load_raw(raw_dir: Path = RAW_DIR, extra_benign: int = EXTRA_BENIGN, seed: int = 42) -> pd.DataFrame:
    benign = pd.read_csv(raw_dir / "benign_iscx.csv", header=None, names=["url"])
    benign["source"] = "iscx_benign"
    benign["label"] = 0

    phish = pd.read_csv(raw_dir / "phishtank_online_valid.csv", usecols=["url", "target"])
    phish = phish.rename(columns={"target": "brand_target"})
    phish["source"] = "phishtank"
    phish["label"] = 1

    domains = pd.read_csv(raw_dir / "top10k_domains.txt", header=None, names=["domain"])
    homepages = pd.DataFrame({"url": domains["domain"], "source": "top_domains", "label": 0})

    extra = pd.read_csv(raw_dir / "faizann_urls.csv")
    extra = extra[extra["label"] == "good"][["url"]]
    # At most 3 URLs per host before sampling, so the sample spans many sites.
    host = extra["url"].str.split("/", n=1).str[0]
    extra = extra.groupby(host, group_keys=False).head(3)
    extra = extra.sample(min(extra_benign, len(extra)), random_state=seed)
    extra = extra.assign(source="faizann_benign", label=0)

    return pd.concat([benign, phish, homepages, extra], ignore_index=True)


def build_dataset(raw_dir: Path = RAW_DIR, max_per_domain: int = MAX_PER_DOMAIN,
                  seed: int = 42) -> pd.DataFrame:
    df = load_raw(raw_dir, seed=seed)
    df["url"] = df["url"].astype(str).str.strip()
    df = df[df["url"].str.len().between(4, 2000)]
    df["norm"] = df["url"].map(normalize_url)
    df["domain"] = df["url"].map(registered_domain)
    df = df[df["domain"].str.len() > 0]

    # A URL listed as both benign and phishing is ambiguous: drop it entirely.
    conflicted = df.groupby("norm")["label"].nunique()
    df = df[~df["norm"].isin(conflicted[conflicted > 1].index)]
    df = df.drop_duplicates("norm")

    # Cap URLs per (domain, label) so no single site dominates training.
    df = (df.sample(frac=1.0, random_state=seed)
            .groupby(["domain", "label"], group_keys=False)
            .head(max_per_domain))
    return df.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def grouped_split(df: pd.DataFrame, test_size: float = 0.2, seed: int = 42):
    """Split so every registered domain lands entirely in train or in test."""
    rng = np.random.default_rng(seed)
    domains = np.array(sorted(df["domain"].unique()))
    rng.shuffle(domains)
    test_domains = set(domains[: int(len(domains) * test_size)])
    is_test = df["domain"].isin(test_domains)
    return df[~is_test].reset_index(drop=True), df[is_test].reset_index(drop=True)
