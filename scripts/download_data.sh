#!/usr/bin/env bash
# Download the three public datasets PhishGuard trains on into data/raw/.
set -euo pipefail
cd "$(dirname "$0")/../data/raw"
REPO=https://raw.githubusercontent.com/shreyagopal/Phishing-Website-Detection-by-Machine-Learning-Techniques/master/DataFiles
curl -fsSL -o benign_iscx.csv           "$REPO/1.Benign_list_big_final.csv"   # ISCX-URL2016 benign URLs
curl -fsSL -o phishtank_online_valid.csv "$REPO/2.online-valid.csv"           # verified PhishTank URLs
curl -fsSL -o top10k_domains.txt https://raw.githubusercontent.com/zer0h/top-1000000-domains/master/top-10000-domains
# 420k labelled URLs; only the 345k "good" ones are used, for benign-site diversity
curl -fsSL -o faizann_urls.csv https://raw.githubusercontent.com/faizann24/Using-machine-learning-to-detect-malicious-URLs/master/data/data.csv
wc -l benign_iscx.csv phishtank_online_valid.csv top10k_domains.txt faizann_urls.csv
