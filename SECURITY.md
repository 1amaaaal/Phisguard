# Security policy

## Reporting a vulnerability

If you find a security problem in PhishGuard itself, for example a crash or hang caused by
a crafted URL, a way to make the API misbehave, or an unsafe dependency, please report it
privately:

1. Go to the repository's **Security** tab.
2. Click **Report a vulnerability** and describe the issue and how to reproduce it.

Please don't open a public issue for security problems. You should get a reply within a
week.

## What is not a vulnerability

A URL that PhishGuard classifies wrongly is a model limitation, not a security
vulnerability. Report it with the *Wrong verdict* issue template, and defang the URL
first (see [CONTRIBUTING.md](CONTRIBUTING.md#handling-phishing-urls-safely)).

## Model files

`PhishGuard.load()` uses `joblib`, which is built on Python's `pickle`. Loading a pickle
file can run arbitrary code, so only load model files you trained yourself or that come
from this repository.
