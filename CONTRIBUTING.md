# Contributing to PhishGuard

Thanks for your interest in improving PhishGuard! Bug reports, wrong-verdict reports,
new features, better data and documentation fixes are all welcome.

By taking part you agree to follow the [Code of Conduct](CODE_OF_CONDUCT.md).

## Ways to contribute

* **Report a wrong verdict.** A legitimate site flagged as phishing, or a phishing link
  marked safe, is the most useful feedback this project can get. Use the
  *Wrong verdict* issue template.
* **Report a bug** in the code, CLI or API with the *Bug report* template.
* **Suggest a feature** with the *Feature request* template.
* **Send a pull request.** For anything larger than a small fix, open an issue first so we
  can agree on the approach before you spend time on it.

Found a security problem in PhishGuard itself? Please don't open a public issue. See
[SECURITY.md](SECURITY.md).

## Handling phishing URLs safely

Live phishing links in issues and pull requests can be clicked by accident. **Defang**
them before posting:

| Instead of | Write |
|---|---|
| `http://evil.example/login` | `hxxp://evil[.]example/login` |
| `https://` | `hxxps://` |

Never post URLs that contain someone's personal data (email addresses, tokens, account
numbers). Replace those parts with `REDACTED`.

## Development setup

```bash
git clone https://github.com/1amaaaal/phisguard.git
cd phisguard
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
./scripts/download_data.sh      # needed only if you retrain
pytest                          # should pass before you start
```

The repository ships with a trained model in `models/`, so `predict` and the API work
without retraining.

## Project rules

These rules keep the reported metrics honest. Pull requests that break them will be asked
to change.

1. **Features must not see the scheme or `www.`** Everything goes through
   `normalize_url()` first. `test_scheme_and_www_do_not_change_features` enforces this.
   The scheme is a dataset artifact, not a phishing signal.
2. **Evaluate on unseen domains.** Use `grouped_split()` or the domain-grouped CV in
   `train.py`. Never use a random row split, because it lets the model memorise sites.
3. **Never tune on the test set.** Thresholds and hyper-parameters are chosen on the
   validation split inside the training data.
4. **No network calls at prediction time.** PhishGuard must stay offline and must never
   visit the URL it is classifying.
5. **Every new feature needs an explanation.** Add it to `FEATURES` in `features.py` with
   a plain-English description, so it can appear in the `reasons` output.

## Making a change

1. Create a branch: `git checkout -b fix/short-description`.
2. Make your change and add or update tests in `tests/`.
3. Run the checks:
   ```bash
   pytest
   python -m phishguard.predict --file scripts/red_team.txt
   ```
4. **If you changed features, data handling or the model**, retrain and include the new
   numbers in your pull request:
   ```bash
   python -m phishguard.train          # full run with CV and ablation (~13 min)
   ```
   Commit the updated `reports/` files and `models/phishguard.joblib`. A change that
   raises one metric but lowers recall at 2% FPR needs a good explanation.
5. Open a pull request and fill in the template.

## Code style

* Python 3.10+, type hints on public functions, and docstrings that explain *why*, not
  just *what*.
* Keep the dependency list short. Ask before adding a new package.
* Match the style of the surrounding code: naming, comment density and module layout.

## Commit messages

Use a short summary line in the imperative mood ("Add punycode homoglyph check", not
"Added stuff"), then a blank line and more detail if needed.

## Questions

Open a [discussion or issue](https://github.com/1amaaaal/phisguard/issues). There are no
silly questions.
