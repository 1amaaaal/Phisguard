## What does this change?

<!-- A short summary. Link the issue it fixes, e.g. "Fixes #12". -->

## Type of change

- [ ] Bug fix
- [ ] New feature
- [ ] Model, feature or data change (affects the metrics)
- [ ] Documentation

## Checklist

- [ ] `pytest` passes
- [ ] `python -m phishguard.predict --file scripts/red_team.txt` still looks right
- [ ] New or changed behaviour has tests
- [ ] Any phishing URLs in this PR are defanged (`hxxp://evil[.]example`)

## Metrics (model, feature or data changes only)

<!-- Paste the before/after numbers from reports/report.md. -->

| Metric | Before | After |
|---|---|---|
| ROC-AUC (test) | | |
| Recall @ 2% FPR (test) | | |
| CV recall @ 2% FPR | | |
