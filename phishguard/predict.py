"""Classify URLs from the command line.

    python -m phishguard.predict "http://paypal.com.secure-login.xyz/verify"
    python -m phishguard.predict --file urls.txt --json
"""

from __future__ import annotations

import argparse
import json
import sys

from .model import MODEL_PATH, PhishGuard

COLORS = {"safe": "\033[32m", "suspicious": "\033[33m", "phishing": "\033[31m"}
RESET = "\033[0m"


def render(pred, color: bool) -> str:
    tag = pred.verdict.upper()
    if color:
        tag = f"{COLORS[pred.verdict]}{tag}{RESET}"
    lines = [f"{tag:<12} {pred.probability:6.1%}  {pred.url}"]
    lines += [f"    - {r}" for r in pred.reasons]
    if pred.suspicious_fragments:
        lines.append("    - Suspicious fragments: " + ", ".join(repr(f) for f in pred.suspicious_fragments))
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("urls", nargs="*")
    ap.add_argument("--file", help="text file with one URL per line")
    ap.add_argument("--json", action="store_true", help="print JSON lines")
    ap.add_argument("--model", default=str(MODEL_PATH))
    args = ap.parse_args(argv)

    urls = list(args.urls)
    if args.file:
        with open(args.file, encoding="utf-8") as fh:
            urls += [line.strip() for line in fh if line.strip() and not line.startswith("#")]
    if not urls:
        ap.error("give at least one URL or --file")

    model = PhishGuard.load(args.model)
    color = sys.stdout.isatty()
    for url in urls:
        pred = model.explain(url)
        print(json.dumps(pred.to_dict()) if args.json else render(pred, color))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
