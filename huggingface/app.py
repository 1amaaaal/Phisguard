"""Hugging Face Spaces demo for PhishGuard (Gradio)."""

import gradio as gr

from phishguard.model import PhishGuard

model = PhishGuard.load()

ICONS = {"safe": "🟢", "suspicious": "🟠", "phishing": "🔴"}
EXAMPLES = [
    "http://paypal.com.account-verify.xyz/webscr/login.php",
    "http://paypa1.com/signin",
    "https://office365-login.000webhostapp.com/owa/auth.php",
    "http://www.google.com@evil-site.ru/login",
    "https://xn--pypal-4ve.com/",
    "https://github.com/login",
    "https://en.wikipedia.org/wiki/Phishing",
]


def code(text: str) -> str:
    """Show user-supplied text literally inside Markdown."""
    return "`" + text.replace("`", "'") + "`"


def check(url: str) -> tuple[str, dict]:
    url = (url or "").strip()
    if not url:
        return "Paste a URL to check.", {}
    if len(url) > 4096:
        return "That URL is too long (max 4,096 characters).", {}
    pred = model.explain(url)
    lines = [
        f"## {ICONS[pred.verdict]} {pred.verdict.upper()}",
        f"**ML phishing score:** {pred.probability:.1%}  ",
        f"**URL:** {code(pred.url)}",
        "",
    ]
    if pred.reasons:
        lines += ["### Why", *[f"- {r}" for r in pred.reasons]]
    if pred.suspicious_fragments:
        lines += ["", "**Suspicious fragments:** " + ", ".join(code(f) for f in pred.suspicious_fragments)]
    return "\n".join(lines), pred.to_dict()


with gr.Blocks(title="PhishGuard") as demo:
    gr.Markdown(
        "# PhishGuard: explainable phishing URL detection\n"
        "Paste a link to see whether it looks like phishing, and why. Only the URL text is "
        "analysed: **the site is never visited**, so it is safe to paste live phishing links.\n\n"
        "[Source code and evaluation on GitHub](https://github.com/1amaaaal/Phisguard)"
    )
    with gr.Row():
        url_in = gr.Textbox(label="URL", placeholder="http://paypal.com.account-verify.xyz/login",
                            scale=5, autofocus=True)
        button = gr.Button("Check", variant="primary", scale=1)
    result = gr.Markdown()
    with gr.Accordion("Raw output (JSON)", open=False):
        raw = gr.JSON()
    gr.Examples(EXAMPLES, inputs=url_in)
    gr.Markdown(
        "<small>The verdict is a model prediction, not a guarantee. A URL classifier can't "
        "see the page content, so stay careful even with a 🟢 result.</small>"
    )
    button.click(check, inputs=url_in, outputs=[result, raw])
    url_in.submit(check, inputs=url_in, outputs=[result, raw])

if __name__ == "__main__":
    demo.launch()
