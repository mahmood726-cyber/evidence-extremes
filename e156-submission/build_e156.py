"""Generate index.html for EvidenceExtremes E156 submission."""

import json
import os
import sys
import io

if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

TEMPLATE = "C:/E156/templates/e156_interactive_template.html"
PAPER_JSON = os.path.join(os.path.dirname(__file__), "paper.json")
OUT_HTML = os.path.join(os.path.dirname(__file__), "index.html")


def build():
    with open(TEMPLATE, encoding="utf-8") as f:
        template = f.read()

    with open(PAPER_JSON, encoding="utf-8") as f:
        paper = json.load(f)

    # Inject paper.json into template
    article_json = json.dumps(paper, indent=2, ensure_ascii=False)
    html = template.replace("__E156_JSON__", article_json)

    with open(OUT_HTML, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"Generated: {OUT_HTML}")


if __name__ == "__main__":
    build()
