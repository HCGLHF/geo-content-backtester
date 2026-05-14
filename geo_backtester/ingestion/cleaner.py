from __future__ import annotations

import re

from bs4 import BeautifulSoup


NOISE_PATTERNS = [
    re.compile(r"^\s*(home|privacy policy|terms of service|cookie policy)\s*$", re.I),
    re.compile(r"^\s*(copyright|all rights reserved).*$", re.I),
    re.compile(r"^\s*(subscribe|newsletter|sign up).*$", re.I),
]


def _html_to_text(text: str) -> str:
    soup = BeautifulSoup(text, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "aside"]):
        tag.decompose()

    lines: list[str] = []
    for element in soup.find_all(["h1", "h2", "h3", "p", "li"]):
        content = element.get_text(" ", strip=True)
        if not content:
            continue
        if element.name == "h1":
            lines.append(f"# {content}")
        elif element.name == "h2":
            lines.append(f"## {content}")
        elif element.name == "h3":
            lines.append(f"### {content}")
        elif element.name == "li":
            lines.append(f"- {content}")
        else:
            lines.append(content)
    return "\n\n".join(lines)


def clean_text(text: str) -> str:
    if re.search(r"</?[a-z][\s\S]*>", text, re.I):
        text = _html_to_text(text)

    cleaned_lines: list[str] = []
    for line in text.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        stripped = re.sub(r"[ \t]+", " ", line).strip()
        if any(pattern.match(stripped) for pattern in NOISE_PATTERNS):
            continue
        cleaned_lines.append(stripped)

    text = "\n".join(cleaned_lines)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()
