from __future__ import annotations

import re


def evaluate_structure(text: str) -> dict[str, object]:
    lines = text.splitlines()
    headings = [line.strip() for line in lines if re.match(r"^#{1,6}\s+\S+", line.strip())]
    h1s = [heading for heading in headings if heading.startswith("# ") and not heading.startswith("##")]
    h2_h3 = [heading for heading in headings if heading.startswith("##")]
    paragraphs = [block.strip() for block in re.split(r"\n\s*\n", text) if block.strip() and not block.strip().startswith("#")]
    issues: list[str] = []
    recommendations: list[str] = []
    score = 100

    if not h1s:
        score -= 15
        issues.append("No H1 detected")
        recommendations.append("Add one descriptive H1 near the top")
    elif len(h1s[0].replace("#", "").strip().split()) < 4:
        score -= 7
        issues.append("H1 may be too short to be descriptive")

    if len(h2_h3) < 2:
        score -= 15
        issues.append("Limited H2/H3 structure")
        recommendations.append("Break the article into clear semantic H2/H3 sections")

    long_paragraphs = [p for p in paragraphs if len(p.split()) > 140]
    if long_paragraphs:
        score -= min(15, len(long_paragraphs) * 5)
        issues.append("Some paragraphs are too long")
        recommendations.append("Split long paragraphs into shorter answer-ready blocks")

    section_lengths = []
    current_count = 0
    for line in lines:
        if line.startswith("## ") and current_count:
            section_lengths.append(current_count)
            current_count = 0
        current_count += len(line.split())
    if current_count:
        section_lengths.append(current_count)
    if any(length > 450 for length in section_lengths):
        score -= 10
        issues.append("Some sections are too long")
        recommendations.append("Break long sections into smaller H2/H3 blocks")

    if not re.search(r"\bFAQ\b|frequently asked|^##\s+.+\?", text, re.I | re.M):
        score -= 10
        issues.append("No FAQ section detected")
        recommendations.append("Add FAQ section with direct questions")

    if not re.search(r"\b(summary|conclusion|key takeaways|in short)\b", text, re.I):
        score -= 8
        issues.append("No summary block detected")
        recommendations.append("Add a short summary or key takeaways section")

    if not re.search(r"\[[^\]]+\]\([^)]+\)|<a\s+", text, re.I):
        score -= 5
        issues.append("No internal links detected")

    if not re.search(r"\b(schema|structured data|FAQPage|Article schema)\b", text, re.I):
        score -= 4
        issues.append("No schema-related content detected")

    if not re.search(r"^(>|\*\*Answer|\*\*Definition|Definition:)", text, re.I | re.M):
        score -= 6
        issues.append("No direct answer block detected")
        recommendations.append("Add a short definition paragraph near the top")

    return {
        "structure_score": round(max(0, min(100, score)), 2),
        "issues": issues,
        "recommendations": recommendations,
        "heading_count": len(headings),
        "paragraph_count": len(paragraphs),
        "faq_detected": not any(issue == "No FAQ section detected" for issue in issues),
    }
