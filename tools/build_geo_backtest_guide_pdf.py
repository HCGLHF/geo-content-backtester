from __future__ import annotations

from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    HRFlowable,
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "GEO_Content_Backtester_核心指标与逻辑指南.pdf"


def register_fonts() -> tuple[str, str]:
    font_dir = Path("C:/Windows/Fonts")
    normal_candidates = [
        font_dir / "NotoSansSC-VF.ttf",
        font_dir / "simhei.ttf",
        font_dir / "msyh.ttc",
    ]
    bold_candidates = [
        font_dir / "msyhbd.ttc",
        font_dir / "simhei.ttf",
        font_dir / "NotoSansSC-VF.ttf",
    ]
    normal = next(path for path in normal_candidates if path.exists())
    bold = next(path for path in bold_candidates if path.exists())
    pdfmetrics.registerFont(TTFont("CJK", str(normal)))
    pdfmetrics.registerFont(TTFont("CJK-Bold", str(bold)))
    return "CJK", "CJK-Bold"


FONT, FONT_BOLD = register_fonts()


def styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    palette = {
        "ink": colors.HexColor("#1F2937"),
        "muted": colors.HexColor("#6B7280"),
        "accent": colors.HexColor("#0F766E"),
        "accent_dark": colors.HexColor("#134E4A"),
        "line": colors.HexColor("#D1D5DB"),
        "soft": colors.HexColor("#F3F7F6"),
        "warn": colors.HexColor("#B45309"),
    }
    return {
        "title": ParagraphStyle(
            "Title",
            parent=base["Title"],
            fontName=FONT_BOLD,
            fontSize=24,
            leading=30,
            textColor=palette["accent_dark"],
            alignment=TA_CENTER,
            spaceAfter=10,
            wordWrap="CJK",
        ),
        "subtitle": ParagraphStyle(
            "Subtitle",
            parent=base["Normal"],
            fontName=FONT,
            fontSize=11.5,
            leading=17,
            textColor=palette["muted"],
            alignment=TA_CENTER,
            spaceAfter=16,
            wordWrap="CJK",
        ),
        "h1": ParagraphStyle(
            "Heading1",
            parent=base["Heading1"],
            fontName=FONT_BOLD,
            fontSize=16,
            leading=22,
            textColor=palette["accent_dark"],
            spaceBefore=14,
            spaceAfter=8,
            wordWrap="CJK",
        ),
        "h2": ParagraphStyle(
            "Heading2",
            parent=base["Heading2"],
            fontName=FONT_BOLD,
            fontSize=12.8,
            leading=18,
            textColor=palette["ink"],
            spaceBefore=10,
            spaceAfter=5,
            wordWrap="CJK",
        ),
        "body": ParagraphStyle(
            "Body",
            parent=base["BodyText"],
            fontName=FONT,
            fontSize=10.2,
            leading=16,
            textColor=palette["ink"],
            spaceAfter=6,
            wordWrap="CJK",
        ),
        "small": ParagraphStyle(
            "Small",
            parent=base["BodyText"],
            fontName=FONT,
            fontSize=8.6,
            leading=12.5,
            textColor=palette["muted"],
            spaceAfter=4,
            wordWrap="CJK",
        ),
        "table_header": ParagraphStyle(
            "TableHeader",
            parent=base["BodyText"],
            fontName=FONT_BOLD,
            fontSize=8.8,
            leading=12,
            textColor=colors.white,
            alignment=TA_LEFT,
            wordWrap="CJK",
        ),
        "table_cell": ParagraphStyle(
            "TableCell",
            parent=base["BodyText"],
            fontName=FONT,
            fontSize=8.4,
            leading=12.2,
            textColor=palette["ink"],
            wordWrap="CJK",
        ),
        "callout": ParagraphStyle(
            "Callout",
            parent=base["BodyText"],
            fontName=FONT,
            fontSize=9.6,
            leading=15,
            textColor=palette["accent_dark"],
            wordWrap="CJK",
        ),
    }


S = styles()


def P(text: str, style: str = "body") -> Paragraph:
    return Paragraph(text, S[style])


def bullet(text: str) -> Paragraph:
    return Paragraph(f"• {text}", ParagraphStyle("Bullet", parent=S["body"], leftIndent=12, firstLineIndent=-8))


def make_table(rows: list[list[str]], widths: list[float]) -> Table:
    data = []
    for i, row in enumerate(rows):
        style = "table_header" if i == 0 else "table_cell"
        data.append([P(cell, style) for cell in row])
    table = Table(data, colWidths=widths, repeatRows=1, hAlign="LEFT")
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0F766E")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#D1D5DB")),
                ("BACKGROUND", (0, 1), (-1, -1), colors.white),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )
    return table


def callout(text: str) -> Table:
    table = Table([[P(text, "callout")]], colWidths=[170 * mm], hAlign="LEFT")
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F3F7F6")),
                ("BOX", (0, 0), (-1, -1), 0.6, colors.HexColor("#99C9C2")),
                ("LEFTPADDING", (0, 0), (-1, -1), 9),
                ("RIGHTPADDING", (0, 0), (-1, -1), 9),
                ("TOPPADDING", (0, 0), (-1, -1), 8),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
            ]
        )
    )
    return table


def header_footer(canvas, doc) -> None:  # type: ignore[no-untyped-def]
    canvas.saveState()
    canvas.setFont(FONT, 8)
    canvas.setFillColor(colors.HexColor("#6B7280"))
    canvas.drawString(20 * mm, 284 * mm, "GEO Content Backtester · 核心指标与逻辑指南")
    canvas.setStrokeColor(colors.HexColor("#D1D5DB"))
    canvas.line(20 * mm, 281.5 * mm, 190 * mm, 281.5 * mm)
    canvas.drawRightString(190 * mm, 12 * mm, f"第 {doc.page} 页")
    canvas.restoreState()


def build_story() -> list:
    story: list = []
    story.append(Spacer(1, 28 * mm))
    story.append(P("GEO Content Backtester", "title"))
    story.append(P("核心批判指标与判分逻辑指南", "title"))
    story.append(P("内部使用版 · 面向内容、策略与产品团队 · 生产域名：alphaxxxx.com", "subtitle"))
    story.append(HRFlowable(width="70%", thickness=1.2, color=colors.HexColor("#0F766E"), spaceBefore=10, spaceAfter=18))
    story.append(
        callout(
            "这份指南解释 backtest 系统在上线前批判一篇新文章的方式：它不是等 Google、AI crawler 或 LLM 系统重新抓取，而是在本地模拟检索、排序、引用和回答生成，判断新版本是否比旧版本更适合作为 AI 答案的可检索证据。"
        )
    )
    story.append(Spacer(1, 12 * mm))
    story.append(P("MVP 的目标不是做一个完整 SaaS，而是给内容团队一个能立即使用的本地评估闭环：输入 old_article.md、new_article.md 和 queries.csv，输出 CSV、JSON 与 HTML/PDF 报告，明确新稿赢在哪里、输在哪里、该改哪一段。", "body"))
    story.append(PageBreak())

    story.append(P("1. 这个 backtest 到底在批判什么", "h1"))
    story.append(P("GEO Content Backtester 的核心不是“文章写得好不好”的主观评审，而是用一套近似 RAG/AI 搜索的本地流程，批判文章是否能被 AI 系统发现、取回、引用并用于生成可靠答案。", "body"))
    for item in [
        "可检索性：目标文章或目标 chunk 是否能在相关 query 下进入 Top K。",
        "排序优势：新版本相对旧版本，目标 chunk 的 rank 是否更靠前。",
        "引用价值：chunk 是否有明确主体、定义句、实体、结构化 claim，能被 LLM 当作证据引用。",
        "答案支撑：检索到的上下文是否足以让 LLM 生成 grounded、完整、少幻觉的回答。",
        "可改进定位：报告必须指出弱 query、弱 chunk、缺失实体和结构问题，而不是只给一个总分。",
    ]:
        story.append(bullet(item))

    story.append(P("2. 本地评估流水线", "h1"))
    story.append(
        make_table(
            [
                ["阶段", "输入/动作", "批判重点", "主要输出"],
                ["加载与清洗", "读取 old/new markdown 或 HTML，保留标题、段落、列表和 section 边界。", "避免过度清洗，保留影响检索的标题与语义结构。", "Article 对象"],
                ["Chunking", "按约 500 tokens、80 overlap 切分，并把 title 与 heading path prepend 到 chunk。", "chunk 是否能独立表达；标题上下文是否进入检索文本。", "Chunk 列表"],
                ["BM25 检索", "低成本关键词/词面匹配。", "query 关键词、实体、术语是否能命中文章。", "BM25 Top K"],
                ["Embedding 检索", "sentence-transformers/OpenAI/TF-IDF fallback 做语义相似度。", "query 改写后语义是否仍能命中。", "Embedding Top K"],
                ["Hybrid 排序", "归一化 BM25 与 embedding，按 alpha 融合。", "综合词面与语义信号后的真实候选排名。", "Hybrid Top K"],
                ["评估与报告", "计算 retrieval、citation、entity、structure、answer 与总分。", "新稿是否胜出，失败点在哪里。", "CSV/JSON/report.html/PDF"],
            ],
            [24 * mm, 53 * mm, 55 * mm, 38 * mm],
        )
    )

    story.append(P("3. 检索指标：判断“能不能被拿出来”", "h1"))
    story.append(P("检索层回答两个最硬的问题：目标文章是否出现，以及出现得够不够靠前。MVP 暂无人工 relevance label 时，相关性由 target_article 与 expected_answer_points 的关键词重合共同判断。", "body"))
    story.append(
        make_table(
            [
                ["指标", "定义", "如何解读"],
                ["Hit@1 / Hit@3 / Hit@5", "Top K 内是否出现相关 chunk。出现记 1，否则 0。", "Hit@3 是最关键的内容可见性信号：如果目标证据进不了前三，AI 回答很可能引用别的内容。"],
                ["MRR", "第一个相关 chunk 的倒数排名：rank=1 得 1，rank=2 得 0.5，没有命中得 0。", "比 Hit@K 更敏感。rank 从 3 提到 1，会明显提升 MRR。"],
                ["average_rank", "所有 query 下第一个相关 chunk 的平均排名。", "越低越好，用来判断整体检索稳定性。"],
                ["top_chunk_score", "Top chunk 的 BM25、embedding 或 hybrid 分数。", "用于解释为什么某个 chunk 被推到前面，但跨 query 不宜孤立比较。"],
                ["rank_delta", "old_rank - new_rank。", "正数代表新稿排名上升；负数代表新稿退步。"],
                ["winner", "按 rank_delta 与命中情况判定 new、old 或 tie。", "用于 query 级诊断，而不是替代总分。"],
            ],
            [32 * mm, 61 * mm, 77 * mm],
        )
    )
    story.append(P("Hybrid 排序公式", "h2"))
    story.append(P("先对每个 query 的 BM25 分数与 embedding 相似度做 min-max normalization，再按默认权重融合：", "body"))
    story.append(callout("hybrid_score = alpha × bm25_norm + (1 - alpha) × embedding_norm；默认 alpha = 0.45，即 45% 词面匹配 + 55% 语义匹配。"))

    story.append(PageBreak())
    story.append(P("4. 引用价值指标：判断“值不值得被 LLM 引用”", "h1"))
    story.append(P("citation_score 是非 LLM 启发式评分，范围 0-100。它批判的是 chunk 是否像一个可以放进 AI 答案里的证据片段，而不是是否有营销感。", "body"))
    story.append(
        make_table(
            [
                ["维度", "加分信号", "扣分信号/风险"],
                ["定义句", "包含“is the process of / refers to / means / is defined as / is a”等定义型句式。", "没有直接定义，或开头只写抽象愿景。"],
                ["具体 claim", "有可核查的解释、对比、步骤、例子、测量口径。", "出现 unlock your potential、future-ready、transform your business、revolutionary、seamless experience、new era 等空泛表达。"],
                ["实体密度", "明确提到 GEO-ALPHA、Generative Engine Optimization、ChatGPT、Perplexity、Gemini、Google AI Overviews 等。", "主语不清，品牌、平台、概念缺位。"],
                ["答案结构", "列表、编号步骤、比较段、FAQ、直接解释。", "大段散文式描述，难以被截取成答案证据。"],
                ["独立性", "chunk 单独拿出来仍能明白“谁做什么、为什么重要”。", "开头大量 this / it / they，没有清晰 antecedent。"],
            ],
            [30 * mm, 70 * mm, 70 * mm],
        )
    )
    story.append(P("引用强度分层", "h2"))
    for item in [
        "75-100：strong，可优先作为 AI answer 的支持证据。",
        "55-74：moderate，信息有用，但可能缺少定义、实体或结构化表达。",
        "0-54：weak，通常需要重写，尤其是营销化、主语不清或缺少可引用 claim 的段落。",
    ]:
        story.append(bullet(item))

    story.append(P("5. 实体指标：判断“AI 是否知道这篇文章在讲谁和什么”", "h1"))
    story.append(P("entity_score 检查品牌实体、核心主题实体、平台实体是否覆盖，以及命名是否一致。GEO 类内容尤其依赖实体清晰度，因为 AI retrieval 和 citation 常常围绕实体关系展开。", "body"))
    story.append(
        make_table(
            [
                ["实体组", "示例", "批判逻辑"],
                ["brand_entities", "GEO-ALPHA、alphaxxxx.com", "品牌与生产域名是否被清楚连接，避免只说“we”。"],
                ["core_topic_entities", "Generative Engine Optimization、AI search、citation readiness、retrieval ranking analysis、schema markup", "主题语义场是否完整，是否覆盖产品真正想占领的概念。"],
                ["platform_entities", "ChatGPT、Claude、Gemini、Perplexity、Google AI Overviews", "是否明确面向 AI answer/retrieval 发生的平台。"],
                ["inconsistent_terms", "GEO optimization、AI SEO、Generative SEO", "不稳定命名会稀释实体一致性，降低可引用与可归因能力。"],
            ],
            [37 * mm, 58 * mm, 75 * mm],
        )
    )
    story.append(callout("推荐口径：GEO-ALPHA helps businesses improve AI search visibility through retrieval analysis, citation readiness, and entity optimization. 这种句子同时连接品牌、服务类别和方法论，比“We help brands grow in the AI era”更适合 GEO。"))

    story.append(PageBreak())
    story.append(P("6. 结构指标：判断“页面是否适合被切块和回答”", "h1"))
    story.append(P("structure_score 批判的是文章结构对检索与 chunking 是否友好。AI 检索并不只看正文词频，标题层级、FAQ、总结、段落长度都会影响 chunk 的可理解性。", "body"))
    for item in [
        "Has H1：必须有清晰、描述性 H1，而不是只写短标签。",
        "Has H2/H3 structure：文章需要可切分的语义区块。",
        "Sections are not too long：长 section 会让 chunk 混入多个主题，影响命中和引用。",
        "Paragraphs are not too long：长段落通常不适合直接进入 AI answer。",
        "FAQ readiness：问题式标题能直接匹配用户 query。",
        "Summary/conclusion：帮助 LLM 获取高密度归纳信息。",
        "Internal links / schema-related content：用于后续扩展到真实站点 SEO/GEO 结构检查。",
        "Direct answer blocks：例如 Definition、Answer、Key takeaways，能提高引用可用性。",
    ]:
        story.append(bullet(item))

    story.append(P("7. 答案质量指标：判断“检索结果能不能支撑 grounded answer”", "h1"))
    story.append(P("answer_eval 是可选层。如果存在 OPENAI_API_KEY，系统会分别用 old/new 的 Top 5 hybrid chunks 生成回答，并让 LLM judge 只基于 retrieved context 打分。没有 API key 时，answer_score = null，系统继续运行。", "body"))
    story.append(
        make_table(
            [
                ["评分项", "含义"],
                ["faithfulness", "回答是否忠实于提供的上下文。"],
                ["relevance", "是否直接回答 query。"],
                ["completeness", "是否覆盖 expected_answer_points。"],
                ["citation_support", "是否用 chunk_id 支撑关键陈述。"],
                ["hallucination_risk", "是否引入上下文外信息，风险越高越差。"],
                ["overall_answer_score", "综合答案质量分，进入总分公式。"],
            ],
            [48 * mm, 122 * mm],
        )
    )
    story.append(callout("重要原则：没有 API key 时不阻塞 MVP。内容团队仍然可以依赖 retrieval、citation、entity、structure 四类指标做上线前改稿。"))

    story.append(P("8. 总 GEO Score：如何把批判结果合成一个分数", "h1"))
    story.append(P("总分满分 100。它不是绝对真理，而是为了让旧稿、新稿、不同版本之间可以稳定比较。", "body"))
    story.append(
        make_table(
            [
                ["场景", "Retrieval", "Citation", "Answer", "Entity", "Structure"],
                ["answer_score 可用", "35%", "25%", "20%", "10%", "10%"],
                ["answer_score 不可用", "45%", "30%", "0%", "12.5%", "12.5%"],
            ],
            [45 * mm, 25 * mm, 25 * mm, 25 * mm, 25 * mm, 25 * mm],
        )
    )
    story.append(P("提升判断：absolute_delta = new_total - old_total；relative_delta_percent = absolute_delta / old_total × 100%。winner 按 delta 大于、小于或等于 0 判定。", "body"))

    story.append(PageBreak())
    story.append(P("9. 如何读报告并指导改稿", "h1"))
    story.append(
        make_table(
            [
                ["报告区域", "内容团队应该看什么", "典型行动"],
                ["Executive Summary", "总分、赢家、提升幅度。", "判断新稿是否值得进入发布候选。"],
                ["Retrieval Performance", "哪些 query 的 new_rank 没有改善，哪些 query 的 MRR 低。", "把对应用户问题改写成 H2/FAQ/direct answer。"],
                ["Query-Level Diagnosis", "old/new Top chunks 的文本差异。", "补充缺失 claim，重写命中但不可引用的 chunk。"],
                ["Citation Readiness", "强/弱 chunk、vague phrase、reasons。", "把营销句改成定义、对比、步骤、可测量表述。"],
                ["Entity Coverage", "missing_entities 与 inconsistent_terms。", "补齐平台、品牌、主题实体，并统一命名。"],
                ["Structure Analysis", "长段落、缺 FAQ、缺 summary、缺 direct answer。", "拆 section、加 FAQ、加 Definition 或 Key takeaways。"],
                ["Raw Outputs", "CSV/JSON 明细。", "交给策略或产品做二次分析、仪表盘或版本追踪。"],
            ],
            [36 * mm, 68 * mm, 66 * mm],
        )
    )
    story.append(P("推荐改稿顺序", "h2"))
    for item in [
        "先修 query 级失败：new_rank 低于 old_rank、Hit@3=0、MRR 显著下降的 query 优先。",
        "再修 citation weak chunks：尤其是 Top K 中被检索到但 citation_score 低的 chunk。",
        "补实体：缺平台实体、品牌实体或核心 topic entity 会影响 AI 对主题归因。",
        "最后修结构：FAQ、summary、H2/H3、direct answer blocks 能提高下一轮 chunking 与 retrieval 稳定性。",
    ]:
        story.append(bullet(item))

    story.append(P("10. MVP 的边界与后续升级", "h1"))
    story.append(P("这个系统故意不依赖真实 crawler indexing，也不修改生产网站。它批判的是“如果这篇内容被拿去做检索语料，它是否更可能被取回、引用并支撑答案”。", "body"))
    story.append(
        make_table(
            [
                ["当前 MVP 边界", "后续可升级方向"],
                ["相关性主要依赖 target_article 与关键词重合。", "加入人工 relevance label、query intent 权重、负样本和 competitor 页面。"],
                ["BM25 tokenizer 简单 lowercase split。", "加入分词、同义词、实体词典、query expansion。"],
                ["Embedding 默认 local/fallback，可能与真实 LLM retrieval 不完全一致。", "接入 OpenAI embeddings、向量库、真实 RAG 配置回放。"],
                ["Citation score 是启发式。", "加入 LLM citation judge 与 factual claim verifier。"],
                ["Answer eval 依赖 OPENAI_API_KEY。", "接入多模型 judge、RAGAS、人工审核闭环。"],
            ],
            [82 * mm, 88 * mm],
        )
    )

    story.append(P("附录：核心文件与输出", "h1"))
    story.append(
        make_table(
            [
                ["模块/文件", "作用"],
                ["geo_backtester/chunking/chunker.py", "按 token/overlap 切分文章并保留 heading context。"],
                ["geo_backtester/retrieval/bm25_retriever.py", "BM25 词面检索。"],
                ["geo_backtester/retrieval/embedding_retriever.py", "语义检索；OpenAI、sentence-transformers、TF-IDF fallback。"],
                ["geo_backtester/retrieval/hybrid_retriever.py", "BM25 + embedding 归一化融合。"],
                ["geo_backtester/evaluation/retrieval_metrics.py", "Hit@K、MRR、average_rank、rank_delta、winner。"],
                ["geo_backtester/evaluation/citation_eval.py", "引用价值启发式评分。"],
                ["geo_backtester/evaluation/entity_eval.py", "实体覆盖与命名一致性检查。"],
                ["geo_backtester/evaluation/structure_eval.py", "标题、段落、FAQ、summary、direct answer 结构评分。"],
                ["geo_backtester/evaluation/geo_score.py", "总 GEO score 权重公式。"],
                ["outputs/runs/{run_id}/", "retrieval_results.csv、citation_results.csv、answer_results.csv、score_summary.json、report.html。"],
            ],
            [72 * mm, 98 * mm],
        )
    )
    return story


def build_pdf() -> Path:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(
        str(OUT),
        pagesize=A4,
        rightMargin=20 * mm,
        leftMargin=20 * mm,
        topMargin=22 * mm,
        bottomMargin=18 * mm,
        title="GEO Content Backtester 核心指标与逻辑指南",
        author="GEO-ALPHA",
    )
    doc.build(build_story(), onFirstPage=header_footer, onLaterPages=header_footer)
    return OUT


if __name__ == "__main__":
    print(build_pdf())
