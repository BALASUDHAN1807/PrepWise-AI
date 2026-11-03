"""Downloadable PDF mock-interview report (ReportLab).

The report is rendered from the canonical English feedback (``content_en``) so it works with
the built-in fonts. If ``PDF_FONT_PATH`` points to a Unicode TTF (for example Noto Sans Tamil
or Noto Sans Devanagari) and the stored feedback is Tamil/Hindi, the translated text is used.
"""
from __future__ import annotations

import io
import os
from datetime import datetime
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from backend.core.config import get_settings


def _font() -> tuple[str, bool]:
    path = get_settings().pdf_font_path
    if path and os.path.exists(path):
        try:
            pdfmetrics.registerFont(TTFont("PWUnicode", path))
            return "PWUnicode", True
        except Exception:
            pass
    return "Helvetica", False


def build_pdf(user, session, feedback, answers: list) -> bytes:
    font, unicode_ok = _font()
    content = feedback.content if (unicode_ok and feedback.language != "en") else feedback.content_en
    ss = getSampleStyleSheet()
    body = ParagraphStyle("body", parent=ss["BodyText"], fontName=font, fontSize=9.5, leading=13)
    h1 = ParagraphStyle("h1", parent=ss["Title"], fontName=font, fontSize=20, textColor=colors.HexColor("#1f3a5f"))
    h2 = ParagraphStyle("h2", parent=ss["Heading2"], fontName=font, fontSize=13, textColor=colors.HexColor("#1f3a5f"),
                        spaceBefore=10)
    small = ParagraphStyle("small", parent=body, fontSize=8, textColor=colors.HexColor("#555555"))
    P = lambda t, st=body: Paragraph(escape(str(t)).replace("\n", "<br/>"), st)  # noqa: E731

    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm, topMargin=16 * mm,
                            bottomMargin=16 * mm, title="PrepWise AI - Mock Interview Report",
                            author="PrepWise AI")
    story = [P("PrepWise AI - Mock Interview Report", h1),
             P(f"Candidate: {user.name}   |   Target role: {user.target_role}   |   Session #{session.id} "
               f"({session.kind})   |   {datetime.now().strftime('%d %b %Y')}", small), Spacer(1, 6)]
    overall = content.get("overall_score", session.overall_score or 0)
    t = Table([[P("Overall session score"), P(f"{overall:.0f} / 100", h2)]], colWidths=[60 * mm, 40 * mm])
    t.setStyle(TableStyle([("BOX", (0, 0), (-1, -1), 0.6, colors.HexColor("#1f3a5f")),
                           ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#eef3fa")), ("VALIGN", (0, 0), (-1, -1), "MIDDLE")]))
    story += [t, P("Performance summary", h2), P(content["summary"])]
    for title, key in (("Strengths", "strengths"), ("Weaknesses", "weaknesses")):
        story.append(P(title, h2))
        story += [P(f"• {x}") for x in content[key]]
    story.append(P("Skill gaps (mapped to the PrepWise skills taxonomy)", h2))
    if content["skill_gaps"]:
        rows = [[P("Skill"), P("Severity"), P("Why")]] + [[P(g.get("name", g["skill_id"])), P(g.get("severity", "")),
                                                          P(g.get("reason", ""))] for g in content["skill_gaps"]]
        tbl = Table(rows, colWidths=[48 * mm, 22 * mm, 100 * mm], repeatRows=1)
        tbl.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.3, colors.grey), ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#dde7f5")),
                                 ("VALIGN", (0, 0), (-1, -1), "TOP")]))
        story.append(tbl)
    else:
        story.append(P("No significant skill gaps detected."))
    story.append(P("Question-by-question feedback", h2))
    for i, q in enumerate(content["per_question"], 1):
        story += [P(f"Q{i}. {q['question']}  -  {q['score']:.0f}/100", ParagraphStyle("q", parent=body, fontName=font, textColor=colors.HexColor("#1f3a5f"))),
                  P(q["feedback"])]
        if q.get("model_answer"):
            story.append(P("Ideal model answer: " + q["model_answer"], ParagraphStyle("m", parent=body, fontName=font, leftIndent=8,
                                                                                       textColor=colors.HexColor("#245c2a"))))
        story.append(Spacer(1, 4))
    story.append(P("Improvement suggestions", h2))
    story += [P(f"{i}. {s}") for i, s in enumerate(content["suggestions"], 1)]
    story.append(P("How the scores were computed", h2))
    for e in content.get("explanations", []):
        if e["type"] == "dsa" and e.get("breakdown"):
            c = e["breakdown"].get("components", {})
            story.append(P(f"DSA - {e['question']}: tests {c.get('tests')}/60, logic {c.get('logic')}/20, complexity "
                           f"{c.get('complexity')}/10, code quality {c.get('quality')}/10, hint penalty "
                           f"-{e['breakdown'].get('hint_penalty')}", small))
        elif e["type"] == "soft" and e.get("rubric"):
            story.append(P(f"Soft skills - {e['question'][:80]}: " + ", ".join(f"{k.replace('_', ' ')} {v}/10"
                                                                              for k, v in e["rubric"].items()), small))
    story += [Spacer(1, 8), P("Disclaimer: " + content["disclaimer"], small)]
    doc.build(story)
    return buf.getvalue()
