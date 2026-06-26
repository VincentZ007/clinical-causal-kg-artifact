#!/usr/bin/env python3
"""Redraw Fig. 1 as a clean vector PDF for the ICKG paper."""
from pathlib import Path
from reportlab.pdfgen import canvas
from reportlab.lib import colors


OUT = Path("ickg_paper/fig1_redraw.pdf")
PNG = Path("ickg_paper/fig1_redraw_preview.png")
W, H = 760, 300


def hex_color(s):
    return colors.HexColor(s)


C = {
    "ink": hex_color("#1f2937"),
    "muted": hex_color("#5b6777"),
    "line": hex_color("#c8d2df"),
    "panel": hex_color("#f7f9fc"),
    "blue": hex_color("#2563a9"),
    "blue_fill": hex_color("#e6f0fb"),
    "green": hex_color("#2f855a"),
    "green_fill": hex_color("#e3f3e9"),
    "red": hex_color("#c2410c"),
    "red_fill": hex_color("#fde6dc"),
    "amber": hex_color("#b7791f"),
    "amber_fill": hex_color("#fff1cc"),
    "gray_fill": hex_color("#f1f5f9"),
    "white": colors.white,
}


def round_rect(c, x, y, w, h, r=8, fill=None, stroke=None, lw=1):
    c.setLineWidth(lw)
    c.setStrokeColor(stroke or C["line"])
    c.setFillColor(fill or C["white"])
    c.roundRect(x, y, w, h, r, stroke=1, fill=1)


def text(c, x, y, s, size=8.5, color=None, bold=False, align="left", leading=None):
    c.setFillColor(color or C["ink"])
    c.setFont("Helvetica-Bold" if bold else "Helvetica", size)
    lines = str(s).split("\n")
    leading = leading or size * 1.18
    for i, line in enumerate(lines):
        yy = y - i * leading
        if align == "center":
            c.drawCentredString(x, yy, line)
        elif align == "right":
            c.drawRightString(x, yy, line)
        else:
            c.drawString(x, yy, line)


def label(c, x, y, s, fill, color, w=54):
    round_rect(c, x, y, w, 18, r=9, fill=fill, stroke=fill, lw=0.5)
    text(c, x + w / 2, y + 5.4, s, size=7.6, color=color, bold=True, align="center")


def arrow(c, x1, y1, x2, y2, color=None, lw=1.6):
    color = color or C["line"]
    c.setStrokeColor(color)
    c.setFillColor(color)
    c.setLineWidth(lw)
    c.line(x1, y1, x2, y2)
    if x2 >= x1:
        pts = [(x2, y2), (x2 - 7, y2 + 4), (x2 - 7, y2 - 4)]
    else:
        pts = [(x2, y2), (x2 + 7, y2 + 4), (x2 + 7, y2 - 4)]
    p = c.beginPath()
    p.moveTo(*pts[0])
    p.lineTo(*pts[1])
    p.lineTo(*pts[2])
    p.close()
    c.drawPath(p, stroke=0, fill=1)


def draw():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    c = canvas.Canvas(str(OUT), pagesize=(W, H))
    c.setTitle("Fig. 1 Overview redraw")
    c.setAuthor("Anonymous")

    # Background
    c.setFillColor(colors.white)
    c.rect(0, 0, W, H, fill=1, stroke=0)

    # Stage panels
    build = (18, 36, 166, 228)
    audit = (210, 22, 342, 252)
    apply = (579, 36, 163, 228)
    for x, y, w, h in (build, audit, apply):
        round_rect(c, x, y, w, h, r=12, fill=C["panel"], stroke=C["line"], lw=1.2)

    label(c, 32, 245, "BUILD", C["blue_fill"], C["blue"], w=58)
    label(c, 226, 250, "AUDIT", C["amber_fill"], C["amber"], w=58)
    label(c, 594, 245, "APPLY", C["green_fill"], C["green"], w=58)

    # Build lane
    round_rect(c, 38, 204, 126, 38, r=7, fill=C["blue_fill"], stroke=hex_color("#9bbce3"))
    text(c, 101, 225, "Discharge notes", 8.8, C["ink"], True, "center")
    text(c, 101, 213, "328,839 MIMIC-IV", 7.7, C["blue"], False, "center")
    arrow(c, 101, 198, 101, 181, hex_color("#9bbce3"))

    round_rect(c, 38, 144, 126, 40, r=7, fill=C["white"], stroke=C["line"])
    text(c, 101, 166, "Qwen extraction", 8.3, C["ink"], True, "center")
    text(c, 101, 154, "cause -> effect pairs", 7.5, C["muted"], False, "center")
    arrow(c, 101, 137, 101, 120, hex_color("#9bbce3"))

    round_rect(c, 38, 83, 126, 42, r=7, fill=C["white"], stroke=C["line"])
    text(c, 101, 106, "UMLS normalize", 8.3, C["ink"], True, "center")
    text(c, 101, 94, "merge surface forms", 7.3, C["muted"], False, "center")
    arrow(c, 101, 76, 101, 61, hex_color("#9bbce3"))

    round_rect(c, 38, 39, 126, 34, r=7, fill=hex_color("#f4f9ff"), stroke=hex_color("#9bbce3"))
    text(c, 101, 59, "Causal KG", 7.8, C["blue"], True, "center")
    text(c, 101, 48, "4.3K concepts / 19K edges", 7.0, C["blue"], True, "center")

    # Interstage arrows
    arrow(c, 184, 150, 210, 150, C["line"], lw=2.4)
    arrow(c, 552, 150, 579, 150, C["line"], lw=2.4)

    # Audit title and reference
    text(c, 381, 239, "Audit each extracted edge against an independent reference", 10.5, C["ink"], True, "center")

    rows = [
        (232, 180, C["green_fill"], C["green"], "Co-occurrence lift", "existence signal", "WORKS", "removes spurious co-mentions"),
        (232, 119, C["red_fill"], C["red"], "Coding-order time", "direction signal", "FAILS", "54.2% vs 73.6%; coding time != onset"),
        (232, 58, C["amber_fill"], C["amber"], "Literature corroboration", "direction + confidence", "PARTIAL", "74% agreement; ~30% edge coverage"),
    ]
    for x, y, fill, col, name, role, verdict, desc in rows:
        round_rect(c, x, y, 296, 46, r=9, fill=fill, stroke=fill)
        c.setFillColor(col)
        c.rect(x + 12, y + 8, 10, 30, fill=1, stroke=0)
        round_rect(c, x + 34, y + 12, 42, 22, r=11, fill=colors.white, stroke=col, lw=1)
        text(c, x + 55, y + 19, verdict, 7.2, col, True, "center")
        text(c, x + 90, y + 29, name, 8.6, C["ink"], True)
        text(c, x + 90, y + 18, role, 7.1, C["muted"])
        text(c, x + 90, y + 8, desc, 7.1, col)

    # Audit takeaway
    round_rect(c, 250, 27, 254, 20, r=10, fill=colors.white, stroke=C["line"], lw=0.8)
    text(c, 377, 33, "Existence is cheap to validate; direction remains uncertain", 7.9, C["ink"], True, "center")

    # Apply lane
    round_rect(c, 600, 204, 121, 42, r=8, fill=C["green_fill"], stroke=hex_color("#91caa7"))
    text(c, 660.5, 226, "Lift-validated", 8.8, C["green"], True, "center")
    text(c, 660.5, 214, "causal KG", 8.8, C["green"], True, "center")
    arrow(c, 660.5, 197, 660.5, 178, hex_color("#91caa7"))

    round_rect(c, 600, 138, 121, 42, r=8, fill=C["white"], stroke=C["line"])
    text(c, 660.5, 161, "Causal RAG", 8.8, C["ink"], True, "center")
    text(c, 660.5, 149, "retriever + Qwen", 7.3, C["muted"], False, "center")
    arrow(c, 660.5, 131, 660.5, 112, hex_color("#91caa7"))

    round_rect(c, 600, 72, 121, 42, r=8, fill=C["white"], stroke=C["line"])
    text(c, 660.5, 94, "Held-out patient", 8.5, C["ink"], True, "center")
    text(c, 660.5, 82, "causal answer", 7.4, C["muted"], False, "center")

    round_rect(c, 592, 43, 138, 20, r=10, fill=hex_color("#f0fff4"), stroke=hex_color("#91caa7"), lw=0.8)
    text(c, 661, 49, "+7.9 pts vs unvalidated graph", 7.4, C["green"], True, "center")

    # Bottom banner
    round_rect(c, 56, 8, 648, 18, r=9, fill=hex_color("#fbfcfe"), stroke=C["line"], lw=0.8)
    text(c, 380, 14, "Recommendation: validate existence with co-occurrence lift; do not orient edges by diagnosis-coding order.", 7.7, C["ink"], True, "center")

    c.showPage()
    c.save()


if __name__ == "__main__":
    draw()
    try:
        import pypdfium2 as pdfium
        pdf = pdfium.PdfDocument(str(OUT))
        bitmap = pdf[0].render(scale=2).to_pil()
        bitmap.save(PNG)
        print(PNG)
    except Exception as exc:
        print(f"preview skipped: {exc}")
    print(OUT)
