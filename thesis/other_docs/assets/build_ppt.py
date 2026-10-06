# -*- coding: utf-8 -*-
from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE
from pptx.oxml.ns import qn
import copy

ASSETS = r"C:\Users\Radu Savin\facultate\anomaly_detection\thesis\other_docs\assets"
OUT = r"C:\Users\Radu Savin\facultate\anomaly_detection\thesis\other_docs\prezentare_licenta.pptx"

# ---------- palette ----------
NAVY = RGBColor(0x1B, 0x2A, 0x4A)
BLUE = RGBColor(0x2E, 0x5C, 0x8A)
ACCENT = RGBColor(0xE0, 0x7A, 0x2C)
LIGHT = RGBColor(0xF4, 0xF6, 0xF9)
GRAY = RGBColor(0x5A, 0x63, 0x70)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
DARK_TEXT = RGBColor(0x22, 0x27, 0x2E)

prs = Presentation()
prs.slide_width = Inches(13.333)
prs.slide_height = Inches(7.5)
SW, SH = prs.slide_width, prs.slide_height
blank = prs.slide_layouts[6]


def add_slide():
    return prs.slides.add_slide(blank)


def set_bg(slide, color=WHITE):
    bg = slide.background
    bg.fill.solid()
    bg.fill.fore_color.rgb = color


def add_rect(slide, l, t, w, h, color, line=False):
    shp = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, l, t, w, h)
    shp.fill.solid()
    shp.fill.fore_color.rgb = color
    if not line:
        shp.line.fill.background()
    else:
        shp.line.color.rgb = color
    shp.shadow.inherit = False
    return shp


def add_text(slide, l, t, w, h, text, size=18, color=DARK_TEXT, bold=False,
             align=PP_ALIGN.LEFT, font="Calibri", anchor=MSO_ANCHOR.TOP, italic=False,
             line_spacing=1.0):
    tb = slide.shapes.add_textbox(l, t, w, h)
    tf = tb.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    lines = text.split("\n")
    for i, line in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = line
        p.alignment = align
        p.line_spacing = line_spacing
        for r in p.runs:
            r.font.size = Pt(size)
            r.font.color.rgb = color
            r.font.bold = bold
            r.font.italic = italic
            r.font.name = font
    return tb


def add_bullets(slide, l, t, w, h, items, size=16, color=DARK_TEXT, font="Calibri",
                 space_after=10, bullet_color=ACCENT, bold_lead=False, line_spacing=1.08):
    """items: list of (text, level) or plain strings (level 0), or (text, level, bold)"""
    tb = slide.shapes.add_textbox(l, t, w, h)
    tf = tb.text_frame
    tf.word_wrap = True
    first = True
    for item in items:
        if isinstance(item, tuple):
            if len(item) == 3:
                text, level, is_bold = item
            else:
                text, level = item
                is_bold = False
        else:
            text, level, is_bold = item, 0, False
        p = tf.paragraphs[0] if first else tf.add_paragraph()
        first = False
        p.level = level
        p.space_after = Pt(space_after)
        p.line_spacing = line_spacing
        marker = "\u25B8  " if level == 0 else "\u2013  "
        run_marker = p.add_run()
        run_marker.text = marker
        run_marker.font.size = Pt(size - (2 if level else 0))
        run_marker.font.color.rgb = bullet_color if level == 0 else GRAY
        run_marker.font.bold = True
        run_marker.font.name = font
        run = p.add_run()
        run.text = text
        run.font.size = Pt(size - (2 if level else 0))
        run.font.color.rgb = color
        run.font.bold = is_bold
        run.font.name = font
    return tb


def add_header(slide, kicker, title, num, total=20):
    add_rect(slide, 0, 0, SW, Inches(1.15), NAVY)
    add_rect(slide, 0, Inches(1.15), SW, Pt(3), ACCENT)
    add_text(slide, Inches(0.55), Inches(0.10), Inches(9.5), Inches(0.35), kicker,
              size=13, color=RGBColor(0xC9, 0xD6, 0xE8), bold=True, font="Calibri")
    add_text(slide, Inches(0.5), Inches(0.38), Inches(10.8), Inches(0.72), title,
              size=28, color=WHITE, bold=True, font="Calibri")
    add_text(slide, Inches(12.15), Inches(0.40), Inches(1.0), Inches(0.4), f"{num:02d}",
              size=20, color=ACCENT, bold=True, align=PP_ALIGN.RIGHT)


def add_footer(slide, note=""):
    add_text(slide, Inches(0.55), Inches(7.14), Inches(9), Inches(0.3), note,
              size=10.5, color=GRAY, italic=True)


def title_slide():
    s = add_slide()
    set_bg(s, NAVY)
    add_rect(s, 0, Inches(6.7), SW, Inches(0.06), ACCENT)
    add_text(s, Inches(1.0), Inches(2.35), Inches(11.3), Inches(1.9),
              "Analiza Paradigmelor de Anomaly Detection\npe Seturi de Date cu Structur\u0103 Temporal\u0103 Diferit\u0103",
              size=34, color=WHITE, bold=True, line_spacing=1.15)
    add_text(s, Inches(1.0), Inches(4.35), Inches(9), Inches(0.5),
              "Lucrare de licen\u021b\u0103 \u2014 Facultatea de Matematic\u0103 \u0219i Informatic\u0103, Universitatea din Bucure\u0219ti",
              size=16, color=RGBColor(0xC9, 0xD6, 0xE8))
    add_text(s, Inches(1.0), Inches(5.5), Inches(6), Inches(0.9),
              "Savin Radu-Andrei\nCoordonator \u0219tiin\u021bific: Paul Irofti",
              size=16, color=WHITE, line_spacing=1.3)
    return s


def section_slide(num, title, subtitle):
    s = add_slide()
    set_bg(s, NAVY)
    add_rect(s, 0, 0, Inches(0.18), SH, ACCENT)
    add_text(s, Inches(1.0), Inches(2.7), Inches(2.0), Inches(1.2), f"{num:02d}",
              size=72, color=ACCENT, bold=True)
    add_text(s, Inches(1.0), Inches(3.75), Inches(11), Inches(1.0), title,
              size=36, color=WHITE, bold=True)
    add_text(s, Inches(1.0), Inches(4.55), Inches(10.5), Inches(0.8), subtitle,
              size=16, color=RGBColor(0xC9, 0xD6, 0xE8))
    return s


def add_picture_framed(slide, path, l, t, w, h, caption=None, cap_size=11):
    pic = slide.shapes.add_picture(path, l, t, height=h)
    # if too wide, rescale by width instead
    if pic.width > w:
        slide.shapes._spTree.remove(pic._element)
        pic = slide.shapes.add_picture(path, l, t, width=w)
    # center within box
    pic.left = int(l + (w - pic.width) / 2)
    pic.top = int(t + (h - pic.height) / 2)
    if caption:
        add_text(slide, l, Emu(int(t + h + Pt(4))), w, Inches(0.4), caption,
                  size=cap_size, color=GRAY, italic=True, align=PP_ALIGN.CENTER)
    return pic


def stat_card(slide, l, t, w, h, value, label, color=BLUE):
    add_rect(slide, l, t, w, h, LIGHT)
    add_rect(slide, l, t, Pt(4), h, color)
    add_text(slide, Emu(int(l + Pt(14))), Emu(int(t + Pt(6))), Emu(int(w - Pt(20))), Emu(int(h*0.55)),
              value, size=22, color=color, bold=True)
    add_text(slide, Emu(int(l + Pt(14))), Emu(int(t + h*0.55)), Emu(int(w - Pt(20))), Emu(int(h*0.4)),
              label, size=11.5, color=DARK_TEXT)


def table_slide_generic(slide, l, t, w, headers, rows, col_widths_ratio,
                          font_size=13, header_size=13, row_h=Inches(0.42),
                          highlight_rows=None, bold_cols=None):
    highlight_rows = highlight_rows or {}
    n_cols = len(headers)
    n_rows = len(rows) + 1
    h = row_h * n_rows
    table_shape = slide.shapes.add_table(n_rows, n_cols, l, t, w, h)
    table = table_shape.table
    total_ratio = sum(col_widths_ratio)
    for i, ratio in enumerate(col_widths_ratio):
        table.columns[i].width = Emu(int(w * ratio / total_ratio))
    # header
    for c, htext in enumerate(headers):
        cell = table.cell(0, c)
        cell.text = htext
        cell.fill.solid()
        cell.fill.fore_color.rgb = NAVY
        cell.vertical_anchor = MSO_ANCHOR.MIDDLE
        for p in cell.text_frame.paragraphs:
            p.alignment = PP_ALIGN.CENTER if c > 0 else PP_ALIGN.LEFT
            for r in p.runs:
                r.font.size = Pt(header_size)
                r.font.bold = True
                r.font.color.rgb = WHITE
    for ridx, row in enumerate(rows, start=1):
        for c, val in enumerate(row):
            cell = table.cell(ridx, c)
            cell.text = str(val)
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            cell.fill.solid()
            cell.fill.fore_color.rgb = LIGHT if ridx % 2 == 0 else WHITE
            if ridx in highlight_rows:
                cell.fill.fore_color.rgb = RGBColor(0xFC, 0xE7, 0xD2)
            for p in cell.text_frame.paragraphs:
                p.alignment = PP_ALIGN.CENTER if c > 0 else PP_ALIGN.LEFT
                for r in p.runs:
                    r.font.size = Pt(font_size)
                    r.font.color.rgb = DARK_TEXT
                    if ridx in highlight_rows or (bold_cols and c in bold_cols):
                        r.font.bold = True
    # remove default table style banding via style id (keep simple)
    tbl = table_shape.table
    tbl_pr = tbl._tbl.find(qn('a:tblPr'))
    if tbl_pr is not None:
        tbl_pr.set('firstRow', '0')
        tbl_pr.set('bandRow', '0')
    return table_shape


# =========================================================
# SLIDE 1 - Title
# =========================================================
title_slide()

# =========================================================
# SLIDE 2 - Introduction: what the paper does
# =========================================================
s = add_slide(); set_bg(s)
add_header(s, "INTRODUCERE", "Ce \u00eentreab\u0103 aceast\u0103 lucrare?", 2)
add_text(s, Inches(0.55), Inches(1.45), Inches(12.2), Inches(0.75),
          "\u00cen ce m\u0103sur\u0103 modul \u00een care un model de detec\u021bie a anomaliilor reprezint\u0103\nstructura latent\u0103 a datelor \u201enormale\u201d determin\u0103 tipurile de anomalii pe care le poate detecta?",
          size=20, color=BLUE, bold=True, line_spacing=1.2)
add_bullets(s, Inches(0.55), Inches(2.5), Inches(12.2), Inches(4.3), [
    ("Trei componente principale implementate", 0, True),
    ("Trei pipeline-uri standardizate de preprocesare / antrenare / evaluare", 1),
    ("Analiz\u0103 sistematic\u0103 a rezultatelor, pe fiecare set de date", 1),
    ("Studii de caz de explicabilitate, ce testeaz\u0103 direct ipotezele formulate", 1),
], size=18, space_after=16)
add_footer(s, "Capitolul 1 \u2014 Introducere \u0219i \u00a72.1 Proiectarea \u0219i Abordarea Cercet\u0103rii")

# =========================================================
# SLIDE 3 - the two axes diagram
# =========================================================
s = add_slide(); set_bg(s)
add_header(s, "INTRODUCERE", "Design experimental: dou\u0103 axe simultane", 3)
add_text(s, Inches(0.55), Inches(1.4), Inches(12.2), Inches(0.5),
          "Fiecare paradigm\u0103 este rulat\u0103 pe fiecare punct al spectrului de structur\u0103 temporal\u0103 \u2192 rezultatul e atribuit paradigmei, nu setului de date.",
          size=15, color=GRAY, italic=True)

# Axis 1 - paradigms
p_l = Inches(0.55); p_t = Inches(2.1); p_w = Inches(5.9); p_h = Inches(4.7)
add_rect(s, p_l, p_t, p_w, Inches(0.5), NAVY)
add_text(s, p_l, p_t, p_w, Inches(0.5), "AXA 1 \u2014 Paradigma modelului", size=16, color=WHITE, bold=True, align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
paradigms = [
    ("Grani\u021b\u0103", "One-Class SVM", "granit\u0103 \u00eentr-un spa\u021biu indus de kernel"),
    ("Parti\u021bionare", "Isolation Forest", "\u00eempar\u021bire aleatoare recursiv\u0103 a spa\u021biului"),
    ("Bottleneck de compresie", "AE (MLP, LSTM, Transformer)", "re\u021bine doar structura dominant\u0103 benign\u0103"),
    ("Obiectiv predictiv", "LSTM Predictiv, Causal Transformer", "\u00eenva\u021b\u0103 s\u0103 prezic\u0103 pasul urm\u0103tor"),
]
y = p_t + Inches(0.65)
row_h = Inches(0.97)
for i, (name, models, desc) in enumerate(paradigms):
    add_rect(s, p_l, y, p_w, row_h - Pt(6), LIGHT if i % 2 == 0 else WHITE)
    add_text(s, Emu(int(p_l+Pt(10))), Emu(int(y+Pt(4))), Emu(int(p_w-Pt(20))), Inches(0.3), name, size=14.5, color=ACCENT, bold=True)
    add_text(s, Emu(int(p_l+Pt(10))), Emu(int(y+Pt(28))), Emu(int(p_w-Pt(20))), Inches(0.3), models, size=12.5, color=NAVY, bold=True)
    add_text(s, Emu(int(p_l+Pt(10))), Emu(int(y+Pt(50))), Emu(int(p_w-Pt(20))), Inches(0.3), desc, size=11, color=GRAY)
    y += row_h

# Axis 2 - datasets
d_l = Inches(6.85); d_t = p_t; d_w = Inches(5.9); d_h = p_h
add_rect(s, d_l, d_t, d_w, Inches(0.5), NAVY)
add_text(s, d_l, d_t, d_w, Inches(0.5), "AXA 2 \u2014 Structura temporal\u0103 a datelor", size=16, color=WHITE, bold=True, align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
datasets = [
    ("Credit Card Fraud", "tabelar, f\u0103r\u0103 ordine temporal\u0103 real\u0103 \u2014 caz de control"),
    ("Yahoo Webscope S5", "serii de timp cu adev\u0103rat secven\u021biale"),
    ("CIC-IDS2017", "flux de re\u021bea \u2014 admite vedere plat\u0103 \u0219i temporal\u0103"),
]
y = d_t + Inches(0.65)
row_h2 = Inches(1.28)
for i, (name, desc) in enumerate(datasets):
    add_rect(s, d_l, y, d_w, row_h2 - Pt(8), LIGHT if i % 2 == 0 else WHITE)
    add_text(s, Emu(int(d_l+Pt(10))), Emu(int(y+Pt(8))), Emu(int(d_w-Pt(20))), Inches(0.35), name, size=15.5, color=BLUE, bold=True)
    add_text(s, Emu(int(d_l+Pt(10))), Emu(int(y+Pt(40))), Emu(int(d_w-Pt(20))), Inches(0.6), desc, size=12.5, color=DARK_TEXT)
    y += row_h2
add_text(s, Emu(int(d_l+Pt(10))), y, Emu(int(d_w-Pt(20))), Inches(0.85),
          "Spectru: f\u0103r\u0103 ordine \u2192 secven\u021bial \u2192 hibrid (plat + temporal)", size=12, color=GRAY, italic=True)
add_footer(s, "\u00a72.1 Proiectarea \u0219i Abordarea Cercet\u0103rii")

# =========================================================
# SLIDE 4 - Models overview
# =========================================================
s = add_slide(); set_bg(s)
add_header(s, "PRELIMINARII", "Modelele: 7 algoritmi, 4 paradigme structurale", 4)
models_tbl = [
    ("One-Class SVM", "Grani\u021b\u0103", "\u00eenva\u021b\u0103 o frontier\u0103 \u00een jurul datelor normale \u00eentr-un spa\u021biu indus de kernel RBF"),
    ("Isolation Forest", "Parti\u021bionare", "izoleaz\u0103 punctele prin diviziuni aleatoare; anomaliile cer mai pu\u021bine t\u0103ieturi"),
    ("Autoencoder MLP", "Bottleneck / reconstruc\u021bie", "comprim\u0103 vectorul printr-un bottleneck \u0219i \u00eencearc\u0103 s\u0103-l reconstruiasc\u0103"),
    ("Autoencoder LSTM", "Bottleneck / reconstruc\u021bie (secven\u021bial)", "comprim\u0103 o secven\u021b\u0103 \u00eentr-o singur\u0103 stare ascuns\u0103 sumar\u0103"),
    ("LSTM Predictiv", "Obiectiv predictiv", "\u00eenva\u021b\u0103 s\u0103 prezic\u0103 pasul t+1 din pa\u0219ii anteriori"),
    ("Bottleneck Transformer AE", "Bottleneck / reconstruc\u021bie (secven\u021bial)", "auto-aten\u021bie non-cauzal\u0103 + mediere global\u0103 printr-un bottleneck liniar"),
    ("Causal Transformer", "Obiectiv predictiv", "echivalentul predictiv, bazat pe aten\u021bie cauzal\u0103, al LSTM Predictiv"),
]
table_slide_generic(s, Inches(0.55), Inches(1.45), Inches(12.25),
                     ["Model", "Paradigm\u0103", "Idee de baz\u0103"], models_tbl,
                     [2.2, 2.6, 5.5], font_size=12.5, row_h=Inches(0.565))
add_text(s, Inches(0.55), Inches(6.42), Inches(12.2), Inches(0.5),
          "Toate scorurile sunt convertite la conven\u021bia \u201escor mai mare = mai anormal\u201d, indiferent de paradigm\u0103.",
          size=12.5, color=GRAY, italic=True)
add_footer(s, "\u00a72.2 Algoritmi de Detec\u021bie a Anomaliilor")

# =========================================================
# SLIDE 5 - Datasets overview
# =========================================================
s = add_slide(); set_bg(s)
add_header(s, "PRELIMINARII", "Seturile de date: un spectru de structur\u0103 temporal\u0103", 5)
cw = Inches(3.95); gap = Inches(0.2); l0 = Inches(0.55); t0 = Inches(1.6); ch = Inches(3.6)
cards = [
    ("A \u2014 Credit Card Fraud", "284.807 tranzac\u021bii, 0.172% frauduloase",
     ["V1\u2013V28 (PCA) + Amount", "F\u0103r\u0103 ordine temporal\u0103 real\u0103 \u2192 caz de control"]),
    ("B \u2014 Yahoo Webscope S5", "4 familii de benchmark (A1\u2013A4), serii univariate",
     ["A1: valori punctuale (real, zgomotos)", "A3: contextual / sezonier", "A4: schimbare de nivel"]),
    ("C \u2014 CIC-IDS2017", "~2.8M fluxuri de re\u021bea, 14 tipuri de atac",
     ["~80 caracteristici statistice (CICFlowMeter)", "Admite vedere plat\u0103 \u015Fi temporal\u0103 (caz de tranzi\u021bie)"]),
]
for i, (name, sub, pts) in enumerate(cards):
    l = l0 + i * (cw + gap)
    add_rect(s, l, t0, cw, ch, LIGHT)
    add_rect(s, l, t0, cw, Inches(0.08), ACCENT)
    add_text(s, Emu(int(l+Pt(14))), Emu(int(t0+Pt(16))), Emu(int(cw-Pt(28))), Inches(0.6), name, size=16.5, color=NAVY, bold=True)
    add_text(s, Emu(int(l+Pt(14))), Emu(int(t0+Pt(58))), Emu(int(cw-Pt(28))), Inches(0.6), sub, size=12.5, color=BLUE, bold=True)
    add_bullets(s, Emu(int(l+Pt(14))), Emu(int(t0+Pt(105))), Emu(int(cw-Pt(28))), Inches(2.2), pts, size=12.5, space_after=8)
add_rect(s, Inches(0.55), Inches(5.5), Inches(12.25), Inches(1.15), NAVY)
add_text(s, Inches(0.85), Inches(5.65), Inches(11.7), Inches(0.9),
          "Metrica principal\u0103 raportat\u0103: PR-AUC (Precizia Medie) \u2014 independent\u0103 de prag, spre deosebire de F1, "
          "care depinde de un prag ales manual \u0219i poate fi \u00een\u0219el\u0103tor la clase dezechilibrate.",
          size=15.5, color=WHITE, bold=True, anchor=MSO_ANCHOR.MIDDLE, line_spacing=1.2)
add_footer(s, "\u00a72.3 Seturile de Date \u0219i \u00a73.2.1 Metrici Raportate")

# =========================================================
# SLIDE 6 - Ipoteze \u0219i Comportament A\u0219teptat
# =========================================================
s = add_slide(); set_bg(s)
add_header(s, "PRELIMINARII", "Ipoteze \u0219i comportament a\u0219teptat", 6)
add_text(s, Inches(0.55), Inches(1.5), Inches(12.2), Inches(0.6),
          "Pe baza celor dou\u0103 axe (paradigm\u0103 \u00d7 set de date), sunt anticipate dou\u0103 tipare \u00een rezultate.",
          size=16, color=GRAY, italic=True)

hyp_l = Inches(0.55); hyp_w = Inches(12.25); hyp_h = Inches(2.15)
add_rect(s, hyp_l, Inches(2.3), hyp_w, hyp_h, LIGHT)
add_rect(s, hyp_l, Inches(2.3), Pt(6), hyp_h, ACCENT)
add_text(s, Emu(int(hyp_l+Pt(24))), Inches(2.48), Emu(int(hyp_w-Pt(48))), Inches(0.4),
          "Ipoteza 1 \u2014 structura temporal\u0103 avantajeaz\u0103 datele secven\u021biale", size=16.5, color=NAVY, bold=True)
add_text(s, Emu(int(hyp_l+Pt(24))), Inches(2.95), Emu(int(hyp_w-Pt(48))), Inches(1.4),
          "Paradigmele care ofer\u0103 modelului o structur\u0103 temporal\u0103 inerent\u0103 (prin construc\u021bie, nu doar prin ferestre de intrare) ar trebui "
          "s\u0103 performeze mai bine pe seturile de date cu adev\u0103rat secven\u021biale (Yahoo S5 \u0219i, \u00eentr-o m\u0103sur\u0103 mai mic\u0103, varianta temporal\u0103 a CIC-IDS2017), "
          "comparativ cu paradigmele care trateaz\u0103 fiecare e\u0219antion ca independent.",
          size=14.5, color=DARK_TEXT, line_spacing=1.25)

add_rect(s, hyp_l, Inches(4.75), hyp_w, hyp_h, LIGHT)
add_rect(s, hyp_l, Inches(4.75), Pt(6), hyp_h, ACCENT)
add_text(s, Emu(int(hyp_l+Pt(24))), Inches(4.93), Emu(int(hyp_w-Pt(48))), Inches(0.4),
          "Ipoteza 2 \u2014 diferite paradigme detecteaz\u0103 diferite tipuri de anomalii", size=16.5, color=NAVY, bold=True)
add_text(s, Emu(int(hyp_l+Pt(24))), Inches(5.4), Emu(int(hyp_w-Pt(48))), Inches(1.4),
          "Dac\u0103 presupunerea e corect\u0103, compunerea mai multor paradigme complementare \u00eentr-un ansamblu ar trebui s\u0103 duc\u0103 la o \u00eembun\u0103t\u0103\u021bire "
          "a performan\u021bei fa\u021b\u0103 de oricare model individual, at\u00e2ta timp c\u00e2t modelele componente exploateaz\u0103 rela\u021bii diferite din date.",
          size=14.5, color=DARK_TEXT, line_spacing=1.25)
add_footer(s, "\u00a72.1.1 Ipoteze \u0219i Comportament A\u0219teptat")

# =========================================================
# SLIDE 7 - Section divider: Results
# =========================================================
section_slide(3, "Rezultate pe cele trei seturi de date",
               "Tabele din capitolul de rezultate \u0219i principalele concluzii pentru fiecare set")

# =========================================================
# SLIDE 8 - Credit Card results table
# =========================================================
s = add_slide(); set_bg(s)
add_header(s, "REZULTATE \u2014 SETUL A", "Detectarea Fraudei cu Cardul de Credit", 8)
cc_rows = [
    ["MLP Autoencoder", "Reconstruc\u021bie", "0.9649", "0.7105", "0.6992"],
    ["Isolation Forest", "Parti\u021bionare", "0.9587", "0.6725", "0.6070"],
    ["LSTM Autoencoder", "Reconstruc\u021bie (pseudo-secv.)", "0.9476", "0.6223", "0.7362"],
    ["OC-SVM", "Frontier\u0103", "0.9354", "0.5959", "0.6454"],
    ["Transformer Bottleneck AE", "Reconstruc\u021bie (pseudo-secv.)", "0.9583", "0.5947", "0.6137"],
    ["Predictive LSTM", "Predictiv (pseudo-secv.)", "0.9581", "0.5190", "0.5244"],
]
table_slide_generic(s, Inches(0.55), Inches(1.45), Inches(12.25),
                     ["Model", "Paradigm\u0103", "ROC-AUC", "Precizie Medie", "F1 (fraud\u0103)"], cc_rows,
                     [2.3, 3.0, 1.5, 1.7, 1.5], font_size=13.5, row_h=Inches(0.55), highlight_rows={1})
add_bullets(s, Inches(0.55), Inches(5.35), Inches(12.2), Inches(1.7), [
    ("Nu exist\u0103 secven\u021b\u0103 autentic\u0103 \u00een acest set \u2014 rândurile sunt independente, Time e eliminat", 0, True),
    ("Reconstruc\u021bia domin\u0103, dar câ\u0219tig\u0103 varianta care trateaz\u0103 datele ca vector plat (MLP AE) \u2014 impunerea unei structuri secven\u021biale acolo unde nu exist\u0103 e un handicap", 0, True),
    ("Predictive LSTM e cel mai slab: obiectivul de predic\u021bie presupune o rela\u021bie cauzal\u0103 \u00eentre pa\u0219i, dar componentele PCA sunt decorelate prin construc\u021bie", 0, True),
], size=15, space_after=10)
add_footer(s, "\u00a74.1 Detectarea Fraudei cu Cardul de Credit")

# =========================================================
# SLIDE 9 - Yahoo results table
# =========================================================
s = add_slide(); set_bg(s)
add_header(s, "REZULTATE \u2014 SETUL B", "Yahoo Webscope S5", 9)
yahoo_rows = [
    ["A1 \u2014 Punctuale", "MLP Autoencoder", "Reconstruc\u021bie", "0.792", "0.611"],
    ["A3 \u2014 Contextuale/sezoniere", "Predictive LSTM", "Predictiv", "0.868", "0.812"],
    ["A4 \u2014 Schimbare-de-nivel", "Predictive LSTM", "Predictiv", "0.745", "0.505"],
]
table_slide_generic(s, Inches(0.55), Inches(1.5), Inches(12.25),
                     ["Benchmark", "Cel mai bun model", "Paradigm\u0103", "ROC-AUC", "Precizie Medie"], yahoo_rows,
                     [2.7, 2.6, 2.0, 1.5, 1.9], font_size=14.5, row_h=Inches(0.65), highlight_rows={1,2,3})
add_bullets(s, Inches(0.55), Inches(4.15), Inches(12.2), Inches(2.8), [
    ("A1 (punctuale): toate modelele \u00eentr-o band\u0103 \u00eengust\u0103 (AP 0.51\u20130.61) \u2014 nu e nevoie de \u00een\u021belegere temporal\u0103 pentru un v\u00e2rf izolat", 0),
    ("A3 (contextuale): paradigma predictiv\u0103 domin\u0103 clar \u2014 Predictive LSTM exploateaz\u0103 direct predictibilitatea seriei sezoniere; modelele clasice (IForest, OC-SVM) se pr\u0103bu\u0219esc aproape la random", 0, True),
    ("A4 (schimbare de nivel): fiecare model se degradeaz\u0103, dar Predictive LSTM \u00eenc\u0103 conduce", 0),
], size=15, space_after=10)
add_footer(s, "\u00a74.2 Yahoo Webscope S5 (Setul de Date B)")

# =========================================================
# SLIDE 10 - A3: Transformer AE vs LSTM AE (sets up explainability tie-in)
# =========================================================
s = add_slide(); set_bg(s)
add_header(s, "REZULTATE \u2014 SETUL B", "A3: Transformer domin\u0103 arhitecturile de reconstruc\u021bie", 10)

a3_rows = [
    ["Transformer Bottleneck AE", "Reconstruc\u021bie", "0.762", "0.663"],
    ["LSTM Autoencoder", "Reconstruc\u021bie (pseudo-secv.)", "0.566", "0.392"],
]
table_slide_generic(s, Inches(0.55), Inches(1.8), Inches(12.25),
                     ["Model", "Paradigm\u0103", "ROC-AUC", "Precizie Medie"], a3_rows,
                     [3.5, 3.0, 1.9, 1.9], font_size=16, row_h=Inches(0.75), highlight_rows={1})

add_rect(s, Inches(0.55), Inches(4.0), Inches(12.25), Inches(2.2), LIGHT)
add_rect(s, Inches(0.55), Inches(4.0), Pt(6), Inches(2.2), ACCENT)
add_text(s, Inches(0.85), Inches(4.25), Inches(11.7), Inches(1.7),
          "Ambele sunt autoencodere cu bottleneck, dar difer\u0103 fundamental \u00een cum comprim\u0103 fereastra:\n"
          "auto-aten\u021bia global\u0103 a Transformer-ului asupra \u00eentregii ferestre recupereaz\u0103 probabil periodicitatea "
          "sezonier\u0103 pe care blocajul recurent al LSTM AE o distruge \u2014 ceea ce probabil explic\u0103 diferen\u021ba.",
          size=17, color=DARK_TEXT, line_spacing=1.35)
add_footer(s, "\u00a74.2.2 A3 \u2014 Anomalii Contextuale/Sezoniere")

# =========================================================
# SLIDE 11 - CIC-IDS2017 results table
# =========================================================
s = add_slide(); set_bg(s)
add_header(s, "REZULTATE \u2014 SETUL C", "CIC-IDS2017 \u2014 flux de re\u021bea", 11)
cic_rows = [
    ["MLP Autoencoder", "Context", "Reconstruc\u021bie", "0.9860", "0.9487", "0.9020"],
    ["OC-SVM", "Context", "Frontier\u0103", "0.9745", "0.9396", "0.9229"],
    ["Isolation Forest", "Context", "Parti\u021bionare", "0.9785", "0.9328", "0.8367"],
    ["LSTM Autoencoder", "Temporal", "Reconstruc\u021bie", "0.9976", "0.7955", "0.6249"],
    ["MLP Autoencoder", "Flat", "Reconstruc\u021bie", "0.9189", "0.7921", "0.8022"],
    ["Isolation Forest", "Flat", "Parti\u021bionare", "0.9049", "0.7612", "0.6639"],
]
table_slide_generic(s, Inches(0.55), Inches(1.45), Inches(12.25),
                     ["Model", "Set caracteristici", "Paradigm\u0103", "ROC-AUC", "Precizie Medie", "F1 (atac)"], cic_rows,
                     [2.1, 1.6, 2.2, 1.4, 1.8, 1.3], font_size=12.5, row_h=Inches(0.5), highlight_rows={1,2,3})
add_bullets(s, Inches(0.55), Inches(5.05), Inches(12.2), Inches(2.0), [
    ("PortScan, DoS Hulk \u0219i DDoS reprezint\u0103 \u00eempreun\u0103 95.1% din fluxurile de atac \u2014 AP-ul agregat e practic un proxy pentru performan\u021ba pe aceste trei clase", 0, True),
    ("Caracteristicile de context al gazdei adaug\u0103 cel mai mult: +0.157 la +0.255 AP fa\u021b\u0103 de vederea plat\u0103, dep\u0103\u0219ind orice model f\u0103r\u0103 context", 0, True),
    ("Concluzie: alegerea caracteristicilor conteaz\u0103 la fel de mult ca alegerea modelului", 0, True),
], size=15, space_after=9)
add_footer(s, "\u00a74.3 CIC-IDS2017 (Setul de Date C)")

# =========================================================
# SLIDE 12 - CIC-IDS2017: a practical ensemble (sets up explainability tie-in)
# =========================================================
s = add_slide(); set_bg(s)
add_header(s, "REZULTATE \u2014 SETUL C", "Un ansamblu practic: acoperire per clas\u0103", 12)

ens_setup_rows = [
    ["AE+Context singur", "0.882", "0.924", "0.292", "0.013", "0.025"],
    ["Temporal IForest singur", "0.956", "0.123", "0.611", "0.578", "0.560"],
]
table_slide_generic(s, Inches(0.55), Inches(1.8), Inches(12.25),
                     ["Model", "DDoS", "PortScan", "GoldenEye", "Slowhttptest", "slowloris"], ens_setup_rows,
                     [2.6, 1.4, 1.5, 1.6, 1.9, 1.6], font_size=15, row_h=Inches(0.7))

add_rect(s, Inches(0.55), Inches(4.0), Inches(12.25), Inches(2.2), LIGHT)
add_rect(s, Inches(0.55), Inches(4.0), Pt(6), Inches(2.2), ACCENT)
add_text(s, Inches(0.85), Inches(4.25), Inches(11.7), Inches(1.7),
          "Niciun model sau set de caracteristici, luat singur, nu acoper\u0103 fiecare clas\u0103 de atac din acest set de date. "
          "Combinarea AE+Context (puternic pe DDoS \u0219i PortScan, clasele care domin\u0103 setul de test ca volum) cu "
          "Temporal IForest (puternic pe clasele DoS lente \u0219i, cu titlu provizoriu, pe Heartbleed) ar acoperi cea mai "
          "larg\u0103 gam\u0103 de clase de atac detectabile \u00een cadrul acestui set de caracteristici.",
          size=16, color=DARK_TEXT, line_spacing=1.3)
add_footer(s, "\u00a74.3.3 Discu\u021bie \u2014 CIC-IDS2017 (Setul de Date C)")

# =========================================================
# SLIDE 13 - Section divider: Explainability
# =========================================================
section_slide(4, "Studii de caz de explicabilitate",
               "Testarea direct\u0103 a ipotezelor din discu\u021bia rezultatelor, prin diagnostice \u021bintite")

# =========================================================
# SLIDE 14 - Yahoo reconstruction error localization (intro + hypothesis)
# =========================================================
s = add_slide(); set_bg(s)
add_header(s, "EXPLICABILITATE 4.1", "Localizarea erorii de reconstruc\u021bie pe Yahoo A3", 14)
add_bullets(s, Inches(0.55), Inches(1.55), Inches(6.1), Inches(3.5), [
    ("Ipotez\u0103: LSTM AE comprim\u0103 \u00eentreaga fereastr\u0103 \u00eentr-o singur\u0103 stare final\u0103 \u2192 eroare difuz\u0103, uniform\u0103", 0),
    ("Transformer AE p\u0103streaz\u0103 informa\u021bie pozi\u021bional\u0103 prin auto-aten\u021bie \u2192 eroare concentrat\u0103, localizat\u0103 exact unde faza sezonier\u0103 e \u00eenc\u0103lcat\u0103", 0),
    ("M\u0103sur\u0103: coeficientul Gini al celor 64 erori p\u0103tratice per fereastr\u0103 (0 = uniform, 1 = concentrat total)", 0, True),
], size=15.5, space_after=14)
add_rect(s, Inches(0.55), Inches(4.6), Inches(6.1), Inches(1.6), LIGHT)
add_text(s, Inches(0.8), Inches(4.75), Inches(5.6), Inches(1.3),
          "Gini mediu:\nLSTM AE = 0.566   vs.   Transformer AE = 0.721",
          size=16.5, color=NAVY, bold=True, line_spacing=1.3)
add_picture_framed(s, f"{ASSETS}\\gini_hist.png", Inches(6.95), Inches(1.55), Inches(5.9), Inches(4.6),
                    caption="Fig. 5.1 \u2014 distribu\u021bia Gini pe cele 14.449 ferestre A3 anormale")
add_footer(s, "\u00a75.1 Localizarea Erorii de Reconstruc\u021bie pe Yahoo A3")

# =========================================================
# SLIDE 15 - Yahoo heatmap visual test
# =========================================================
s = add_slide(); set_bg(s)
add_header(s, "EXPLICABILITATE 4.1", "Verificare vizual\u0103: h\u0103r\u021bi termice ale erorii", 15)
add_picture_framed(s, f"{ASSETS}\\yahoo_heatmap.png", Inches(0.55), Inches(1.35), Inches(3.75), Inches(5.35),
                    caption="Fig. 7.1 \u2014 300 ferestre A3 anormale", cap_size=10.5)
add_picture_framed(s, f"{ASSETS}\\yahoo_heatmap_zoom.png", Inches(4.6), Inches(1.35), Inches(8.2), Inches(3.15),
                    caption="Detaliu \u2014 treimea a treia a benzii (r\u00e2ndurile ~150\u2013225 din 300)", cap_size=10.5)
add_bullets(s, Inches(4.6), Inches(4.85), Inches(8.2), Inches(1.9), [
    ("Fiecare r\u00e2nd = o fereastr\u0103; fiecare coloan\u0103 = un pas temporal; culoarea = eroarea p\u0103tratic\u0103", 0),
    ("LSTM AE (st\u00e2nga): eroare ridicat\u0103 mai des r\u0103sp\u00e2ndit\u0103 sub\u021bire pe toat\u0103 l\u0103\u021bimea", 0),
    ("Transformer AE (dreapta): eroare vizibil mai rar\u0103 \u0219i limitat\u0103 la pa\u0219i izola\u021bi", 0),
    ("Confirm\u0103 ipoteza calitativ, dar coeficientul Gini o cuantific\u0103 riguros", 0, True),
], size=14, space_after=10)
add_footer(s, "Anexa 7 \u2014 completeaz\u0103 verificarea calitativ\u0103 din \u00a75.1")

# =========================================================
# SLIDE 16 - SHAP intro
# =========================================================
s = add_slide(); set_bg(s)
add_header(s, "EXPLICABILITATE 4.2", "SHAP: o scurt\u0103 introducere", 16)
add_text(s, Inches(0.55), Inches(1.55), Inches(12.2), Inches(0.5),
          "SHapley Additive exPlanations \u2014 provine din teoria jocurilor cooperative (valorile Shapley)",
          size=17, color=BLUE, bold=True)
add_rect(s, Inches(0.55), Inches(2.3), Inches(12.25), Inches(1.15), NAVY)
add_text(s, Inches(0.55), Inches(2.3), Inches(12.25), Inches(1.15),
          "\u03D5\u1D62(f) = contribu\u021bia caracteristicii i la ie\u0219irea modelului f, mediat\u0103 peste toate coali\u021biile posibile de caracteristici",
          size=18, color=WHITE, bold=True, align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
add_bullets(s, Inches(0.55), Inches(3.75), Inches(12.2), Inches(2.9), [
    ("Fiec\u0103rui e\u0219antion i se atribuie c\u00e2te o valoare per caracteristic\u0103 \u2014 c\u00e2t a influen\u021bat acea caracteristic\u0103 ie\u0219irea", 0),
    ("Semnul arat\u0103 direc\u021bia (spre normal sau anormal); magnitudinea arat\u0103 contribu\u021bia relativ\u0103", 0),
    ("Diagram\u0103 beeswarm: un r\u00e2nd per caracteristic\u0103, un punct per e\u0219antion, pozi\u021bionat dup\u0103 valoarea SHAP \u0219i colorat dup\u0103 valoarea caracteristicii (albastru = mic, ro\u0219u = mare)", 0, True),
], size=16, space_after=14)
add_footer(s, "\u00a75.2 Regimuri de Caracteristici Disjuncte \u2014 introducere SHAP")

# =========================================================
# SLIDE 17 - SHAP Temporal IForest
# =========================================================
s = add_slide(); set_bg(s)
add_header(s, "EXPLICABILITATE 4.2", "SHAP \u2014 Temporal IForest: aten\u021bie difuzat\u0103 pe fereastr\u0103", 17)
add_picture_framed(s, f"{ASSETS}\\shap_temporal_iforest.png", Inches(0.55), Inches(1.5), Inches(12.25), Inches(3.6),
                    caption="Fig. 5.2 \u2014 valoarea medie SHAP per pas temporal, ferestre DDoS (st\u00e2nga) \u0219i DoS slowloris (dreapta)")
add_bullets(s, Inches(0.55), Inches(5.4), Inches(12.2), Inches(1.7), [
    ("Atribuirea e negativ\u0103 (anormal) \u0219i recurent\u0103 pe majoritatea celor 16 pa\u0219i temporali \u2014 nu se concentreaz\u0103 doar pe fluxul final (linia ro\u0219ie)", 0, True),
    ("Caracteristici principale confirm\u0103 mecanismul cunoscut al atacului: lungimea pachetelor pentru DDoS, timpul \u00eentre sosiri pentru atacurile lente", 0),
], size=15, space_after=10)
add_footer(s, "\u00a75.2.1 \u2014 SHAP calculat cu TreeExplainer pe fereastra apla\u021bizat\u0103")

# =========================================================
# SLIDE 18 - SHAP AE+Context
# =========================================================
s = add_slide(); set_bg(s)
add_header(s, "EXPLICABILITATE 4.2", "SHAP \u2014 AE+Context: dominat de context de gazd\u0103", 18)
add_picture_framed(s, f"{ASSETS}\\shap_ae_context.png", Inches(0.55), Inches(1.35), Inches(5.4), Inches(5.6),
                    caption="Fig. 5.3 \u2014 diagram\u0103 beeswarm SHAP, e\u0219antioane PortScan")
add_bullets(s, Inches(6.3), Inches(1.6), Inches(6.5), Inches(3.3), [
    ("ctx_min_port_cnt domin\u0103 printr-un cluster str\u00e2ns, uniform pozitiv \u2014 dep\u0103\u0219e\u0219te orice alt\u0103 caracteristic\u0103", 0, True),
    ("8/15 (PortScan) \u0219i 7/15 (DDoS) dintre caracteristicile de top sunt caracteristici de context al gazdei", 0),
    ("Cum se cite\u0219te: fiecare punct = un e\u0219antion; pozi\u021bie orizontal\u0103 = impact SHAP; culoare = valoarea caracteristicii (ro\u0219u = mare, albastru = mic)", 0),
], size=15, space_after=12)
add_rect(s, Inches(6.3), Inches(5.1), Inches(6.5), Inches(1.5), LIGHT)
add_text(s, Inches(6.55), Inches(5.25), Inches(6.0), Inches(1.2),
          "Confirm\u0103 c\u0103 avantajul lui AE+Context pe clasele sale puternice (DDoS, PortScan) e determinat aproape \u00een \u00eentregime de dovezi de context al gazdei \u2014 indisponibile lui Temporal IForest.",
          size=13.5, color=DARK_TEXT, line_spacing=1.2)
add_footer(s, "\u00a75.2.2 \u2014 SHAP calculat cu GradientExplainer pe eroarea de reconstruc\u021bie")

# =========================================================
# SLIDE 19 - Ensemble score construction + results
# =========================================================
s = add_slide(); set_bg(s)
add_header(s, "EXPLICABILITATE 4.2", "Ansamblul: cum se combin\u0103 scorurile \u0219i rezultat", 19)
add_text(s, Inches(0.55), Inches(1.5), Inches(12.2), Inches(0.5),
          "AE+Context (puternic pe DDoS, PortScan) + Temporal IForest (puternic pe DoS lente) \u2014 evidente demonstrabil disjuncte",
          size=15, color=BLUE, bold=True)
add_bullets(s, Inches(0.55), Inches(2.15), Inches(5.9), Inches(3.2), [
    ("Scorurile brute nu se combin\u0103 direct (scale diferite) \u2192 normalizare pe rang: fiecare scor mapat la percentila sa \u00een distribu\u021bia de antrenare benign\u0103", 0),
    ("Votare ponderat\u0103 (soft voting): s = w\u00b7s_AE + (1-w)\u00b7s_IF", 0, True),
    ("Regula maximului: s = max(s_AE, s_IF) \u2014 un combinator de tip OR", 0, True),
], size=15, space_after=13)

ens_rows = [
    ["AE+Context singur", "0.882", "0.924", "0.292", "0.013", "0.025"],
    ["Temporal IForest singur", "0.956", "0.123", "0.611", "0.578", "0.560"],
    ["Votare ponderat\u0103 (w=0.4)", "0.977", "0.228", "0.801", "0.205", "0.319"],
]
table_slide_generic(s, Inches(6.65), Inches(2.05), Inches(6.15),
                     ["Model", "DDoS", "PortScan", "GoldenEye", "Slowhttptest", "slowloris"], ens_rows,
                     [2.0, 1.0, 1.1, 1.15, 1.35, 1.1], font_size=11.5, row_h=Inches(0.55), highlight_rows={3})
add_text(s, Inches(0.55), Inches(5.6), Inches(12.2), Inches(1.4),
          "Votarea ponderat\u0103 ridic\u0103 pragul minim de acoperire pe clas\u0103 \u2014 clasa cea mai slab\u0103 (0.205) dep\u0103\u0219e\u0219te propriile clase cele mai slabe ale ambelor modele singure (0.013 \u0219i 0.123), cu costul unui AP agregat mai mic.",
          size=14, color=DARK_TEXT, line_spacing=1.2)
add_footer(s, "\u00a75.2.3\u20134 \u2014 Construirea Scorului Combinat \u0219i Acoperire per Clas\u0103")

# =========================================================
# SLIDE 20 - Conclusions
# =========================================================
s = add_slide(); set_bg(s)
add_header(s, "CONCLUZII", "R\u0103spuns la \u00eentrebarea de cercetare", 20)
add_text(s, Inches(0.55), Inches(1.5), Inches(12.2), Inches(0.7),
          "R\u0103spuns afirmativ: aceea\u0219i paradigm\u0103 produce rezultate diferite pe seturi de date diferite \u2014 pentru c\u0103 presupunerea sa despre normalitate se potrive\u0219te sau nu structurii reale a anomaliei c\u0103utate.",
          size=17, color=BLUE, bold=True, line_spacing=1.25)
add_bullets(s, Inches(0.55), Inches(2.75), Inches(12.2), Inches(3.0), [
    ("Ipoteza 1 (structura temporal\u0103 avantajeaz\u0103 date secven\u021biale) se confirm\u0103 pe Yahoo A3/A4, dar se inverseaz\u0103 pe Setul A \u0219i e dep\u0103\u0219it\u0103 de caracteristicile de context la CIC-IDS2017", 0, True),
    ("Ipoteza 2 (paradigme complementare, combinate, \u00eembun\u0103t\u0103\u021besc performan\u021ba) se confirm\u0103: AE+Context \u0219i Temporal IForest exploateaz\u0103 evidente disjuncte, iar votarea ponderat\u0103 extinde acoperirea", 0, True),
    ("Ambele ipoteze sunt confirmate, dar nuan\u021bate \u2014 nu necondi\u021bionat", 0),
], size=16, space_after=14)
add_rect(s, Inches(0.55), Inches(5.9), Inches(12.25), Inches(1.05), LIGHT)
add_text(s, Inches(0.8), Inches(6.03), Inches(11.7), Inches(0.8),
          "Direc\u021bii viitoare: testarea sistematic\u0103 a celorlalte ipoteze (de ce ferestrele temporale d\u0103uneaz\u0103 uneori) \u0219i extinderea ansamblului la celelalte seturi de date.",
          size=13.5, color=GRAY, italic=True)
add_footer(s, "Capitolul 6 \u2014 Concluzii")

# =========================================================
# SLIDE 21 - Thank you
# =========================================================
s = add_slide(); set_bg(s, NAVY)
add_rect(s, 0, Inches(3.55), SW, Pt(3), ACCENT)
add_text(s, Inches(1.0), Inches(2.7), Inches(11), Inches(0.8), "V\u0103 mul\u021bumesc!", size=44, color=WHITE, bold=True)
add_text(s, Inches(1.0), Inches(3.75), Inches(11), Inches(0.6), "\u00centreb\u0103ri?", size=22, color=RGBColor(0xC9, 0xD6, 0xE8))
add_text(s, Inches(1.0), Inches(6.6), Inches(10), Inches(0.5), "Savin Radu-Andrei \u2014 Coordonator: Paul Irofti", size=14, color=RGBColor(0x9A, 0xA8, 0xC0))

prs.save(OUT)
print("Saved", OUT, "slides:", len(prs.slides.__iter__.__self__._sldIdLst))
