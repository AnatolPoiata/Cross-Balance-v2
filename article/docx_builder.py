"""Helpers to write an ECOFORUM-formatted .docx with python-docx.

The paragraph/table styles (Title, Authors, Abstract, Heading 2, Subsection,
Text, Table Title, Figure Caption, References, Note) come from the
ECOFORUM template; ``new_document`` re-uses the A4 page set-up, styles,
header and footer of an existing template-based file and empties its body.
"""

import re

import docx
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt

M_NS = "http://schemas.openxmlformats.org/officeDocument/2006/math"


def new_document(base_path):
    doc = docx.Document(base_path)
    body = doc.element.body
    sect = body.find(qn("w:sectPr"))
    for child in list(body):
        if child is not sect:
            body.remove(child)
    # drop image parts that are no longer referenced (template leftovers)
    for rid, rel in list(doc.part.rels.items()):
        if rel.reltype.endswith("/image"):
            doc.part.drop_rel(rid)
    return doc


# **bold** / *italic* only when the markers hug non-space text and are not part of a longer run
# of asterisks, so literal "***" (anonymous references), "F1*" and "* note" stay as typed.
_TOKEN = re.compile(r"((?<!\*)\*\*(?=[^\s*]).+?(?<=[^\s*])\*\*(?!\*)"
                    r"|(?<!\*)\*(?=[^\s*]).+?(?<=[^\s*])\*(?!\*)|_\{.+?\}|\^\{.+?\})")


def add_runs(par, text, size=None, bold=None, italic=None):
    """Minimal inline markup: **bold**, *italic*, _{subscript}, ^{superscript}."""
    for part in _TOKEN.split(text):
        if not part:
            continue
        b, i, sub, sup = bold, italic, False, False
        if _TOKEN.fullmatch(part) and part.startswith("**"):
            part, b = part[2:-2], True
        elif _TOKEN.fullmatch(part) and part.startswith("*"):
            part, i = part[1:-1], True
        elif part.startswith("_{"):
            part, sub = part[2:-1], True
        elif part.startswith("^{"):
            part, sup = part[2:-1], True
        run = par.add_run(part)
        if b is not None:
            run.bold = b
        if i is not None:
            run.italic = i
        if sub:
            run.font.subscript = True
        if sup:
            run.font.superscript = True
        if size:
            run.font.size = Pt(size)
    return par


def para(doc, style, text="", **kw):
    p = doc.add_paragraph(style=style)
    add_runs(p, text, **kw)
    return p


def keep_with_next(p):
    p.paragraph_format.keep_with_next = True
    return p


# ---------------------------------------------------------------- equations
def _m(tag, **attrs):
    el = OxmlElement(f"m:{tag}")
    for k, v in attrs.items():
        el.set(qn(f"m:{k}"), v)
    return el


def m_run(text, italic=False):
    r = _m("r")
    rpr = _m("rPr")
    rpr.append(_m("sty", val="i" if italic else "p"))
    r.append(rpr)
    wrpr = OxmlElement("w:rPr")
    fonts = OxmlElement("w:rFonts")
    fonts.set(qn("w:ascii"), "Cambria Math")
    fonts.set(qn("w:hAnsi"), "Cambria Math")
    wrpr.append(fonts)
    sz = OxmlElement("w:sz")
    sz.set(qn("w:val"), "20")
    wrpr.append(sz)
    r.append(wrpr)
    t = _m("t")
    t.text = text
    t.set("{http://www.w3.org/XML/1998/namespace}space", "preserve")
    r.append(t)
    return r


def _wrap(tag, *children):
    el = _m(tag)
    for c in children:
        el.append(c)
    return el


_SUB = re.compile(r"([A-Za-z\u0391-\u03c9][\u0300-\u036f]?)_\{([^}]+)\}")
_TOK = re.compile(r"([A-Za-z\u0391-\u03c9][\u0300-\u036f]?[A-Za-z]*|\d+(?:[.,]\d+)*|\s+|.)")
_UPRIGHT_SYMBOLS = {"π"}


def m_text(text):
    """Plain math text -> runs: single-letter variables (Latin or Greek, possibly
    with a combining accent) in italic; multi-letter words (min, tan, if, flag),
    numbers, operators and spaces upright."""
    out = []
    for tok in _TOK.findall(text):
        letters = re.sub(r"[\u0300-\u036f]", "", tok)
        is_var = len(letters) == 1 and letters.isalpha() and tok not in _UPRIGHT_SYMBOLS
        out.append(m_run(tok, italic=is_var))
    return out


def m_seq(items):
    """items: list of str | element. Strings are parsed into italic variables
    and upright words; 'x_{sub}' (x = one letter, optionally accented) becomes a
    real OMML subscript object."""
    out = []
    for it in items:
        if not isinstance(it, str):
            out.append(it)
            continue
        pos = 0
        for mm in _SUB.finditer(it):
            out += m_text(it[pos:mm.start()])
            out.append(m_sub(mm.group(1), mm.group(2)))
            pos = mm.end()
        out += m_text(it[pos:])
    return out


def m_sub(base, sub):
    return _wrap("sSub", _wrap("e", *m_text(base)), _wrap("sub", *m_text(sub)))


def m_frac(num, den):
    return _wrap("f", _wrap("num", *m_seq(num)), _wrap("den", *m_seq(den)))


def equation(doc, items, number):
    """Centred editable OMML equation followed by a right-hand '(n)'."""
    p = doc.add_paragraph(style="Text")
    pf = p.paragraph_format
    pf.first_line_indent = Cm(0)
    pf.space_before = Pt(5)
    pf.space_after = Pt(5)
    pf.keep_together = True
    p.alignment = WD_ALIGN_PARAGRAPH.LEFT
    # centre tab at text middle, right tab at margin for the number
    tabs = pf.tab_stops
    tabs.add_tab_stop(Cm(8.0), alignment=1)
    tabs.add_tab_stop(Cm(16.0), alignment=2)
    p.add_run("\t")
    omath = _m("oMath")
    for el in m_seq(items):
        omath.append(el)
    p._p.append(omath)
    p.add_run(f"\t({number})")
    return p


# ------------------------------------------------------------------- tables
def _cell_shading(cell, fill):
    tcpr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), fill)
    tcpr.append(shd)


def _borders(table, color="808080"):
    tblpr = table._tbl.tblPr
    borders = OxmlElement("w:tblBorders")
    for edge in ("top", "bottom", "insideH"):
        el = OxmlElement(f"w:{edge}")
        el.set(qn("w:val"), "single")
        el.set(qn("w:sz"), "4")
        el.set(qn("w:color"), color)
        borders.append(el)
    for edge in ("left", "right", "insideV"):
        el = OxmlElement(f"w:{edge}")
        el.set(qn("w:val"), "nil")
        borders.append(el)
    tblpr.append(borders)
    mar = OxmlElement("w:tblCellMar")
    for side in ("top", "bottom", "left", "right"):
        el = OxmlElement(f"w:{side}")
        el.set(qn("w:w"), "28" if side in ("top", "bottom") else "70")
        el.set(qn("w:type"), "dxa")
        mar.append(el)
    tblpr.append(mar)


def table(doc, rows, widths_cm, font_pt=8, header_rows=1, align=None, bold_cells=()):
    """rows: list of lists of str (inline markup allowed). ``align``: per
    column 'l'/'c'/'r'. ``bold_cells``: set of (row, col) to embolden."""
    t = doc.add_table(rows=len(rows), cols=len(rows[0]))
    t.alignment = 1
    t.autofit = False
    _borders(t)
    align = align or ["l"] + ["c"] * (len(rows[0]) - 1)
    amap = {"l": WD_ALIGN_PARAGRAPH.LEFT, "c": WD_ALIGN_PARAGRAPH.CENTER, "r": WD_ALIGN_PARAGRAPH.RIGHT}
    for i, row in enumerate(rows):
        tr = t.rows[i]
        if i < header_rows:
            trpr = tr._tr.get_or_add_trPr()
            h = OxmlElement("w:tblHeader")
            trpr.append(h)
        cant = OxmlElement("w:cantSplit")
        tr._tr.get_or_add_trPr().append(cant)
        for j, txt in enumerate(row):
            cell = tr.cells[j]
            cell.width = Cm(widths_cm[j])
            p = cell.paragraphs[0]
            p.style = doc.styles["Normal"]
            pf = p.paragraph_format
            pf.space_before = Pt(0)
            pf.space_after = Pt(0)
            pf.first_line_indent = Cm(0)
            pf.line_spacing = 1.0
            p.alignment = amap[align[j]]
            bold = True if (i < header_rows or (i, j) in bold_cells) else None
            add_runs(p, str(txt), size=font_pt, bold=bold)
            for r in p.runs:
                r.font.name = "Times New Roman"
            if i < header_rows:
                _cell_shading(cell, "EDEDED")
    # grid widths (Word honours tblGrid)
    grid = t._tbl.tblGrid
    for j, gc in enumerate(grid.findall(qn("w:gridCol"))):
        gc.set(qn("w:w"), str(int(widths_cm[j] * 567)))
    return t


def caption(doc, style, text):
    p = para(doc, style, text)
    p.paragraph_format.space_before = Pt(6)
    p.paragraph_format.space_after = Pt(3)
    if style == "Table Title":
        keep_with_next(p)
    return p


def note(doc, text, size=8):
    p = para(doc, "Text", text, size=size)
    p.paragraph_format.first_line_indent = Cm(0)
    p.paragraph_format.space_before = Pt(2)
    p.paragraph_format.space_after = Pt(4)
    return p


def figure(doc, path, width_cm):
    p = doc.add_paragraph(style="Normal")
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(4)
    p.paragraph_format.keep_with_next = True
    p.add_run().add_picture(str(path), width=Cm(width_cm))
    return p


def hyperlink_text(p, text, url, size=8):
    part = p.part
    rid = part.relate_to(url, "http://schemas.openxmlformats.org/officeDocument/2006/relationships/hyperlink",
                         is_external=True)
    h = OxmlElement("w:hyperlink")
    h.set(qn("r:id"), rid)
    r = OxmlElement("w:r")
    rpr = OxmlElement("w:rPr")
    sz = OxmlElement("w:sz")
    sz.set(qn("w:val"), str(size * 2))
    rpr.append(sz)
    r.append(rpr)
    t = OxmlElement("w:t")
    t.text = text
    t.set("{http://www.w3.org/XML/1998/namespace}space", "preserve")
    r.append(t)
    h.append(r)
    p._p.append(h)


def reference(doc, text, url=None):
    p = para(doc, "References", text)
    if url:
        p.add_run(" ")
        hyperlink_text(p, url, url)
    return p


