"""Convert PROJECT_DOSSIER.md into a formatted Word document.

WHY A SCRIPT RATHER THAN A HAND-MADE DOCX
    The Markdown file is the single source of truth. Regenerating the .docx from
    it means the two can never disagree, and if the dossier is updated the Word
    file is one command away from being current.

    Deliberately dependency-light (python-docx only) and deliberately tolerant:
    unknown Markdown is emitted as normal paragraphs rather than crashing, so a
    small formatting edit to the .md can never block producing the .docx.

HANDLES
    #/##/###/#### headings, pipe tables, bullet + numbered lists, fenced code
    blocks, horizontal rules, bold/italic/code spans, and blockquotes.
"""
import re
from pathlib import Path

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Pt, RGBColor, Inches

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "PROJECT_DOSSIER.md"
OUT = ROOT / "PROJECT_DOSSIER.docx"

ACCENT = RGBColor(0x1F, 0x4E, 0x79)
GREY = RGBColor(0x59, 0x59, 0x59)

# Order matters: bold first, then italic, then code, so nested spans survive.
SPAN = [
    (re.compile(r"\*\*(.+?)\*\*"), "b"),
    (re.compile(r"(?<!\*)\*([^*]+?)\*(?!\*)"), "i"),
    (re.compile(r"`([^`]+?)`"), "code"),
]


def add_runs(par, text):
    """Add text to a paragraph, converting inline Markdown spans to formatting."""
    # Split on the spans first so their content is never re-processed.
    tokens = re.split(r"(\*\*.+?\*\*|`[^`]+?`|(?<!\*)\*[^*]+?\*(?!\*))", text)
    for tok in tokens:
        if not tok:
            continue
        if tok.startswith("**") and tok.endswith("**") and len(tok) > 4:
            par.add_run(tok[2:-2]).bold = True
        elif tok.startswith("`") and tok.endswith("`") and len(tok) > 2:
            r = par.add_run(tok[1:-1])
            r.font.name = "Consolas"
            r.font.size = Pt(9)
        elif tok.startswith("*") and tok.endswith("*") and len(tok) > 2:
            par.add_run(tok[1:-1]).italic = True
        else:
            par.add_run(tok)


def split_row(line):
    return [c.strip() for c in line.strip().strip("|").split("|")]


def is_table_sep(line):
    return bool(re.match(r"^\s*\|[\s:|-]+\|\s*$", line))


def build():
    doc = Document()

    # Base typography.
    normal = doc.styles["Normal"]
    normal.font.name = "Calibri"
    normal.font.size = Pt(10.5)
    normal.paragraph_format.space_after = Pt(6)
    normal.paragraph_format.line_spacing = 1.15

    for level, size in ((1, 20), (2, 15), (3, 12.5), (4, 11)):
        st = doc.styles[f"Heading {level}"]
        st.font.name = "Calibri"
        st.font.size = Pt(size)
        st.font.color.rgb = ACCENT
        st.font.bold = True

    # Reasonable margins.
    for s in doc.sections:
        s.left_margin = s.right_margin = Inches(0.85)

    lines = SRC.read_text(encoding="utf-8").split("\n")
    i, n = 0, len(lines)

    while i < n:
        line = lines[i]
        stripped = line.strip()

        # Blank line: nothing to do.
        if not stripped:
            i += 1
            continue

        # Fenced code block.
        if stripped.startswith("```"):
            i += 1
            block = []
            while i < n and not lines[i].strip().startswith("```"):
                block.append(lines[i])
                i += 1
            i += 1
            para = doc.add_paragraph()
            para.paragraph_format.left_indent = Inches(0.25)
            para.paragraph_format.space_before = Pt(6)
            para.paragraph_format.space_after = Pt(8)
            run = para.add_run("\n".join(block))
            run.font.name = "Consolas"
            run.font.size = Pt(9)
            run.font.color.rgb = RGBColor(0x1A, 0x1A, 0x1A)
            continue

        # Headings.
        m = re.match(r"^(#{1,4})\s+(.*)$", stripped)
        if m:
            level, text = len(m.group(1)), m.group(2).strip()
            if level == 1:
                # Title paragraph, styled distinctly from section headings.
                p = doc.add_paragraph()
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
                r = p.add_run(re.sub(r"\*\*", "", text))
                r.bold = True
                r.font.size = Pt(24)
                r.font.color.rgb = ACCENT
                p.paragraph_format.space_after = Pt(2)
            else:
                p = doc.add_heading(level=level - 1 if level > 1 else 1)
                p.clear()
                add_runs(p, text)
            i += 1
            continue

        # Horizontal rule: a real paragraph border rather than a run of underscores.
        if re.match(r"^-{3,}$", stripped) or re.match(r"^\*{3,}$", stripped):
            p = doc.add_paragraph()
            p.paragraph_format.space_before = Pt(4)
            p.paragraph_format.space_after = Pt(4)
            pPr = p._p.get_or_add_pPr()
            borders = OxmlElement("w:pBdr")
            bottom = OxmlElement("w:bottom")
            bottom.set(qn("w:val"), "single")
            bottom.set(qn("w:sz"), "6")
            bottom.set(qn("w:space"), "1")
            bottom.set(qn("w:color"), "BFBFBF")
            borders.append(bottom)
            pPr.append(borders)
            i += 1
            continue

        # Table: header row, separator row, then data rows.
        if stripped.startswith("|") and i + 1 < n and is_table_sep(lines[i + 1]):
            header = split_row(stripped)
            i += 2
            body = []
            while i < n and lines[i].strip().startswith("|"):
                body.append(split_row(lines[i].strip()))
                i += 1

            table = doc.add_table(rows=1, cols=len(header))
            table.style = "Light Grid Accent 1"
            table.alignment = WD_TABLE_ALIGNMENT.CENTER
            for cell, text in zip(table.rows[0].cells, header):
                cell.text = ""
                p = cell.paragraphs[0]
                add_runs(p, text)
                for r in p.runs:
                    r.bold = True
                    r.font.size = Pt(9.5)
            for row in body:
                cells = table.add_row().cells
                for cell, text in zip(cells, row):
                    cell.text = ""
                    p = cell.paragraphs[0]
                    add_runs(p, text)
                    for r in p.runs:
                        r.font.size = Pt(9.5)
            doc.add_paragraph()
            continue

        # Blockquote.
        if stripped.startswith(">"):
            p = doc.add_paragraph()
            p.paragraph_format.left_indent = Inches(0.3)
            p.paragraph_format.space_before = Pt(4)
            add_runs(p, re.sub(r"^>\s?", "", stripped))
            for r in p.runs:
                r.font.color.rgb = GREY
                r.italic = True
            i += 1
            continue

        # Bullet list.
        m = re.match(r"^(\s*)[-*]\s+(.*)$", line)
        if m:
            indent = len(m.group(1))
            p = doc.add_paragraph(style="List Bullet")
            if indent >= 2:
                p.paragraph_format.left_indent = Inches(0.75)
            add_runs(p, m.group(2))
            i += 1
            continue

        # Numbered list.
        m = re.match(r"^(\s*)(\d+)\.\s+(.*)$", line)
        if m:
            p = doc.add_paragraph(style="List Number")
            if len(m.group(1)) >= 2:
                p.paragraph_format.left_indent = Inches(0.75)
            add_runs(p, m.group(3))
            i += 1
            continue

        # Plain paragraph (merge consecutive plain lines into one paragraph).
        buf = [stripped]
        i += 1
        while i < n:
            nxt = lines[i].strip()
            if (not nxt or nxt.startswith(("#", "|", ">", "-", "*", "```"))
                    or re.match(r"^\d+\.\s", nxt)):
                break
            buf.append(nxt)
            i += 1
        p = doc.add_paragraph()
        add_runs(p, " ".join(buf))

    doc.save(OUT)
    return OUT


if __name__ == "__main__":
    out = build()
    print("wrote", out, f"({out.stat().st_size / 1024:.1f} KB)")
