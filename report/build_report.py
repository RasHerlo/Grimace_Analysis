"""Assemble all saved analysis blocks into one navigable PDF.

Run:  python report/build_report.py
Output: report/output/grimace_report.pdf

The report is rebuilt from scratch every time from the blocks that the analysis scripts have saved
(report/output/content/*/block.json). Run `python report/run_all.py` to re-run all analyses first.
"""

from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from xml.sax.saxutils import escape

import matplotlib
import yaml
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (BaseDocTemplate, Frame, Image, KeepTogether, PageBreak, PageTemplate, Paragraph,
                                Spacer, Table, TableStyle)
from reportlab.platypus.tableofcontents import TableOfContents

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from grimace.report import content_dir, git_commit  # noqa: E402

OUT = ROOT / "report" / "output"
OUT.mkdir(parents=True, exist_ok=True)
PDF = OUT / "grimace_report.pdf"

# ---------------------------------------------------------------- fonts (DejaVu ships with matplotlib: full Unicode)
fd = Path(matplotlib.get_data_path()) / "fonts" / "ttf"
for name, file in [("DV", "DejaVuSans.ttf"), ("DV-B", "DejaVuSans-Bold.ttf"), ("DV-I", "DejaVuSans-Oblique.ttf"),
                   ("DV-BI", "DejaVuSans-BoldOblique.ttf"), ("DVM", "DejaVuSansMono.ttf")]:
    pdfmetrics.registerFont(TTFont(name, str(fd / file)))
pdfmetrics.registerFontFamily("DV", normal="DV", bold="DV-B", italic="DV-I", boldItalic="DV-BI")

# ---------------------------------------------------------------- styles
NAVY, GREY, LIGHT, ACC = colors.HexColor("#1f3a5f"), colors.HexColor("#666666"), colors.HexColor("#eef2f7"), colors.HexColor("#c0392b")
ST = {
    "body": ParagraphStyle("body", fontName="DV", fontSize=9, leading=12.5, alignment=TA_LEFT, spaceAfter=3),
    "small": ParagraphStyle("small", fontName="DV", fontSize=7.5, leading=10, textColor=GREY),
    "cell": ParagraphStyle("cell", fontName="DV", fontSize=7.5, leading=9.5),
    "cellb": ParagraphStyle("cellb", fontName="DV-B", fontSize=7.5, leading=9.5, textColor=colors.white),
    "cellw": ParagraphStyle("cellw", fontName="DV-B", fontSize=7, leading=9),
    "bullet": ParagraphStyle("bullet", fontName="DV", fontSize=9, leading=12.5, leftIndent=12, bulletIndent=2, spaceAfter=1.5),
    "h0": ParagraphStyle("h0", fontName="DV-B", fontSize=17, leading=21, textColor=NAVY, spaceBefore=4, spaceAfter=6),
    "h1": ParagraphStyle("h1", fontName="DV-B", fontSize=12.5, leading=16, textColor=NAVY, spaceBefore=10, spaceAfter=3),
    "h2": ParagraphStyle("h2", fontName="DV-B", fontSize=10.5, leading=14, textColor=colors.black, spaceBefore=8, spaceAfter=3),
    "lab": ParagraphStyle("lab", fontName="DV-B", fontSize=8.5, leading=11, textColor=NAVY, spaceBefore=6, spaceAfter=1),
    "cap": ParagraphStyle("cap", fontName="DV", fontSize=8.2, leading=11, spaceBefore=2, spaceAfter=1),
    "how": ParagraphStyle("how", fontName="DV-I", fontSize=8, leading=10.8, textColor=colors.HexColor("#333333"), spaceAfter=7),
    "title": ParagraphStyle("title", fontName="DV-B", fontSize=26, leading=31, textColor=NAVY, spaceAfter=6),
    "sub": ParagraphStyle("sub", fontName="DV", fontSize=11, leading=15, textColor=GREY, spaceAfter=10),
    "take": ParagraphStyle("take", fontName="DV-B", fontSize=9.2, leading=12.8),
}
STATUS_COL = {"not analysed": colors.HexColor("#999999"), "preliminary": colors.HexColor("#d68910"), "final": colors.HexColor("#1e8449")}


def fmt(text: str) -> str:
    """Tiny inline markup: **bold**, *italic*, `code`; everything else is escaped."""
    t = escape(str(text))
    t = re.sub(r"`([^`]+)`", r'<font face="DVM" size="8">\1</font>', t)
    t = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", t)
    t = re.sub(r"(?<![\w*])\*([^*\s][^*]*)\*(?![\w*])", r"<i>\1</i>", t)
    return t


def P(text, style="body"):
    return Paragraph(fmt(text), ST[style])


def bullets(items, style="bullet"):
    return [Paragraph(fmt(i), ST[style], bulletText="•") for i in items]


class Heading(Paragraph):
    """Paragraph that registers itself in the outline/TOC."""

    def __init__(self, text, style, level, key):
        super().__init__(fmt(text), ST[style])
        self.level, self.key, self.plain = level, key, text


# ---------------------------------------------------------------- load blocks and questions
cfg = yaml.safe_load((ROOT / "report" / "questions.yaml").read_text(encoding="utf-8"))
chapters = cfg["chapters"]
questions = {q["id"]: q for q in cfg["questions"]}
blocks = []
for f in sorted(content_dir().glob("*/block.json")):
    b = json.loads(f.read_text(encoding="utf-8"))
    b["dir"] = f.parent
    blocks.append(b)
blocks.sort(key=lambda b: (list(questions).index(b["meta"]["questions"][0]) if b["meta"]["questions"][0] in questions else 99, b["meta"]["order"], b["meta"]["id"]))
for b in blocks:
    for q in b["meta"]["questions"]:
        assert q in questions, f"block {b['meta']['id']} refers to unknown question {q}"
primary = defaultdict(list)
secondary = defaultdict(list)
for b in blocks:
    primary[b["meta"]["questions"][0]].append(b)
    for q in b["meta"]["questions"][1:]:
        secondary[q].append(b)


def q_status(qid):
    if not primary[qid]:
        return "not analysed"
    return "final" if all(b["meta"]["status"] == "final" for b in primary[qid]) else "preliminary"


def status_tag(s):
    return f'<font color="{STATUS_COL[s].hexval().replace("0x", "#")}"><b>{s.upper()}</b></font>'


# ---------------------------------------------------------------- open items from RESEARCH_QUESTIONS.md
open_md = defaultdict(list)
section = ""
for line in (ROOT / "RESEARCH_QUESTIONS.md").read_text(encoding="utf-8").splitlines():
    if line.startswith("#"):
        section = line.lstrip("# ").strip()
    elif line.startswith("- [ ]"):
        open_md[section].append(line[5:].strip())

# ---------------------------------------------------------------- document with TOC + outline
class Doc(BaseDocTemplate):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.cur = ""
        frame = Frame(self.leftMargin, self.bottomMargin, self.width, self.height, id="f", leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0)
        self.addPageTemplates([PageTemplate(id="p", frames=[frame], onPage=self.deco, onPageEnd=self.footer)])

    def beforeDocument(self):
        self.cur = ""  # multiBuild runs several passes; do not carry the footer text over

    def deco(self, c, d):
        c.saveState()
        c.setFont("DV", 7.5)
        c.setFillColor(GREY)
        c.drawString(self.leftMargin, A4[1] - 11 * mm, "Grimace analysis report")
        c.drawRightString(A4[0] - self.rightMargin, A4[1] - 11 * mm, "PRELIMINARY - living document")
        c.setStrokeColor(colors.HexColor("#cccccc"))
        c.line(self.leftMargin, A4[1] - 13 * mm, A4[0] - self.rightMargin, A4[1] - 13 * mm)
        c.restoreState()

    def footer(self, c, d):
        c.saveState()
        c.setFont("DV", 7.5)
        c.setFillColor(GREY)
        c.drawString(self.leftMargin, 9 * mm, self.cur[:95])
        c.drawRightString(A4[0] - self.rightMargin, 9 * mm, f"page {c.getPageNumber()}")
        c.restoreState()

    def afterFlowable(self, fl):
        if isinstance(fl, Heading):
            self.canv.bookmarkPage(fl.key)
            self.canv.addOutlineEntry(fl.plain, fl.key, fl.level, closed=fl.level >= 1)
            if fl.level <= 1:
                self.cur = fl.plain
            self.notify("TOCEntry", (fl.level, fl.plain, self.page, fl.key))


doc = Doc(str(PDF), pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm, topMargin=18 * mm, bottomMargin=16 * mm,
          title="Grimace analysis report", author="Grimace_Analysis")
W = doc.width
S = []  # story

# ---------------------------------------------------------------- cover
n_final = sum(b["meta"]["status"] == "final" for b in blocks)
S += [Spacer(1, 25 * mm), P("Grimace analysis report", "title"),
      P("Nav1.8 / Piezo2 / Levcromakalim: are subsets or states of the grimace action units more informative than the total score?", "sub"),
      P(f"Generated {datetime.now():%Y-%m-%d %H:%M} from repository state `{git_commit()}`.  "
        f"{len(blocks)} analyses included ({n_final} final, {len(blocks) - n_final} preliminary).", "small"), Spacer(1, 8 * mm)]

how = [
    "**What this is.** A living document. Every analysis script saves a structured *block*; this report is rebuilt from all blocks, "
    "so it grows with the analysis and the numbers in the text always come from the code that made the figures.",
    "**How it is organised.** Chapters follow `RESEARCH_QUESTIONS.md`: data and QC, the sub-questions (Q1-Q6), Part A (A1-A4) and Part B (B1-B5). "
    "Every question has its own subsection, also when nothing has been done for it yet (marked *not analysed*).",
    "**Where to start.** *Status overview* (one line per question, clickable) and *Decision queue* (what is open and needs a decision). "
    "The PDF bookmarks (side panel) mirror the table of contents.",
    "**How each analysis is laid out.** Takeaway (the current answer) → Question → Data used → Method (numbered steps with parameters) → "
    "Figures/tables, each with a caption and a *how to read* note → Findings (what the numbers show) → Interpretation (what we conclude, and what we do not) "
    "→ Caveats → Open points. The script and repository commit are named under each title so that every result can be re-run.",
    "**Conventions.** Pattern notation is always `Orbital-Nose-Ears-Whiskers`, e.g. `0-2-1-0` = Orbital 0, Nose 2, Ears 1, Whiskers 0. "
    "Scores are 0-2 per unit; `-1` (not scorable) is treated as missing. Null models and statistical terms are explained in the glossary at the end.",
    "**Status.** *Preliminary* = descriptive, exploratory, not yet corrected for multiple testing or tied to a final choice of method. "
    "Nothing in this report is a final result until marked *final*.",
]
S += [Paragraph(fmt(t), ST["body"]) for t in how]
S += [PageBreak(), Paragraph("Contents", ST["h0"])]
toc = TableOfContents(dotsMinLevel=0)
toc.levelStyles = [
    ParagraphStyle("t0", fontName="DV-B", fontSize=10, leading=15, spaceBefore=5, textColor=NAVY),
    ParagraphStyle("t1", fontName="DV", fontSize=8.8, leading=12, leftIndent=12),
    ParagraphStyle("t2", fontName="DV", fontSize=8, leading=11, leftIndent=26, textColor=GREY),
]
S += [toc, PageBreak()]

# ---------------------------------------------------------------- status overview
S.append(Heading("Status overview", "h0", 0, "status_overview"))
S.append(P("One line per research question. Click a question ID to jump to its subsection."))
rows = [[Paragraph(h, ST["cellb"]) for h in ("ID", "Question", "Status", "Current answer (takeaway)")]]
for ch in chapters:
    rows.append([Paragraph(f"<b>{escape(ch['title'])}</b>", ST["cell"]), "", "", ""])
    for q in [q for q in questions.values() if q["chapter"] == ch["id"]]:
        s = q_status(q["id"])
        take = " ".join(f"[{b['meta']['id'].split('_')[0]}] {b['sections'].get('takeaway', '')}" for b in primary[q["id"]]) or "-"
        rows.append([Paragraph(f'<a href="#q_{q["id"]}" color="#1f3a5f"><b>{q["id"]}</b></a>', ST["cell"]), P(q["short"], "cell"),
                     Paragraph(status_tag(s), ST["cellw"]), P(take, "cell")])
t = Table(rows, colWidths=[10 * mm, 38 * mm, 25 * mm, W - 73 * mm], repeatRows=1)
sty = [("BACKGROUND", (0, 0), (-1, 0), NAVY), ("VALIGN", (0, 0), (-1, -1), "TOP"), ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#cccccc")),
       ("LEFTPADDING", (0, 0), (-1, -1), 3), ("RIGHTPADDING", (0, 0), (-1, -1), 3)]
for i, r in enumerate(rows):
    if i and r[1] == "":
        sty += [("SPAN", (0, i), (-1, i)), ("BACKGROUND", (0, i), (-1, i), LIGHT)]
t.setStyle(TableStyle(sty))
S += [t, PageBreak()]

# ---------------------------------------------------------------- decision queue
S.append(Heading("Decision queue: what is open", "h0", 0, "decision_queue"))
S.append(P("Collected automatically from the *open points* of every analysis and from the unchecked items in `RESEARCH_QUESTIONS.md`. "
           "This is the list to go through when deciding the next direction."))
S.append(Heading("Raised by the analyses", "h1", 1, "dq_blocks"))
rows = [[Paragraph(h, ST["cellb"]) for h in ("From", "Open point")]]
for b in blocks:
    for p in b["sections"].get("open_points", []):
        rows.append([Paragraph(f'<a href="#blk_{b["meta"]["id"]}" color="#1f3a5f"><b>{b["meta"]["id"].split("_")[0]}</b></a>', ST["cell"]), P(p, "cell")])
if len(rows) == 1:
    rows.append(["", P("none yet", "cell")])
t = Table(rows, colWidths=[16 * mm, W - 16 * mm], repeatRows=1)
t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), NAVY), ("VALIGN", (0, 0), (-1, -1), "TOP"), ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#cccccc"))]))
S.append(t)
for sec, items in open_md.items():
    key = "dq_" + re.sub(r"\W+", "_", sec)
    S.append(Heading(f"From RESEARCH_QUESTIONS.md: {sec}", "h1", 1, key))
    S += bullets(items)
S.append(PageBreak())


# ---------------------------------------------------------------- block renderer
def fit_image(path, max_w, max_h):
    w, h = ImageReader(str(path)).getSize()
    s = min(max_w / w, max_h / h)
    return Image(str(path), width=w * s, height=h * s)


def render_block(b):
    m, sec = b["meta"], b["sections"]
    bid = m["id"]
    out = [Heading(f"{bid.split('_')[0]}: {m['title']}", "h2", 2, f"blk_{bid}")]
    also = ", ".join('<a href="#q_%s" color="#1f3a5f">%s</a>' % (q, q) for q in m["questions"][1:])
    meta_line = (f"{status_tag(m['status'])} &nbsp; script {escape(m['script'])} &nbsp; commit {escape(m['commit'])} &nbsp; run {m['created'].replace('T', ' ')}"
                 + (f" &nbsp; also answers: {also}" if also else ""))
    out.append(Paragraph(meta_line, ST["small"]))
    out.append(Spacer(1, 3))
    if "takeaway" in sec:
        box = Table([[Paragraph(f"Takeaway: {fmt(sec['takeaway'])}", ST["take"])]], colWidths=[W])
        box.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), LIGHT), ("LINEBEFORE", (0, 0), (0, -1), 3, NAVY),
                                 ("LEFTPADDING", (0, 0), (-1, -1), 8), ("TOPPADDING", (0, 0), (-1, -1), 5), ("BOTTOMPADDING", (0, 0), (-1, -1), 5)]))
        out += [box]
    for key, label in [("question", "Question"), ("data", "Data used")]:
        if key in sec:
            out += [P(label, "lab"), P(sec[key])]
    if "method" in sec:
        out.append(P("Method", "lab"))
        out += [Paragraph(fmt(s), ST["bullet"], bulletText=f"{i}.") for i, s in enumerate(sec["method"], 1)]
    for it in b["items"]:
        if it["type"] == "figure":
            img = fit_image(b["dir"] / it["file"], W, (195 if it.get("tall") else 105) * mm)
            parts = [img, Paragraph(f"<b>{it['label']}.</b> {fmt(it['caption'])}", ST["cap"])]
            if it["how_to_read"]:
                parts.append(Paragraph(f"<b>How to read:</b> {fmt(it['how_to_read'])}", ST["how"]))
            out += [Spacer(1, 4), KeepTogether(parts)]
        elif it["type"] == "table":
            data = [[Paragraph(fmt(c), ST["cellb"]) for c in it["columns"]]] + [[Paragraph(fmt(v), ST["cell"]) for v in r] for r in it["rows"]]
            tb = Table(data, repeatRows=1)
            tb.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), NAVY), ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#cccccc")),
                                    ("VALIGN", (0, 0), (-1, -1), "TOP"), ("LEFTPADDING", (0, 0), (-1, -1), 3), ("RIGHTPADDING", (0, 0), (-1, -1), 3),
                                    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f6f8fb")])]))
            parts = [Paragraph(f"<b>{it['label']}.</b> {fmt(it['caption'])}", ST["cap"]), tb]
            if it["how_to_read"]:
                parts.append(Paragraph(f"<b>How to read:</b> {fmt(it['how_to_read'])}", ST["how"]))
            out += [Spacer(1, 4), KeepTogether(parts) if len(it["rows"]) < 22 else parts[0]] + ([] if len(it["rows"]) < 22 else parts[1:])
        else:
            out.append(P(it["text"]))
    for key, label in [("findings", "Findings (what the numbers show)"), ("interpretation", "Interpretation (what we conclude, and what we do not)"),
                       ("caveats", "Caveats"), ("open_points", "Open points / decisions needed")]:
        if sec.get(key):
            out += [P(label, "lab")] + bullets(sec[key])
    return out


# ---------------------------------------------------------------- chapters
for ch in chapters:
    S.append(Heading(ch["title"], "h0", 0, f"ch_{ch['id']}"))
    S.append(P(ch["intro"]))
    for q in [q for q in questions.values() if q["chapter"] == ch["id"]]:
        qid = q["id"]
        s = q_status(qid)
        S.append(Heading(f"{qid}: {q['short']}", "h1", 1, f"q_{qid}"))
        S.append(Paragraph(f"{fmt(q['text'])} &nbsp;&nbsp; {status_tag(s)}", ST["body"]))
        if secondary[qid]:
            S.append(Paragraph("See also: " + ", ".join(f'<a href="#blk_{b["meta"]["id"]}" color="#1f3a5f">{b["meta"]["id"].split("_")[0]}: {escape(b["meta"]["title"])}</a>' for b in secondary[qid]), ST["small"]))
        if not primary[qid]:
            S.append(Paragraph("<i>Not analysed yet.</i>", ST["small"]))
        for b in primary[qid]:
            S += render_block(b)
    S.append(PageBreak())

# ---------------------------------------------------------------- appendix
S.append(Heading("Appendix", "h0", 0, "appendix"))
S.append(Heading("Analyses included", "h1", 1, "app_blocks"))
rows = [[Paragraph(h, ST["cellb"]) for h in ("Block", "Questions", "Status", "Script", "Commit", "Run")]]
for b in blocks:
    m = b["meta"]
    rows.append([P(m["id"], "cell"), P(", ".join(m["questions"]), "cell"), P(m["status"], "cell"), P(m["script"], "cell"), P(m["commit"], "cell"), P(m["created"].replace("T", " "), "cell")])
t = Table(rows, colWidths=[34 * mm, 20 * mm, 20 * mm, W - 34 * mm - 20 * mm - 20 * mm - 28 * mm - 28 * mm, 28 * mm, 28 * mm], repeatRows=1)
t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), NAVY), ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#cccccc")), ("VALIGN", (0, 0), (-1, -1), "TOP")]))
S.append(t)

S.append(Heading("Glossary: terms and null models used", "h1", 1, "app_glossary"))
GLOSS = [
    ("Total Grimace Score", "Sum of the four unit scores (0-8) in a frame; undefined (NaN) if any unit is not scorable."),
    ("Pattern / state", "The combination of the four unit scores in one frame, written O-N-E-W (e.g. `0-2-1-0`). 'State' is used for recurring patterns once they have been shown to recur beyond chance (not yet)."),
    ("Complete frame", "A frame in which all four units are scorable (no `-1`). 82% of frames."),
    ("Recording / mouse", "One video = one recording (baseline or stimulated). 71 mice x 2 sessions = 142 recordings. The mouse, not the frame, is the independent unit; frames within a recording are autocorrelated."),
    ("Within-recording shuffle (null model)", "Each unit's scores are permuted independently inside each recording. This keeps every recording's own score distribution (so differences between mice are preserved) but destroys any covariation between units within the recording. What is left of the observed structure after this is **state-like covariation**, not mouse-level differences."),
    ("Time shuffle (null model)", "Frames are permuted in time inside each recording. Keeps how often each pattern occurs, destroys temporal order. Used to ask whether patterns persist or follow each other more than chance."),
    ("Expected under independence (pooled / within)", "Product of the unit marginals, either over all frames (pooled) or computed per recording (within). 'Observed / expected' > 1 means the pattern is over-represented."),
    ("Mutual information, total correlation (bits)", "Information shared between two units (mutual information) or among all four (total correlation: sum of the individual entropies minus the joint entropy). 0 = independent. Plug-in estimates are biased upwards, so they are always compared with the shuffle null."),
    ("Entropy of the pattern vs of the total", "H(pattern) - H(total) = bits needed to tell the patterns apart once the total is known, i.e. what the total discards."),
    ("Spearman / within-recording / between-recording correlation", "Rank correlation over all frames; Pearson correlation of scores after subtracting each recording's mean (state-like); correlation of the recordings' mean scores (trait-like, mouse level)."),
    ("PCA, PC1", "Principal component analysis of the correlation matrix. If PC1 points equally along all four units it is the same as the total score."),
    ("Autocorrelation", "Correlation between a score and the score k seconds later within a recording (after subtracting the recording mean; frames with a missing value are skipped). 0 = no memory."),
    ("Dwell time / run", "Number of consecutive seconds with an identical pattern; an unscorable frame ends a run."),
    ("GEE, odds ratio", "Generalised estimating equation (logistic, independence working correlation, robust standard errors clustered on recording). The odds ratio is the multiplicative change in the odds of the outcome per one-point increase in the predictor."),
    ("Kruskal-Wallis / Wilcoxon", "Rank-based tests: Kruskal-Wallis compares several groups, the (paired) Wilcoxon signed-rank test compares two conditions in the same mice."),
    ("Missing at random?", "Missingness is *informative* if the chance that a unit is unscorable depends on the scores of the other units; dropping such frames then biases the remaining data."),
]
rows = [[Paragraph(h, ST["cellb"]) for h in ("Term", "Meaning")]] + [[P(a, "cell"), P(b, "cell")] for a, b in GLOSS]
t = Table(rows, colWidths=[48 * mm, W - 48 * mm], repeatRows=1)
t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), NAVY), ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#cccccc")), ("VALIGN", (0, 0), (-1, -1), "TOP"),
                       ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f6f8fb")])]))
S.append(t)

S.append(Heading("Reproducing the report", "h1", 1, "app_repro"))
S += bullets([
    "Environment: `python -m pip install -r requirements.txt` (pinned versions; see the file for the venv recipe).",
    "Run all analyses and rebuild: `python report/run_all.py`. Rebuild the PDF only: `python report/build_report.py`.",
    "Raw data stay on the data drive (`config.yaml`); derived data and all outputs are git-ignored and regenerable.",
    "To add an analysis: build a `grimace.report.Block` in the script (see the docstring in `src/grimace/report.py`), add the script to `report/run_all.py`, "
    "and, if it addresses a new question, add the question to `report/questions.yaml`.",
])
import platform  # noqa: E402
import numpy, pandas, scipy, statsmodels, sklearn, reportlab  # noqa: E402,F401
S.append(P(f"Software: Python {platform.python_version()}, numpy {numpy.__version__}, pandas {pandas.__version__}, scipy {scipy.__version__}, "
           f"statsmodels {statsmodels.__version__}, scikit-learn {sklearn.__version__}, matplotlib {matplotlib.__version__}, reportlab {reportlab.Version}.", "small"))

doc.multiBuild(S)
print("Report written to", PDF)
