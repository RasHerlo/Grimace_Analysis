"""Collect the content of one analysis (a "block") for the running PDF report.

Every analysis script builds one or more `Block`s and calls `.save()`. The report builder
(`report/build_report.py`) assembles all saved blocks, ordered by the research question they answer.
Numbers in the text should be computed in the script (f-strings), never typed by hand, so that the
report cannot go stale.

    blk = Block("A1_patterns", questions=["A1", "Q3"], title="Which 4-unit patterns occur?", script=__file__)
    blk.takeaway("One-sentence answer shown at the top and in the status overview.")
    blk.data("Which frames / mice were used.")
    blk.method(["Step 1 ...", "Step 2 ..."])
    blk.figure(fig, "Caption.", how_to_read="Explain axes, colours, what a 'good' result would look like.")
    blk.table(df, "Caption.")
    blk.findings([...]); blk.interpretation([...]); blk.caveats([...]); blk.open_points([...])
    blk.save()

The first entry of `questions` is the primary question (the block is printed there in full); the others
get a "see also" link.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from datetime import datetime
from pathlib import Path

import pandas as pd

from .config import repo_root

STATUSES = ("preliminary", "final")


def use_style():
    """Common matplotlib look for all report figures."""
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.size": 9, "axes.titlesize": 10, "axes.titleweight": "bold", "axes.labelsize": 9,
        "legend.fontsize": 8, "legend.frameon": False, "axes.spines.top": False, "axes.spines.right": False,
        "figure.dpi": 100, "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.6,
    })


TREATMENT_LABELS = {1: "Cre- VEH", 2: "Cre- LEV", 3: "Cre+ VEH", 4: "Cre+ LEV"}
TREATMENT_COLORS = {1: "#7f8c8d", 2: "#e67e22", 3: "#2980b9", 4: "#c0392b"}


def content_dir() -> Path:
    d = repo_root() / "report" / "output" / "content"
    d.mkdir(parents=True, exist_ok=True)
    return d


def git_commit() -> str:
    try:
        sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=repo_root(), capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain"], cwd=repo_root(), capture_output=True, text=True, check=True).stdout.strip()
        return sha + (" + uncommitted changes" if dirty else "")
    except Exception:
        return "unknown"


class Block:
    def __init__(self, block_id: str, questions: list[str], title: str, script: str, status: str = "preliminary", order: float = 0.0):
        assert status in STATUSES
        self.id = block_id
        self.dir = content_dir() / block_id
        if self.dir.exists():
            shutil.rmtree(self.dir)  # a re-run replaces the old block completely
        self.dir.mkdir(parents=True)
        self.meta = {
            "id": block_id, "questions": questions, "title": title, "status": status, "order": order,
            "script": str(Path(script).resolve().relative_to(repo_root())).replace("\\", "/"),
        }
        self.items: list[dict] = []
        self.sections: dict[str, object] = {}
        self._nfig = 0
        self._ntab = 0

    # ------------------------------------------------------------------ text parts
    def takeaway(self, text: str):
        """One or two sentences: the current answer to the question."""
        self.sections["takeaway"] = text

    def question(self, text: str):
        """What exactly this analysis asks (more specific than the research question)."""
        self.sections["question"] = text

    def data(self, text: str):
        """Which frames / recordings / mice went into the analysis, and what was excluded."""
        self.sections["data"] = text

    def method(self, steps: list[str]):
        """Numbered, reproducible description incl. parameters, null models and software choices."""
        self.sections["method"] = steps

    def findings(self, bullets: list[str]):
        """Plain statements of what the numbers/figures show (no interpretation)."""
        self.sections["findings"] = bullets

    def interpretation(self, bullets: list[str]):
        """What we conclude from the findings, and explicitly what we do NOT conclude."""
        self.sections["interpretation"] = bullets

    def caveats(self, bullets: list[str]):
        self.sections["caveats"] = bullets

    def open_points(self, bullets: list[str]):
        """Decisions or checks needed before going on. Collected in the 'decision queue' of the report."""
        self.sections["open_points"] = bullets

    # ------------------------------------------------------------------ figures / tables, in reading order
    def figure(self, fig, caption: str, how_to_read: str = "", dpi: int = 170, tall: bool = False):
        """`tall=True` lets a figure use most of a page (for multi-row panels); default is at most ~105 mm high."""
        self._nfig += 1
        fname = f"fig{self._nfig}.png"
        fig.savefig(self.dir / fname, dpi=dpi, bbox_inches="tight", facecolor="white")
        self.items.append({
            "type": "figure", "file": fname, "label": f"Figure {self.id.split('_')[0]}.{self._nfig}",
            "caption": caption, "how_to_read": how_to_read, "tall": tall,
        })

    def table(self, df: pd.DataFrame, caption: str, how_to_read: str = "", floatfmt: str = "{:.3g}", index: bool = False):
        self._ntab += 1
        d = df.reset_index() if index else df
        rows = [[floatfmt.format(v) if isinstance(v, float) else str(v) for v in r] for r in d.itertuples(index=False)]
        self.items.append({
            "type": "table", "label": f"Table {self.id.split('_')[0]}.{self._ntab}", "caption": caption,
            "how_to_read": how_to_read, "columns": [str(c) for c in d.columns], "rows": rows,
        })

    def text(self, text: str):
        self.items.append({"type": "text", "text": text})

    # ------------------------------------------------------------------ save
    def save(self):
        self.meta.update({"created": datetime.now().isoformat(timespec="seconds"), "commit": git_commit()})
        (self.dir / "block.json").write_text(json.dumps({"meta": self.meta, "sections": self.sections, "items": self.items}, indent=1), encoding="utf-8")
        print(f"[report] saved block '{self.id}' ({self._nfig} figures, {self._ntab} tables) -> {self.dir}")
