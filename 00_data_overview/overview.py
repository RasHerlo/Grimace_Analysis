"""Data overview: what is on disk, how it maps to the overview Excel, and basic QC.

Run:  python 00_data_overview/overview.py
Outputs go to 00_data_overview/output/ (git-ignored).
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from grimace import build_manifest, load_all_frames, load_config, load_metadata

cfg = load_config()
AUS = cfg["action_units"]
out = Path(__file__).parent / "output"
out.mkdir(exist_ok=True)

meta = load_metadata(cfg)
man = build_manifest(cfg)

# ---------------------------------------------------------------- manifest vs Excel
print("== Recordings on disk ==")
print(man.groupby("session").size().to_string())
print("unparsed file names:", int((~man["parsed"]).sum()))
print("recordings with animal_no NOT in overview Excel:", int((~man["in_overview"]).sum()))
print(man.loc[~man["in_overview"], ["session", "id", "animal_no", "path"]].to_string())

per_mouse = man.pivot_table(index="animal_no", columns="session", values="uuid", aggfunc="count").fillna(0).astype(int)
missing = meta.set_index("animal_no").join(per_mouse).fillna({"baseline": 0, "stimulated": 0})
print("\n== Mice in Excel without exactly 1 baseline and 1 stimulated recording ==")
print(missing.loc[(missing["baseline"] != 1) | (missing["stimulated"] != 1), ["id", "baseline", "stimulated"]].to_string())

print("\n== Recordings with a note in the file name (e.g. repeats) ==")
print(man.loc[man["note"].fillna("") != "", ["session", "id", "animal_no", "note"]].to_string())

man.drop(columns=["path"]).to_csv(out / "manifest.csv", index=False)

# ---------------------------------------------------------------- frames / QC
frames = load_all_frames(cfg)
print(f"\n== Frames == {len(frames)} rows, {frames['recording_id'].nunique()} recordings")
print(frames.groupby("session").size().to_string())

conf_cols = [f"Confidence({a})" for a in AUS]
print("\nNaN per column:")
print(frames[AUS + ["Total Grimace Score"] + conf_cols].isna().sum().to_string())
print("\nAU Scored value counts:")
print(frames["AU Scored"].value_counts(dropna=False).to_string())
print("\nScore value counts per unit:")
for a in AUS:
    print(a, frames[a].value_counts(dropna=False).sort_index().to_dict())
print("\nConfidence summary:")
print(frames[conf_cols].describe().T.to_string())

# Does the total equal the sum of the four units?
sum4 = frames[AUS].sum(axis=1, min_count=1)
diff = (frames["Total Grimace Score"] - sum4).abs()
print("\nTotal == sum of 4 units in", f"{(diff < 1e-9).mean():.1%}", "of frames")
print("Rows where it differs:", frames.loc[diff >= 1e-9, ["AU Scored", "Total Grimace Score"]].assign(sum4=sum4).head(10).to_string())

rec = frames.groupby(["recording_id", "session"]).agg(
    n_frames=("Frame Index", "size"),
    n_valid=("Total Grimace Score", "count"),
    duration_s=("Timestamp(x)", "max"),
    mean_total=("Total Grimace Score", "mean"),
).reset_index()
rec.to_csv(out / "recordings_summary.csv", index=False)
print("\nFrames per recording:\n", rec.groupby("session")[["n_frames", "n_valid", "duration_s"]].describe().T.to_string())

# ---------------------------------------------------------------- plots
fig, axes = plt.subplots(1, 3, figsize=(13, 3.5))
for a in AUS:
    axes[0].hist(frames[f"Confidence({a})"].dropna(), bins=40, alpha=0.5, label=a)
axes[0].set(xlabel="model confidence", ylabel="frames", title="Confidence per unit")
axes[0].legend(fontsize=7)

counts = pd.DataFrame({a: frames[a].value_counts().sort_index() for a in AUS}).fillna(0)
counts.plot.bar(ax=axes[1], width=0.8)
axes[1].set(xlabel="score", ylabel="frames", title="Score distribution per unit")

axes[2].hist(frames["Total Grimace Score"].dropna(), bins=range(0, 10), align="left", rwidth=0.9)
axes[2].set(xlabel="total score", ylabel="frames", title="Total grimace score")
fig.tight_layout()
fig.savefig(out / "qc_distributions.png", dpi=150)
print("\nSaved outputs to", out)


# ---------------------------------------------------------------- report block D1
import numpy as np

from grimace import load_analysis_frames
from grimace.report import Block, use_style

use_style()
af = load_analysis_frames(cfg)
n_mice = len(meta)
n_baseline, n_stim = int((man["session"] == "baseline").sum()), int((man["session"] == "stimulated").sum())
extra = man.loc[~man["in_overview"]]
cc = af[af["complete"]]
total_ok = bool(((cc["Total Grimace Score"] - cc[AUS].sum(axis=1)).abs() < 1e-9).all())
n_exact = int(((per_mouse["baseline"] == 1) & (per_mouse["stimulated"] == 1)).reindex(meta["animal_no"]).fillna(False).sum())

blk = Block("D1_overview", ["D1"], "Dataset overview: mice, recordings and score distributions", __file__, order=0)
blk.takeaway(f"{n_mice} mice (2x2: genotype x drug, {int((meta['sex'] == 'Female').sum())} female / {int((meta['sex'] == 'Male').sum())} male, "
             f"{meta['cohort'].nunique()} cohorts) each with one baseline and one stimulated recording of about {af.groupby('recording_id')['t'].max().median() / 60:.0f} minutes, scored once per second on four action units.")
blk.question("What data exist, how do they map to the overview Excel, and what do the raw scores look like?")
blk.data(f"{n_baseline} baseline and {n_stim} stimulated recordings (zip files with a per-frame CSV) were found. {n_exact} of the {n_mice} mice in the overview Excel have exactly one of each. "
         f"{len(extra)} recordings from {extra['animal_no'].nunique()} animals that are **not** in the Excel are excluded from all analyses "
         f"(animal numbers {', '.join(str(int(a)) for a in sorted(extra['animal_no'].unique()))}); whether they were excluded on purpose is an open data question. "
         f"After exclusion: {af['recording_id'].nunique()} recordings, {len(af):,} frames.")
blk.method([
    "The Excel overview gives mouse metadata (genotype, drug, treatment code, sex, cohort, weight). Recordings are matched to mice by the animal number in the file name.",
    "Each recording contains one row per analysed video frame (every 50th frame = 1 s), with a 0-2 score and a model confidence for Orbital, Nose, Ears and Whiskers (Cheek is not scored in this dataset), "
    "and the Total Grimace Score. Score -1 means 'not scorable' and is treated as missing in all analyses (see D2).",
    "The distributions below are over all frames of the 71 mice (scores) or over complete frames (total score).",
])
g = meta.groupby(["genotype", "drug"]).agg(mice=("animal_no", "size"), females=("sex", lambda s: int((s == "Female").sum())), males=("sex", lambda s: int((s == "Male").sum())),
                                          cohorts=("cohort", "nunique"), weight=("weight", "mean")).reset_index()
g["weight"] = g["weight"].map("{:.1f}".format)
g.insert(2, "treatment code", [int(meta[(meta["genotype"] == a) & (meta["drug"] == b)]["treatment"].iloc[0]) for a, b in zip(g["genotype"], g["drug"])])
g.columns = ["genotype", "drug", "treatment code", "mice", "females", "males", "cohorts", "mean weight (g)"]
blk.table(g, "Study design: number of mice per group.", "Treatment code: 1 = Cre- VEH, 2 = Cre- LEV, 3 = Cre+ VEH, 4 = Cre+ LEV. Groups are balanced within about one mouse.")

fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
sh = pd.DataFrame({a: af[a].value_counts(dropna=False, normalize=True) for a in AUS}).fillna(0)
sh = sh.rename(index={np.nan: "not scorable (-1)"})
sh = sh.loc[[i for i in [0.0, 1.0, 2.0, "not scorable (-1)"] if i in sh.index]]
x = np.arange(len(AUS))
colors_ = {0.0: "#bdc3c7", 1.0: "#f39c12", 2.0: "#c0392b", "not scorable (-1)": "#2c3e50"}
bottom = np.zeros(len(AUS))
for idx, r in sh.iterrows():
    ax[0].bar(x, r.to_numpy() * 100, bottom=bottom, color=colors_[idx], label=f"score {int(idx)}" if idx != "not scorable (-1)" else idx)
    bottom += r.to_numpy() * 100
ax[0].set_xticks(x, AUS)
ax[0].set(ylabel="% of frames", title="Score distribution per unit")
ax[0].legend(fontsize=7, loc="center left", bbox_to_anchor=(1.0, 0.5))
ax[1].hist(cc["Total Grimace Score"], bins=np.arange(-0.5, 9.5, 1), color="#2980b9", rwidth=0.85)
ax[1].set(xlabel="Total Grimace Score", ylabel="complete frames", title="Total score")
ax[2].hist(af.groupby("recording_id").size(), bins=25, color="#7f8c8d")
ax[2].set(xlabel="frames (seconds) per recording", ylabel="recordings", title="Recording length")
fig.tight_layout()
blk.figure(fig, "What the raw scores look like.", "Left: share of frames at each score per unit (dark = not scorable). Nose is only ever scored 0 or 2. Middle: total score of complete frames. Right: how many one-second frames each recording has.")
plt.close(fig)

conf_ok = {a: af.loc[~af[f"miss_{a}"], f"Confidence({a})"].mean() for a in AUS}
blk.findings([
    "Score levels: " + "; ".join(f"{a}: " + ", ".join(f"{int(k)}={v:.0%}" for k, v in af[a].value_counts(normalize=True).sort_index().items()) for a in AUS) + " (of scorable frames).",
    f"Nose takes only the values {sorted(int(v) for v in af['Nose'].dropna().unique())}; all other units use 0, 1 and 2.",
    f"The Total Grimace Score equals the sum of the four units in {'all' if total_ok else 'NOT all'} complete frames and is undefined (NaN) in the {1 - af['complete'].mean():.0%} of frames where any unit is unscorable.",
    f"Recordings contain {af.groupby('recording_id').size().min()}-{af.groupby('recording_id').size().max()} frames (median {int(af.groupby('recording_id').size().median())}).",
    "Mean model confidence of scorable frames: " + ", ".join(f"{a} {v:.2f}" for a, v in conf_ok.items()) + ".",
])
blk.interpretation([
    "The scores are coarse (3 levels, Nose 2 levels) and skewed towards 0-1, so the total score takes only nine values and most single-frame information is in *which* units are raised, not in how much.",
    "With 71 mice (about 18 per group) the mouse, not the frame, is the unit that limits statistical power; frames are only useful for within-mouse structure.",
])
blk.caveats(["Group sizes are for the 71 mice in the Excel; the 10 additional animals are not analysed (see open data questions)."])
blk.open_points([
    "Excel and disk: are the 11 extra recordings (10 animals) intentionally excluded?",
    "Unit of analysis (per-mouse mean, per-frame, or both): per-frame structure is used for state analyses, per-mouse summaries for group comparisons - confirm this split.",
])
blk.save()
