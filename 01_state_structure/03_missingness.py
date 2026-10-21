"""Step 3: is the `-1` ("not scorable") missingness random, or does it depend on state, time, group or confidence?

Report block produced: D2 (also Q6)

Run:  python 01_state_structure/03_missingness.py      (then: python report/build_report.py)
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

from grimace import load_analysis_frames, load_config
from grimace.report import TREATMENT_COLORS, TREATMENT_LABELS, Block, use_style

use_style()
cfg = load_config()
AUS = cfg["action_units"]
MISS = [f"miss_{a}" for a in AUS]
SESS = ["baseline", "stimulated"]

df = load_analysis_frames(cfg)
df["tbin"] = (df["t"] // 60).astype(int)
n_frames, n_rec, n_mice = len(df), df["recording_id"].nunique(), df["animal_no"].nunique()
full_minutes = (df.groupby("recording_id")["t"].max() + 1) // 60
last_full_bin = int(max(b for b in range(20) if (full_minutes > b).mean() >= 0.9))  # last minute that >= 90% of recordings cover completely

# ------------------------------------------------------------------ how much / which units together
by_sess = df.groupby("session")[MISS].mean()
n_miss = df[MISS].sum(axis=1)
assert ((4 - n_miss) == df["AU Scored"]).all(), "AU Scored != number of scorable units"
combo = df[MISS].apply(lambda r: "+".join(a for a, m in zip(AUS, r) if m) or "none", axis=1)
combo_share = combo.value_counts(normalize=True)
phi = df[MISS].astype(float).corr()
phi.index = phi.columns = AUS
share_incomplete = 1 - df["complete"].mean()

# ------------------------------------------------------------------ per recording / group
rec = df.groupby(["recording_id", "animal_no", "session", "treatment", "drug"])[["complete"]].mean().reset_index()
rec["inc"] = 1 - rec["complete"]
kw = {s: stats.kruskal(*[g["inc"] for _, g in d.groupby("treatment")]) for s, d in rec.groupby("session")}
stim = rec[rec["session"] == "stimulated"]
lev_vs_veh = stats.mannwhitneyu(stim.loc[stim["drug"] == stim["drug"].unique()[0], "inc"], stim.loc[stim["drug"] == stim["drug"].unique()[1], "inc"])
drug_levels = list(stim["drug"].unique())
med_by_drug = {d: stim.loc[stim["drug"] == d, "inc"].median() for d in drug_levels}
wide = rec.pivot(index="animal_no", columns="session", values="inc").dropna()
wil = stats.wilcoxon(wide["baseline"], wide["stimulated"])

# ------------------------------------------------------------------ temporal clustering
rows = []
for a in AUS:
    m = df[f"miss_{a}"]
    nxt = df.groupby("recording_id")[f"miss_{a}"].shift(-1)
    ok = nxt.notna()
    rows.append({"unit": a, "P(unscorable)": m.mean(), "P(unscorable next s | unscorable now)": nxt[ok & m].mean()})
clust = pd.DataFrame(rows)

# ------------------------------------------------------------------ does missingness depend on the other units' scores?
rows, curve = [], {}
for a in AUS:
    others = [b for b in AUS if b != a]
    d = df[df[others].notna().all(axis=1)].copy()
    d["osum"] = d[others].sum(axis=1)
    d["o_between"] = d.groupby("recording_id")["osum"].transform("mean")
    d["o_within"] = d["osum"] - d["o_between"]
    curve[a] = d.groupby("osum")[f"miss_{a}"].agg(["mean", "size"])
    X = sm.add_constant(d[["o_within", "o_between"]])
    res = sm.GEE(d[f"miss_{a}"].astype(float), X, groups=d["recording_id"], family=sm.families.Binomial(), cov_struct=sm.cov_struct.Independence()).fit()
    ci = res.conf_int()
    for term in ["o_within", "o_between"]:
        rows.append({"unit": a, "term": term, "OR": np.exp(res.params[term]), "lo": np.exp(ci.loc[term, 0]), "hi": np.exp(ci.loc[term, 1]), "p": res.pvalues[term], "n_frames": len(d), "n_events": int(d[f"miss_{a}"].sum())})
gee = pd.DataFrame(rows)

# ------------------------------------------------------------------ confidence
rows = []
for a in AUS:
    raw = df[a].fillna(-1)
    c = df[f"Confidence({a})"]
    for s in [-1, 0, 1, 2]:
        sel = raw == s
        if sel.sum():
            rows.append({"unit": a, "score": s, "n": int(sel.sum()), "mean_conf": c[sel].mean(), "share_lt_0.8": (c[sel] < 0.8).mean()})
conf = pd.DataFrame(rows)
conf_scorable = pd.DataFrame({a: df.loc[~df[f"miss_{a}"], f"Confidence({a})"] for a in AUS})
low_conf = {th: (conf_scorable < th).mean() for th in (0.5, 0.8, 0.95)}
conf_missing_max = max(df.loc[df[f"miss_{a}"], f"Confidence({a})"].max() for a in AUS)

# ================================================================== block
blk = Block("D2_unscorable", ["D2", "Q6"], "Unscorable frames (score -1) and model confidence", __file__, order=2)
w = gee[gee["term"] == "o_within"].set_index("unit")
informative = [a for a in AUS if w.loc[a, "lo"] > 1]
blk.takeaway(f"{share_incomplete:.0%} of frames have at least one unscorable unit, and this is **not random**: for {', '.join(informative)} the chance of being unscorable rises by "
             f"{(w.loc[informative, 'OR'].min() - 1):.0%}-{(w.loc[informative, 'OR'].max() - 1):.0%} per point of the other units' summed score within a recording, so dropping incomplete frames under-represents high-score moments.")
blk.question("How much of the data cannot be scored (`-1`), which units fail together, does it depend on time, group or on the scores of the other units, and what does the model confidence tell about the scores?")
blk.data(f"{n_mice} mice, {n_rec} recordings, {n_frames:,} frames. Score `-1` is taken as 'not scorable' (treated as missing). The model's own `AU Scored` column equals the number of scorable units in every frame (checked).")
blk.method([
    "Share of frames with `-1` per unit and session; which units are unscorable together (combinations) and the correlation (phi) between the 'unscorable' indicators.",
    "Per recording: fraction of incomplete frames (>= 1 unit unscorable). Groups compared with Kruskal-Wallis (4 treatment groups, per session), Mann-Whitney (Levcromakalim vs vehicle, stimulated session) "
    "and a paired Wilcoxon test (baseline vs stimulated, per mouse).",
    "Temporal clustering: probability that a unit is unscorable in the next second given that it is unscorable now, compared with the overall probability.",
    "**Is it random?** For each unit, among frames where the other three units are scorable: logistic regression (GEE, independence working correlation, robust standard errors clustered on recording) of "
    "'this unit is unscorable' on the summed score of the other three units. The predictor is split into (a) its deviation from the recording's mean (within recording: does the unit drop out when the others are high *right now*?) "
    "and (b) the recording's mean (between recordings: do high-scoring recordings have more dropout?). Odds ratios are per +1 point of the summed score.",
    "Confidence: mean model confidence and share of frames with confidence < 0.8, split by the assigned score (including `-1`).",
])

fig, ax = plt.subplots(1, 2, figsize=(11, 3.6), gridspec_kw={"width_ratios": [1, 1.3]})
x = np.arange(4)
for k, s in enumerate(SESS):
    ax[0].bar(x + (k - 0.5) * 0.38, by_sess.loc[s].to_numpy() * 100, 0.36, label=s, color=["#2980b9", "#c0392b"][k])
ax[0].set_xticks(x, AUS)
ax[0].set(ylabel="% of frames unscorable (-1)", title="Unscorable frames per unit")
ax[0].legend()
top = combo_share.drop("none").head(8)[::-1]
ax[1].barh(top.index, top.values * 100, color="#34495e")
ax[1].set(xlabel="% of all frames", title=f"Which units are unscorable together (no unit missing: {combo_share['none']:.0%})")
fig.tight_layout()
blk.figure(fig, f"Unscorable frames. {share_incomplete:.0%} of frames lack at least one unit.",
           "Left: share of frames with score -1 per unit (Whiskers is the most often unscorable). Right: the most frequent combinations of unscorable units; "
           "a long bar for 'Orbital+Nose+Ears+Whiskers' means the whole face was not scored (face not found), a bar for a single unit means only that unit failed.")
plt.close(fig)

fig, ax = plt.subplots(1, 3, figsize=(13, 3.9), constrained_layout=True)
tm = df[df["tbin"] <= last_full_bin].groupby(["session", "tbin"])[MISS].mean().reset_index()
for i, (a, m) in enumerate(zip(AUS, MISS)):
    for s, ls in zip(SESS, ["-", "--"]):
        d = tm[tm["session"] == s]
        ax[0].plot(d["tbin"], d[m] * 100, ls, color=f"C{i}", label=f"{a}" if s == "baseline" else None)
ax[0].set(xlabel="minute of session", ylabel="% unscorable", title="Over time (solid baseline, dashed stimulated)")
ax[0].legend(fontsize=7)
pos, labels = [], []
for k, s in enumerate(SESS):
    for tr in (1, 2, 3, 4):
        d = rec[(rec["session"] == s) & (rec["treatment"] == tr)]["inc"] * 100
        p = k * 5 + tr - 1
        bp = ax[1].boxplot([d], positions=[p], widths=0.7, patch_artist=True, showfliers=False)
        bp["boxes"][0].set(facecolor=TREATMENT_COLORS[tr], alpha=0.5)
        ax[1].plot(p + np.random.default_rng(p).uniform(-0.2, 0.2, len(d)), d, ".", color="k", ms=3, alpha=0.5)
        pos.append(p)
        labels.append(TREATMENT_LABELS[tr])
ax[1].set_xticks(pos, labels, rotation=60, ha="right", fontsize=7)
ax[1].set(ylabel="% incomplete frames per recording", title="By group (left baseline, right stimulated)")
ax[2].plot(wide["baseline"] * 100, wide["stimulated"] * 100, "o", ms=4, alpha=0.7)
lim = [0, max(wide.max()) * 100 * 1.05]
ax[2].plot(lim, lim, "k--", lw=0.8)
ax[2].set(xlabel="baseline: % incomplete frames", ylabel="stimulated: % incomplete frames", title="Same mouse, two sessions")
fig.tight_layout()
blk.figure(fig, "Does the amount of unscorable data depend on time in the session, on group, or on session?",
           "Left: fraction of unscorable frames per minute. Middle: one dot per recording, boxes = quartiles, by treatment group and session. Right: each dot is one mouse; "
           "dots below the dashed diagonal have fewer incomplete frames in the stimulated than in the baseline recording.")
plt.close(fig)

fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
for i, a in enumerate(AUS):
    c = curve[a][curve[a]["size"] >= 300]
    ax[0].plot(c.index, c["mean"] * 100, "o-", ms=4, label=a, color=f"C{i}")
ax[0].set(xlabel="summed score of the other three units (only values with >= 300 frames)", ylabel="% of frames where the unit is unscorable", title="Dropout vs the other units' scores")
ax[0].text(0.02, 0.97, "frames where the other three units are scorable;\nOrbital/Ears are rarely unscorable on their own", transform=ax[0].transAxes, va="top", fontsize=7, color="0.35")
ax[0].legend()
yy = 0
ticks, tl = [], []
for i, a in enumerate(AUS):
    for term, col, off_ in [("o_within", "#c0392b", 0.12), ("o_between", "#7f8c8d", -0.12)]:
        r = gee[(gee["unit"] == a) & (gee["term"] == term)].iloc[0]
        ax[1].errorbar(r["OR"], i + off_, xerr=[[r["OR"] - r["lo"]], [r["hi"] - r["OR"]]], fmt="o", color=col, capsize=3,
                       label=("within recording" if term == "o_within" else "between recordings") if i == 0 else None)
ax[1].axvline(1, color="k", lw=0.8)
ax[1].set_xscale("log")
ax[1].set_yticks(range(4), AUS)
ax[1].invert_yaxis()
ax[1].set(xlabel="odds ratio of being unscorable per +1 in the others' summed score (log scale)", title="Is dropout independent of the other units?")
ax[1].legend()
fig.tight_layout()
blk.figure(fig, "Missingness is related to the other units' scores. Left: raw dropout rate by the summed score of the other three units. Right: odds ratios from the logistic model, split into within- and between-recording parts, with 95% intervals.",
           "Left: a rising line means the unit is more often unscorable when the others score high. Right: dots right of the vertical line (odds ratio > 1) = more dropout when the others are higher; "
           "an interval crossing 1 means no reliable relation. If dropout were independent of the face's state, all intervals would include 1.")
plt.close(fig)

fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
for i, a in enumerate(AUS):
    d = conf[(conf["unit"] == a) & (conf["score"] >= 0)]
    ax[0].plot(d["score"], d["mean_conf"], "o-", color=f"C{i}", label=a)
    ax[1].plot(d["score"], d["share_lt_0.8"] * 100, "o-", color=f"C{i}", label=a)
for a_ in ax:
    a_.set_xticks([0, 1, 2])
    a_.set_xlabel("assigned score")
ax[0].set(ylabel="mean model confidence", title="Confidence by score")
ax[1].set(ylabel="% of frames with confidence < 0.8", title="Low-confidence frames by score")
ax[0].legend()
fig.tight_layout()
blk.figure(fig, "Model confidence of scorable frames by assigned score. (Unscorable frames always have confidence 0 and are not shown.)",
           "Lines that fall with the score mean that elevated scores (1-2) are assigned with less confidence than 0. Nose has no score-1 frames (only 0 and 2).")
plt.close(fig)

t_inc = pd.DataFrame([{"session": s, "group": TREATMENT_LABELS[tr], "recordings": len(d), "mean % incomplete": f"{d['inc'].mean() * 100:.1f}", "range %": f"{d['inc'].min() * 100:.1f}-{d['inc'].max() * 100:.1f}"}
                      for (s, tr), d in rec.groupby(["session", "treatment"])])
blk.table(t_inc, "Fraction of incomplete frames per recording, by session and treatment group.", "One value per recording; 'range' is the lowest and highest recording.")
tg = gee.copy()
tg = pd.DataFrame({"unit": tg["unit"], "predictor": tg["term"].map({"o_within": "others' sum, within recording", "o_between": "others' sum, recording mean"}),
                   "odds ratio": tg["OR"].map("{:.2f}".format), "95% CI": [f"{a:.2f}-{b:.2f}" for a, b in zip(tg["lo"], tg["hi"])], "p": tg["p"].map("{:.2g}".format), "frames": tg["n_frames"].map("{:,}".format), "unscorable events": tg["n_events"].map("{:,}".format)})
blk.table(tg, "Logistic (GEE) models of 'unit unscorable' on the other units' summed score.", "Odds ratio > 1 with a CI above 1 = the unit is more often unscorable when the others score higher. 'Within recording' compares seconds of the same recording; 'recording mean' compares recordings.")

lc = lambda th, a: low_conf[th][a]  # noqa: E731
bt = gee[gee["term"] == "o_between"].set_index("unit")
between_sig = [a for a in AUS if bt.loc[a, "hi"] < 1 or bt.loc[a, "lo"] > 1]
lowest_conf = {a: int(conf[(conf["unit"] == a) & (conf["score"] >= 0)].sort_values("mean_conf").iloc[0]["score"]) for a in AUS}
blk.findings([
    f"**Amount.** {share_incomplete:.1%} of frames have at least one unscorable unit. Per unit: " + ", ".join(f"{a} {df[f'miss_{a}'].mean():.1%}" for a in AUS) + ". "
    f"Whiskers is typically unscorable on its own ({combo_share.get('Whiskers', 0):.1%} of frames); {combo_share.get('Orbital+Nose+Ears+Whiskers', 0):.1%} of frames have all four units unscorable (whole face).",
    f"The 'unscorable' indicators are correlated (phi {phi.to_numpy()[np.triu_indices(4, 1)].min():.2f}-{phi.to_numpy()[np.triu_indices(4, 1)].max():.2f}); the strongest pair is Orbital-Ears ({phi.loc['Orbital', 'Ears']:.2f}).",
    "**Clustering in time.** " + "; ".join(f"{r.unit}: {r._2:.1%} overall vs {r._3:.0%} in the next second if unscorable now" for r in clust.rename(columns={"P(unscorable)": "_2", "P(unscorable next s | unscorable now)": "_3"}).itertuples()) + " - dropout comes in runs, not as isolated frames.",
    f"**Groups/sessions.** Incomplete frames per recording: baseline median {wide['baseline'].median():.1%}, stimulated {wide['stimulated'].median():.1%} (paired Wilcoxon p = {wil.pvalue:.2g}). "
    f"Groups differ in the stimulated session (Kruskal-Wallis p = {kw['stimulated'].pvalue:.3g}; medians by drug: " + ", ".join(f"{d} {v:.1%}" for d, v in med_by_drug.items()) + f"; Mann-Whitney p = {lev_vs_veh.pvalue:.2g}) but not at baseline (p = {kw['baseline'].pvalue:.2g}).",
    "**Dependence on state.** Within a recording, the odds of being unscorable increase per +1 in the other units' summed score by: " + ", ".join(f"{a} {w.loc[a, 'OR']:.2f} ({w.loc[a, 'lo']:.2f}-{w.loc[a, 'hi']:.2f})" for a in AUS) + ". "
    + ("Orbital shows no relation." if "Orbital" not in informative else ""),
    f"**Confidence.** Unscorable frames have confidence {conf_missing_max:.1f} by construction. Among scorable frames {lc(0.5, 'Orbital'):.1%}/{lc(0.5, 'Nose'):.1%}/{lc(0.5, 'Ears'):.1%}/{lc(0.5, 'Whiskers'):.1%} (Orbital/Nose/Ears/Whiskers) have confidence < 0.5 and "
    f"{lc(0.8, 'Orbital'):.0%}/{lc(0.8, 'Nose'):.0%}/{lc(0.8, 'Ears'):.0%}/{lc(0.8, 'Whiskers'):.0%} have confidence < 0.8. The score with the lowest mean confidence is " + ", ".join(f"{a} {s}" for a, s in lowest_conf.items()) + ".",
    "**Between recordings.** " + (", ".join(f"{a} OR {bt.loc[a, 'OR']:.2f} ({bt.loc[a, 'lo']:.2f}-{bt.loc[a, 'hi']:.2f})" for a in between_sig) + " per +1 in the recording's mean of the others' sum."
                                 if between_sig else "No unit shows a reliable relation between a recording's mean score and its dropout."),
])
blk.interpretation([
    f"Dropout is **not missing completely at random**: {len(informative)} of 4 units ({', '.join(informative)}) drop out more when the rest of the face scores high. A plausible cause (not tested here) is that grimacing changes head/face position or "
    "appearance so the model cannot score a unit. In either case, analysing complete frames only systematically removes some of the highest-score moments, so effects of drug/state may be **underestimated** and "
    "pattern frequencies (A1, A4) are biased towards low scores.",
    f"Dropout also differs between treatment groups during stimulation (Kruskal-Wallis p = {kw['stimulated'].pvalue:.2g}), so complete-case comparisons between groups are not on equal footing: groups differ in *which* frames survive.",
    "The between-recording relation has a different sign from the within-recording one for some units (recordings with higher mean scores have *less* dropout). That is not a contradiction: "
    "it can arise from mice or sessions with difficult recording conditions, and it is not evidence about states. The within-recording comparison is the relevant one for the state analysis.",
    "The confidence column tells which frames the model is less sure about, but low confidence is not what makes a unit unscorable (those have confidence 0 by definition), so it cannot be used to *predict* the unscorable frames.",
])
ev = gee[gee["term"] == "o_within"].set_index("unit")["n_events"]
blk.caveats([
    "The dependence analysis conditions on the other three units being scorable, which removes the 'whole-face' dropouts (probably a different mechanism: face not found). "
    "It therefore addresses *partial* dropout only, and for Orbital and Ears there are few such events left (" + ", ".join(f"{a}: {ev[a]:,}" for a in AUS) + " unscorable frames), "
    "so their estimates are less reliable than those of Nose and Whiskers. Whether whole-face dropouts depend on the state before and after them is not examined.",
    "We do not know *why* the model returns -1 (face not detected, occlusion, pose); the interpretation above is a hypothesis.",
    "GEE odds ratios are per summed-score point and assume a linear effect on the log-odds scale; the left panel of the figure shows the raw rates for a check of that assumption.",
    "Tests across groups are not corrected for multiple testing and are descriptive.",
    "The Excel 'Counted frames' (an open data question) use an unknown confidence rule; the share of scorable frames reported here is therefore not directly comparable with them.",
])
blk.open_points([
    "How to handle `-1` in the state analysis: model as missing (e.g. a latent-class model that uses only the observed units in each frame, assuming the unit is missing at random *given the other units*), or sensitivity analyses with and without incomplete frames?",
    "Should frames with all four units unscorable (face not found) be treated differently from frames where only one unit fails?",
    "Can the cause of `-1` be checked (a few videos, or the model's raw output in the acquisition JSON)?",
])
blk.save()
