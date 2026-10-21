"""Step 1: which 4-unit patterns occur, and do the units covary beyond what independence predicts?

Report blocks produced:
  A1  over/under-represented patterns       (also Q3)
  Q3  covariation between the four units    (also A4)
  A4  what the total score discards         (also Q2)

Run:  python 01_state_structure/01_patterns.py      (then: python report/build_report.py)
"""

from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from grimace import load_analysis_frames, load_config
from grimace.report import Block, use_style

use_style()
cfg = load_config()
AUS = cfg["action_units"]
rng = np.random.default_rng(0)
N_SHUFFLE, N_BOOT, MIN_EXP = 200, 1000, 50

df = load_analysis_frames(cfg)
cc = df[df["complete"]].reset_index(drop=True)
n_frames, n_cc = len(df), len(cc)
n_rec = df["recording_id"].nunique()
n_mice = df["animal_no"].nunique()

X = cc[AUS].to_numpy().astype(int)
K = int(X.max()) + 1
assert X.min() == 0 and K == 3
n_pat = K ** len(AUS)
POW = K ** np.arange(len(AUS))
codes = X @ POW
rec_ids = cc["recording_id"].to_numpy()
rec_names = list(dict.fromkeys(rec_ids))
rec_idx = [np.flatnonzero(rec_ids == r) for r in rec_names]
rec_mouse = np.array([cc["animal_no"].to_numpy()[i[0]] for i in rec_idx])
mouse_ids = np.unique(rec_mouse)
all_pat = np.arange(n_pat)
digits = np.stack([(all_pat // K**i) % K for i in range(len(AUS))], axis=1)  # (81, 4)
pat_total = digits.sum(1)
label = lambda c: "-".join(str(d) for d in digits[c])  # noqa: E731
DATA_TXT = (f"{n_mice} mice in the overview Excel, {n_rec} recordings (baseline + stimulated), {n_frames:,} frames at 1 s spacing. "
            f"Only **complete frames** (all four units scorable) are used here: {n_cc:,} frames ({n_cc / n_frames:.1%}). "
            "Incomplete frames are excluded for now; whether that biases the result is examined in D2.")

# ================================================================== A1: pattern table with two independence baselines
obs_r = np.stack([np.bincount(codes[i], minlength=n_pat) for i in rec_idx]).astype(float)  # (rec, 81)
exp_r = np.zeros_like(obs_r)
for r, i in enumerate(rec_idx):
    p_r = np.stack([np.bincount(X[i, u], minlength=K) / len(i) for u in range(len(AUS))])
    exp_r[r] = len(i) * np.prod(p_r[np.arange(len(AUS)), digits], axis=1)
obs, exp_within = obs_r.sum(0), exp_r.sum(0)
p_pool = np.stack([np.bincount(X[:, u], minlength=K) / n_cc for u in range(len(AUS))])
exp_pool = n_cc * np.prod(p_pool[np.arange(len(AUS)), digits], axis=1)

tab = pd.DataFrame({"pattern": [label(c) for c in all_pat], "total": pat_total, "n": obs.astype(int), "freq": obs / n_cc,
                    "exp_pooled": exp_pool, "exp_within": exp_within})
tab["ratio_within"] = tab["n"] / tab["exp_within"]
tab["z_within"] = (tab["n"] - tab["exp_within"]) / np.sqrt(tab["exp_within"])

# bootstrap over MICE (both recordings of a mouse move together): CI of the observed/expected ratio
mouse_of_row = np.searchsorted(mouse_ids, rec_mouse)
boot = np.empty((N_BOOT, n_pat))
for b in range(N_BOOT):
    w = np.bincount(rng.integers(0, len(mouse_ids), len(mouse_ids)), minlength=len(mouse_ids))[mouse_of_row]
    boot[b] = (w @ obs_r) / np.maximum(w @ exp_r, 1e-9)
tab["ratio_lo"], tab["ratio_hi"] = np.percentile(boot, [2.5, 97.5], axis=0)
n_seen = int((obs > 0).sum())
n90 = int((np.cumsum(np.sort(obs)[::-1]) / n_cc < 0.9).sum() + 1)

ok = tab[tab["exp_within"] >= MIN_EXP].copy()
ok["l2"] = np.log2(ok["ratio_within"])
over = ok.sort_values("l2", ascending=False).head(12)
under = ok.sort_values("l2").head(12)
sig_over = ok[(ok["ratio_lo"] > 1)]
sig_under = ok[(ok["ratio_hi"] < 1)]

blk = Block("A1_patterns", ["A1", "Q3"], "Which 4-unit patterns are over- or under-represented?", __file__, order=1)
zero = tab.set_index("pattern").loc["0-0-0-0"]
allhi = tab.set_index("pattern").loc["2-2-1-1"]
max_over = sig_over["ratio_within"].max()
blk.takeaway(f"Patterns are not combinations of independent units: the all-zero pattern occurs {zero['ratio_within']:.1f}x and several multi-unit co-activations up to {max_over:.1f}x as often as expected, "
             f"while isolated single-unit activations occur less often than expected - also after accounting for each recording's own score levels.")
blk.question("If the four units fluctuated independently of each other, which of the 54 possible patterns would we expect how often? Which patterns deviate, and in which direction?")
blk.data(DATA_TXT + f" Nose is only ever scored 0 or 2 (never 1), so 3x2x3x3 = {int(((tab['n']>0)).sum())} of the 81 coded patterns are possible, and all of them occur.")
blk.method([
    "Each complete frame is coded as one pattern `Orbital-Nose-Ears-Whiskers`.",
    "**Expected count under independence, pooled:** N x P(Orbital=a) x P(Nose=b) x P(Ears=c) x P(Whiskers=d), with the P's taken over all complete frames. "
    "This mixes in differences between mice (a high-scoring mouse raises all units at once).",
    "**Expected count under independence, within recording (main baseline):** the same product, but computed separately inside every recording from that recording's own "
    "score distribution, then summed over recordings. What deviates from this baseline cannot be explained by some mice/sessions simply scoring higher.",
    "Ratio = observed / expected(within). z = (observed - expected) / sqrt(expected) is shown only for ranking; frames are autocorrelated (see Q4), so z is **not** a p-value.",
    f"**Uncertainty:** {N_BOOT} bootstrap resamples of the {len(mouse_ids)} mice (both recordings of a mouse are resampled together); the ratio is recomputed each time; bars show the 2.5-97.5 percentile.",
    f"Over/under-representation is only plotted for patterns with expected count >= {MIN_EXP} frames, to avoid ratios dominated by very rare patterns.",
])

# Figure 1: most frequent patterns
top = tab.sort_values("n", ascending=False).head(25).reset_index(drop=True)
fig, ax = plt.subplots(figsize=(10, 3.6))
ax.bar(top.index, top["n"], color="#bdc3c7", label="observed")
ax.plot(top.index, top["exp_within"], "o", color="#c0392b", ms=5, label="expected, independence within recording")
ax.plot(top.index, top["exp_pooled"], "x", color="#2980b9", ms=5, label="expected, independence pooled")
ax.set_xticks(top.index, top["pattern"], rotation=90, fontsize=7.5)
ax.set(ylabel="number of frames", title="The 25 most frequent patterns (Orbital-Nose-Ears-Whiskers)")
ax.legend()
fig.tight_layout()
blk.figure(fig, f"The 25 most frequent of the {n_seen} patterns that occur; together with the next ones they cover 90% of frames after {n90} patterns.",
           "Grey bars: observed frames per pattern. Red dots: how many frames would be expected if the units were independent inside each recording; blue crosses: same but "
           "independent across all frames. A bar clearly above (below) its red dot means the pattern is over- (under-) represented.")
plt.close(fig)

# Figure 2: log2 ratio with bootstrap CI
sel = pd.concat([over, under]).drop_duplicates("pattern").sort_values("l2")
fig, ax = plt.subplots(figsize=(8, 6))
y = np.arange(len(sel))
col = np.where(sel["l2"] > 0, "#c0392b", "#2980b9")
sure = ((sel["ratio_lo"] > 1) | (sel["ratio_hi"] < 1)).to_numpy()
for yi, (_, r), c, s in zip(y, sel.iterrows(), col, sure):
    ax.barh(yi, r["l2"], color=c, alpha=0.9 if s else 0.35)
    ax.plot([np.log2(r["ratio_lo"]), np.log2(r["ratio_hi"])], [yi, yi], "k-", lw=1.2)
    ax.text(np.log2(r["ratio_hi"]) + 0.05 if r["l2"] > 0 else np.log2(r["ratio_lo"]) - 0.05, yi,
            f"n={int(r['n']):,}", va="center", ha="left" if r["l2"] > 0 else "right", fontsize=7, color="0.3")
ax.set_yticks(y, sel["pattern"], fontsize=8)
ax.axvline(0, color="k", lw=0.8)
ax.set(xlabel="log2(observed / expected)    <- fewer than expected | more than expected ->",
       title=f"Patterns that deviate most from independence (expected >= {MIN_EXP} frames)")
ax.margins(x=0.18)
fig.tight_layout()
blk.figure(fig, f"Largest deviations from within-recording independence, with 95% bootstrap interval over mice (black line) and observed frame count. "
                f"{len(sig_over)} patterns are over- and {len(sig_under)} under-represented with an interval that excludes 'no deviation' (among {len(ok)} patterns with enough expected frames).",
           "Bars to the right of 0 = pattern more frequent than independence predicts (+1 = twice as often); to the left = less frequent (-1 = half as often). "
           "A black interval that crosses 0 means the deviation is not reliable across mice. Pattern labels are Orbital-Nose-Ears-Whiskers.")
plt.close(fig)

show = tab.sort_values("n", ascending=False).head(12).copy()
show = pd.DataFrame({"pattern": show["pattern"], "total": show["total"], "frames": show["n"].map("{:,}".format), "share": show["freq"].map("{:.1%}".format),
                     "expected (within)": show["exp_within"].map("{:,.0f}".format), "ratio": show["ratio_within"].map("{:.2f}".format),
                     "ratio 2.5%": show["ratio_lo"].map("{:.2f}".format), "ratio 97.5%": show["ratio_hi"].map("{:.2f}".format)})
blk.table(show, "The 12 most frequent patterns with observed and expected counts.", "Ratio > 1 = more frequent than expected under within-recording independence; the last two columns are the bootstrap interval over mice.")
(Path(__file__).parent / "output").mkdir(exist_ok=True)
tab.sort_values("n", ascending=False).to_csv(Path(__file__).parent / "output" / "pattern_table.csv", index=False)

sel_txt = lambda p: f"`{p}` ({tab.set_index('pattern').loc[p, 'ratio_within']:.1f}x, n={int(tab.set_index('pattern').loc[p, 'n']):,})"  # noqa: E731
blk.findings([
    f"{n_seen} of the 54 possible patterns occur; {n90} patterns cover 90% of frames. The most common pattern `0-0-1-0` (only Ears = 1) makes up {tab['freq'].max():.0%} of frames.",
    f"The all-zero pattern occurs {int(zero['n']):,} times, {zero['ratio_within']:.2f}x its within-recording expectation (95% interval {zero['ratio_lo']:.2f}-{zero['ratio_hi']:.2f}).",
    f"Strong co-activations are over-represented, e.g. {sel_txt('2-2-1-1')}, {sel_txt('1-2-1-1')} and {sel_txt('2-0-2-2')}.",
    f"Isolated activation of one unit is under-represented, e.g. {sel_txt('2-0-0-0')}, {sel_txt('0-2-0-0')}, {sel_txt('1-0-0-0')} and {sel_txt('0-0-0-2')}.",
    f"The within-recording baseline already removes part of the deviation relative to the pooled baseline (e.g. `2-2-1-1`: expected {exp_pool[all_pat[[label(c) == '2-2-1-1' for c in all_pat]][0]]:.0f} pooled vs "
    f"{allhi['exp_within']:.0f} within, observed {int(allhi['n'])}), but most of it remains.",
])
blk.interpretation([
    "The units do not vary independently of each other even within a single recording: there is covariation that is **not** just 'some mice score higher than others'.",
    "The structure is one of *co-activation* (units move up together, or all stay low) rather than of units trading off against each other: the over-represented patterns are the all-zero pattern and "
    "patterns with several units raised, while patterns with a single unit at 2 (and Orbital or Whiskers alone at 1) are under-represented. Ears = 1 on its own is the usual resting pattern "
    "(Ears is mostly scored 1). This is consistent with the total score capturing the main shared signal (see Q3), and leaves the question of what the *deviations* from it are (see A4, A2).",
    "These are descriptive over/under-representations. They do **not** yet show that the patterns are recurring *states*: a state must also persist or recur in time (see Q4, B1), and be a "
    "natural grouping rather than just the extreme of a continuum (A2).",
])
blk.caveats([
    "Complete frames only; missingness is informative (D2), so rare high-score patterns with an unscorable unit may be under-counted here.",
    "Independence of units is a baseline, not a model of the data: it does not say *which* structure (clusters, a single severity axis, ...) generates the deviations.",
    "Intervals resample mice but ignore that baseline and stimulated recordings come from different conditions; the groups (drug, genotype) are pooled here.",
])
blk.open_points(["Should the over/under-represented patterns be compared between baseline and stimulated and between the drug/genotype groups (would feed A3), or first be turned into states by clustering (A2)?"])
blk.save()

# ================================================================== Q3: covariation between units
def entropy(counts):
    p = counts[counts > 0] / counts.sum()
    return float(-(p * np.log2(p)).sum())


def total_corr(Xm):
    return sum(entropy(np.bincount(Xm[:, i], minlength=K)) for i in range(Xm.shape[1])) - entropy(np.bincount(Xm @ POW, minlength=n_pat))


def pair_mi(Xm, i, j):
    return entropy(np.bincount(Xm[:, i], minlength=K)) + entropy(np.bincount(Xm[:, j], minlength=K)) - entropy(np.bincount(Xm[:, i] * K + Xm[:, j], minlength=K * K))


def shuffle_within(Xm):
    Xs = Xm.copy()
    for i in rec_idx:
        for u in range(Xm.shape[1]):
            Xs[i, u] = Xm[rng.permutation(i), u]
    return Xs


pairs = list(combinations(range(len(AUS)), 2))
obs_tc = total_corr(X)
obs_mi = {p: pair_mi(X, *p) for p in pairs}
null_tc, null_mi = [], {p: [] for p in pairs}
for _ in range(N_SHUFFLE):
    Xs = shuffle_within(X)
    null_tc.append(total_corr(Xs))
    for p in pairs:
        null_mi[p].append(pair_mi(Xs, *p))
null_tc = np.array(null_tc)


def corr_tables(d):
    cen = d[AUS] - d.groupby("recording_id")[AUS].transform("mean")
    return {"pooled over all frames\n(Spearman)": d[AUS].corr(method="spearman"), "within recording\n(state-like)": cen.corr(),
            "between recordings\n(mouse-level means)": d.groupby("recording_id")[AUS].mean().corr()}


corrs = corr_tables(cc)
within = corrs["within recording\n(state-like)"]
between = corrs["between recordings\n(mouse-level means)"]
off = lambda m: m.to_numpy()[np.triu_indices(4, 1)]  # noqa: E731
w_base = corr_tables(cc[cc["session"] == "baseline"])["within recording\n(state-like)"]
w_stim = corr_tables(cc[cc["session"] == "stimulated"])["within recording\n(state-like)"]

ev, evec = np.linalg.eigh(within.to_numpy())
o = np.argsort(ev)[::-1]
ev, evec = ev[o], evec[:, o]
pc1 = evec[:, 0] * np.sign(evec[:, 0].sum())
cos_total = float(pc1 @ np.ones(4) / 2)
share_pc = ev / ev.sum()

blk = Block("Q3_covariation", ["Q3", "A4"], "How strongly do the four units covary?", __file__, order=1)
blk.takeaway(f"The units covary positively but weakly within a recording (r = {off(within).min():.2f}-{off(within).max():.2f}) and strongly at the level of recordings/mice "
             f"(r = {off(between).min():.2f}-{off(between).max():.2f}). The main shared axis is the total score, but {1 - share_pc[0]:.0%} of the within-recording variance lies outside it.")
blk.question("How correlated are the four units, separately for state-like variation inside a recording and trait-like differences between mice? Is the total score redundant with the units, or does it mix distinct signals?")
blk.data(DATA_TXT)
blk.method([
    "**Pooled correlation:** Spearman rank correlation over all complete frames (mixes both sources of covariation).",
    "**Within-recording correlation:** each unit's score minus its recording mean, then Pearson correlation over frames. Measures whether the units move together *from second to second* in the same recording.",
    f"**Between-recording correlation:** Pearson correlation of the {n_rec} recording means (one value per unit per recording). Measures whether a mouse/session that scores high on one unit also scores high on the others.",
    f"**Mutual information (bits)** between each pair of units, and **total correlation** among all four (sum of the single-unit entropies minus the joint entropy). Plug-in estimates are biased upwards, "
    f"so they are compared with {N_SHUFFLE} within-recording shuffles (each unit permuted independently inside each recording); the shuffle keeps each recording's score levels and removes the covariation.",
    "**PCA** of the within-recording correlation matrix. If PC1 gives equal weight to all four units it coincides with the total score (all weights equal); the cosine between PC1 and (1,1,1,1) quantifies that.",
])
fig, axes = plt.subplots(1, 5, figsize=(13, 3.4), constrained_layout=True, sharey=True)
mats = list(corrs.items()) + [("within recording,\nbaseline only", w_base), ("within recording,\nstimulated only", w_stim)]
for a, (k, v) in zip(axes, mats):
    im = a.imshow(v.to_numpy(), vmin=-1, vmax=1, cmap="RdBu_r")
    a.set_xticks(range(4), AUS, rotation=45, ha="right", fontsize=8)
    a.set_yticks(range(4), AUS, fontsize=8)
    a.set_title(k, fontsize=8.5)
    a.grid(False)
    for i in range(4):
        for j in range(4):
            a.text(j, i, f"{v.iloc[i, j]:.2f}", ha="center", va="center", fontsize=8, color="white" if abs(v.iloc[i, j]) > 0.6 else "black")
fig.colorbar(im, ax=axes, shrink=0.8, label="correlation")
blk.figure(fig, "Correlation between the four units, computed three ways (left three) and within recordings split by session (right two).",
           "Each cell is the correlation between two units; red = positive. Compare the panels: if within-recording values are much smaller than between-recording values, "
           "most of the covariation is a mouse/session-level difference in overall score rather than moment-to-moment co-movement.")
plt.close(fig)

fig, ax = plt.subplots(1, 2, figsize=(10, 3.4), gridspec_kw={"width_ratios": [2, 1]})
names = [f"{AUS[i]}-{AUS[j]}" for i, j in pairs]
ob = np.array([obs_mi[p] for p in pairs])
nm = np.array([np.mean(null_mi[p]) for p in pairs])
nh = np.array([np.percentile(null_mi[p], 97.5) for p in pairs])
x = np.arange(len(pairs))
ax[0].bar(x, ob, color="#2980b9", label="observed")
ax[0].errorbar(x, nm, yerr=[np.zeros(len(x)), nh - nm], fmt="o", color="k", capsize=3, label="shuffle null (mean, upper 97.5%)")
ax[0].set_xticks(x, names, rotation=30, ha="right")
ax[0].set(ylabel="mutual information (bits)", title="Pairwise mutual information")
ax[0].legend()
ax[1].bar([0], [obs_tc], color="#2980b9")
ax[1].errorbar([0], [null_tc.mean()], yerr=[[0], [np.percentile(null_tc, 97.5) - null_tc.mean()]], fmt="o", color="k", capsize=3)
ax[1].set_xticks([0], ["all four units"])
ax[1].set(ylabel="total correlation (bits)", title="Total correlation")
fig.tight_layout()
blk.figure(fig, f"Information shared between units within recordings. Total correlation {obs_tc:.3f} bits observed vs {null_tc.mean():.3f} (upper 97.5% {np.percentile(null_tc, 97.5):.3f}) expected without covariation.",
           "Blue bars = observed; black dot with whisker = what the within-recording shuffle produces (mean and upper limit). Observed far above the whisker = real covariation. "
           "The scale is bits (0 = independent); values are small in absolute terms because each unit has only 2-3 levels.")
plt.close(fig)

fig, ax = plt.subplots(1, 2, figsize=(9, 3.2))
ax[0].bar(range(1, 5), share_pc * 100, color="#2980b9")
ax[0].axhline(25, color="k", ls="--", lw=0.8)
ax[0].text(4.45, 26, "equal share (25%)", ha="right", fontsize=7.5)
ax[0].set(xlabel="principal component", ylabel="% of within-recording variance", title="Variance per component", xticks=range(1, 5))
ax[1].bar(range(4), pc1, color="#2980b9", label="PC1 loading")
ax[1].plot(range(4), [0.5] * 4, "k--", lw=1, label="total score direction (0.5 each)")
ax[1].set_xticks(range(4), AUS)
ax[1].set(ylabel="loading", title=f"PC1 vs total score (cosine {cos_total:.3f})", ylim=(0, 0.75))
ax[1].legend(loc="lower right")
fig.tight_layout()
blk.figure(fig, f"PCA of the within-recording correlation matrix. PC1 explains {share_pc[0]:.0%} of the variance; its loadings are nearly equal across units.",
           "Left: share of variance per component (25% each would mean no shared structure at all). Right: PC1 loadings against the equal weights (0.5) that define the total score; "
           "bars matching the dashed line means 'the first shared axis is the total score'.")
plt.close(fig)

def extreme_pair(m, fn):
    v = {(AUS[i], AUS[j]): m.iloc[i, j] for i, j in pairs}
    k = fn(v, key=v.get)
    return f"{k[0]}-{k[1]} ({v[k]:.2f})"


mean_between = {a: (between.loc[a].sum() - 1) / 3 for a in AUS}
top_unit = max(mean_between, key=mean_between.get)
all_above_null = all(obs_mi[p] > np.percentile(null_mi[p], 97.5) for p in pairs)
blk.findings([
    f"Within recordings, all six pairs correlate positively but weakly (r = {off(within).min():.2f}-{off(within).max():.2f}); strongest for {extreme_pair(within, max)}, weakest for {extreme_pair(within, min)}.",
    f"Between recordings the correlations are much larger (r = {off(between).min():.2f}-{off(between).max():.2f}; strongest {extreme_pair(between, max)}, weakest {extreme_pair(between, min)}); "
    f"{top_unit} has the highest average correlation with the other units ({mean_between[top_unit]:.2f}).",
    f"Baseline and stimulated recordings show nearly the same within-recording structure (differences in single cells <= {np.abs(off(w_base) - off(w_stim)).max():.2f}).",
    f"Total correlation among the four units is {obs_tc:.3f} bits vs {null_tc.mean():.3f} under the shuffle null (excess {obs_tc - null_tc.mean():.3f} bits); " + ("every pair has mutual information above the 97.5th percentile of its null" if all_above_null else "NOT every pair exceeds its null") +
    f" (largest: {names[int(np.argmax(ob))]} {ob.max():.3f} bits vs null {nm[int(np.argmax(ob))]:.3f}).",
    f"PC1 explains {share_pc[0]:.0%} of the within-recording variance, with loadings {', '.join(f'{a} {w:.2f}' for a, w in zip(AUS, pc1))}; cosine to the total-score direction = {cos_total:.3f}. PC2-PC4 explain {share_pc[1]:.0%}, {share_pc[2]:.0%}, {share_pc[3]:.0%}.",
])
blk.interpretation([
    "The total score is **not redundant** with the units but also not far from a single common factor: PC1 *is* essentially the total score, so summing the units follows the dominant shared axis.",
    f"Most of the variation within a recording ({1 - share_pc[0]:.0%}) is **not** along that axis, and the units are only weakly coupled from second to second. This is what leaves room for unit-specific effects (Q1) or for patterns that differ at equal total score (A4).",
    "The large between-recording correlations say that mice/sessions differ in an overall severity level that affects all units; part of that is the expected drug effect, part is mouse-level variation (Q1, Q5).",
    "Not concluded: the correlations do not say that any particular unit combination *carries the group effect*; that requires comparing groups (Q1, A3).",
])
blk.caveats([
    "Correlations of ordinal 0-2 scores (Nose effectively binary) understate the association of the underlying continuous quantity; polychoric correlations would be a more appropriate (not yet used) measure.",
    "Frames are autocorrelated, so no p-values are given for correlations; shuffles (null model) are used for the information measures, and intervals between mice are not yet computed for the correlations.",
    "Between-recording correlations are based on only 142 recordings (71 mice, two sessions each; the two sessions of one mouse are not independent).",
])
blk.open_points(["Use polychoric correlations / a latent-factor model instead of Pearson/Spearman for the final correlation analysis? (relevant if this becomes a figure in the paper)"])
blk.save()

# ================================================================== A4: what does the total discard?
h_pat = entropy(obs)
h_tot = entropy(np.bincount(pat_total[codes], minlength=9))
rows = []
for tot, g in tab.groupby("total"):
    g = g[g["n"] > 0]
    rows.append({"total": int(tot), "frames": int(g["n"].sum()), "patterns seen": len(g), "patterns >= 5% of total": int((g["n"] / g["n"].sum() >= 0.05).sum()),
                 "most common pattern": g.sort_values("n").iloc[-1]["pattern"], "its share": g["n"].max() / g["n"].sum()})
collapse = pd.DataFrame(rows)
mix = collapse[(collapse["total"] >= 2) & (collapse["total"] <= 6)]
k5 = mix["patterns >= 5% of total"]

blk = Block("A4_total_discards", ["A4", "Q2"], "What does the total score discard?", __file__, order=1)
blk.takeaway(f"Totals 2-6 each hide {k5.min()}-{k5.max()} different unit combinations that make up at least 5% of their frames; knowing the total leaves {h_pat - h_tot:.2f} bits ({(h_pat - h_tot) / h_pat:.0%}) of the pattern information unresolved.")
blk.question("Which different 4-unit patterns end up with the same total score, and how much information about the pattern is lost by summing?")
blk.data(DATA_TXT)
blk.method([
    "For every total score value 0-8, list the patterns that sum to it and their share of the frames with that total.",
    "Compute the Shannon entropy (bits) of the pattern distribution H(pattern) and of the total H(total). Because the total is a function of the pattern, "
    "H(pattern | total) = H(pattern) - H(total) is the information that remains once the total is known.",
])
fig, ax = plt.subplots(figsize=(10, 4.2))
cmap = plt.get_cmap("tab20")
for tot in range(9):
    g = tab[(tab["total"] == tot) & (tab["n"] > 0)].sort_values("n", ascending=False)
    sh = g["n"] / g["n"].sum()
    bottom = 0
    for k, (p, s) in enumerate(zip(g["pattern"], sh)):
        ax.bar(tot, s, bottom=bottom, color=cmap(k % 20), edgecolor="white", lw=0.5, width=0.8)
        if s >= 0.06:
            ax.text(tot, bottom + s / 2, f"{p}\n{s:.0%}", ha="center", va="center", fontsize=6.5)
        bottom += s
    ax.text(tot, 1.02, f"n={int(g['n'].sum()):,}", ha="center", fontsize=7)
ax.set_xticks(range(9))
ax.set(xlabel="Total Grimace Score (sum of the four units)", ylabel="share of frames with that total", ylim=(0, 1.1),
       title="Which patterns hide behind each total (labels: Orbital-Nose-Ears-Whiskers, patterns >= 6%)")
ax.grid(False)
fig.tight_layout()
blk.figure(fig, "Composition of every total-score value in terms of unit patterns (complete frames).",
           "Each column is one total score. Colour blocks are different patterns that sum to that total, labelled when they hold at least 6% of the frames. "
           "One block filling the column = the total identifies the pattern; many blocks = the total mixes several unit combinations.")
plt.close(fig)
ct = collapse.copy()
ct["frames"] = ct["frames"].map("{:,}".format)
ct["its share"] = ct["its share"].map("{:.0%}".format)
blk.table(ct, "Number of different patterns behind each total score.", "'patterns >= 5% of total' counts patterns that are not negligible for that total. Totals 0 and 8 each contain a single pattern.")

blk.findings([
    f"H(pattern) = {h_pat:.2f} bits, H(total) = {h_tot:.2f} bits; the total therefore discards {h_pat - h_tot:.2f} bits = {(h_pat - h_tot) / h_pat:.0%} of the pattern entropy.",
    "Total 0 and total 8 correspond to one pattern each; total 1 is mostly (" + f"{collapse.loc[collapse['total'] == 1, 'its share'].iloc[0]:.0%}) `0-0-1-0` (Ears only).",
    "Totals 2-6 are mixtures: " + "; ".join(f"total {int(r['total'])}: {r['patterns >= 5% of total']} patterns >= 5% (largest `{r['most common pattern']}` {r['its share']:.0%})" for _, r in mix.iterrows()) + ".",
    "The mixtures consist of qualitatively different combinations, e.g. total 2 = " + ", ".join(
        f"`{r.pattern}` ({r.n / tab.loc[tab['total'] == 2, 'n'].sum():.0%})" for r in tab[tab["total"] == 2].sort_values("n", ascending=False).head(4).itertuples()) + "; total 3 = " + ", ".join(
        f"`{r.pattern}` ({r.n / tab.loc[tab['total'] == 3, 'n'].sum():.0%})" for r in tab[tab["total"] == 3].sort_values("n", ascending=False).head(4).itertuples()) + ".",
])
blk.interpretation([
    "The total score is a many-to-one mapping for the mid-range, where most frames lie; **information about which units are active is demonstrably lost**.",
    "This shows that unit-specific information *exists*; it does **not** show that the lost information is *useful*, e.g. that the patterns behind one total differ between drug groups. That is the test in A3 (and Q1/Q2).",
    "Part of the entropy difference is expected simply because scores have several levels; the interesting quantity is whether it is *structured* (patterns recurring, differing between groups).",
])
blk.caveats(["Entropies are plug-in estimates from 76k frames and 54 patterns (small bias); frames are autocorrelated, so the effective sample is smaller than the frame count.",
             "Group (drug/genotype) and session are pooled here."])
blk.open_points(["Which outcome should decide whether the discarded information is 'useful': group separation (cross-validated), temporal structure, or both? (needed for A3)"])
blk.save()
print("done")
