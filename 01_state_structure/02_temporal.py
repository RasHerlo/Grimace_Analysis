"""Step 2: time course within a session, and is the 1 s resolution adequate for states and transitions?

Report block produced: Q4 (also B1, B4)

Run:  python 01_state_structure/02_temporal.py      (then: python report/build_report.py)
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from grimace import load_analysis_frames, load_config
from grimace.report import TREATMENT_COLORS, TREATMENT_LABELS, Block, use_style

use_style()
cfg = load_config()
AUS = cfg["action_units"]
TOT = "Total Grimace Score"
VARS = AUS + [TOT]
rng = np.random.default_rng(0)
MAX_LAG, N_SHUFFLE = 30, 20
SESS = ["baseline", "stimulated"]

df = load_analysis_frames(cfg)
n_rec, n_mice = df["recording_id"].nunique(), df["animal_no"].nunique()
step = df.groupby("recording_id")["t"].diff().dropna()
assert (step == 1.0).all(), "frames are not exactly 1 s apart"
df["tbin"] = (df["t"] // 60).astype(int)
full_minutes = (df.groupby("recording_id")["t"].max() + 1) // 60  # complete minutes per recording
last_full_bin = int(max(b for b in range(20) if (full_minutes > b).mean() >= 0.9))  # last minute that >= 90% of recordings cover completely
recs = {rid: g for rid, g in df.groupby("recording_id", sort=False)}
rec_session = df.groupby("recording_id")["session"].first()
rec_len = df.groupby("recording_id")["t"].max()

# ------------------------------------------------------------------ autocorrelation
def acf(x, max_lag):
    x = x - np.nanmean(x)
    var = np.nanmean(x**2)
    res = np.full(max_lag, np.nan)
    if not var > 0:
        return res
    for k in range(1, max_lag + 1):
        a, b = x[:-k], x[k:]
        ok = ~np.isnan(a) & ~np.isnan(b)
        if ok.sum() > 30:
            res[k - 1] = np.mean(a[ok] * b[ok]) / var
    return res


rows = [(rid, rec_session[rid], v, k, r) for rid, g in recs.items() for v in VARS for k, r in enumerate(acf(g[v].to_numpy(float), MAX_LAG), start=1)]
ac = pd.DataFrame(rows, columns=["recording_id", "session", "var", "lag", "acf"])
acm = ac.groupby(["session", "var", "lag"])["acf"].agg(["mean", "sem"]).reset_index()

# ------------------------------------------------------------------ persistence and dwell times
def persistence(values):
    a, b = values[:-1], values[1:]
    ok = ~np.isnan(a) & ~np.isnan(b)
    if ok.sum() < 30:
        return np.nan, np.nan, 0
    v = values[~np.isnan(values)]
    _, c = np.unique(v, return_counts=True)
    p = c / c.sum()
    return float(np.mean(a[ok] == b[ok])), float((p**2).sum()), int(ok.sum())


pat_codes, rows = {}, []
for rid, g in recs.items():
    code = np.where(g["complete"], (g[AUS].fillna(0).to_numpy() * 3 ** np.arange(4)).sum(1), -1).astype(int)
    pat_codes[rid] = code
    pat = np.where(g["complete"], code, np.nan).astype(float)
    for name, vals in [(a, g[a].to_numpy(float)) for a in AUS] + [("4-unit pattern", pat)]:
        o, c, n = persistence(vals)
        rows.append((rid, rec_session[rid], name, o, c, n))
per = pd.DataFrame(rows, columns=["recording_id", "session", "what", "obs", "chance", "n"]).dropna()
pers = pd.DataFrame([{"session": s, "what": w, "obs": np.average(d["obs"], weights=d["n"]), "chance": np.average(d["chance"], weights=d["n"])}
                     for (s, w), d in per.groupby(["session", "what"])])
pers["ratio"] = pers["obs"] / pers["chance"]


def run_lengths(codes):
    change = np.r_[True, codes[1:] != codes[:-1]]
    idx = np.flatnonzero(change)
    return np.diff(np.r_[idx, len(codes)])[codes[idx] >= 0]


obs_runs = np.concatenate([run_lengths(c) for c in pat_codes.values()])
null_runs = [np.concatenate([run_lengths(rng.permutation(c)) for c in pat_codes.values()]) for _ in range(N_SHUFFLE)]
null_mean = np.mean([r.mean() for r in null_runs])
null_len1 = np.mean([np.mean(r == 1) for r in null_runs])

# ------------------------------------------------------------------ drift over the session
per_mouse = df[df["tbin"] <= last_full_bin].groupby(["session", "treatment", "animal_no", "tbin"])[VARS].mean().reset_index()
drift = per_mouse.groupby(["session", "treatment", "tbin"])[VARS].agg(["mean", "sem"])
early = per_mouse[per_mouse["tbin"] <= 2].groupby(["session", "treatment", "animal_no"])[VARS].mean()
late = per_mouse[per_mouse["tbin"] >= last_full_bin - 2].groupby(["session", "treatment", "animal_no"])[VARS].mean()
dl = (late - early).dropna()

# ================================================================== block
blk = Block("Q4_time_course", ["Q4", "B1", "B4"], "Time course and temporal persistence of the scores", __file__, order=1)
lag1 = acm[acm["lag"] == 1].set_index(["session", "var"])["mean"]
lag1_u = lag1[[(s, a) for s in SESS for a in AUS]]
pp = pers[pers["what"] == "4-unit pattern"].set_index("session")
tot_groups = per_mouse.groupby(["session", "treatment"])[TOT].mean().unstack()
blk.takeaway(f"At 1 s resolution the scores are dominated by frame-to-frame noise (lag-1 autocorrelation {lag1_u.min():.2f}-{lag1_u.max():.2f}, ~80% of pattern runs last one second), "
             f"so a 'state' cannot be read off single frames; the Levcromakalim effect is present from the first minutes and stays flat, i.e. a sustained shift rather than a time-limited event.")
blk.question("Do the units change over the course of a session (onset, drift), and how long does a score or a 4-unit pattern persist from one second to the next? "
             "The second part decides whether transitions between states (B1) and prediction of states (B4) can be studied on the raw 1 s frames.")
blk.data(f"{n_mice} mice, {n_rec} recordings of median {rec_len.median():.0f} s (frames exactly 1 s apart, checked); the last partial minute is dropped for the minute-wise curves "
         f"(minutes 0-{last_full_bin}; later minutes are covered by fewer than 90% of the recordings). Autocorrelation and persistence use all frames with the relevant values scorable (unscorable frames are skipped, not filled); "
         "dwell times use complete frames and an unscorable frame ends a run.")
blk.method([
    f"**Minute curves:** for each mouse, session and minute the mean score (unscorable units ignored); curves show the mean over mice per treatment group (Cre-/Cre+ x VEH/LEV) with the standard error of the mean over mice. "
    f"'Early' = minutes 0-2, 'late' = minutes {last_full_bin - 2}-{last_full_bin}; the change late minus early is computed per mouse.",
    f"**Autocorrelation:** per recording, subtract the recording's mean score, then correlate the score at time t with the score at t+k for k = 1..{MAX_LAG} s using only pairs where both are scorable, divided by the variance. "
    "Averaged over recordings (mean and SEM over recordings), separately per session.",
    "**Persistence:** among pairs of consecutive scorable frames, the fraction with an identical value (single unit) or identical complete 4-unit pattern at t and t+1. "
    "'Chance' = the probability that two randomly drawn frames of the same recording are identical, sum of squared category shares, computed per recording. Averaged over recordings weighted by number of pairs.",
    f"**Dwell time:** length of runs of identical consecutive complete patterns. Null: the order of all frames of a recording is randomly permuted ({N_SHUFFLE} permutations, unscorable frames included so the number and share of gaps stays the same).",
])

fig, axes = plt.subplots(len(VARS), 2, figsize=(10, 11), sharex=True)
for j, s in enumerate(SESS):
    for i, v in enumerate(VARS):
        for tr in (1, 2, 3, 4):
            m = drift.loc[(s, tr)][(v, "mean")]
            e = drift.loc[(s, tr)][(v, "sem")]
            axes[i, j].plot(m.index, m, color=TREATMENT_COLORS[tr], label=TREATMENT_LABELS[tr], lw=1.4)
            axes[i, j].fill_between(m.index, m - e, m + e, color=TREATMENT_COLORS[tr], alpha=0.15, lw=0)
        axes[i, j].set_ylabel("total (0-8)" if v == TOT else f"{v} (0-2)", fontsize=8)
    axes[0, j].set_title(s)
    axes[-1, j].set_xlabel("minute of session")
axes[0, 0].legend(ncol=2, fontsize=7)
fig.tight_layout()
blk.figure(fig, "Mean score per minute of the session for the four treatment groups, per unit and for the total (rows), baseline (left) and stimulated (right) recordings.",
           "Lines = mean over mice, bands = +-1 SEM over mice. Flat lines: no change during the session. Separated lines: group differences. "
           "A group effect that only appears later in the session would show as lines that diverge to the right; here we look for that.", tall=True)
plt.close(fig)

fig, ax = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
for a, s in zip(ax, SESS):
    for v in VARS:
        d = acm[(acm["session"] == s) & (acm["var"] == v)]
        a.plot(d["lag"], d["mean"], "o-", ms=3, lw=1.1, color="k" if v == TOT else None, label=v)
        a.fill_between(d["lag"], d["mean"] - d["sem"], d["mean"] + d["sem"], alpha=0.15, lw=0, color="k" if v == TOT else None)
    a.axhline(0, color="0.4", lw=0.7)
    a.set(xlabel="lag (seconds)", title=s)
ax[0].set_ylabel("autocorrelation")
ax[0].legend(fontsize=7)
fig.tight_layout()
blk.figure(fig, "Autocorrelation of each unit and of the total score as a function of the time lag, averaged over recordings.",
           "Value at lag k = correlation between the score now and k seconds later within a recording (1 = identical, 0 = unrelated). A curve that drops to ~0 within a few seconds means a "
           "fast-fluctuating signal with little memory; a slowly decaying curve would mean persistent states.")
plt.close(fig)

fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
order = AUS + ["4-unit pattern"]
x = np.arange(len(order))
for k, s in enumerate(SESS):
    d = pers[pers["session"] == s].set_index("what").loc[order]
    ax[0].bar(x + (k - 0.5) * 0.38, d["obs"], 0.36, color=["#2980b9", "#c0392b"][k], label=f"observed, {s}")
    ax[0].plot(x + (k - 0.5) * 0.38, d["chance"], "k_", ms=14, mew=2, label="chance" if k == 0 else None)
ax[0].set_xticks(x, order, rotation=25, ha="right")
ax[0].set(ylabel="P(same value one second later)", title="Persistence vs chance")
ax[0].legend(fontsize=7)
lmax = int(np.percentile(obs_runs, 99.9))
bins = np.arange(1, lmax + 2) - 0.5
ax[1].hist(obs_runs, bins=bins, alpha=0.65, density=True, label="observed", color="#2980b9")
ax[1].hist(null_runs[0], bins=bins, alpha=0.5, density=True, label="time-shuffled", color="#e67e22")
ax[1].set(yscale="log", xlabel="run length of an identical 4-unit pattern (s)", ylabel="share of runs", title="Dwell times")
ax[1].legend()
ax[2].bar(["observed", "time-shuffled"], [obs_runs.mean(), null_mean], color=["#2980b9", "#e67e22"])
ax[2].set(ylabel="mean run length (s)", title="Mean dwell time")
fig.tight_layout()
blk.figure(fig, f"How long values persist. Left: probability that a unit (or the whole pattern) is unchanged one second later, versus the chance level (black bars). "
                f"Middle/right: duration of runs of an identical pattern versus the same frames in random order.",
           "Left: a bar far above the black mark means a value tends to stay the same; equal to the mark means no memory. Middle: if patterns were stable states the observed histogram would "
           "have a heavy tail of long runs compared with the shuffled one. Right: average run length.")
plt.close(fig)

ac_tab = acm[acm["lag"].isin([1, 2, 5, 10, 30])].pivot_table(index=["session", "var"], columns="lag", values="mean").reset_index()
ac_tab.columns = ["session", "variable"] + [f"lag {int(c)} s" for c in ac_tab.columns[2:]]
for c in ac_tab.columns[2:]:
    ac_tab[c] = ac_tab[c].map("{:.2f}".format)
blk.table(ac_tab, "Mean autocorrelation at selected lags.", "Rows: session and variable. Values near 0 at lag 10-30 s mean no long-range memory beyond the recording mean.")

rows = []
for s in SESS:
    for tr in (1, 2, 3, 4):
        d = dl.loc[(s, tr)][TOT]
        e = early.loc[(s, tr)][TOT]
        l = late.loc[(s, tr)][TOT]
        rows.append({"session": s, "group": TREATMENT_LABELS[tr], "mice": len(d), "total, early": f"{e.mean():.2f}", "total, late": f"{l.mean():.2f}", "late - early (mean +- SEM)": f"{d.mean():+.2f} +- {d.sem():.2f}"})
blk.table(pd.DataFrame(rows), f"Total score early (minutes 0-2) vs late (minutes {last_full_bin - 2}-{last_full_bin}) in the session, per group.",
          "The last column is the mean of the per-mouse change (late minus early) with its standard error; an interval that does not include 0 would indicate drift within the session.")

dlt = dl[TOT]
drifting = [(s, tr) for s in SESS for tr in (1, 2, 3, 4) if abs(dlt.loc[(s, tr)].mean()) > 2 * dlt.loc[(s, tr)].sem()]
lev, veh = tot_groups.loc["stimulated", [2, 4]].mean(), tot_groups.loc["stimulated", [1, 3]].mean()
lev_b, veh_b = tot_groups.loc["baseline", [2, 4]].mean(), tot_groups.loc["baseline", [1, 3]].mean()
ac10 = acm[(acm["lag"] == 10)].set_index(["session", "var"])["mean"]
ac30 = acm[(acm["lag"] == 30)].set_index(["session", "var"])["mean"]
blk.findings([
    f"**Time course.** Averaged over the session the total score of stimulated recordings is {lev:.2f} in the two Levcromakalim groups and {veh:.2f} in the two vehicle groups; at baseline it is {lev_b:.2f} vs {veh_b:.2f}. "
    "The group curves are separated from the first minutes and run roughly parallel through the session (Figure Q4.1).",
    "Change within the session (late minus early total): " + ("; ".join(f"{s} {TREATMENT_LABELS[tr]} {dlt.loc[(s, tr)].mean():+.2f}" for s, tr in drifting) + " differ from 0 by more than 2 SEM; all others do not."
                                                           if drifting else "no group differs from 0 by more than 2 SEM (Table Q4.2)."),
    f"**Short-term memory.** Lag-1 autocorrelation is {lag1_u.min():.2f}-{lag1_u.max():.2f} for the units and {lag1[('baseline', TOT)]:.2f} / {lag1[('stimulated', TOT)]:.2f} for the total (baseline / stimulated); it is higher in stimulated than baseline recordings for every unit. "
    f"At 10 s it is {ac10.loc[[(s, a) for s in SESS for a in AUS]].min():.2f}-{ac10.loc[[(s, a) for s in SESS for a in AUS]].max():.2f} and at 30 s {ac30.loc[[(s, a) for s in SESS for a in AUS]].min():.2f}-{ac30.loc[[(s, a) for s in SESS for a in AUS]].max():.2f}.",
    f"**Persistence.** The complete 4-unit pattern is identical one second later in {pp.loc['baseline', 'obs']:.0%} (baseline) and {pp.loc['stimulated', 'obs']:.0%} (stimulated) of cases vs a chance level of {pp.loc['baseline', 'chance']:.0%} / {pp.loc['stimulated', 'chance']:.0%} "
    f"({pp.loc['baseline', 'ratio']:.1f}x / {pp.loc['stimulated', 'ratio']:.1f}x chance). Single units are {pers[pers['what'].isin(AUS)]['ratio'].min():.2f}-{pers[pers['what'].isin(AUS)]['ratio'].max():.2f}x chance "
    "(see Figure Q4.3, left).",
    f"**Dwell time.** Mean run of an identical pattern is {obs_runs.mean():.2f} s (median {np.median(obs_runs):.0f}, 95th percentile {np.percentile(obs_runs, 95):.0f}); the time-shuffled null gives {null_mean:.2f} s. "
    f"{np.mean(obs_runs == 1):.0%} of runs are a single second (null {null_len1:.0%}).",
])
blk.interpretation([
    "There is **temporal structure, but it is weak at 1 s**: scores persist more than chance (relative to chance more so in stimulated recordings), yet most of the second-to-second variation is not predictable from the previous second. "
    "Either the scoring model is noisy frame by frame, or the face really fluctuates faster than 1 s; this analysis cannot tell which.",
    "Consequence for Part B: transitions between *raw 1 s patterns* would mostly describe noise. Realistic options are (i) smoothing / windows of several seconds before defining states, "
    "(ii) a model that treats each frame as a noisy readout of a slower hidden state (e.g. a hidden Markov model), or (iii) restricting the transition analysis to well-separated, persistent patterns. "
    "Small but non-zero autocorrelation out to ~10 s is consistent with a slower component underneath the noise, but is not proof of discrete states.",
    "The Levcromakalim effect is a sustained shift in the score level (visible from the first minutes), not an event that builds up or fades during the recording; for Q4 this means that "
    "'onset' cannot be resolved on the scale of minutes in this data. Whether the units differ in their course would need the per-unit curves compared formally (Figure Q4.1 shows them side by side but no test is done).",
    "Raw persistence and persistence relative to chance point in different directions between sessions for several variables: in stimulated recordings the scores are more varied (lower chance level), "
    "so the raw share of unchanged values is often lower while the ratio to chance and the autocorrelation are higher. Whether the *dynamics* differ between sessions or only the score levels do has not been separated (B2).",
])
blk.caveats([
    "Standard errors in the minute curves are over mice (n = 17-18 per group); no formal test of group differences is made here (that belongs to Q1).",
    "Autocorrelation is computed on ordinal 0-2 scores; unscorable frames are skipped, which shortens the effective gaps between valid frames and, since missingness is informative (D2), may bias persistence slightly.",
    "The shuffle null for dwell times permutes whole recordings, so it also destroys any slow drift; a slow drift alone would therefore show up as 'longer than null runs'.",
])
blk.open_points([
    "Temporal resolution for Part B: smooth over a window (which length, e.g. 5 or 10 s?), use an HMM on the raw frames, or both?",
    "Do we know whether the frame-to-frame jumps reflect real behaviour or classifier noise (e.g. from video of a few mice, or a scoring-model test-retest)? This matters for how much smoothing is justified (also Q6).",
    "Is there a known event timeline in the stimulated recordings (injection/stimulus times)? Without it, 'onset' of an effect cannot be defined.",
])
blk.save()
