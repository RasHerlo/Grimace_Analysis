# Research questions

Living document for the overall questions and sub-questions of the paper.
Question IDs (Q1-Q6, A1-A4, B1-B5, D1-D2) are used as section names in the PDF report (eport/questions.yaml\ must be kept in sync).
Analytical approaches are deliberately left open for now.

## Overall question

Are there specific subsets of the grimace action units (**Orbital, Nose, Ears, Whiskers**) that are more informative than the summed **Total Grimace Score**, which is currently what gets plotted?

## Study design (from the data overview)

- 71 mice, 2x2: Genotype (Cre+ / Cre-) x Drug (Levcromakalim / Vehicle).
- Treatment code: 1 = Cre- VEH, 2 = Cre- LEV, 3 = Cre+ VEH, 4 = Cre+ LEV.
- Covariates: Sex, Cohort, Weight.
- Each mouse has a baseline and a stimulated (test) recording.
- Per-frame scores (every 50th frame, 1 s apart): 4 action units (Cheek is not scored), each with a model confidence value.

## Sub-questions

- **Q1.** Do individual action units, or combinations of them, separate the groups better than the total score?
  - Which units, or which combinations, carry the Levcromakalim effect?
  - Do they differ by genotype (Cre+ vs Cre-)?
- **Q2.** Does the total score hide effects, for example when units move in opposite directions or one unit dominates?
- **Q3.** How correlated are the four units with each other? Is the total score redundant, or does it mix distinct signals?
- **Q4.** Do the units differ in their time course within a session (onset, persistence)?
- **Q5.** Are the effects consistent across sex and cohort?
- **Q6.** How reliable is each unit (model confidence, baseline variability, test-retest)?

## Analysis plan: states and dynamics of the 4-unit vector

Approach per step is still open; candidate methods are discussed in chat and will be fixed before implementation.

### Part A. Subclusters (states) of [Orbital, Nose, Ears, Whiskers]

Do the 4-unit vectors form recurring subclusters (states) over time? Do the covariations between the units carry more information than the Total Grimace Score?

- A1. Which unit combinations occur more (or less) often than expected if the units were independent?
- A2. Is there a small number of recurring states, and what are their profiles?
- A3. Do the states separate baseline vs stimulated, Vehicle vs Levcromakalim and Cre+ vs Cre- better than the total score?
- A4. What does the total score discard (different states that map to the same total)?

### Part B. Transitions and prediction

Do some states follow each other with high probability, and can a state be predicted before it occurs?

- B1. Which transitions between states are more likely than expected from state frequencies alone?
- B2. Do transition structure and dwell times differ between groups (drug, genotype, baseline vs stimulated)?
- B3. Do some units lead others (e.g. a change in one unit precedes changes in the others)?
- B4. Can the onset of a state be predicted k seconds ahead, better than a "same as now" baseline?
- B5. Does a mouse's baseline dynamics predict its response after stimulation?

### Where the results are

Results, methods and interpretation are collected in the running PDF report (eport/output/grimace_report.pdf\, built by \python report/run_all.py\).
Its chapters and subsections follow the question IDs of this document (D1-D2 data, Q1-Q6, A1-A4, B1-B5). Do not copy numbers into this file; they live in the report so that they stay tied to the code.

## Open analysis decisions

- [ ] Definition of "informative": group discrimination (cross-validated, leave-mice-out), interpretability of states, or both?
- [ ] States shared across baseline and stimulated (needed to compare occupancy), or fitted separately?
- [ ] Handling of `-1` (not scorable) frames: drop, treat as missing in the model, or own state? Is missingness related to state or confidence?
- [ ] Confidence: weight or filter frames? (ties to the Excel confidence-rule question below)
- [ ] Is there a known stimulus timeline in the stimulated recordings (time since stimulation as a predictor)?
- [ ] Prediction horizon, and whether the 1 s sampling is fine enough for the state dynamics.

## Open data questions

- [ ] `Baselines/Raw data` has 82 recordings; 71 belong to the mice in the Excel (exactly 1 baseline + 1 stimulated each). The other 11 are from 10 animals NOT in the Excel (91 x2 incl. one "omkorsel", 142, 149, 151, 153, 201, 205, 217, 274, 302). Excluded animals, or missing from the sheet?
- [ ] Score `-1` means "not scorable" (4-24% of frames per unit; Whiskers highest). The Total Grimace Score is NaN in exactly those frames where any unit is -1 (18% of frames).
- [ ] Excel "Counted frames" are much lower than the number of valid frames (e.g. mouse 83: 263 vs 501), and I cannot reproduce the Excel means. So which confidence rule was used? It is not a single min/mean confidence threshold.
- [ ] The "Test" columns in the overview Excel are empty. They can be filled from the stimulated raw data.
- [ ] Unit of analysis: per-mouse mean, per-frame, or both?

## Ideas for later (not decided)

- Parts-based decomposition of the 4-unit vectors (e.g. NMF, Lee & Seung 1999) as one possible way to find informative subgroups.
- Latent class analysis (ordinal indicators) for the states, and a hidden Markov model as a joint model of states and transitions.
