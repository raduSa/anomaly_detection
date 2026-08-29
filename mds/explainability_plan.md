# Explainability Chapter — Narrative Plan

Goal: not a standalone SHAP/heatmap demo, but a chapter that visually **stress-tests specific
claims already made in `results.tex`**. Each subsection below is organized around one existing
claim, the tool used to test it, and what a confirming vs. contradicting result would mean.

## SHAP — feature-attribution claims (CIC-IDS2017)

SHAP fits claims about *which features* drove a decision. Two claims in `results.tex` currently
rest on plausible-but-unverified mechanism stories:

### 1. Host-context features / PortScan blind spot
- **Claim being tested** (`results.tex`, "Host-context features dominate" / per-attack breakdown):
  `ctx_max_port_cnt` / `ctx_max_port_rate` are asserted to be "unmistakable during a scan" and to
  explain why context models jump from AP 0.07-0.28 to >0.60 on PortScan.
- **Test:** SHAP summary plot on IForest+Context and AE+Context for PortScan samples.
- **Confirms:** context port-count/port-rate features rank as top contributors.
- **Complicates:** some other feature (or a flat feature interacting with context) actually drives
  the gain — equally interesting, would need the prose corrected.

### 2. DoS GoldenEye split between IForest+Context and AE+Context
- **Claim being tested:** IForest+Context isolates the "high `ctx_max_flow_cnt`, low
  `ctx_max_port_rate`" combination cleanly; AE+Context reconstructs it with error spread across
  many dimensions instead, explaining the AP 0.655 vs. ~0.22 split.
- **Test:** SHAP on IForest+Context (TreeExplainer) for GoldenEye samples, compared against the
  reconstruction-error heatmap for AE+Context on the same samples (see below) — a direct
  cross-model comparison on the same attack.

### 3. Practical ensemble — does pairing context models actually cover more classes?
- **Claim being tested** (`results.tex`, CIC Discussion, "A practical ensemble"): combining
  complementary context models covers more attack classes than any single model.
- **Test:** restrict to the three same-feature-set context models (IForest+Context,
  OC-SVM+Context, AE+Context — row-aligned, same `processed_cicids_context` test arrays), combine
  normalized scores (max, 2-way and 3-way), and recompute overall + per-class AP.
- **Confirms:** the combined score beats every individual model's per-class AP on its weak classes
  without eroding AP elsewhere.
- **Complicates:** if confirmed, follow up with per-class SHAP on each member to check whether the
  gain traces to genuinely disjoint feature evidence or just noise cancellation.

Also can look at the shap values of Temporal IF - Temporal IForest's TreeExplainer SHAP attribution pointing at something that only exists because of the 16-step window (a feature varying/repeating/trending across
     stacked timesteps) rather than just the same base features flat models already see

## Reconstruction-error heatmaps — localization claims (Yahoo S5, Credit Card)

Heatmaps fit claims about *when/where* within a sample or window the error concentrates, which
maps onto the temporal/bottleneck claims rather than the feature-importance ones.

### 4. Yahoo A3 — LSTM AE bottleneck destroys phase vs. Transformer recovers it
- **Claim being tested** (`results.tex`, A3 Discussion): LSTM AE's single final hidden state
  "destroys" seasonal phase information (ROC-AUC collapses to 0.566), while Transformer Bottleneck
  AE's global self-attention "likely recovers periodicity," explaining its AP 0.663 lead over LSTM
  AE's 0.392.
- **Test:** per-timestep reconstruction-error heatmap on the same anomalous A3 window, LSTM AE vs.
  Transformer Bottleneck AE side by side.
- **Confirms:** LSTM AE shows diffuse/flat error across the window; Transformer shows a sharp
  spike localized at the phase-violation point.
- **Complicates:** if LSTM AE's error is also localized, the "compression destroys phase" story
  needs revision.

### 5. Credit Card — pseudo-sequence models don't rely on artificial row order
- **Claim being tested** (`results.tex`, Credit Card Discussion): the three "temporal"
  architectures (Predictive LSTM, LSTM AE, Transformer Bottleneck AE) reshape 29 independent
  PCA features into an artificial sequence, and results rank models by how much each depends on
  that artificial ordering being meaningful.
- **Test:** per-"timestep" (i.e. per-V-feature-position) reconstruction-error heatmap for LSTM AE
  and Transformer Bottleneck AE on fraud samples.
- **Confirms:** no coherent temporal/positional pattern in the error — supports "there is no real
  next-step signal to learn."
- **Complicates:** a structured pattern would suggest the models are picking up on something
  beyond the null "no genuine order" story (e.g. an artifact of column ordering in the source CSV).

## Structure

Organize the chapter by claim (as numbered above), not by technique — each subsection states the
Results-chapter claim it targets, applies the relevant tool, and reports confirm/complicate.
A subsection noting any claim that did *not* hold up under inspection is worth keeping in, not
smoothing over — it's a stronger thesis contribution than a chapter of pure confirmations.

## Lower priority (stretch goals, not core to this framing)

Deferred because current `results.tex` Discussion sections don't make claims these would test:

- **Latent-space UMAP/t-SNE** (`IMPLEMENTATION_PLAN.md` §8.4) — answers "is the representation
  separated," which nothing in the current Discussion asserts.
- **Attention-weight visualization / gradient-weighted attention / Integrated Gradients**
  (`IMPLEMENTATION_PLAN.md` §8.6-8.7) — answers "is attention faithful," a question not raised in
  current results prose. Worth revisiting if the chapter needs to grow, or if SHAP/heatmap
  findings raise a "which model actually attends to the right thing" follow-up question.
- **Counterfactual / sensitivity analysis** (`IMPLEMENTATION_PLAN.md` §8.5) — orthogonal axis
  (boundary robustness) not currently claimed about in `results.tex`.
