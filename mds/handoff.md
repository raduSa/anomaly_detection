# Session Handoff

## Project Purpose

Thesis project comparing unsupervised anomaly detection models across three domains:
- **Dataset A (tabular):** Credit card fraud — `data/raw/credit_card/creditcard.csv`
- **Dataset B (time series):** Yahoo S5 — `data/raw/yahoo/`
- **Dataset C (network flows):** CIC-IDS2017 — `data/raw/cicids2017/` (Engelen et al. cleaned version)

All models are trained exclusively on **benign samples** (novelty detection, not outlier detection). Evaluation uses both classes. Primary metric is **PR-AUC (Average Precision)** due to severe class imbalance. Threshold is selected by maximising F2 score (recall-weighted) on the PR curve via `plot_results.sweep_threshold()`.

---

## File Structure

```
anomaly_detection/
├── plot_results.py              ← shared eval: sweep_threshold, evaluate, plot_results,
│                                   plot_loss_curves, evaluate_per_class
├── explainability/               ← explainability chapter scripts (see "Explainability Chapter" below)
│   ├── heatmap_utils.py          ← next_experiment_dir, per_position_error, plot_error_heatmap,
│   │                                gini_coefficient, per_window_gini, plot_gini_comparison,
│   │                                plot_paired_window_comparison
│   ├── shap_utils.py             ← next_experiment_dir, tree_explainer_shap_values,
│   │                                gradient_explainer_shap_values, save_summary_plot,
│   │                                top_features_by_mean_abs_shap, _squeeze_singleton_output
│   ├── yahoo_heatmap.py          ← claim: LSTM AE vs. Transformer Bottleneck AE per-timestep
│   │                                error localization on Yahoo A3 (heatmaps + Gini)
│   ├── credit_card_heatmap.py    ← claim: pseudo-sequence models on credit card fraud
│   │                                (per-feature heatmaps + column-shuffle experiment)
│   ├── cicids_shap.py            ← claims: host-context/PortScan SHAP attribution,
│   │                                DoS GoldenEye IForest+Context SHAP vs. AE+Context heatmap,
│   │                                AE+Context ctx_* dominance on DDoS/PortScan (disjoint-
│   │                                feature-regime claim, AE+Context half)
│   ├── cicids_temporal_lastflow_check.py  ← sanity check: does Temporal IForest's per-class AP
│   │                                survive relabeling windows by the last flow's own label
│   │                                instead of the lenient any-of-16 window label
│   ├── cicids_temporal_shap.py   ← disjoint-feature-regime claim, Temporal IForest half:
│   │                                per-(timestep,feature) SHAP decomposition on flattened
│   │                                windows, position-concentration + top-feature profiles
│   └── cc_fraud_notebook.md      ← pasted-in community Kaggle notebook (correlation/outlier
│                                    findings), used as an external cross-check
├── mds/
│   ├── handoff.md               ← this file
│   ├── architecture_diffs.md    ← all per-dataset parameter differences + justifications
│   ├── IMPLEMENTATION_PLAN.md   ← original thesis plan (explainability chapter etc.)
│   ├── cicids2017.md            ← dataset-specific notes (Engelen fixes, attack taxonomy)
│   ├── plan.md / general_strucutre.md / look_at.md / preprocess_suggestions.md
│   │                            ← informal brainstorming notes, not authoritative
│
├── cicids2017/
│   ├── preprocess.py            ← flat pipeline → processed_cicids/ (77 features)
│   ├── preprocess_temporal.py   ← temporal pipeline → processed_cicids_temporal/ (N,16,77)
│   ├── preprocess_context.py    ← flat + host-context → processed_cicids_context/ (86 features)
│   ├── cicids_dataset.py        ← EDA plots
│   ├── plot_cicids_results.py   ← overall metrics table + per-class AP heatmap
│   ├── results_analysis.md      ← full analysis of all 13 models, per-class breakdown
│   ├── train_iforest.py         ← flat IForest
│   ├── train_ocsvm.py           ← flat OC-SVM
│   ├── train_autoencoder.py     ← flat MLP AE (77→64→32→16)
│   ├── train_iforest_context.py ← IForest on 86-feature context data
│   ├── train_ocsvm_context.py   ← OC-SVM on 86-feature context data
│   ├── train_autoencoder_context.py ← MLP AE on 86-feature context data
│   ├── ts_train_iforest.py      ← temporal IForest (windows flattened)
│   ├── ts_train_ocsvm.py        ← temporal OC-SVM
│   ├── ts_train_autoencoder.py  ← temporal MLP AE
│   ├── ts_train_lstm.py         ← predictive LSTM
│   ├── ts_train_lstm_ae.py      ← LSTM AE
│   ├── ts_train_transformer_bottleneck.py
│   └── ts_train_transformer_causal.py
│
├── yahoo/
│   ├── preprocess_timeseries.py ← sliding window preprocessor (BENCHMARK constant edited per run)
│   ├── yahoo_dataset.py         ← EDA plots
│   ├── results_notes.md         ← full per-benchmark analysis (A1/A3/A4), 2×2 paradigm×architecture discussion
│   ├── ts_train_iforest.py / ts_train_ocsvm.py
│   ├── ts_train_autoencoder.py / ts_train_lstm_ae.py
│   ├── ts_train_lstm.py                ← predictive LSTM (next-step), not an autoencoder
│   ├── ts_train_trivial_baseline.py    ← Kim et al. ‖w‖₂ sanity-check baseline
│   ├── ts_train_transformer_ae.py      ← masked transformer (superseded by bottleneck/causal below — no results yet)
│   ├── ts_train_transformer_bottleneck.py
│   └── ts_train_transformer_causal.py
│
└── credit_card/
    ├── preprocess_tabular.py    ← NOTE: RAW_PATH currently points at data/creditcard.csv, which
    │                              does not exist (actual file: data/raw/credit_card/creditcard.csv) — broken, needs a path fix
    ├── transaction_dateset.py   ← EDA plots (filename typo: "dateset")
    ├── train_iforest.py / train_ocsvm.py / train_autoencoder.py
    ├── train_lstm.py / train_lstm_ae.py    ← predictive LSTM / LSTM autoencoder
    ├── train_transformer_ae.py / train_transformer_bottleneck.py
```

---

## Data Directory Structure

```
data/
├── raw/cicids2017/          ← original Engelen CSVs (Mon–Fri)
├── raw/yahoo/               ← ydata-labeled-time-series-anomalies-v1_0/
├── processed_data/
│   ├── processed_cicids/          ← flat 77-feature arrays
│   ├── processed_cicids_temporal/ ← (N,16,77) windows
│   ├── processed_cicids_context/  ← flat 86-feature arrays (77 + 9 context)
│   ├── processed_yahoo/{A1,A2,A3,A4}Benchmark/  ← all four generated; A2 only partially used so far
│   └── processed_credit_card/
├── models/models_cicids/ / models_yahoo/ / models_credit_card/
├── results/results_cicids/ / results_yahoo/ / results_credit_card/
├── graphs/
│   ├── EDA/cicids/ / EDA/yahoo/ / EDA/credit_card/
│   └── results/cicids/ / results/yahoo/ / results/credit_card/
└── explainability/<experiment_name>/<n>/  ← outputs from explainability/*.py, auto-incrementing
    run number per experiment (next_experiment_dir()) so re-runs never clobber earlier ones.
    Current experiment_name dirs: yahoo_A3Benchmark_lstm_vs_transformer/, credit_card_lstm_vs_transformer/,
    cicids_shap_portscan/, cicids_shap_goldeneye/, cicids_temporal_shap/, cicids_context_disjoint_regime/
```

---

## Architecture Standardisation

All parameter choices are documented and justified in `mds/architecture_diffs.md`. Key points:

- **Epochs**: evidence-based per model from loss curves. Fast-converging models (flat AE, Causal Transformer) use 20. Slow-converging (CIC LSTM AE, CIC/CC Bottleneck Transformer, CC LSTM) use 50. Full table in architecture_diffs.md.
- **Validation split**: all neural nets hold out 10% of training data (fixed seed 0) for monitoring only — never used for stopping or tuning. Loss curves saved to `data/graphs/results/<dataset>/<model>_loss_curve.png`.
- **Dropout**: 0.1 across all MLP AE and LSTM models.
- **IForest**: N_ESTIMATORS=150, RANDOM_SEED=123 everywhere.
- **OC-SVM**: NU=0.01, KERNEL='rbf', GAMMA='scale' everywhere.

---

## CIC-IDS2017 Experiments — COMPLETE

Full results and analysis in `cicids2017/results_analysis.md`. Summary table:

| Model | ROC-AUC | Avg Precision | Notes |
|---|---|---|---|
| **AE+Context** | 0.9860 | **0.9487** | Best overall |
| OC-SVM+Context | 0.9745 | 0.9396 | Best F1/balanced acc |
| IForest+Context | 0.9785 | 0.9328 | Best for slow DoS |
| Temporal LSTM AE | 0.8976 | 0.7955 | 50 epochs |
| Flat MLP AE | 0.9189 | 0.7921 | Previous best before context |
| Flat IForest | 0.9049 | 0.7612 | |
| Temporal OC-SVM | 0.8629 | 0.7816 | |
| Temporal AE | 0.8831 | 0.7509 | |
| Temporal IForest | 0.8438 | 0.7604 | |
| Flat OC-SVM | 0.7336 | 0.6845 | |
| Bottleneck Transformer | 0.8077 | 0.6539 | Re-run needed (LATENT_DIM fixed) |
| Causal Transformer | 0.7099 | 0.5449 | Predictive failure |
| Predictive LSTM | 0.6126 | 0.4472 | Predictive failure |

### Context features (key finding)

`preprocess_context.py` appends 9 host-context features to the standard 77. For each flow it snapshots, using a 60-second sliding window per host IP, how many flows and unique ports that host has generated recently. This directly encodes:
- **DDoS**: high `ctx_max_flow_cnt`
- **PortScan**: high `ctx_max_port_cnt` / `ctx_max_port_rate`

PortScan jumped from AP=0.07–0.28 (all flat/temporal models) to AP=0.84 (AE+Context). This was the largest unexplained weakness across the original suite. The 9 features follow Raskovalov et al. (2024) adapted for anomaly detection (unsupervised — no attack labels used).

### Bottleneck Transformer note

The CIC bottleneck transformer results in the table above were obtained with LATENT_DIM=64 = D_MODEL, providing no actual bottleneck. This has been fixed to LATENT_DIM=32 and EPOCHS=50. **Re-run needed** for valid results.

---

## Yahoo S5 Experiments — A1, A3, A4 RUN; A2 MOSTLY PENDING

A1, A3 and A4 benchmarks are run across IForest, OC-SVM, MLP AE, LSTM AE, Predictive LSTM, and (A3/A4 only) Causal Transformer + Bottleneck Transformer, plus an `‖w‖₂` trivial baseline (A3). A2 has only an autoencoder run so far. All models converge by epoch ~5–7 — very fast due to small training set size. No overfitting observed. Full model-by-model analysis is in `yahoo/results_notes.md`; summary:

- **A1 (point anomalies, real server metrics):** all models cluster at AP 0.51–0.61 — point spikes need no temporal understanding; MLP AE narrowly best (AP 0.611).
- **A3 (contextual/seasonal anomalies):** results diverge sharply. Predictive LSTM dominates (AP 0.812) — its recurrent hidden state acts as a phase tracker. Classical models (IForest, OC-SVM) and LSTM AE collapse near-random. Transformer Bottleneck is the best *reconstruction* model (AP 0.663).
- **A4 (level-shift anomalies):** all models degrade vs. A3; Predictive LSTM still leads (AP 0.505) but the expected reconstruction-model reversal did not materialise (confounded by inter-series baseline variance).
- The `ts_train_transformer_ae.py` masked transformer (per `IMPLEMENTATION_PLAN.md` §2.5) has not been run for Yahoo yet — superseded in practice by the causal/bottleneck transformer variants, which have full results.
- Trivial-baseline check (Kim et al. 2022, Case 2: score = `‖w‖₂`) confirms A3 gains are real signal, not just window-magnitude artifacts (baseline AP 0.348 vs. Predictive LSTM's 0.812).

### Preprocessing (`yahoo/preprocess_timeseries.py`)

```
BENCHMARK  = 'A3Benchmark'   # change per benchmark
WINDOW     = 64
STRIDE_TR  = 4
STRIDE_TE  = 1
TRAIN_FRAC = 0.70
```

- MinMaxScaler fit on normal training points only per series
- Output: `X_train (N,64,1)`, `X_test (M,64,1)`, `y_test (M,)`
- Output dir: `data/processed_data/processed_yahoo/{BENCHMARK}/`

### Benchmark overview

| Benchmark | Series | Origin | Anomaly rate |
|---|---|---|---|
| A1 | 67 | Real server metrics | 1.76% |
| A2 | 100 | Synthetic simple | 0.33% |
| A3 | 100 | Synthetic + trend/seasonality | 0.56% |
| A4 | 100 | Same, harder variant | 0.50% |

---

## Credit Card Experiments — COMPLETE (6 of 7 implemented models evaluated)

Full results and analysis in `credit_card/credit_card_results.md` (written this session, same style as `yahoo/results_notes.md` / `cicids2017/results_analysis.md`). Summary table:

| Model | Paradigm | ROC-AUC | Avg Precision | F1 (fraud) | Notes |
|---|---|---|---|---|---|
| **MLP Autoencoder** | Reconstruction | 0.9649 | **0.7105** | 0.6992 | Best overall |
| Isolation Forest | Partition | 0.9587 | 0.6725 | 0.6070 | Best non-reconstruction model |
| LSTM Autoencoder | Reconstruction (pseudo-seq.) | 0.9476 | 0.6223 | **0.7362** | Best F1; pooled hidden state softens pseudo-seq. handicap |
| OC-SVM | Boundary | 0.9354 | 0.5959 | 0.6454 | |
| Transformer Bottleneck AE | Reconstruction (pseudo-seq.) | 0.9583 | 0.5947 | 0.6137 | |
| Predictive LSTM | Predictive (pseudo-seq.) | 0.9581 | 0.5190 | 0.5244 | Worst — no real next-step signal to learn |

Masked Transformer AE (`train_transformer_ae.py`) has been implemented but is **excluded from evaluation for now**; no Causal Transformer variant exists for this dataset (predictive paradigm already shown to fail via Predictive LSTM, so a second predictive architecture wasn't worth training — see below).

### Key preprocessing

`data/processed_data/processed_credit_card/` (via `preprocess_tabular.py`):
- `Time` dropped (an arbitrary offset from the dataset's first transaction, not meaningful for any model); V1–V28 (PCA features) pass through **unscaled** by default (`V_SCALING='none'`, with `'standard'`/`'quantile'` available as toggles); `Amount` gets `RobustScaler`
- Train: ~227k normal samples (80/20 split, seed 123); Test: held-out 20% normal + all 492 frauds (56,863 / 492)
- **Known issue:** `RAW_PATH` points at `data/creditcard.csv`, which no longer exists — the raw file lives at `data/raw/credit_card/creditcard.csv`. The script needs that path fixed before it can be re-run.

### Central finding: no genuine sequence exists in this dataset

Unlike Yahoo (real time series) or CIC's temporal/context variants (real per-host flow ordering), credit card rows have no temporal or causal structure across features. The three "temporal" architectures (Predictive LSTM, LSTM AE, Transformer Bottleneck AE) reshape each 29-feature row into an artificial sequence of 29 scalar steps (`as_sequences()`: `(N,29) → (N,29,1)`), one feature per step, in whatever order the PCA components appear in the CSV — there is no causal/generative relationship between adjacent components (PCA axes are orthogonal/decorrelated by construction).

This produces a clear ranking by how much each architecture *depends* on that artificial ordering being meaningful:
- **MLP AE** (no windowing at all) wins outright (AP 0.7105).
- **LSTM AE** and **Transformer Bottleneck AE** hold up reasonably (AP 0.622 / 0.595) because both compress the full input into one pooled/summary representation before decoding — closer in spirit to the MLP AE's bottleneck than to genuine sequence modelling.
- **Predictive LSTM** is worst (AP 0.519) — its next-step objective has no real signal to learn when consecutive "steps" are decorrelated PCA axes, reflected in its highest false-positive count (FP=566).

This is the mirror image of the CIC finding (real host-context features are a decisive win) and the Yahoo finding (the predictive objective wins under genuine periodic structure): here, imposing sequence structure where none exists is a handicap, not an advantage. Neural training curves are noisier than Yahoo/CIC as a result (confirmed by loss curves); LSTM and LSTM AE required 50 epochs to converge vs. 20 for the flat MLP AE.

---

## Shared Evaluation Module (`plot_results.py`)

```python
sweep_threshold(y_test, scores, beta=2.0)   # returns best threshold maximising F2
evaluate(y_test, scores, threshold, name, out_path=None)
plot_results(y_test, y_pred, scores, name, out_path)     # CM + ROC + PR curve
plot_loss_curves(train_losses, val_losses, title='', out_path=None)  # train/val curve
evaluate_per_class(attack_types, scores, skip_labels=None, out_path=None)  # per-class AP
```

---

## Explainability Chapter — IN PROGRESS

Narrative plan lives in `mds/explainability_plan.md`: organized by *claim* (specific mechanism asserted in `results.tex`), not by technique — each subsection states the claim being tested, applies a tool, and reports confirm/complicate rather than treating SHAP/heatmaps as a standalone demo. Original technique catalogue is `mds/IMPLEMENTATION_PLAN.md` §8.1–8.7; only a subset has been pursued so far, deliberately (see "Lower priority" in `explainability_plan.md` — latent-space UMAP, attention visualisation, and counterfactual/sensitivity analysis are deferred because no current `results.tex` claim needs them).

**Implemented and written up in `thesis/methodology-results-draft/explainability.tex`** (`\chapter{Explainability Case Studies}`, included in `main.tex` after `results.tex`):

- **§3.1 Yahoo A3 — reconstruction-error localization** (`explainability/yahoo_heatmap.py`). Tests the claim that the LSTM AE's pooled hidden state destroys seasonal phase (diffuse error) vs. the Transformer Bottleneck AE's attention localizing error at the phase-violation point. Per-timestep error heatmaps (visual, suggestive but confounded by independent per-panel colour scales) followed by a Gini-coefficient-of-concentration metric computed over all 14,449 anomalous A3 windows (scale-invariant, so it isolates shape from magnitude). Result: **confirms on average** (mean Gini 0.566 LSTM AE vs. 0.721 Transformer) but the LSTM AE's Gini distribution is **bimodal** — a majority of its windows localize about as well as the Transformer's, and it's a distinct low-Gini subpopulation, not a uniform degradation, that drags its aggregate AP/ROC-AUC down. This nuance only shows up in the full distribution, not the two means.
- **§3.2 Credit Card — pseudo-sequence reconstruction error** (`explainability/credit_card_heatmap.py`). Tests the claim that the "temporal" architectures (LSTM AE, Transformer Bottleneck AE) gain nothing from the artificial, CSV-order-only column sequence. Per-feature-position heatmaps on fraud samples showed a structured (non-noise) pattern, which **complicated** the naive "no structure" expectation — resolved with a column-shuffle experiment (fixed permutation, re-run both models, compare top-10 highest-error features before/after): 6/10 overlap for both models (well above the ~3–4 chance baseline), with **V14 and V17** stable across both orderings for both models and independently corroborated as the strongest fraud-correlates in `explainability/cc_fraud_notebook.md` (pasted-in community Kaggle notebook, also cited in `methodology.tex`). But overall error also rose substantially after shuffling (LSTM AE 47.7→71.7, Transformer 12.9→136.8), showing the models additionally leaned on incidental (non-causal) adjacency in the one fixed column order. Net conclusion: no genuine causal/temporal order exists (supports the original claim and the Predictive LSTM failure), but "the order carries no information the models use" is too strong — a refinement, not an overturn.
- **§"Disjoint Feature Regimes: AE+Context vs. Temporal IForest"** (new section, `sec:explain-disjoint-regime`) — tests the premise behind `results.tex`'s CIC Discussion "A practical ensemble" paragraph (pairing AE+Context with Temporal IForest): that the two models' complementary class coverage is backed by genuinely non-overlapping feature evidence, not two rankings that just happen to differ.
  - **Temporal IForest half** (`explainability/cicids_temporal_lastflow_check.py` + `cicids_temporal_shap.py`): first confirmed per-class AP is essentially unchanged when test windows are relabeled by the last flow's own true label instead of the lenient any-of-16 window label (DDoS 0.9518→0.9515, GoldenEye 0.6278→0.6265, Slowhttptest 0.6851→0.6849, slowloris 0.6091→0.6077; Heartbleed drops 0.5927→0.2892 on only n=11, excluded/tentative). Then, since Temporal IForest trains on flattened 16-flow windows, `TreeExplainer`'s SHAP values decompose cleanly into a (window, timestep, feature) cube: attribution on the final timestep (the flow that would pair with AE+Context) carries only 6–8% of total mass across all four classes, near the 6.25% uniform baseline — not concentrated there. Top features match each attack's mechanism (backward-packet-length stats for DDoS/GoldenEye; inter-arrival-time stats for Slowhttptest/slowloris), and recur with consistent sign across most of the window rather than spiking only near the end. **Confirms** the premise for Temporal IForest.
  - **AE+Context half** (`explainability/cicids_shap.py`, new claim #3 block): SHAP (`GradientExplainer`) on AE+Context's reconstruction error, DDoS and PortScan (its two strong classes). `ctx_min_port_cnt`/`ctx_min_flow_cnt` dominate both classes by roughly an order of magnitude over any ordinary flow feature (PortScan: `ctx_min_port_cnt` mean|SHAP|=0.327, ~70× the next-highest ordinary feature; DDoS: 0.234, ~6× the leading ordinary feature). 8/15 (PortScan) and 7/15 (DDoS) of the top-15 features are `ctx_*`. **Confirms** the mirror premise for AE+Context.
  - **Bug found and fixed while running this**: `shap.GradientExplainer` returns SHAP values shaped `(n, F, 1)` (trailing singleton output dim) rather than TreeExplainer's plain `(n, F)`. `save_summary_plot` was passing this raw 3D array straight to `shap.summary_plot`, which silently misread it and rendered a garbage single-feature plot for every AE+Context SHAP figure generated this session (PortScan, GoldenEye, DDoS) — the numeric rankings (`top_features_by_mean_abs_shap`) were unaffected since that function already squeezed the shape correctly. Fixed with a shared `_squeeze_singleton_output()` helper in `shap_utils.py`; all affected plots regenerated (latest runs: `cicids_shap_portscan/5/`, `cicids_context_disjoint_regime/4/`).
  - Written up in `explainability.tex` with the PortScan AE+Context SHAP summary plot included as a figure; the DDoS SHAP plot exists (`cicids_context_disjoint_regime/4/autoencoder_context_ddos_shap.png`) but is not yet included as a figure, only its numeric ranking.

**Not yet done** (per `explainability_plan.md`):
- **SHAP for CIC-IDS2017** (`explainability/cicids_shap.py`, `explainability/shap_utils.py`) — scripts exist and run (TreeExplainer for IForest+Context, GradientExplainer-wrapped reconstruction-error for AE+Context on PortScan; TreeExplainer + reconstruction-error heatmap, not SHAP, for AE+Context on DoS GoldenEye per explicit instruction not to force a SHAP explainer onto a reconstruction-error question). Findings **complicate** the `results.tex` claims as currently written: PortScan's SHAP ranking is topped by `ctx_min_port_cnt`/`ctx_min_flow_cnt`/`ctx_max_flow_cnt`, not `ctx_max_port_cnt`/`ctx_max_port_rate` as asserted; GoldenEye's `ctx_max_flow_cnt` appears but `ctx_max_port_rate` doesn't make the top 15. **Not yet written into `explainability.tex`** — outputs sit in `data/explainability/cicids_shap_portscan/1/` and `cicids_shap_goldeneye/1/` only; `results.tex`'s host-context prose may also need correcting once this is written up.
- 8.3 Anomaly taxonomy breakdown, 8.4 Latent space UMAP/t-SNE, 8.5 Sensitivity/counterfactual analysis, 8.6 Attention weight visualisation, 8.7 Integrated Gradients — all deferred per `explainability_plan.md`'s "Lower priority" section (no current `results.tex` claim motivates them yet).
- **Practical ensemble (AE+Context + Temporal IForest) — DONE, built and written up.** Script: `cicids2017/ensemble_context_ae_temporal_iforest.py`. Row-alignment (recovering, for every temporal test window, which position in AE+Context's X_test corresponds to that window's last flow, via a shared `base_id` reconstructed by replaying both preprocessing pipelines' row-ordering logic) confirmed exact: standalone Temporal IForest per-class AP recomputed through this alignment matches the earlier validated last-flow-labeling numbers (`cicids_temporal_lastflow_check.py`) almost exactly (e.g. GoldenEye 0.6265 both ways).
  - Two fusion rules (weighted soft-vote swept over $w\in\{0.3,...,0.7\}$; max-rule) × two score normalizations (z-score; rank/percentile) were tested — results in `data/results/results_cicids/ensemble_*`.
  - **z-score normalization is a dead end**, not a negative result about fusion: AE+Context's reconstruction-error z-scores are extremely heavy-tailed (test-set max z≈2366, benign-only 99.9th percentile already z≈227) vs. Temporal IForest's tightly bounded z-scores (max≈21), so a handful of AE+Context outliers dominate any z-scored linear/max combination regardless of weight — every z-score variant just reproduces AE+Context's own ranking. Switching to rank normalization (percentile within each model's own benign training-score distribution, bounded to [0,1]) fixes this.
  - **Key finding, framed by per-class coverage rather than aggregate AP** (aggregate AP is dominated by DDoS/PortScan's raw sample volume and misleadingly favors whichever variant changes least): max-rule (rank) inherits the *union* of both models' false positives and barely moves off AE+Context's own weak GoldenEye/Slowhttptest/slowloris scores (0.29/0.06/0.08) despite Temporal IForest's much stronger standalone numbers there (0.61/0.58/0.56). Soft-vote (rank) shows the opposite failure mode on PortScan: averaging with a model that is actively poor there (Temporal IForest AP 0.123, not just uninformative) drags PortScan down substantially (0.924→0.228 at w=0.4) while lifting GoldenEye from 0.292→0.801.
  - **Conclusion:** rank-normalized soft-vote at w≈0.4 is the most defensible ensemble config — its *weakest* covered class (0.205, Slowhttptest) clears both standalone models' own weakest class (AE+Context 0.013 Slowhttptest; Temporal IForest 0.123 PortScan), a genuine if partial win — but it is an explicit tradeoff (PortScan detection quality drops substantially), not the free coverage win the original aggregate-framed proposal implied. Written up in `explainability.tex` (new subsections under `sec:explain-disjoint-regime`: "Building the Combined Score", "Two Fusion Rules, Two Normalizations", "Per-Class Coverage vs. Aggregate AP"); `results.tex`'s "A practical ensemble" paragraph now points forward to this result instead of leaving it as an untested proposal. Thesis recompiles cleanly (47 pages, xelatex, no undefined refs).

---

## Thesis Writing Progress (`thesis/`)

```
thesis/
├── Thesis_template__Faculty_of_Mathematics_and_Informatics__University_of_Bucharest__1_/
│   ← full FMI/UB bachelor's thesis template (0-title, 0-abstract, 1-introducere,
│     2-preliminarii, 3-methodology, 4-results, 5-concluzii). Chapters 3/4 are the
│     same skeletons described below; 0/1/2/5 remain untouched template placeholders.
└── methodology-results-draft/
    ← standalone 2-chapter extract (methodology.tex + results.tex + its own main.tex/
      bibliography.bib) used to iterate on just these chapters without recompiling
      the full thesis. Source of truth for current content; changes should eventually
      be folded back into 3-methodology.tex/4-results.tex above.
```

Language: English (explicit choice, template originally Romanian). Abstract/introduction are intentionally out of scope for now — work is focused on Methods and Experimental Results only.

**`methodology.tex` ("Methods and Experimental Setup") — fully written, no placeholders remaining:**
- Anomaly Detection Algorithms (one subsection per model — OC-SVM, IForest, MLP AE, LSTM AE, Predictive LSTM, Bottleneck Transformer AE, Causal Transformer — covering architecture, scoring, and score-sign conventions, checked directly against each `train_*.py`/`ts_train_*.py` implementation; masked Transformer AE and CIC's missing credit-card Causal Transformer are both noted as excluded/nonexistent rather than silently omitted). The Transformer-Based Autoencoders subsection now also has a short note contrasting the project's deliberate encoder-only designs against the full encoder-decoder structures common in the transformer time-series literature (Informer's generative decoder; Kim et al.'s Transformer-encoder + 1D-conv-decoder design), framed as a simplification justified by the short sequence lengths used here (16–64).
- Datasets (per-dataset nuances pulled from EDA scripts/notes: credit card PCA/kurtosis, Yahoo A1–A4 table, CIC data-quality issues from Engelen et al.; credit card subsection now also links the Kaggle dataset page and a community exploratory-analysis notebook)
- Preprocessing (tabular/windowing/network-flow pipelines described directly from the preprocessing scripts, including the known `RAW_PATH` bug and the `V_SCALING='none'` default; the `RobustScaler`-on-`Amount` justification now cites the same Kaggle notebook). The Yahoo windowing subsection gained a **"Comparison with DeepAnt"** paragraph (Munir et al., 2019) — verified directly against the paper (`papers/DeepAnt.pdf`) rather than taken on faith from `yahoo/results_notes.md` — covering three concrete differences: DeepAnt's proposed predictor is CNN-based and many-to-one (their LSTM baseline is narrower/shallower than this project's Predictive LSTM and still many-to-one, vs. this project's simultaneous every-step prediction); DeepAnt trains a separate model per series vs. this project's pooled training across ~100 series; and DeepAnt's per-benchmark-tuned window sizes ($w\in\{25,35,45\}$) vs. this project's fixed $w=64$. The label-blur bullet also now points to DeepAnt's window-size ablation as a concrete untried mitigation ($w=35$).
- Hyperparameter and Training-Related Choices (renamed from "Hyperparameter Choices and Architecture Standardization"; ported from `mds/architecture_diffs.md` — one table per architecture) — gained a new **Optimizer** bullet documenting Adam at a fixed `lr=1e-3`, confirmed uniform across every model and dataset by grepping all `train_*.py`/`ts_train_*.py` files.
- Evaluation Protocol (metrics table from `plot_results.py`; PR-AUC justified as the primary metric with a direct quote from Kim et al. 2022's Discussion section recommending AUROC/AUPR over threshold-dependent F1; F2 oracle-threshold note; no-PA justification). The **Trivial Baseline Sanity Check** subsection that used to live here has been moved into `results.tex`'s Yahoo section (it's a result, not a protocol description).

**`results.tex` ("Experimental Results") — all three dataset sections now fully written, no placeholders remaining:**
- Credit Card Fraud Detection: model comparison table (6 models, masked Transformer AE excluded), the pseudo-sequence caveat, per-paradigm discussion, and the F1-vs-AP sanity check. Condensed from `credit_card/credit_card_results.md`.
- **Yahoo Webscope S5 — newly written this session, structured to mirror the Credit Card section:** an intro note explaining A2's exclusion (point anomalies only, same limited-discriminative-power issue as A1, and only partially run); one subsection per benchmark (A1, A3, A4), each with a model-comparison table (ranked by AP) and a `\paragraph{Discussion.}` calling out the best/worst model and the structural reason for the spread; a **Trivial Baseline Sanity Check** subsection (moved from methodology.tex) explaining *why* the check exists (Kim et al. 2022's recommendation to test against a label-free baseline) and reporting the margin each model holds above the `‖w‖₂` baseline on A3, including the finding that the baseline's own AP (0.348) matches the ~33% label-blur-inflated positive-window prior, confirming it is pure noise; and a final **Discussion** subsection synthesising all three benchmarks — the 2×2 paradigm×architecture finding (predictive necessary but not sufficient; LSTM's hidden state as an implicit phase tracker vs. causal attention), a per-anomaly-type summary, real-vs-synthetic seasonality (A1 vs. A3), and the A4 negative result (expected reconstruction-model reversal did not materialise, explained by a single pooled-training confound — windows from ~100 series at different baseline levels being indistinguishable by absolute value alone — that is shown to undermine both the classical/density-based and reconstruction-based approaches for the same underlying reason, while Predictive LSTM is comparatively immune since it conditions on each series' own recent history).
- **CIC-IDS2017 — newly written this session, replacing the placeholder skeleton:**
  - **Overall Comparison**: two attack-class composition tables (flat/context and temporal-windowed) placed *before* the 13-model comparison table, so the reader sees test-set composition before interpreting AP. Building these surfaced a real correction mid-session: **DoS Hulk** (158,469 flows/windows, tied with PortScan for the largest class) is entirely absent from `results_analysis.md`'s per-class writeup. Checked directly against `papers/wtmc2021_Engelen_Troubleshooting.pdf` — Engelen et al. document the DoS Hulk attack tool as misconfigured (sets HTTP `Connection: Close` instead of `Keep-Alive`, so the target server terminates the connection immediately, rendering the attack ineffective by construction, not merely mislabelled like other classes) — and confirmed against every `preprocess*.py`/`train_*.py`/`ts_train_*.py`: DoS Hulk flows *are* counted as positive-class samples in every pooled/aggregate metric (`is_attack` has no per-class filtering), but every training script sets `SKIP_LABELS = {'DoS Hulk'}` before calling `evaluate_per_class()`, deliberately excluding it only from the per-class breakdown. `results.tex` now states this explicitly with the Engelen citation, rather than treating the omission as an unexplained gap. The two composition tables also corrected an over-generalised claim from `results_analysis.md`'s sanity-check note ("temporal counts are 4–5× higher than flat"): the real per-class numbers show that's only true for sparse/isolated classes (Bot ~4.5×, Infiltration ~14×) — dense/bursty classes (PortScan, DDoS, DoS Hulk, the DoS/Patator classes) are close to 1:1 between flat flow counts and temporal window counts.
  - **Per-Attack-Type Breakdown** (new subsection, labelled `sec:cic-per-attack`): a per-class AP table for the three context models, with "Consistently well-detected" (DDoS, PortScan — each its own bolded, line-separated item) and "Mixed context response for application-layer DoS" (DoS GoldenEye; DoS Slowhttptest/slowloris — same bolded-item treatment) mirroring `results_analysis.md`'s section headers and structure directly, plus "Largely undetectable regardless of feature set" for the protocol-level attacks.
  - **Discussion**: five `\paragraph{}`s explicitly built from `results_analysis.md`'s Key Takeaways 1, 2, 3, 6, 7, 8 — host-context as the single most impactful addition (coalescing 1+2+3: the AP 0.79→0.95 gain, PortScan's host-only detectability, and the three context models' differing exploitation of the same 9 features); temporal windowing's limited value on its own; a new "Even the winning strategy has limits" paragraph (coalescing 6+7: slow-DoS still needs Temporal IForest's sequential context, protocol-level attacks are undetectable at the flow level regardless of feature set); the three-dataset reconstruction-vs-prediction synthesis; and a closing "A practical ensemble" paragraph (8: AE+Context + Temporal IForest for widest attack-class coverage).
  - `main.tex` gained `\usepackage{float}` and the main 13-model table now uses `[H]` (was `[h]`) after the two new composition tables pushed it past the "Host-context features dominate" paragraph that follows it in-source — `[h]` is only a placement *hint* LaTeX can override under float congestion, `[H]` forces it.

**New source doc:** `credit_card/credit_card_results.md` — same style as `yahoo/results_notes.md` and `cicids2017/results_analysis.md` (previously the credit card dataset had no equivalent results-summary doc, only raw `*_report.txt` files).

**`explainability.tex` ("Explainability Case Studies") — new chapter, added this session, included in `main.tex` after `results.tex`:** see "Explainability Chapter — IN PROGRESS" above for content (§3.1 Yahoo A3, §3.2 Credit Card). Both sections follow the same structure: initial assumption stated as a checkable prediction, a qualitative heatmap check (two images placed side by side via `\hspace` + two `\caption`/`\label` pairs in one `figure` environment, to avoid burning a page per image), then a quantitative verification with its own table/figure. Two things worth remembering for future edits: (1) the heatmap PNGs are tall (1500×9000, from the 300-row subsample in the generating scripts) — always include `height=0.55\textheight,keepaspectratio` or similar, a plain `width=` include will overflow the page; (2) image paths are relative to `thesis/methodology-results-draft/`, pointing at `../../data/explainability/<experiment_name>/<run_number>/` — the run number is whatever `next_experiment_dir()` produced on the last actual script run, so re-running a script and not updating the `.tex` path will silently reference stale output.

All three `.tex` files (`methodology.tex`, `results.tex`, `explainability.tex`) compile cleanly via `xelatex` (no undefined references) as of this session — 47 pages total.

---

## Expected Thesis Narrative

- **Tabular (credit card):** Classical models ≥ sequential models. Masked transformer unreliable on tabular data (no meaningful feature ordering for masking).
- **Time series (Yahoo):** Sequential models outperform flat models. Predictive LSTM leverages temporal dependencies cleanly.
- **Network flows (CIC-IDS2017):** Host-context features are the decisive addition — PortScan (159k samples) undetectable at per-flow level, trivially detectable at host-context level. Reconstruction-based models outperform prediction-based (attack traffic more predictable than benign). Context models dominate; temporal sequence windowing adds limited value given arbitrary intra-window flow ordering.
