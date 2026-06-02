# Session Handoff

## Project Purpose

Thesis project comparing unsupervised anomaly detection models across three domains:
- **Dataset A (tabular):** Credit card fraud — `data/raw/creditcard.csv`
- **Dataset B (time series):** Yahoo S5 — `data/raw/yahoo/`
- **Dataset C (network flows):** CIC-IDS2017 — `data/raw/cicids2017/` (Engelen et al. cleaned version)

All models are trained exclusively on **benign samples** (novelty detection, not outlier detection). Evaluation uses both classes. Primary metric is **PR-AUC (Average Precision)** due to severe class imbalance. Threshold is selected by maximising F2 score (recall-weighted) on the PR curve via `plot_results.sweep_threshold()`.

---

## File Structure

```
anomaly_detection/
├── plot_results.py              ← shared eval: sweep_threshold, evaluate, plot_results,
│                                   plot_loss_curves, evaluate_per_class
├── mds/
│   ├── handoff.md               ← this file
│   ├── architecture_diffs.md    ← all per-dataset parameter differences + justifications
│   ├── IMPLEMENTATION_PLAN.md   ← original thesis plan (explainability chapter etc.)
│   └── cicids2017.md            ← dataset-specific notes (Engelen fixes, attack taxonomy)
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
│   ├── preprocess_timeseries.py ← sliding window preprocessor
│   ├── yahoo_dataset.py         ← EDA plots
│   ├── ts_train_iforest.py / ts_train_ocsvm.py
│   ├── ts_train_autoencoder.py / ts_train_lstm_ae.py / ts_train_lstm.py
│   ├── ts_train_transformer_ae.py      ← masked transformer
│   ├── ts_train_transformer_bottleneck.py
│   └── ts_train_transformer_causal.py
│
└── credit_card/
    ├── preprocess_tabular.py
    ├── transaction_dataset.py
    ├── train_iforest.py / train_ocsvm.py / train_autoencoder.py
    ├── train_lstm.py / train_lstm_ae.py
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
│   ├── processed_yahoo/{BENCHMARK}/
│   └── processed_credit_card/
├── models/models_cicids/ / models_yahoo/ / models_credit_card/
├── results/results_cicids/ / results_yahoo/ / results_credit_card/
└── graphs/
    ├── EDA/cicids/ / EDA/yahoo/ / EDA/credit_card/
    └── results/cicids/ / results/yahoo/ / results/credit_card/
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

## Yahoo S5 Experiments — PARTIALLY RUN (A3 Benchmark)

A3 benchmark fully run. All models converge by epoch ~5–7 — very fast due to small training set size. No overfitting observed.

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

## Credit Card Experiments — RUN (hyperparameters updated)

Key preprocessing: `data/processed_data/processed_credit_card/`
- StandardScaler on all features
- Train: ~227k normal samples; Test: mixed

Neural models on credit card are architecturally atypical: the 29 PCA features are treated as a pseudo-sequence (no temporal order). This causes noisier training curves than Yahoo/CIC — confirmed by loss curves. LSTM and LSTM AE required 50 epochs to converge.

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

## Explainability Plan (not yet implemented)

See `mds/IMPLEMENTATION_PLAN.md` sections 8.1–8.7:
- **8.1** SHAP for IForest (TreeExplainer) and OC-SVM (KernelExplainer)
- **8.2** Per-feature reconstruction error heatmaps (MLP AE, LSTM AE); per-token 2D heatmap (Transformer)
- **8.3** Anomaly taxonomy: point vs. contextual vs. collective anomaly breakdown by model
- **8.4** Latent space UMAP/t-SNE
- **8.5** Sensitivity / counterfactual analysis
- **8.6** Attention weight visualisation (hook-based)
- **8.7** Integrated Gradients via `captum`

---

## Expected Thesis Narrative

- **Tabular (credit card):** Classical models ≥ sequential models. Masked transformer unreliable on tabular data (no meaningful feature ordering for masking).
- **Time series (Yahoo):** Sequential models outperform flat models. Predictive LSTM leverages temporal dependencies cleanly.
- **Network flows (CIC-IDS2017):** Host-context features are the decisive addition — PortScan (159k samples) undetectable at per-flow level, trivially detectable at host-context level. Reconstruction-based models outperform prediction-based (attack traffic more predictable than benign). Context models dominate; temporal sequence windowing adds limited value given arbitrary intra-window flow ordering.
