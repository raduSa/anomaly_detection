# Yahoo S5 — Results Notes

## Model comparison

| Model | A1 AP | A1 ROC | A3 AP | A3 ROC | A4 AP | A4 ROC |
|---|---|---|---|---|---|---|
| IForest | 0.511 | 0.765 | 0.331 | 0.511 | 0.268 | 0.498 |
| OC-SVM | 0.595 | 0.776 | 0.376 | 0.584 | 0.265 | 0.519 |
| MLP AE | 0.611 | 0.792 | 0.507 | 0.765 | 0.402 | 0.719 |
| LSTM AE | 0.600 | 0.794 | 0.392 | 0.566 | 0.252 | 0.502 |
| Causal Transformer | — | — | 0.623 | 0.738 | 0.436 | 0.695 |
| Predictive LSTM | 0.588 | 0.789 | **0.812** | **0.868** | **0.505** | **0.745** |
| Transformer Bottleneck | — | — | 0.663 | 0.762 | 0.404 | 0.655 |

---

## A1Benchmark — real server metrics, point anomalies

All models cluster at AP 0.51–0.61. Point anomalies (single spikes) don't require temporal understanding to detect — a spike in window position 32 is an outlier in the flat feature space just as much as in a sequence model. The window labeling strategy (positive if *any* point is anomalous) inflates the window-level anomaly rate from ~1.76% (point-level) to ~17–18%, making many windows only marginally anomalous and limiting all models equally.

**Takeaway:** no model has a structural advantage on isolated spikes.

---

## A3Benchmark — synthetic seasonal/trend series, contextual anomalies

Results diverge sharply. A3 anomalies are deviations from the expected seasonal/trend trajectory — the anomalous value may be normal in absolute terms but wrong for that phase of the cycle.

**Classical models collapse (IForest ROC-AUC 0.51, OC-SVM 0.58 — near random).** The flattened 64-dim window carries no phase information. Anomaly detection by absolute value distribution fails when the signal is contextual.

**LSTM AE also collapses (ROC-AUC 0.57, TN = 39 / 29,651).** The encoder compresses a 64-step seasonal window into a single hidden state, then the decoder must recover the exact phase and amplitude. This compression fails — reconstruction error is uniformly high across normal windows, destroying discriminative ability. Temporality alone is not enough; the bottleneck is the encoder compression, not the model family.

**Predictive LSTM dominates (AP 0.81, ROC-AUC 0.87).** Trained to predict step `t+1` from steps `0..t`, it exploits the fact that seasonal series are highly predictable. When an anomaly is injected, the prediction error spikes because the next value violates the expected seasonal trajectory. This is a direct match between model objective and anomaly type.

**Transformer Bottleneck is the best reconstruction model (AP 0.66).** Global self-attention captures periodicity within the 64-step window more effectively than the LSTM AE's sequential bottleneck.

### Trivial baseline check (A3)

| Model | ROC-AUC | AP |
|---|---|---|
| Trivial baseline `‖w‖₂` | 0.508 | 0.348 |
| LSTM AE | 0.566 | 0.392 |
| MLP AE | 0.765 | 0.507 |
| Causal Transformer | 0.744 | 0.634 |
| Transformer Bottleneck | 0.724 | 0.630 |
| Predictive LSTM | **0.885** | **0.818** |

The trivial baseline (L2 norm of the raw window, no model) scores ROC-AUC 0.508 and AP 0.348 on A3 — indistinguishable from random. At its F2-optimal threshold it labels the entire test set as anomalous (FN=0, TN=32, FP=29,619). This is the expected behaviour for contextual anomalies: A3 deviations are phase violations, not amplitude spikes, so `‖w‖₂` carries no discriminative signal.

The AP of 0.348 closely matches the positive window prior (~33%, from label blur over a 33%-positive window pool), confirming the baseline is pure noise on A3.

**Margins above the baseline are the thesis-relevant quantities:**
- LSTM AE (+0.04 AP): barely above noise — encoder compression discards phase, leaving the model effectively using magnitude.
- MLP AE (+0.16): captures some window-shape statistics implicitly.
- Transformers (+0.28–0.29): global attention over the 64-step window recovers periodicity directly; large, meaningful margin.
- Predictive LSTM (+0.47): more than doubles the baseline AP. This is the strongest evidence that the *training objective* (predict next step) is the decisive factor on contextual anomalies, not model capacity alone.

---

## A4Benchmark — synthetic change-points, level shifts

A4 has the same structure as A3 but anomalies are abrupt level shifts rather than seasonal injection spikes. Every model degrades relative to A3.

**Classical models hit rock-bottom (IForest ROC-AUC 0.498 — worse than a coin flip).** A post-shift window at the "new level" looks identical to a normal window from a different training series at that level. No information about the transition survives flattening.

**LSTM AE collapses again (ROC-AUC 0.50).** Same encoder compression failure as A3 — confirming this is a structural issue independent of anomaly type.

**The expected reversal did not materialise.** The prediction was that reconstruction autoencoders (MLP AE, Transformer Bottleneck) would overtake Predictive LSTM on change-points. They did not. Both reach AP ~0.40, below Predictive LSTM's 0.51. The step-function shape (half-window at old level, half at new) is detectable but hard when models are pooled across 100 series with varied baseline levels.

**Predictive LSTM still leads (AP 0.51, ROC-AUC 0.75)** — down sharply from A3's 0.81. After a level shift the model predicts the old baseline for a full window, generating a sustained prediction error over the transition. This signal is real but noisier than the clean seasonal deviation signal in A3.

**Takeaway:** Predictive LSTM's advantage is proportional to how *predictable* normal data is. Change-points produce a weaker, shorter-lived signal than seasonal deviations, so the advantage shrinks but does not flip.

---

## Key finding

The 2×2 experiment (paradigm × architecture) shows that **both** paradigm and architecture matter:

|  | LSTM | Transformer |
|---|---|---|
| **Reconstruction** | 0.39 AP (A3) | 0.66 AP (A3) |
| **Predictive** | **0.81 AP (A3)** | 0.62 AP (A3) |

The predictive paradigm is necessary — both predictive models beat both reconstruction models on A3. But the 0.19 AP gap between Predictive LSTM and Causal Transformer (same objective, same training data) shows the LSTM architecture carries an additional advantage on seasonal data.

The reason: the LSTM hidden state acts as a continuous phase tracker, accumulating a compact summary of where the signal is in the seasonal cycle at each step. When the next value violates the expected phase, prediction error spikes cleanly. The causal transformer can attend to any previous position directly but must learn explicit attention patterns between specific positions to recover phase — a harder learning problem, especially for A3's three interacting frequency components.

On A4 (change-points) the gap shrinks to 0.07 AP — the single abrupt transition does not require phase accumulation, so the LSTM's sequential inductive bias matters less.

Per anomaly type:

- **Point anomalies (A1):** all models equal at ~0.55–0.61 AP — no temporal understanding needed
- **Seasonal anomalies (A3):** Predictive LSTM dominates (0.81) — recurrent phase tracking is the key advantage; Causal Transformer trails (0.62) despite same objective
- **Change-points (A4):** all models degrade; Predictive LSTM still leads (0.51); Causal Transformer (0.44) beats reconstruction models
- **Classical models and LSTM AE:** collapse on any anomaly requiring contextual rather than absolute discrimination

**Refined thesis claim:** the predictive paradigm is necessary but not sufficient — the LSTM's recurrent hidden state carries an inductive bias (implicit phase accumulation) that makes it structurally well-matched to periodic anomaly detection beyond what causal attention achieves with the same training objective.

### DeepAnt comparison note

DeepAnt (Munir et al., 2019) is the closest reference in the literature — a CNN-based next-step predictor evaluated on all four Yahoo S5 benchmarks. Two methodological differences matter when comparing results:

**Per-series vs. pooled training.** DeepAnt fits a separate model to each individual time series at inference time (100 models for A3). All models here are trained once on the pooled training windows from all 100 series. Per-series training gives DeepAnt a structural advantage on clean periodic data because each model can memorise a single frequency. The pooled regime is harder and more realistic — it requires the model to generalise across different periods and amplitudes. Results here should not be compared directly to DeepAnt's reported F-scores without noting this.

**Window size.** DeepAnt uses w = 25–45 (their ablation peaks at w = 35 for A3). The current setup uses w = 64. A larger window increases label blur (more overlapping positive windows per anomaly point) and may dilute the spike signal for point anomalies. **Worth trying:** re-running A3 with w = 35 to check whether the smaller context window improves Predictive LSTM or Causal Transformer scores, as DeepAnt's ablation suggests it might.





---

## Discussion: Benchmark Structure and Architectural Fit

Not all four benchmarks present seasonality. The dataset was specifically constructed to test how models respond to different structural properties of time series.

### Seasonality breakdown

| Benchmark | Type | Seasonality | Anomaly type |
|---|---|---|---|
| A1 | Real server metrics | Yes — messy, organic daily/weekly cycles | Point outliers |
| A2 | Synthetic | Yes — single clean periodic component + linear trend + Gaussian noise | Point outliers |
| A3 | Synthetic | Yes — three interacting frequency components (complex) | Contextual (phase deviations) |
| A4 | Synthetic | No — structural change-points only | Level shifts |

### Reconstruction vs. prediction: why the gap emerges

**Clean seasonality (A2, A3):** Prediction models excel. An LSTM trained to predict step `t+1` quickly approximates the periodic function, driving normal prediction error close to zero. Anomalous steps violate the expected trajectory and produce a clean spike in prediction error. Reconstruction autoencoders are competitive but tend to smooth out small point anomalies because the bottleneck averages across the full window.

**Real-world noisy seasonality (A1):** Prediction models are penalised by phase jitter. A normal daily peak that arrives 30 minutes late generates a false alarm at the expected peak time, then another when the delayed peak appears. Reconstruction models are more robust here: a global 64-step window captures whether the *shape* is normal without caring about exact timing within the window. This accounts for why MLP AE matches or beats Predictive LSTM on A1 despite Predictive LSTM dominating A3.

**Structural change-points (A4):** The divergence is largest. When the baseline level shifts abruptly:
- *Prediction models* continue predicting the old level for a full window after the shift, generating sustained prediction error during the transition. This is a real signal but noisy — it decays once the LSTM's context buffer fills with post-shift values, creating a time-limited anomaly window rather than a clean point detection.
- *Reconstruction models* see a "step-function" window (half at old level, half at new). Having trained only on flat, uniform windows, the autoencoder has no representation for a step and incurs high reconstruction error precisely over the transition window. This should give reconstruction models a structural advantage on A4, but the A4 results show the expected reversal did not materialise — Predictive LSTM still leads. The likely cause is inter-series level variance: across 100 series with different baselines, a step in series X may resemble a normal plateau in series Y, blunting reconstruction error as a discriminator.

### Summary

| Benchmark | Expected best paradigm | Observed best model | Notes |
|---|---|---|---|
| A1 | Reconstruction (handles phase noise) | MLP AE (AP 0.611) | All models cluster; no temporal advantage |
| A2 | Prediction (clean seasonality) | — not yet run — | |
| A3 | Prediction (seasonal phase tracking) | Predictive LSTM (AP 0.812) | LSTM hidden state acts as phase accumulator |
| A4 | Reconstruction (step-function detection) | Predictive LSTM (AP 0.505) | Expected reversal did not materialise |

The A4 finding is the thesis's most interesting negative result: structural advantage does not guarantee observed advantage when inter-series variance confounds the reconstruction signal.

---

## Methodology note: evaluation protocol and the PA critique

### No Point Adjustment applied

Many TAD papers apply a protocol called **Point Adjustment (PA)** before computing F1: if any predicted score in a contiguous anomaly segment exceeds the threshold, the entire segment is retroactively marked as correctly detected. Kim et al. (2022) show this inflates F1 to near-1 even for purely random scores when anomaly segments are long, and that F1PA is nearly uncorrelated with raw F1 (Pearson r = −0.59 on SWaT). This evaluation does not apply PA. Window-level labels are fixed at preprocessing time; predicted labels are derived by thresholding the model's per-window score, with no retroactive adjustment.

### PR-AUC as primary metric

`average_precision_score` (AUPR) is used as the primary metric throughout. Kim et al. (2022) explicitly recommend AUROC/AUPR over PA-adjusted F1 precisely because they are threshold-independent — the entire PR curve is summarised in a single number without committing to a decision boundary. This removes the overestimation risk that PA introduces.

### Oracle threshold — acknowledged limitation

The F2-maximising threshold is selected from the test set's own PR curve (`sweep_threshold` in `plot_results.py`). This is oracle thresholding: it would not be available in deployment. It is acceptable here because the goal is model comparison under the best achievable operating point, not deployment readiness. Since PR-AUC does not depend on threshold selection, it remains the authoritative metric; reported F2/F1 values at the oracle threshold are secondary.

### Window-level label blur for point anomalies

Window labels use the *any* rule: a window is positive if at least one of its 64 timesteps is anomalous. For Yahoo S5 point anomalies (typically 1–3 timestep spikes), a single anomalous timestep at position *t* labels up to 64 overlapping windows as positive, each containing at most 1–3/64 anomalous timesteps. This has two consequences:

- The window-level anomaly rate is substantially higher than the point-level rate. For A1 the point-level rate is ~1.76%; after windowing it rises to ~17–18%.
- Windows near but not at the anomaly peak are labeled positive despite being predominantly normal, penalising models that correctly reconstruct the normal portion and score the window only moderately. This is distinct from PA: the *ground-truth* labels are blurred, not the predicted ones.

This blurring affects all models equally under consistent evaluation, so cross-model comparisons remain valid. It is relevant when interpreting absolute AP values: a ceiling well below 1.0 is expected even for a perfect detector, because some positive windows are indistinguishable from normal windows.

### Trivial baseline check (Kim et al. Case 2)

Kim et al. propose a model-free baseline: score every test window by its L2 norm `‖w‖₂`. Because anomalies in A1/A2 are amplitude spikes, windows containing them have elevated magnitude even without any learned representation. Any trained model should exceed this baseline on PR-AUC; failure to do so indicates that temporal modelling adds no value over raw signal amplitude. The script `ts_train_trivial_baseline.py` implements this for A2. Expected outcome: trained sequential models (Predictive LSTM, Transformer) should comfortably exceed the baseline on A3 (contextual anomalies undetectable by magnitude alone), while the margin may be smaller on A1/A2 (pure point outliers).