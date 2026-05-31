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





No, not all four benchmarks present seasonality, and this architectural divergence is exactly why the Yahoo S5 dataset is so highly regarded. It was specifically built to test how models handle different structural properties of time series data.Seasonality Breakdown of the 4 BenchmarksA1Benchmark (Real Data): Yes. These files contain real production traffic from Yahoo services. They exhibit natural, organic, and messy seasonality (such as daily patterns where traffic peaks during the day and drops at night, or weekly patterns where weekends behave differently).A2Benchmark (Synthetic Data): Yes. These sequences are synthetically generated with a single clean periodic component (usually a basic sine wave), a linear trend, and random Gaussian noise. They focus entirely on point outliers.A3Benchmark (Synthetic Data): Yes (Highly Complex). These are generated with a trend, noise, and three separate interacting periodic components (multiple overlapping seasonal frequencies). This creates complex wave patterns that mimic highly intricate systems.A4Benchmark (Synthetic Data): No. The A4 benchmark intentionally strips away predictable repeating seasonality. Instead, it focuses on structural change-points—abrupt level shifts where the baseline mean or variance of the data suddenly jumps up or down and stays there.The Impact on Architecture: Reconstruction vs. PredictionThe presence or absence of seasonality creates a massive fork in performance between Prediction Models (Forecasting LSTMs that predict the next step t+1) and Reconstruction Models (Autoencoders that compress and reconstruct a global window of length W).1. Performance on Clean Seasonality (A2 & A3)Prediction Models (LSTMs): Excel here. Neural networks are exceptionally good at acting as function approximators for repeating periodic sequences. An LSTM will quickly memorize the frequencies of A2 and A3. Its prediction error for normal data will drop to almost zero, making point anomalies (unexpected spikes) stand out with extreme clarity.Reconstruction Models (Autoencoders): Good, but with a catch. Autoencoders will easily compress the periodic shapes into a lower-dimensional latent space. However, because they optimize for global window reconstruction, they tend to "smooth out" data. If a point anomaly is small, a reconstruction model might accidentally reconstruct it or overlook it, making them slightly less sensitive to tiny point anomalies in seasonal data than predictors.2. Performance on Real-World Noisy Seasonality (A1)Prediction Models: Suffer from "phase lag" or noise oversensitivity. Real traffic has jitter. If a normal daily peak happens 30 minutes later than usual, a forecasting model will predict a drop, see a rise, and throw a massive false positive error.Reconstruction Models: More Robust. Because a reconstruction model looks at a global window (e.g., a 64-step snapshot) rather than a single step, it doesn't care exactly when a peak happens inside that window—it just cares that the general shape looks like a normal daily curve. It handles noisy seasonality with fewer false alarms.3. Performance on Structural Change-Points (A4)This is where the divergence is most dramatic. If the data suddenly jumps from a baseline of 10 to a baseline of 50:Prediction Models: Fail terribly for a brief period. When the level shifts, the model will continue predicting a value near 10. The prediction error will skyrocket. It will continue throwing an anomaly alert for an entire window's length until its historical context buffer fills completely with the new baseline (50). This results in a massive block of false positives right after a structural shift.Reconstruction Models: Succeed brilliantly. When a window rolls over a change-point, it captures a "step function" (half the window is at 10, half is at 50). Because the Autoencoder was trained only on uniform, flat training data, it has no mathematical concept of a step function. The entire reconstruction mechanism breaks down for that window, registering a clean, precise anomaly interval for the structural change without entering a perpetual loop of false alarms.Summary Matrix for ExperimentsBenchmark TypeBest Architecture ChoiceReasonA1 (Real Noise)ReconstructionHandles shifting phases and natural variance globally without micro-panicking.A2 & A3 (Pure Seasonality)PredictionEasily regression-models the exact frequency equations; flags spikes perfectly.A4 (Change-Points)ReconstructionFlags the structural impossibility of a windowed step-change instantly.Given how differently these architectures react to structural change-points versus pure seasonality, are you planning to evaluate a pure forecasting model, a pure reconstruction model, or a hybrid setup (like an LSTM-Autoencoder) across these benchmarks?