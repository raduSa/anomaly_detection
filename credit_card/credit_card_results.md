# Credit Card Fraud — Results Notes

## Model comparison

| Model | Paradigm | ROC-AUC | Avg Precision | Balanced Acc | F1 (fraud) | TP | FP | FN |
|---|---|---|---|---|---|---|---|---|
| MLP Autoencoder | Reconstruction | 0.9649 | **0.7105** | 0.8983 | 0.6992 | 394 | 241 | 98 |
| Isolation Forest | Partition | 0.9587 | 0.6725 | 0.8828 | 0.6070 | 380 | 380 | 112 |
| LSTM Autoencoder | Reconstruction (pseudo-sequence) | 0.9476 | 0.6223 | 0.8928 | **0.7362** | 388 | 174 | 104 |
| OC-SVM | Boundary | 0.9354 | 0.5959 | 0.8875 | 0.6454 | 384 | 314 | 108 |
| Transformer Bottleneck AE | Reconstruction (pseudo-sequence) | 0.9583 | 0.5947 | 0.8849 | 0.6137 | 382 | 371 | 110 |
| Predictive LSTM | Predictive (pseudo-sequence) | 0.9581 | 0.5190 | 0.8771 | 0.5244 | 376 | 566 | 116 |

Test set: 56,863 legit + 492 fraud (57,355 total, 0.86% fraud rate). Primary ranking metric is Average Precision (PR-AUC), consistent with the Yahoo and CIC-IDS2017 analyses — this dataset is even more imbalanced than CIC (~25% attack rate) and closer to Yahoo A1's ~1.76% point-anomaly rate, so AP is essential rather than accuracy or ROC-AUC alone.

A masked-reconstruction Transformer AE (`train_transformer_ae.py`) was also implemented for this dataset but is excluded from this comparison — it is not currently part of the evaluated model set.

---

## The central caveat: there is no real sequence here

Unlike Yahoo S5 (genuine univariate time series) or CIC-IDS2017 temporal/context variants (genuine per-host flow ordering), **credit card transactions have no temporal or causal ordering across features**. Time was dropped during preprocessing (`preprocess_tabular.py` drops the `Time` column entirely), and each remaining row is a single, independent 29-dimensional feature vector: `V1..V28` (anonymised PCA components) + `Amount`.

The three "temporal" architectures evaluated here (Predictive LSTM, LSTM AE, Transformer Bottleneck AE) do not operate on this dataset the way they do on Yahoo or CIC. Instead of predicting or reconstructing actual sequential data points, they reshape each 29-feature row into an artificial sequence of 29 scalar "time steps" — `train_lstm.py`/`train_lstm_ae.py`/`train_transformer_bottleneck.py` all call `as_sequences()`, which does `(N, 29) → (N, 29, 1)` and feeds the model one feature per step, in whatever order the PCA components happen to appear in the CSV:

- **Predictive LSTM** predicts feature index `k+1` from features `0..k`, i.e. it tries to forecast "V7 given V1..V6" the way it would forecast tomorrow's server load given today's. There is no causal or generative relationship between adjacent PCA components — PCA produces orthogonal, decorrelated axes by construction — so this objective has no real signal to learn beyond whatever residual linear correlation survives decorrelation.
- **LSTM AE** likewise encodes/decodes over this arbitrary 29-step ordering rather than a meaningful trajectory.

**Impact on the results:**

- **Predictive LSTM is the worst model in this comparison** (AP 0.519), mirroring the same failure mode seen on Yahoo A1/A2-style point anomalies but for a different reason: on Yahoo, next-step prediction fails on point anomalies because a single spike doesn't need temporal context to detect; here it fails because there is no next-step relationship to predict in the first place. The model still achieves reasonable ROC-AUC (0.958) — the aggregate reconstruction-error magnitude still correlates loosely with fraud — but its confusion matrix (FP=566, the highest of any model here) shows the F2-optimal threshold has to accept far more false alarms to catch fraud, because the underlying score is a much noisier signal than a true next-value prediction error would be.
- **LSTM AE holds up comparatively well** despite the same artificial framing (AP 0.622, and the **best F1 of all evaluated models**, 0.736, with the lowest false-positive count, 174). The likely explanation is architectural rather than about the sequence semantics: the encoder still compresses the full 29-step input into a single final hidden state before the decoder reconstructs from it, so it functions similarly to the MLP AE's bottleneck — a global summary of the whole feature vector — rather than genuinely relying on order-dependent structure. The pseudo-sequence framing is a soft handicap here, not a fatal one.
- **Transformer Bottleneck AE** (AP 0.595) sits close to the classical models, for a similar reason to the LSTM AE: it pools token representations into a single latent vector via global average pooling before decoding, so it is effectively reconstructing the flattened feature vector through a bottleneck rather than depending on positional/sequential coherence between adjacent PCA components.
- **MLP Autoencoder — the one architecture that treats the 29 features as an ordinary flat vector with no artificial windowing — wins overall** (best AP, 0.7105). This is consistent with the dataset's actual structure: there is no sequence to exploit, so the paradigm best suited to a static multivariate feature vector wins.

This is the mirror image of the CIC-IDS2017 finding that host-context features (which encode *real* temporal/behavioural information) dominate, and of the Yahoo finding that the predictive objective wins when there is genuine periodic structure to predict. On this dataset, imposing sequence structure where none exists is a handicap rather than an advantage, and the results rank models roughly in order of "how much does this architecture depend on the artificial ordering being meaningful": MLP AE (none) > LSTM AE / Transformer Bottleneck (soft, pooled/compressed) > Predictive LSTM (hard, depends on order).

---

## Paradigm-level discussion

### Reconstruction-based models
MLP AE (0.711), LSTM AE (0.622), and Transformer Bottleneck AE (0.595) are three of the top four models — reconstruction is the strongest-performing paradigm on this dataset once the artificial-sequence handicap is accounted for (see above). All three learn what a "normal" 29-dimensional legitimate transaction looks like and flag high reconstruction error; fraud transactions, occupying a different region of the PCA + Amount space, reconstruct poorly.

### Partition-based (Isolation Forest)
Isolation Forest is the second-best model overall (AP 0.6725) and the best non-reconstruction model. With `N_ESTIMATORS=150`, `MAX_SAMPLES=200000`, it isolates fraud via axis-aligned splits directly on the 29 raw features (no artificial sequence framing applies to it or OC-SVM at all — both operate on flat feature vectors natively). Its relatively high false-positive count (380, tied with true positives) reflects tree-based isolation's tendency to flag more borderline points than a smooth reconstruction error surface would.

### Boundary-based (OC-SVM)
OC-SVM (AP 0.5959, `NU=0.01`, RBF kernel) sits mid-table. Its recall (0.7805) is close to the other top models, but its precision is lower, giving the second-highest false-positive count (314) after Predictive LSTM and the two weaker transformer-adjacent models. The RBF hypersphere boundary is more sensitive to the raw, unscaled `V1-V28` features (recall `V_SCALING='none'` is the pipeline default) than the tree-based or reconstruction-based approaches.

### Predictive
Predictive LSTM is the weakest model in this comparison (AP 0.519) for the structural reason discussed above: the "next-step" framing has no real target to predict when the 29 steps are unordered PCA components. There is no causal-attention analogue implemented for this dataset (unlike Yahoo/CIC, this project does not train a Causal Transformer variant here), so Predictive LSTM is the only representative of the predictive paradigm on this dataset.

---

## Outliers and sanity checks

- **Isolation Forest outperforming OC-SVM and two of the three "temporal" architectures (Predictive LSTM, Transformer Bottleneck AE)** is notable: on CIC-IDS2017, IForest without context features (`Flat IForest`, AP 0.7612 relative to that dataset's models) similarly proves to be a strong, robust baseline. Simple, low-assumption models remain competitive whenever the imposed structure (sequence framing, kernel choice) is a poor match for the data.
- **LSTM AE has the best F1 (0.7362) despite a middling AP (0.6223)** — a reminder that F1 is computed at a single oracle F2-threshold and can favor a different operating point than the one that maximises area-under-the-PR-curve. AP remains the primary metric for cross-model ranking (see Evaluation Protocol notes below); F1 differences at the oracle threshold should be read as a secondary, threshold-dependent signal.
- All six models achieve ROC-AUC > 0.93, but AP spans from 0.519 to 0.711 — a reminder (also made in the Yahoo and CIC notes) that ROC-AUC is far less sensitive to class imbalance than AP and can look deceptively uniform across models that behave very differently once the positive class is rare (~0.86% here).

---

## Evaluation protocol notes (shared with Yahoo and CIC-IDS2017)

The same `plot_results.py` module and evaluation protocol used for the other two datasets applies here unchanged:

- **PR-AUC (Average Precision) is the primary metric**, for the same class-imbalance reasons as Yahoo/CIC (Kim et al., 2022).
- **Threshold selection uses the F2-maximising oracle threshold** (`sweep_threshold`, beta=2), selected from the test set's own PR curve — an oracle threshold, not a deployment-ready one, used purely for model comparison at a consistent operating point.
- **No Point Adjustment is applied.** Since credit card fraud detection operates on independent transaction rows rather than contiguous anomaly segments, PA (a segment-level adjustment) is not even a meaningful concept here — every prediction is evaluated per-transaction, which if anything makes this dataset's reported metrics the least susceptible of the three to the PA-inflation critique.
- **Train/test split**: all models are trained exclusively on the 80% split of *legitimate* transactions (`X_train_normal.npy`, `TEST_SIZE=0.2`, `RANDOM_SEED=123`); the test set combines the held-out 20% of legitimate transactions with all 492 fraud transactions, giving the 56,863/492 split reported above.
