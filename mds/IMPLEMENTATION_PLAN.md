# Anomaly Detection — Implementation & Testing Plan

## 1. Project Structure

```
licenta/
├── data/
│   ├── raw/          # original downloaded files
│   └── processed/    # cleaned, split, windowed versions
├── models/
│   ├── ocsvm.py
│   ├── isolation_forest.py
│   ├── autoencoder.py
│   └── lstm_ae.py
├── experiments/
│   ├── tabular/      # runs on Dataset A
│   └── timeseries/   # runs on Dataset B
├── explainability/
│   ├── shap_analysis.py
│   ├── reconstruction_heatmap.py
│   └── attention_weights.py
├── evaluation/
│   ├── metrics.py
│   └── plots.py
├── notebooks/        # EDA, result exploration
└── results/          # saved scores, figures, tables
```

---

## 2. The Four Models

### 2.1 One-Class SVM (OC-SVM)
**Paradigm:** Boundary-based  
**Logic:** Learns a hypersphere (in kernel-mapped space) that contains the normal data. Anything outside = anomaly.  
**Key hyperparameter:** `nu` (upper bound on fraction of anomalies) and kernel bandwidth `gamma`.

**Learning resources**
- Original paper: Schölkopf et al. (1999) — *Support Vector Method for Novelty Detection*  
  https://proceedings.neurips.cc/paper_files/paper/1999/file/8725fb777f25776ffa9076e44fcfd776-Paper.pdf
- scikit-learn docs: https://scikit-learn.org/stable/modules/generated/sklearn.svm.OneClassSVM.html
- Practical guide (Sebastian Raschka): https://sebastianraschka.com/Articles/2014_intro_supervised_learning.html (sections on SVMs)
- Short intuitive explanation: https://www.youtube.com/watch?v=GBUtPOznKUg

### 2.2 Isolation Forest (iForest)
**Paradigm:** Partition-based  
**Logic:** Builds random trees that recursively split features. Anomalies require fewer splits to isolate → shorter path length = higher anomaly score.  
**Key hyperparameter:** `n_estimators`, `contamination`, `max_samples`.

**Learning resources**
- Original paper: Liu et al. (2008) — *Isolation Forest*  
  https://cs.nju.edu.cn/zhouzh/zhouzh.files/publication/icdm08b.pdf
- scikit-learn docs: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html
- Blog walkthrough (Towards Data Science): https://towardsdatascience.com/isolation-forest-the-key-to-outlier-detection-110f52f5b8ef
- Extended Isolation Forest (EIF — fixes axis-parallel bias): https://arxiv.org/abs/1811.02141

### 2.3 Feedforward Autoencoder (AE)
**Paradigm:** Reconstruction-based  
**Logic:** Encoder compresses input → bottleneck → decoder reconstructs. Normal data is reconstructed well; anomalies are not. Anomaly score = reconstruction error (MSE).  
**Key design choice:** Bottleneck size relative to input dimension.

**Learning resources**
- Foundational chapter (Goodfellow et al., Deep Learning): https://www.deeplearningbook.org/contents/autoencoders.html
- Tutorial (PyTorch): https://lightning.ai/docs/pytorch/stable/notebooks/course_UvA-DL/08-deep-autoencoders.html
- Applied to anomaly detection: https://towardsdatascience.com/anomaly-detection-using-autoencoders-5b032178a1ea
- VAE variant (if extending): https://arxiv.org/abs/1312.6114

### 2.4 LSTM Autoencoder (LSTM-AE)
**Paradigm:** Temporal reconstruction-based  
**Logic:** Encodes a time window into a fixed-length vector via LSTM, then decodes it back. High reconstruction error on a window = anomaly.  
**Key design choice:** Window length, LSTM hidden units, threshold on reconstruction error.

**Learning resources**
- Seminal applied paper: Malhotra et al. (2016) — *LSTM-based Encoder-Decoder for Multi-sensor Anomaly Detection*  
  https://arxiv.org/abs/1607.00148
- PyTorch implementation walkthrough: https://curiousily.com/posts/time-series-anomaly-detection-using-lstm-autoencoder-with-pytorch-in-python/
- Tutorial on sliding window preprocessing: https://machinelearningmastery.com/convert-time-series-supervised-learning-problem-python/
- LSTM fundamentals (Colah's blog): https://colah.github.io/posts/2015-08-Understanding-LSTMs/

### 2.5 Masked Transformer Autoencoder (Transformer AE)
**Paradigm:** Masked self-supervised reconstruction  
**Logic:** Each of the 29 input features is treated as one token. During training ~40% of tokens are replaced with a learned mask embedding and the model reconstructs the originals at those positions using global self-attention. At inference the same masking is applied across multiple passes and the average MSE on masked positions is the anomaly score — matching the training objective exactly.  
**Key design choice:** Mask ratio, d_model, number of attention heads, number of encoder blocks.

**Why it is distinct from the LSTM-AE:**  
- Self-attention is global: any token can attend to any other, regardless of distance. The LSTM degrades over long sequences; the transformer does not.  
- On real time-series data (Dataset B) this makes the transformer the stronger reconstruction model for collective anomalies spanning many timesteps.  
- The token-level output gives a 2D reconstruction error map (timestep × feature), enabling finer-grained explainability than any other model in this project.

**Implementation:** `train_transformer_ae.py`  
**Architecture:** `Linear(1 → d_model) → Sinusoidal PE → TransformerEncoder(1 block, Pre-LN) → Linear(d_model → 1)`

**Learning resources**
- Masked Autoencoders Are Scalable Vision Learners (He et al., 2022): https://arxiv.org/abs/2111.06377
- Anomaly Transformer (Xu et al., 2022): https://arxiv.org/abs/2110.02642
- PyTorch TransformerEncoderLayer docs: https://pytorch.org/docs/stable/generated/torch.nn.TransformerEncoderLayer.html
- The Illustrated Transformer (Jay Alammar): https://jalammar.github.io/illustrated-transformer/

---

## 3. Datasets

### Dataset A — Tabular (no meaningful temporal order)

**Credit Card Fraud Detection (Kaggle)**
- URL: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
- Format: CSV, 284,807 rows, 30 features (PCA-transformed), 0.17% anomalies
- Why: Standard benchmark, extreme imbalance, no temporal dependence — ideal to show that sequence models add no value here
- Preprocessing: RobustScaler (resistant to outliers), no windowing needed

**Alternative: ODDS repository — HTTP / Shuttle / Cardiotocography**
- URL: http://odds.cs.cmu.edu/
- Lighter weight, useful for sanity checks and ablations

### Dataset B — Time-Series (strong temporal dependencies)

**Yahoo Webscope S5**
- URL: https://webscope.sandbox.yahoo.com/catalog.php?datatype=s&did=70 (free academic registration)
- Format: CSV files of univariate time series with labeled anomaly timestamps
- Why: Clean labels, multiple difficulty levels (A1–A4), standard benchmark in recent papers
- Preprocessing: sliding window (length 30–50 steps), MinMaxScaler per series

**Numenta Anomaly Benchmark (NAB)**
- URL: https://github.com/numenta/NAB
- Format: CSV, ~50 real-world time series (server metrics, traffic, tweets)
- Why: Open license, real-world noise, diverse anomaly types
- Preprocessing: same sliding window approach

**NASA CMAPSS (turbofan engine degradation)**
- URL: https://data.nasa.gov/dataset/CMAPSS-Jet-Engine-Simulated-Data/ff5v-kuh6
- Format: multivariate sensor readings, degradation labels
- Why: Multivariate → tests LSTM-AE in its strongest scenario

### Recommended minimal scope
Start with **Credit Card + Yahoo S5**. Add NAB as a third dataset if time allows.

---

## 4. Preprocessing Pipeline

### Tabular (Dataset A)
1. Drop `Time` column (not meaningful for this task)
2. `RobustScaler` on all features
3. Split: 70% train (normals only), 15% val, 15% test (both classes)
4. No windowing

### Time-Series (Dataset B)
1. `MinMaxScaler` per series (fit on train portion only)
2. Sliding window: window = 30, stride = 1
3. Train on anomaly-free windows; test on full set
4. For iForest and OC-SVM: flatten window + add rolling mean/std as hand-crafted features

---

## 5. Experimental Plan

### Phase 1 — Tabular experiments (Dataset A)
Run all four models on Credit Card data.

| Model    | Training input | Anomaly score |
|----------|---------------|---------------|
| OC-SVM   | Normal rows (scaled) | Decision function output |
| iForest  | Normal rows (scaled) | `score_samples()` |
| AE       | Normal rows | MSE reconstruction error |
| LSTM-AE  | Normal rows (as length-1 window, or skip) | MSE reconstruction error |

Expected outcome: OC-SVM and iForest should outperform LSTM-AE here.

### Phase 2 — Time-series experiments (Dataset B)
Run all four models on Yahoo S5 (or NAB).

| Model    | Training input | Anomaly score |
|----------|---------------|---------------|
| OC-SVM   | Flattened windows + lag features | Decision function |
| iForest  | Same as OC-SVM | `score_samples()` |
| AE       | Flattened windows | MSE per window |
| LSTM-AE  | Raw windows (shape: batch × steps × features) | MSE per window |

Expected outcome: LSTM-AE should outperform OC-SVM and iForest.

### Phase 3 — Cross-analysis
- Fix a decision threshold on validation F1 for each model-dataset pair
- Report all metrics on the held-out test set
- For each model, note where it degrades: which anomaly types does it miss?

---

## 6. Evaluation Metrics

**Do not report accuracy** — it is meaningless for imbalanced data.

| Metric | Why it matters |
|--------|---------------|
| **F1-Score** | Harmonic mean of precision and recall — primary metric |
| **PR-AUC** (Area under Precision-Recall curve) | Threshold-independent; superior to ROC-AUC under imbalance |
| **ROC-AUC** | Report for completeness and comparability with literature |
| **MCC** (Matthews Correlation Coefficient) | Fully accounts for all four cells of the confusion matrix |
| **Precision @ fixed recall** (e.g., R=0.8) | Operationally relevant — simulates a "catch 80% of anomalies" target |

**Practical metrics to also record:**

| Metric | How to measure |
|--------|---------------|
| Training time | `time.perf_counter()` around fit call |
| Inference time per sample | Same, divided by test set size |
| Model size | Number of parameters / `sys.getsizeof` of trained object |
| Threshold sensitivity | Plot F1 vs. threshold curve |

---

## 7. Statistical Robustness

- Run each experiment **5 times** with different random seeds; report mean ± std
- For iForest and OC-SVM: vary `contamination` / `nu` over `[0.01, 0.05, 0.1, 0.2]` to show sensitivity
- For AE and LSTM-AE: vary bottleneck size and window length; pick best on validation set

---

## 8. Explainability Direction

This is the recommended extension for the thesis. The goal is to not just rank models by performance, but to explain *why* each paradigm succeeds or fails.

### 8.1 SHAP for OC-SVM and Isolation Forest

SHAP (SHapley Additive exPlanations) assigns a contribution score to each input feature for a given prediction.

**For Isolation Forest:** Use `shap.TreeExplainer` — natively supported.

```python
import shap
explainer = shap.TreeExplainer(iso_forest_model)
shap_values = explainer.shap_values(X_test_anomalies)
shap.summary_plot(shap_values, X_test_anomalies)
```

**For OC-SVM:** Use `shap.KernelExplainer` (slower, model-agnostic).

This answers: *"Which features pushed this sample into the anomaly region?"*

**Resources:**
- SHAP paper: https://arxiv.org/abs/1705.07874
- SHAP docs & tutorials: https://shap.readthedocs.io/en/latest/
- iForest + SHAP example: https://towardsdatascience.com/explain-your-model-with-the-shap-values-bc36aac4de3d

### 8.2 Reconstruction Error Heatmaps for Autoencoders

For AE on tabular data: compute per-feature reconstruction error.

```python
recon = model.predict(X_test)
error_per_feature = np.abs(X_test - recon)  # shape: (n_samples, n_features)
# Plot as heatmap — anomalous samples × features
```

This answers: *"Which features the autoencoder fails to reconstruct → which parts of an anomalous sample are 'unusual'?"*

For LSTM-AE on time-series: compute per-timestep reconstruction error within a window.

```python
recon = model.predict(X_windows)
error_per_step = np.mean(np.abs(X_windows - recon), axis=-1)  # shape: (n_windows, window_len)
# Plot as time-aligned heatmap
```

This answers: *"At which point in time did the anomaly begin inside the window?"*

For the **Transformer AE**: the model outputs a reconstruction at every token position, giving a 2D error map `(timestep × feature)` at no extra cost. This is the richest resolution of any model — you can pinpoint both *when* and *which sensor* drove the anomaly score, which the LSTM AE and MLP AE only give in aggregate.

```python
# recon shape: (N, seq_len, 1) — one error per token
error_map = (orig - recon) ** 2  # (N, seq_len, 1)
# For multivariate windows: (N, seq_len, D) → heatmap over time and features
```

### 8.3 Anomaly Taxonomy Analysis

Categorize test set anomalies by type and measure per-type performance:

| Type | Description | Example |
|------|-------------|---------|
| **Point anomaly** | Single outlier value | A transaction 10× normal amount |
| **Contextual anomaly** | Normal value in wrong context | Normal CPU spike but at 3 AM |
| **Collective anomaly** | Sequence of individually normal values | Gradual sensor drift |

Hypothesis to validate:
- iForest excels at point anomalies (extreme feature values)
- LSTM-AE excels at collective/contextual anomalies (unusual patterns in time)
- AE sits between them

Report a breakdown table: model × anomaly type → F1.

### 8.4 Latent Space Visualization

For the Autoencoder: extract bottleneck activations for the test set, then use t-SNE or UMAP to project to 2D.

```python
encoder = Model(inputs=ae.input, outputs=ae.get_layer('bottleneck').output)
latent = encoder.predict(X_test)
# UMAP/t-SNE → scatter plot colored by true label
```

This shows: *"Does the model actually learn a compact and separated representation of normal vs. anomalous data?"*

A clear visual cluster separation = the model has learned meaningful structure.  
Overlapping clusters = the model is lucky with thresholds, not actually separating.

For the **Transformer AE**: use the mean-pooled encoder output (across all token positions) as the latent vector. This is equivalent to what the MLP AE bottleneck provides, but derived from global attention rather than a single linear bottleneck. Comparing the UMAP plots of both models reveals whether attention-based compression produces better-separated geometry than feedforward compression.

**Resources:**
- UMAP paper: https://arxiv.org/abs/1802.03426
- UMAP library: https://umap-learn.readthedocs.io/en/latest/
- t-SNE guide: https://distill.pub/2016/misread-tsne/

### 8.5 Sensitivity / Counterfactual Analysis

For each model, perturb an anomalous sample back toward normal and find the minimal change that flips it to "normal":

- For iForest: decrease the most SHAP-influential feature by stepwise increments; record when score crosses threshold
- For AE: set individual features to their training-set mean one at a time; record reconstruction error change

This answers: *"How robust is each model's decision boundary? Is it fragile (one feature change flips it) or stable?"*

This is a clean, thesis-quality analysis that shows not just performance numbers but the qualitative character of each paradigm.

### 8.6 Attention Weight Visualization (Transformer AE)

Each transformer block produces an attention weight matrix of shape `(seq_len, seq_len)` per head. Entry `[i, j]` reflects how strongly token `i` attends to token `j` when being reconstructed.

On normal windows the attention pattern is typically structured — nearby tokens or periodic positions attending to each other. On anomalous windows the model cannot find useful context and the pattern becomes diffuse or erratic. Visualizing the mean attention map across heads for a normal vs. anomalous sample side-by-side is an interpretable diagnostic.

```python
# Register a forward hook on the attention layer to capture weights
attention_maps = []
hook = model.encoder.layers[0].self_attn.register_forward_hook(
    lambda m, inp, out: attention_maps.append(out[1])  # out[1] is attn weights
)
_ = model(x_sample)
hook.remove()
# attention_maps[0]: (batch, nhead, seq_len, seq_len)
```

**Caveat:** raw attention weights reflect routing, not necessarily causal importance. Use as a visual diagnostic alongside the reconstruction heatmap, not as a standalone explanation.

### 8.7 Gradient-Weighted Attention and Integrated Gradients (Transformer AE)

Two more faithful attribution methods that go beyond attention weights:

**Gradient-weighted attention** multiplies each attention weight by the gradient of the anomaly score with respect to that weight. Positions with both high attention and high gradient are genuinely influential. This is the transformer analog of GradCAM.

```python
# Requires gradients — do not use torch.no_grad()
score = anomaly_score(model, x)
score.backward()
grad = model.encoder.layers[0].self_attn...  # gradient captured via hook
importance = (attention_weights * grad).abs().mean(dim=1)  # avg over heads
```

**Integrated Gradients** via PyTorch's `captum` library attributes the anomaly score to each input token by integrating gradients along a path from a baseline (training set mean) to the input. The result is a `(seq_len,)` attribution vector — directly comparable to the SHAP values computed for IForest and OC-SVM in 8.1.

```python
from captum.attr import IntegratedGradients
ig   = IntegratedGradients(lambda x: anomaly_score_fn(model, x))
attr = ig.attribute(x_input, baselines=x_baseline, n_steps=50)
# attr: (batch, seq_len, 1) — per-token attribution
```

This enables a cross-model comparison: "all four paradigms agree feature V14 is the most anomalous in this sample" is a much stronger thesis claim than per-model numbers alone.

**Resources:**
- Captum library: https://captum.ai/
- Integrated Gradients paper (Sundararajan et al., 2017): https://arxiv.org/abs/1703.01365
- Attention is not Explanation (Jain & Wallace, 2019): https://arxiv.org/abs/1902.10186
- Attention is not not Explanation (Wiegreffe & Pinter, 2019): https://arxiv.org/abs/1908.04626

---

## 9. Suggested Thesis Narrative Arc

1. **Introduction:** Why anomaly detection is hard (imbalance, no labels, domain shift)
2. **Background:** Taxonomy of methods — boundary, partition, reconstruction, temporal
3. **Experiments — Dataset A:** Show classical models are sufficient for tabular data
4. **Experiments — Dataset B:** Show temporal models are necessary for sequential data
5. **Cross-analysis:** Breakdown by anomaly type, sensitivity analysis
6. **Explainability chapter:** SHAP analysis + reconstruction heatmaps + latent space
7. **Discussion:** When to pick which model; practical deployment considerations
8. **Conclusion:** A decision framework — a simple flowchart for practitioners

---

## 10. Implementation Timeline (Rough)

| Week | Goal |
|------|------|
| 1 | Set up repo, download datasets, write preprocessing pipeline |
| 2 | Implement OC-SVM + iForest, run Phase 1 experiments |
| 3 | Implement AE, run Phase 1 + Phase 2 for classical models |
| 4 | Implement LSTM-AE, run Phase 2 experiments |
| 5 | Run all Phase 3 cross-analysis, compute all metrics, generate tables |
| 6 | SHAP analysis + reconstruction heatmaps |
| 7 | Latent space + anomaly taxonomy breakdown |
| 8 | Write thesis chapters, polish figures |
