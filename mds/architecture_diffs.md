# Architecture Differences Across Datasets

## Statistical Robustness

Run **3 seeds** for the best-performing neural model per dataset only. IForest and OC-SVM do not need seed analysis — IForest variance is negligible at n_estimators=150, and OC-SVM is a convex optimisation (deterministic given fixed data).

**Validation split.** All neural network training scripts hold out a random 10% of the training data for monitoring purposes only. The split uses `np.random.default_rng(0).permutation` so it is deterministic and consistent across runs. The val set is never used to select hyperparameters or stop training early — it exists solely to print val loss alongside train loss each epoch and to produce the loss curve saved to `data/graphs/results/<dataset>/<model>_loss_curve.png`. The model is trained on the remaining 90% and evaluated on the separate held-out test set as before.

Do not sweep model hyperparameters. The goal is paradigm comparison under consistent, principled configurations — not finding the optimal configuration per model. Sweeping would require a proper held-out validation set for each model/dataset combination and would answer a different question. The architectural choices (compression ratios, hidden dims, LR=1e-3) are already principled and defensible as-is. The one exception: if a model performs unexpectedly poorly, check LR ∈ {1e-4, 1e-3, 3e-3} as a sanity check before drawing conclusions.

---

## MLP Autoencoder

| | CIC flat | CIC context | Yahoo | Credit card |
|---|---|---|---|---|
| Input dim | 77 | 86 | 64 (W×1 flattened) | ~30 (PCA) |
| HIDDEN | [64, 32, 16] | [64, 32, 16] | [32, 16] | [16, 8] |
| Bottleneck ratio | 0.21 | 0.19 | 0.25 | ~0.27 |
| Final activation | Sigmoid | Sigmoid | Linear | Linear |

**Layer widths** scale with input size to maintain a consistent bottleneck compression ratio of ~0.20–0.27. Using the same absolute widths across datasets would over-compress small inputs or under-compress large ones.

**Final activation** differs because CIC uses MinMaxScaler (data bounded to [0, 1] → Sigmoid keeps reconstruction in range), while Yahoo uses RobustScaler and credit card uses StandardScaler (both unbounded → linear output).

Everything else — LR, epochs, dropout (0.1), batch size — is identical across datasets.

---

## Isolation Forest

| | CIC flat | CIC context | Yahoo | Credit card |
|---|---|---|---|---|
| N_ESTIMATORS | 150 | 150 | 150 | 150 |
| MAX_SAMPLES | 300,000 | 300,000 | min(n, 50,000) | 200,000 |
| Training set size | ~371k | ~371k | ~few thousand | ~227k |
| **% of train used** | **~81%** | **~81%** | **~100%** | **~88%** |
| RANDOM_SEED | 123 | 123 | 123 | 123 |

**MAX_SAMPLES** is the only meaningful difference. It is set as high as practically feasible for each dataset — IForest accuracy improves with more samples per tree up to a point of diminishing returns. For Yahoo, the training set is smaller than the 50k cap so all samples are used. The percentage used (~80–100%) is consistent enough across datasets to not be a concern.

Yahoo's cap of 50,000 is never reached in practice and is kept as a safety ceiling.

---

## OC-SVM

| | CIC flat | CIC context | Yahoo | Credit card |
|---|---|---|---|---|
| NU | 0.01 | 0.01 | 0.01 | 0.01 |
| KERNEL | `'rbf'` | `'rbf'` | `'rbf'` | `'rbf'` |
| GAMMA | `'scale'` | `'scale'` | `'scale'` | `'scale'` |
| SUBSAMPLE | 50,000 | 50,000 | none (full) | none (full) |
| **% of train used** | **~13.5%** | **~13.5%** | **~100%** | **~100%** |

All parameters are now uniform. The only intentional difference is subsampling (see IForest note above for the same reasoning). Credit card (~227k samples) does not subsample — training will be slow but feasible; add a `SUBSAMPLE` cap if runtime becomes a problem.

---

## Predictive LSTM

| | CIC | Yahoo | Credit card |
|---|---|---|---|
| INPUT_SIZE | 77 | 1 | 1 |
| HIDDEN_DIM | 128 | 64 | 64 |
| NUM_LAYERS | 1 | 1 | 1 |
| EPOCHS | 20 | 20 | 50 |
| BATCH_SIZE | 256 | 256 | 512 |

**HIDDEN_DIM** scales with input size for CIC (128 for INPUT_SIZE=77, a moderate ~1.66× expansion). Both univariate datasets (Yahoo, credit card) use 64, which provides sufficient capacity to learn temporal dependencies in a scalar signal.

**BATCH_SIZE** for credit card is larger (512 vs 256) proportionate to its larger training set — same reasoning as AE.

**EPOCHS**: CIC and Yahoo converge by epoch ~5 on this model; 20 is conservative. Credit card (feature-as-sequence) shows continued decline past 30 epochs due to its non-temporal structure making optimisation noisier — confirmed at 50.

Note: the credit card LSTM treats the 29 PCA feature dimensions as a pseudo-sequence and learns to predict each component from the previous ones. There is no temporal ordering — the sequence axis is feature-space, not time. This is architecturally identical to CIC/Yahoo but conceptually different.

---

## LSTM Autoencoder

| | CIC | Yahoo | Credit card |
|---|---|---|---|
| INPUT_SIZE | 77 | 1 | 1 |
| HIDDEN_DIM | 128 | 64 | 64 |
| NUM_LAYERS | 1 | 1 | 1 |
| EPOCHS | 50 | 30 | 50 |
| BATCH_SIZE | 256 | 256 | 512 |

Architecture is identical across all three: encoder LSTM compresses the sequence to a hidden state, which is repeated as decoder input at every step, then a linear projection reconstructs the original feature at each position. HIDDEN_DIM and BATCH_SIZE follow the same reasoning as the predictive LSTM above.

EPOCHS differs: Yahoo converges by epoch ~5 so 30 is already generous. CIC and credit card both required 50 epochs to plateau — confirmed by loss curves showing continued decline at 30.

---

## Bottleneck Transformer AE

| | CIC | Yahoo | Credit card |
|---|---|---|---|
| D_MODEL | 64 | 64 | 64 |
| NHEAD | 4 | 4 | 4 |
| NUM_LAYERS | 2 | 2 | 2 |
| DIM_FF | 256 | 256 | 256 |
| LATENT_DIM | 32 | 32 | 32 |
| EPOCHS | 50 | 20 | 20 |
| BATCH_SIZE | 256 | 256 | 512 |

**EPOCHS**: CIC was still declining at 30 epochs — confirmed by loss curve to plateau around epoch 40, so 50 is used. Yahoo and credit card both converge by epoch ~15; 20 is conservative. BATCH_SIZE for credit card is larger for the same reason as other models.

**LATENT_DIM** is the bottleneck applied after global average pooling, which always produces a (B, D_MODEL) vector regardless of sequence length or input size. LATENT_DIM must be strictly less than D_MODEL to create a real information constraint — if LATENT_DIM=D_MODEL the squeeze+expand pair is unconstrained and provides no bottleneck. LATENT_DIM=32 gives 50% compression relative to D_MODEL=64 uniformly across all datasets.

Note: the existing CIC results in `results_analysis.md` were obtained with LATENT_DIM=64 (no bottleneck). Re-running is needed for valid comparison.

---

## Causal Transformer

| | CIC | Yahoo |
|---|---|---|
| D_MODEL | 64 | 64 |
| NHEAD | 4 | 4 |
| NUM_LAYERS | 2 | 2 |
| DIM_FF | 256 | 256 |
| DROPOUT | 0.1 | 0.1 |
| EPOCHS | 30 | 30 |
| BATCH_SIZE | 256 | 256 |

No differences — fully consistent. WINDOW (16 vs 64) and INPUT_SIZE (77 vs 1) are the only variations and are dataset-specific by definition. No credit card implementation exists for this architecture.

