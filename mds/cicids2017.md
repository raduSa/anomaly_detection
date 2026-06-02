# CIC-IDS2017 — Dataset Notes

## What it is

Network intrusion detection benchmark from the Canadian Institute for Cybersecurity. Generated over 5 days in a controlled network environment; attacks injected on Tuesday–Friday, Monday is 100% benign.

- ~2.8M rows, each row = one **bidirectional network flow** (5-tuple)
- ~80 statistical features extracted by CICFlowMeter (duration, packet counts, byte stats, flag counts, IAT, window sizes, etc.)
- `Label` column: `BENIGN` or one of 14 attack types (DoS Hulk, DDoS, PortScan, Bot, FTP-Patator, SSH-Patator, DoS GoldenEye, DoS slowloris, DoS Slowhttptest, Heartbleed, Web Attack – Brute Force, Web Attack – XSS, Web Attack – Sql Injection, Infiltration, Bot)
- Class imbalance: ~80% BENIGN; some attack classes have <50 flows

**Monday = BENIGN only → natural novelty detection split** (train on Monday, test on rest).

---

## Dataset problems (Engelen et al., WTMC 2021)

The raw/Kaggle CSVs are substantially corrupted. >25% of all flows are artefacts. Most published results are inflated via shortcut learning.

### 1. TCP appendices (25.9% of dataset)

CICFlowMeter closes a flow on the first FIN packet, not on the full two-way FIN/ACK. Every TCP connection therefore generates a real flow **plus** a tiny leftover "appendix" of 1–2 ACK/FIN packets, labelled the same class as the original. These appendices carry no attack content but are labelled `DoS Hulk`, `Bot`, etc. RF classifiers learn to detect classes by appendix artefacts (`Fwd Header Length`, `Init_Win_bytes_forward`, `SYN Flag Count`) instead of by actual attack patterns.

### 2. Empty attack flows (mislabelled as attacks)

Labelling only checks: *attacker IP + time window*. It does not verify a payload was sent. Result: large fractions of labelled attack flows have zero forward data transfer — no actual attack occurred.

| Class | Original | After payload filter | Drop % |
|---|---|---|---|
| Web Attack – Brute Force | 1507 | 151 | −90% |
| Web Attack – XSS | 652 | 27 | −96% |
| Bot | 1956 | 738 | −62% |
| DoS Slowhttptest | 5499 | 1742 | −68% |

These flows should be labelled `X – Attempted`, not the original attack class.

### 3. DoS Hulk is broken

The attack tool sends `Connection: Close` instead of `Connection: Keep-Alive` — the web server immediately terminates the connection. The attack never works. The entire `DoS Hulk` class is an invalid simulation and results on it are meaningless.

### 4. Shortcut learning via identifier columns

The raw CSVs include `Flow ID`, `Source IP`, `Destination IP`, `Timestamp`. Models that train on these learn to identify attacks by IP address or collection time window, not by traffic characteristics. **These columns must be dropped before training.**

---

## Cleaning approach (Engelen et al.)

Three-stage pipeline applied to re-extracted PCAPs:

1. **Fix CICFlowMeter** — reunite TCP appendices with their parent flows; honour RST as a valid terminator
2. **Payload filter** — relabel attack flows with no forward data transfer as `X – Attempted`
3. **Drop identifier columns** — `Flow ID`, `Source IP`, `Destination IP`, `Timestamp`

Cleaned dataset + corrected CICFlowMeter:
```
https://downloads.distrinet-research.be/WTMC2021
```
Use this, not the Kaggle/UNB CSVs.

---

## Anomaly type mapping (→ Yahoo S5 analogy)

| Yahoo benchmark | Anomaly type | CIC-IDS2017 analogue |
|---|---|---|
| A1 — point anomaly | Single outlier value | Heartbleed, Web Attacks — few flows with extreme feature values |
| A3 — contextual | Normal value, wrong context | Infiltration, Bot — individually normal flows, suspicious as a sequence |
| A4 — level shift | Sustained baseline change | DoS/DDoS floods — sustained high-rate traffic shifts the baseline |

---

## Key implications for novelty detection

- Novelty models trained on Monday BENIGN **cannot exploit IP/timestamp shortcuts by construction** — this is a more honest evaluation than supervised results in the literature
- Use cleaned dataset to avoid artefact-driven results on Bot and Web Attack classes
- Evaluate **per attack type** — aggregated metrics hide all class-level failure (Engelen et al. show 0.99 weighted-avg F1 while Bot F1 = 0.60 on the original data)
- Flag `DoS Hulk` separately or exclude it; the attack never executed
- For temporal models (LSTM, Transformer): group flows by **Src IP** (not 5-tuple) and sort by timestamp — see temporal preprocessing section below

---

## Preprocessing pipeline (implemented)

**Scripts:** `cicids2017/preprocess.py`, `train_iforest.py`, `train_ocsvm.py`, `train_autoencoder.py`
**Output:** `data/processed_cicids/` — `X_train.npy`, `X_test.npy`, `y_test.npy`, `attack_types_test.npy`, `feature_names.npy`

### Steps

1. **Drop identifier columns** — `Flow ID`, `Src IP`, `Dst IP`, `Timestamp`, `Src Port`  
   `Src Port` is ephemeral (random 1024–65535); `Dst Port` is kept as a legitimate service indicator.

2. **One-hot encode `Protocol`** — values {0=HOPOPT, 6=TCP, 17=UDP} carry no ordinal relationship; 1 column → 3 dummy columns.

3. **Inf → NaN** — `Flow Bytes/s` and `Flow Packets/s` produce `÷0` for zero-duration flows; replaced with NaN before imputation. This step was absent from `preprocess_suggestions.md`.

4. **NaN imputation** — per-column median computed on the training set, applied to both train and test. Affected columns: `Flow IAT Mean/Std/Max/Min` and the two rate columns above. Also absent from the suggestions.

5. **Zero-variance check** — checked on the Monday training set. `Fwd URG Flags`, `Bwd URG Flags`, `URG Flag Count` are zero-variance on Monday (URG flags never appear in clean Monday traffic) and are dropped. **80 → 77 features.**

6. **MinMaxScaler** — fit on Monday BENIGN training rows only, applied to test. No leakage.

7. **Train / test split (day-based)**  
   - Training: **Monday BENIGN rows only** (371,749 rows). Monday is 100% BENIGN by dataset design and is the natural reference period.  
   - Test: all Tuesday–Friday BENIGN rows + all real attack rows. `X – Attempted` flows excluded entirely.  
   - Aligning both flat and temporal pipelines to the same training day avoids a confound when comparing model results across paradigms.

### Final dataset shape

| Split | Rows | Features | Attack rate |
|---|---|---|---|
| Train | 371,749 | 77 | 0% (Monday BENIGN only) |
| Test | 1,719,921 | 77 | 25.2% |

### Attack type counts in test set

| Label | Rows | Anomaly type analogue |
|---|---|---|
| BENIGN | 1,285,944 | — |
| PortScan | 159,151 | point (high packet rate, distinctive flags) |
| DoS Hulk | 158,469 | ⚠️ broken simulation — attack never executed |
| DDoS | 95,123 | level shift (sustained flood) |
| DoS GoldenEye | 7,567 | level shift |
| DoS slowloris | 4,001 | level shift |
| FTP-Patator | 3,973 | point (brute-force login attempts) |
| SSH-Patator | 2,980 | point |
| DoS Slowhttptest | 1,742 | level shift |
| Bot | 738 | contextual (individually normal flows, suspicious sequence) |
| Web Attack – Brute Force | 151 | point |
| Infiltration | 32 | contextual |
| Web Attack – XSS | 27 | point |
| Web Attack – Sql Injection | 12 | point |
| Heartbleed | 11 | point (extreme feature values) |

---

## Temporal preprocessing pipeline (implemented)

**Script:** `cicids2017/preprocess_temporal.py`
**Output:** `data/processed_cicids_temporal/` — same files as flat pipeline plus shape `(N, W, F)` instead of `(N, F)`

### Why the Yahoo 5-tuple session approach does not work here

Yahoo's pipeline slides a window over one long continuous series per sensor. The natural CIC-IDS2017 analogue would be one window per TCP connection (5-tuple). This fails completely:

- Median flows per 5-tuple session: **1**. 75th percentile: **1**.
- At W=4, only 8.2% of sessions are long enough — and 0% of PortScan, FTP-Patator, SSH-Patator, and Web Attack sessions qualify.
- These attacks work *by opening many short connections*, not by sustaining one long one. A PortScan SYN sweep produces 159,151 separate 1-flow sessions.

### Grouping key: Src IP stream

Group all flows from the same source host, sorted by timestamp. This works because:

- The testbed has only ~48 machines — each host generates tens of thousands of ordered flows across the week.
- A PortScan attacker (172.16.0.1) generates 159,151 flows in one stream; the repeated short-connection burst is exactly the temporal pattern to detect.
- At W=16, Src IP streams cover **100% of flows** for 12/14 attack classes (only Heartbleed [11 flows] and SQL Injection [12 flows] fall below W).
- Semantically: one stream = the behavioral fingerprint of one host over time, which is the natural unit for host-based anomaly detection.

### Key differences from Yahoo preprocessing

| Property | Yahoo (`preprocess_timeseries.py`) | CIC-IDS2017 temporal |
|---|---|---|
| Stream definition | One sensor series | All flows from one Src IP |
| Input shape | `(W, 1)` | `(W, 77)` |
| Window size | W=64 | W=16 |
| Scaler | Per-series, fit on normal train points | Global MinMaxScaler, fit on Monday BENIGN rows only |
| Train/test split | Temporal 70/30 within each series | **Day-based: Monday = train, Tue–Fri = test** |
| Window label | 1 if any point anomalous | 1 if any flow in window is an attack |

### Two bugs found and fixed during development

**Bug 1 — Timeline gaps (label-split windowing):**  
An earlier version extracted `X_host[benign_pos]` and `X_host[attack_pos]` as separate arrays, then slid windows over each. Rows in a label-filtered subset are not necessarily consecutive in wall-clock time — attack flows between two BENIGN rows are silently skipped, so a window bridges a real time gap. Fix: slide windows over the **full sorted sub-stream** without removing rows first (`yw == 0` filters training windows after the fact, as in Yahoo).

**Bug 2 — Attack classes disappearing (position-split):**  
After Bug 1 was fixed with a position-based 80/20 split, most attack classes vanished from the test set. Root cause: the main attacker `172.16.0.1` generates 434k flows across Mon–Fri. The 80% split lands at row 347,552 — past ALL of PortScan, DoS Hulk, FTP-Patator, and SSH-Patator. Those flows fell in the training portion and were discarded by `yw == 0`. Fix: **day-based split** — attacks are concentrated on specific days; a position fraction cannot account for this. Monday as training, Tue–Fri as test resolves both coverage and alignment with the flat pipeline.

### Steps

1. Load all CSVs, attach `Day` column, drop `X – Attempted` flows.
2. Extract `Src IP`, `Day`, `Timestamp` for grouping/splitting; drop all from the feature matrix along with `Flow ID`, `Dst IP`, `Src Port`.
3. One-hot encode `Protocol`, replace Inf → NaN.
4. Impute NaN with Monday BENIGN medians; drop zero-variance columns on Monday BENIGN rows (removes 3 URG flag columns → 77 features).
5. Fit global MinMaxScaler on Monday BENIGN rows; transform all flows.
6. For each host: sort flows by timestamp → split by day.
   - Training: Monday flows, slide with `STRIDE_TRAIN=4`, keep only `yw == 0` windows.
   - Test: Tuesday–Friday flows, slide with `STRIDE_TEST=1`, label by any-attack rule, carry fine-grained type.

### Final dataset shape

| Split | Windows | Shape | Attack rate |
|---|---|---|---|
| Train | 92,884 | (92 884, 16, 77) | 0% |
| Test | 1,719,616 | (1 719 616, 16, 77) | 25.4% |

### Attack type counts in test windows

| Label | Windows | Note |
|---|---|---|
| BENIGN | 1,282,611 | — |
| PortScan | 159,160 | all 159k flows form windows in one long stream |
| DoS Hulk | 158,469 | ⚠️ broken simulation |
| DDoS | 95,131 | |
| DoS GoldenEye | 7,575 | |
| DoS slowloris | 4,001 | |
| FTP-Patator | 3,966 | |
| Bot | 3,320 | higher than flat — windows span transitions into attack behavior |
| SSH-Patator | 2,980 | |
| DoS Slowhttptest | 1,742 | |
| Infiltration | 445 | higher than flat — context windows capture surrounding normal flows |
| Web Attack – Brute Force | 151 | |
| Web Attack – XSS | 26 | |
| Web Attack – Sql Injection | 20 | |
| Heartbleed | 19 | |

---

## EDA plots (`cicids2017/cicids_dataset.py`)

Four figures saved to `data/`:

**`cicids_temporal_all.png`** — Stacked bar chart, 30-min buckets, all classes. BENIGN forms the base; attack classes stack on top ordered by total count descending. Shows the Monday training period (100% BENIGN) versus the attack injection windows Tuesday–Friday. Useful for visualising class imbalance and the day-based train/test split structure.

**`cicids_temporal_attacks.png`** — Attack-only line traces over the same 30-min buckets (BENIGN omitted so attack volumes are legible on their own y-scale). Reveals each class's temporal injection window: PortScan is a single sustained burst on Friday, DDoS appears Wednesday, Patator attacks run across Tuesday–Thursday, Bot/Infiltration are sparse and intermittent.

**`cicids_feature_profiles.png`** — 2×3 grid of overlapping normalised histograms, one subplot per feature, one histogram per class. Six features chosen to cover distinct axes of variation:
- *Flow Duration*, *Total Fwd Packet*, *Flow Bytes/s*, *Flow IAT Mean* — log x-axis; cover volume and timing axes
- *SYN Flag Count* — linear; near-binary for PortScan vs everything else
- *Fwd Packet Length Mean* — linear; payload size separates brute-force from flood attacks
X-axis clipped to 1st–99th percentile on linear-scale subplots to prevent outliers from collapsing the visible distribution.

**`cicids_feature_structure.png`** — Two-panel figure on BENIGN rows only:
- *Left*: lower-triangle Pearson correlation heatmap (77×77) with feature labels at fontsize 4.5. Shows redundancy clusters (e.g. all `Fwd Packet Length *` stats are highly correlated), motivating bottleneck architectures.
- *Right*: top-20 features by standardised mean difference between BENIGN and all attacks. Red bars = attacks higher; blue = attacks lower. Confirms the feature space has learnable structure before any model is run.

---

## Classical models on temporal windows (IForest, OC-SVM)

CICFlowMeter flows are already session-level aggregates — each row contains temporal statistics computed over the lifetime of a session (IAT mean/std, flow duration, packet counts, etc.). This distinguishes CIC-IDS2017 from Yahoo S5, where each point is an instantaneous scalar and windowing is the only way to give the model context.

Windowing over CIC flows creates a **behavioral episode vector**: 16 consecutive sessions from the same host concatenated into a 1232-dim vector (16 × 77). When IForest or OC-SVM is applied to this flattened vector, it can detect anomalous *patterns of behavior* across consecutive sessions that no single-flow model would catch. A PortScan generates many short connections to sequential ports — any individual flow looks unremarkable, but 16 consecutive identical short flows to different ports is an outlier cluster in 1232-dim space.

However, flattening loses sequential order. IForest and OC-SVM treat all 1232 dimensions symmetrically, so "flow-1-duration followed by flow-2-duration" is indistinguishable from the reverse. The LSTM and Transformer models are the ones that actually exploit ordering.

This creates a deliberate two-level comparison for the temporal pipeline:

| Model class | What it captures |
|---|---|
| IForest / OC-SVM on flat windows | "Bag of flows" — presence of unusual flow statistics, no ordering |
| LSTM AE / Predictive LSTM / Transformer | True sequential structure — the order and transition between flows |

The performance gap (or absence of one) between these two groups directly evidences whether *ordering* within a behavioral episode matters for intrusion detection. If LSTM significantly outperforms flattened IForest on contextual attacks (Bot, Infiltration) but not on point attacks (PortScan, Patator), that maps cleanly onto the thesis claim about inductive bias matching anomaly type.
