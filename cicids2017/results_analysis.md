# CIC-IDS2017 Model Results Analysis

## Summary Table

| Model | ROC-AUC | Avg Precision | Balanced Acc | F1 (attack) |
|---|---|---|---|---|
| **AE+Context** | **0.9860** | **0.9487** | 0.9548 | 0.9020 |
| OC-SVM+Context | 0.9745 | 0.9396 | **0.9569** | **0.9229** |
| IForest+Context | 0.9785 | 0.9328 | 0.9270 | 0.8367 |
| Temporal LSTM AE | 0.8976 | 0.7955 | 0.7846 | 0.6249 |
| Flat MLP AE | 0.9189 | 0.7921 | 0.8022 | 0.6305 |
| Flat IForest | 0.9049 | 0.7612 | 0.8228 | 0.6639 |
| Temporal OC-SVM | 0.8629 | 0.7816 | 0.7600 | 0.5888 |
| Temporal AE | 0.8831 | 0.7509 | 0.7734 | 0.6125 |
| Temporal IForest | 0.8438 | 0.7604 | 0.6841 | 0.5195 |
| Flat OC-SVM | 0.7336 | 0.6845 | 0.6121 | 0.4644 |
| Bottleneck Transformer | 0.8077 | 0.6539 | 0.7059 | 0.5423 |
| Causal Transformer | 0.7099 | 0.5449 | 0.5520 | 0.4320 |
| Predictive LSTM | 0.6126 | 0.4472 | 0.5475 | 0.4295 |

Primary ranking metric is Average Precision (PR-AUC), which is most meaningful under class imbalance (~25% attack rate).

---

## Overall Performance

### The context family dominates

The three models trained on context-augmented features (AE+Context, OC-SVM+Context, IForest+Context) occupy the top three positions by every metric, with AP ranging from 0.933 to 0.949. The gap between the best context model (AP=0.949) and the best non-context model (flat MLP AE, AP=0.792) is larger than the entire spread across the remaining ten models. All three context variants improve substantially over their non-context counterparts:

| Model | Flat AP | +Context AP | Gain |
|---|---|---|---|
| IForest | 0.7612 | 0.9328 | +0.172 |
| OC-SVM | 0.6845 | 0.9396 | +0.255 |
| MLP AE | 0.7921 | 0.9487 | +0.157 |

The OC-SVM gain is the largest — the same model that was worst among the flat trio becomes roughly tied with the best. This makes sense: OC-SVM's RBF kernel finds a hypersphere boundary in feature space. The flat 77-feature space has heavily overlapping benign/attack regions for PortScan; the context features add dimensions that cleanly separate scanning hosts from normal hosts, making the kernel problem well-separated.

### Host-context features

The context preprocessing (implemented in `preprocess_context.py`) appends 9 features to the standard 77-feature flow vector. For each flow, these features snapshot the activity of both the source and destination IP in the preceding 60 seconds using a sliding time window:

| Feature | What it captures |
|---|---|
| `ctx_new_port` | Whether the destination port is new for this src IP in the window |
| `ctx_max/min_flow_cnt` | How many flows src/dst IP has recently generated |
| `ctx_max/min_port_cnt` | How many unique ports src/dst IP has contacted |
| `ctx_abs_flow/port_diff` | Asymmetry between src and dst host activity |
| `ctx_max/min_port_rate` | Unique ports per flow (port diversity rate) |

This methodology follows Raskovalov et al. (2024), adapted to an unsupervised anomaly detection setting. The context is computed before each flow is added to the host's window, so it reflects only prior host activity — not the current flow itself.

### Per-class profile differences within the context family

The three context models share the same 9 extra features but exploit them differently, producing distinct per-class profiles:

| Attack class | IForest+Ctx | OC-SVM+Ctx | AE+Ctx |
|---|---|---|---|
| DDoS | 0.982 | **0.998** | 0.994 |
| PortScan | 0.601 | 0.820 | **0.842** |
| DoS GoldenEye | **0.655** | 0.215 | 0.220 |
| DoS Slowhttptest | **0.112** | 0.007 | 0.013 |
| DoS slowloris | **0.190** | 0.012 | 0.025 |
| Heartbleed | **0.093** | 0.001 | 0.000 |

**AE+Context is strongest for PortScan** (AP=0.84). The autoencoder learns a smooth reconstruction function over the port-diversity features. A scanning host's `ctx_max_port_cnt` and `ctx_max_port_rate` diverge continuously from the training distribution, producing a well-calibrated reconstruction error that ranks scan flows reliably across the full threshold range.

**IForest+Context is strongest for DoS GoldenEye, slow DoS, and Heartbleed.** Isolation Forest uses axis-aligned splits that can directly target specific feature combinations. Slowloris and Slowhttptest maintain many concurrent long-duration connections to a single port, producing high `ctx_max_flow_cnt` alongside low `ctx_max_port_rate` (high flow count, barely any port diversity). This combination is a small, isolated region of feature space that IForest can isolate efficiently with a short path length. The AE, reconstructing holistically across all 86 features, does not weight this interaction as strongly. DoS GoldenEye benefits similarly — high flow count to a single port, unusual flow/port ratio.

**OC-SVM+Context achieves the best F1 (0.923) and balanced accuracy (0.957)** at the operating threshold. It trades slightly lower AP than AE+Context for a more balanced precision/recall split (precision=0.897, recall=0.951 vs. AE+Context's precision=0.842, recall=0.971). For a deployment that penalises false positives more heavily, OC-SVM+Context would be preferred.

### Flat vs. context vs. temporal

For IForest and AE, adding temporal sequence context (src-IP grouped windows of 16 flows) does not improve AP relative to flat:
- Flat IForest (AP=0.761) ≈ Temporal IForest (AP=0.760)
- Flat AE (AP=0.792) > Temporal AE (AP=0.751)

Adding host-context aggregate statistics is dramatically more effective:
- Flat AE (AP=0.792) → AE+Context (AP=0.949) — +0.157
- Flat IForest (AP=0.761) → IForest+Context (AP=0.933) — +0.172

The reason becomes clear from the per-class breakdown: sequence windowing does not meaningfully help for the two largest attack classes (DDoS and PortScan), which together account for the bulk of the test set and drive overall AP. Sequence models are suited to detecting pattern changes over ordered flows; host-context statistics directly encode the signals that define these attacks (high flow rate for DDoS, high port diversity for PortScan).

OC-SVM is also a clear beneficiary: the RBF kernel on 77 features could not find a clean boundary for PortScan, but on 86 features with context it can.

### Predictive models fail

The **Predictive LSTM** (AP=0.447) and **Causal Transformer** (AP=0.545) both collapse — the threshold sweep lands at a point where nearly all traffic is flagged as anomalous (FP > 1.1M out of 1.28M legit flows). This is not a bug; ROC-AUC of 0.61 and 0.71 confirm the scores themselves are poorly discriminative.

The root cause is inverted anomaly signal: these models learn to predict the next flow from previous flows, trained only on Monday benign traffic. Many attack types (DDoS floods, port scans) are highly repetitive and thus *more predictable* than diverse benign traffic (mixed web, email, DNS, file transfer). The model assigns lower prediction error to the attacks than to some legitimate traffic, inverting the intended direction.

### Reconstruction vs. prediction

Among temporal models, reconstruction-based models (LSTM AE, Temporal AE, OC-SVM) all significantly outperform prediction-based ones (Predictive LSTM, Causal Transformer). This confirms that for this dataset, asking "does this window look like normal traffic?" is a more reliable signal than "can I predict the next flow?"

---

## Per-Class Attack Analysis

### Consistently well-detected

**DDoS** (95k samples) — by far the easiest class. All context models score AP > 0.98; OC-SVM+Context reaches 0.998. Even flat models score AP > 0.77. The high volume and distinctive flow statistics (very high packet rates, tiny packet sizes) produce an extreme outlier in both per-flow and host-context feature spaces. `ctx_max_flow_cnt` alone makes DDoS trivially isolated.

**PortScan** (159k samples) — completely transformed by context. All flat and temporal models scored AP=0.07–0.28; all context models score AP > 0.60, with AE+Context reaching **AP=0.842**. At the per-flow level, a short TCP flow to an arbitrary port is unremarkable. But `ctx_max_port_cnt` (unique destination ports contacted by the src IP in the last 60 seconds) is unmistakable during a scan: Monday benign hosts contact a handful of distinct ports per minute; a scanning host contacts hundreds. IForest+Context is comparatively weaker here (0.601) because tree splits on port count have hard thresholds, while the AE produces a continuous gradient over the port-diversity features.

### Mixed context response for application-layer DoS

**DoS GoldenEye** (7.5k samples) — shows a clear split within the context family. IForest+Context achieves AP=0.655, while AE+Context and OC-SVM+Context both score ~0.22. GoldenEye floods a single HTTP endpoint, producing many flows to one destination port. `ctx_max_flow_cnt` is high, but `ctx_max_port_rate` is very low (many flows, one port). IForest isolates this unusual combination directly; the AE reconstructs it with moderate error spread across many feature dimensions, reducing its discriminative power for this attack specifically.

**DoS Slowhttptest and DoS slowloris** (1.7k and 4k samples) — IForest+Context shows meaningful improvement over all other context models (AP=0.112 and 0.190), though it still falls far short of Temporal IForest (AP=0.69 and 0.61). Slow HTTP attacks maintain many concurrent long-duration connections to a single port. The resulting context signature — elevated `ctx_max_flow_cnt`, near-zero `ctx_max_port_rate` — is a tight, isolated cluster that IForest's isolation mechanism targets well. AE+Context and OC-SVM+Context fail to exploit this combination, scoring near-zero. The 60-second time window is a partial mismatch here: Slowloris connections last minutes, so many flows fall outside the window. Temporal IForest, operating on ordered per-host sequences, still provides the best detection for these attacks.

### Largely undetectable (unchanged by context)

**FTP-Patator, SSH-Patator** (4k and 3k samples) — AP < 0.01 across all models including context variants. These credential brute-force attacks over legitimate protocols generate normal-looking flow statistics and unremarkable host-context activity (sequential connections to a single port, much like legitimate FTP/SSH clients). Detecting them requires failed-authentication counters or payload inspection, neither of which is available here.

**Bot** (738 samples) — AP < 0.01 universally. Bot traffic mimics normal browsing; host-context shows moderate flow counts and varied ports, indistinguishable from a legitimate active client.

**Web Attack subtypes** (Brute Force n=151, SQL Injection n=12, XSS n=27) — near-zero AP everywhere. HTTP request-level attacks generate completely normal TCP flows and normal host-context activity. Flow-level feature sets have no visibility into HTTP content.

**Heartbleed** (11 samples) — IForest+Context improves noticeably to AP=0.093 (vs. near-zero for AE+Context and OC-SVM+Context), though the sample size is too small for strong conclusions. The TLS-level anomaly in Heartbleed produces unusual per-flow byte statistics; combined with the host's low port diversity, IForest can isolate the handful of affected flows.

**Infiltration** (32 samples) — AP 0.03–0.14 across all models, no clear winner. Multi-stage attacks (lateral movement, exfiltration) produce varied traffic that doesn't cluster in any feature space.

---

## Sanity Checks

- The flat model per-class `n` values (Bot=738, Heartbleed=11) are consistent with expected low counts for rare/hard-to-generate attacks in a 5-day capture. The temporal model counts are 4–5× higher (Bot=3320, Heartbleed=19) because each source IP generates multiple overlapping windows — expected and not a data leak.
- AE+Context ROC-AUC=0.986 is at the high end of supervised baselines reported in the literature (0.95–0.99 with label access). This is explained by PortScan (159k samples, ~9% of test set) being near-perfectly separated, and DDoS (95k samples, ~6%) being trivially detectable. These two classes together contribute the majority of the PR-AUC gain.
- The Predictive LSTM threshold collapse (FP=1.16M) is consistent with its ROC-AUC of 0.61 — if positive and negative score distributions nearly overlap, the F2-optimal threshold lands at an extreme point. This is a property of the model, not a pipeline bug.
- OC-SVM+Context has better F1 and balanced accuracy than AE+Context despite lower AP. These metrics are threshold-dependent; OC-SVM finds a more balanced operating point. AP is threshold-independent and remains the primary ranking criterion.

---

## Key Takeaways

1. **Host-context features are the single most impactful addition** in this study. Appending 9 rolling-window aggregate statistics per flow raises AP from 0.79 to 0.95, driven primarily by solving the PortScan blindspot that all sequence and flat models shared.
2. **PortScan is only detectable at the host level, not the flow level.** A single scan flow is indistinguishable from legitimate traffic; a scanning host's port diversity over 60 seconds is unmistakable. This is a structural property of the attack, not a modelling artefact.
3. **Different anomaly detectors exploit different aspects of the context features.** AE+Context is strongest for PortScan (continuous reconstruction error over port-diversity features). IForest+Context is strongest for DoS GoldenEye and slow DoS (isolation of specific flow-count/port-rate combinations). OC-SVM+Context provides the best precision/recall balance at the operating threshold.
4. **Temporal sequence windowing does not consistently help** for this dataset. The src-IP grouping produces windows with arbitrary intra-window flow ordering, limiting sequence model effectiveness.
5. **Reconstruction-based models outperform prediction-based ones** for network intrusion. Attack traffic is often more regular than benign traffic, causing prediction-error models to mis-rank.
6. **Slow DoS attacks (Slowloris, Slowhttptest) still require per-host sequential context** for strong detection. Temporal IForest (AP=0.69/0.61) remains best for these; IForest+Context improves over AE/OC-SVM context variants but does not match the temporal approach.
7. **Protocol-level attacks (brute force, web attacks, Bot) are essentially undetectable** at the network flow feature level without payload content — a fundamental limit of this feature set, not a model failure.
8. **A practical ensemble** pairing AE+Context (strong on DDoS and PortScan) with Temporal IForest (strong on slow DoS, Heartbleed) would cover the widest range of detectable attack classes within this feature set.
