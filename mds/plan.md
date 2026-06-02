🧠 1. Your Initial Algorithm Choices — Are They Good?

You mentioned:

Isolation Forest
LSTM-based anomaly detection

👉 This is actually an excellent contrast:

Model	Strength	Weakness
Isolation Forest	Works well on tabular/static data, fast, interpretable-ish	Ignores temporal structure
LSTM	Captures time dependencies, good for sequences	Harder to train, needs more data

💡 This already gives you a core thesis question:

“How do classical vs sequence-based anomaly detection methods behave on different data types?”

📊 2. Choosing the Right Datasets (This Is Crucial)

You want datasets that highlight differences, not hide them.

🟢 Best Strategy: Use 2–3 Complementary Dataset Types
🔹 A. Financial Transaction Dataset (Tabular, Non-Temporal)
Good option:
Credit Card Fraud Detection (Kaggle)

Why it’s perfect:

Highly imbalanced (real anomaly scenario)
Tabular (no temporal structure needed)
Benchmark dataset → easy to justify

What you’ll see:

Isolation Forest → works well
LSTM → probably overkill or underperforms

👉 This shows: sequence models aren’t always needed

🔹 B. Time-Series Dataset (Sequential)
Excellent options:
Yahoo Webscope S5 (time-series anomalies)
NAB (Numenta Anomaly Benchmark)
NASA Turbofan Engine (CMAPSS)

Why:

Clear temporal dependencies
Known anomaly points
Widely used in research

What you’ll see:

LSTM → shines
Isolation Forest → struggles unless features are engineered

👉 This shows: temporal modeling matters

🔹 C. Network / System Metrics (Hybrid)
Options:
CICIDS2017 (network intrusion dataset)
Server metrics (CPU, memory logs)

Why:

Semi-temporal (can be both tabular + sequential)
Real-world complexity

👉 This is your bridge dataset — where neither method is perfect.

⚙️ 3. Additional Algorithms You Should Consider

To make your comparison more meaningful, add 1–2 more models:

🔸 Classical (easy to implement)
Local Outlier Factor (LOF)
One-Class SVM

👉 Good baselines, widely cited.

🔸 Strong modern baseline
Autoencoder (feedforward)

Why:

Works on both tabular and time-series (with slight changes)
Bridges gap between classical and deep learning
🔸 For time-series
LSTM Autoencoder (better than raw LSTM)

Instead of predicting:
👉 reconstruct sequence → anomaly = reconstruction error

💡 Suggested final set:
Isolation Forest
LOF or One-Class SVM
Feedforward Autoencoder
LSTM Autoencoder

👉 This gives you a clean narrative across paradigms

🧪 4. What You Should Actually Compare

Don’t just compare accuracy — that’s too shallow.

Instead, compare:

📌 Performance:
Precision / Recall / F1
ROC-AUC
📌 Behavior:
Sensitivity to noise
Sensitivity to anomaly type (point vs contextual)
📌 Practical aspects:
Training time
Inference time
Data requirements
📌 Stability:
Does it overfit?
Does it generalize across datasets?
🔍 5. How to Structure Your Experiments

A clean experimental design:

Phase 1:
Train all models on Dataset A (financial)
Phase 2:
Train all models on Dataset B (time-series)
Phase 3:
Cross-evaluate insights:
What breaks?
What generalizes?
🚀 6. Extension Direction (Choose One Later)

You already identified two excellent directions:

🧭 Option 1: Explainability (Highly Recommended)

This fits very naturally after your comparison.

Ideas:
Why did Isolation Forest flag this transaction?
What features contributed most?
Visualize:
Reconstruction error heatmaps (for autoencoders)
Feature importance (SHAP, LIME)
Output:
“Explainable anomaly detection framework”

👉 This is:

Very thesis-friendly
Not too heavy computationally
Academically appealing
⚡ Option 2: Online / Real-Time Detection System
Ideas:
Process streaming data (Kafka / simulated stream)
Sliding window anomaly detection
Compare batch vs online models
Challenges:
More engineering-heavy
Time-consuming

👉 Better if you enjoy systems / backend work.

🧠 My Honest Recommendation

Based on your plan:

👉 Best path:

Start with:
Isolation Forest
LSTM Autoencoder
1 classical baseline
Use:
1 tabular dataset
1 time-series dataset
Then extend into:
✅ Explainability
🧩 What Your Thesis Could Become

A strong final title could look like:

“Comparative and Explainable Anomaly Detection Across Tabular and Time-Series Data”

moving from classical baselines to deep learning architectures. By testing these "Core Four" models across different data structures, you can scientifically demonstrate when complexity is an asset and when it is a liability.1. The Core Four: Model RundownThese four models represent the four fundamental "philosophies" of detecting outliers.ModelParadigmDetection LogicBest Suited ForOne-Class SVMBoundary-basedLearns a soft boundary (hypersphere) around normal data in a high-dimensional kernel space.Low-to-medium dimensional tabular data.Isolation ForestPartition-basedRandomly splits data; anomalies are "isolated" faster (fewer splits) than normal points.High-dimensional tabular data; fast & scalable.AutoencoderReconstruction-basedCompresses data and tries to rebuild it. Anomalies have high Reconstruction Error.Tabular and image data; non-linear patterns.LSTM (Predictive)Temporal-basedLearns to predict the next value in a sequence. Anomalies have high Prediction Residuals.Time-series and sequential logs.2. Experimental Plan & DatasetsTo make your thesis "concrete," you must test each model on two contrasting datasets.Dataset A: The Tabular ChallengeTarget: Credit Card Fraud Detection (Kaggle) or the macrOData benchmark (a 2026 standard for tabular outliers).Preprocessing: Robust scaling (to handle outliers) and handling the massive class imbalance (e.g., 0.1% anomalies).Expected Outcome: Isolation Forest and Autoencoders typically dominate here. LSTM will likely fail or underperform because there is no meaningful "past" to predict.Dataset B: The Sequential ChallengeTarget: Yahoo Webscope S5 or the Numenta Anomaly Benchmark (NAB).Preprocessing: Sliding window transformation. You must convert raw time steps into overlapping sequences (e.g., windows of 10–50 time steps).Expected Outcome: LSTM and Autoencoders (with temporal layers) will excel. iForest and OC-SVM will struggle unless you manually create "lag features" (e.g., the rolling average of the last hour).3. Evaluation Metrics (The Thesis "Teeth")In anomaly detection, Accuracy is a trap. If 99% of your data is normal, a model that says "everything is normal" has 99% accuracy but is useless. You must use:Precision-Recall (PR) AUC: Superior to ROC-AUC for highly imbalanced datasets.F1-Score: The harmonic mean of Precision and Recall.$$F_1 = 2 \cdot \frac{Precision \cdot Recall}{Precision + Recall}$$Matthews Correlation Coefficient (MCC): Increasingly favored in 2026 research for its ability to provide a balanced score even with extreme imbalance.$$MCC = \frac{TP \cdot TN - FP \cdot FN}{\sqrt{(TP+FP)(TP+FN)(TN+FP)(TN+FN)}}$$4. Variations and Hybrid ParadigmsTo elevate your thesis, you can include one or two "advanced" variations that combine these philosophies.A. Variations on the Base ModelsExtended Isolation Forest (EIF): Uses random slopes instead of axis-parallel splits. This removes the "ghosting" artifacts found in the standard version.LSTM-Autoencoder (LSTM-AE): Instead of just predicting the next step, it reconstructs the entire sequence. This is currently a state-of-the-art standard for industrial sensor data (Moussa & Alazzawi, 2026).Variational Autoencoder (VAE): Instead of a fixed reconstruction, it provides a probability distribution, allowing you to quantify uncertainty.B. Hybrid Combinations (Paradigm Merging)Deep SVDD: Combines the Deep Learning power of an Autoencoder with the Boundary logic of an OC-SVM. It trains a network to map data into the smallest possible hypersphere.CNN-LSTM: Uses Convolutional layers to extract features across sensors (spatial) and LSTM to find patterns over time (temporal).AE + Forest (Feature Extraction): Use the Autoencoder to reduce the dimensionality of your data into a "latent code," then run Isolation Forest on that code. This often yields higher precision than either model alone (Longari et al., 2026).