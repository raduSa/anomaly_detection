For neural nets (like MLP AE):
- Feature Pruning (Preventing Data Leakage)
Autoencoders are excellent at memorizing patterns. If you give them trivial identifiers, they will memorize the network topology rather than the behavior of the traffic.

Drop Identifiers: Remove Flow ID, Source IP, Destination IP, Source Port, and Timestamp. (Note: Destination Port is sometimes kept as it defines the service being accessed, but be careful that it doesn't bias your model).

Drop Zero-Variance Features: Remove features that have the exact same value across the entire dataset (e.g., Bwd PSH Flags, Fwd URG Flags in some splits), as they provide zero mathematical value to an MLP and only add noise.

- Encoding and Scaling
Neural networks require strictly numerical, scaled inputs to calculate gradients effectively.

Categorical Encoding: The Protocol feature is typically represented as integers (0, 6, 17 for HOPOPT, TCP, UDP). You should One-Hot Encode this feature so the network doesn't falsely assume that UDP (17) is "greater" than TCP (6).

Feature Scaling: This is arguably the most critical step for an MLP AE. Network flow features have wildly different ranges (e.g., Flow Duration is in the millions, while Fwd Packet Length Max might be a few thousand).

Use a MinMaxScaler (scaling values between 0 and 1) or a StandardScaler (scaling to a mean of 0 and variance of 1).

Crucial Rule: Fit the scaler only on your Training Set (Benign data), and then Transform both your Training Set and Test Set.

-Beware of "Day-of-the-Week" Bias
If you train your MLP AE exclusively on Monday's traffic, it will only learn what normal traffic looks like on a Monday.

Network traffic patterns often change throughout the week. If you later test your model against Tuesday or Friday traffic, your Autoencoder might flag perfectly normal traffic as anomalous simply because it didn't see those specific user behaviors during training.

The Fix: It is highly recommended to take the Monday dataset and combine it with the Benign labeled rows from Tuesday through Friday. This ensures your model learns a complete picture of a normal workweek.

- Verify the "Benign" Labels (Even on Monday)
Even though Monday is inherently attack-free, the Engelen et al. improved dataset still labels these rows as Benign.

When loading the Monday CSV into your pipeline, make sure you still explicitly filter by the Benign label. This is just a good programmatic safeguard in case a stray malformed row or header slipped into the CSV during extraction.