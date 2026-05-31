import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('data/creditcard.csv')

print('Shape: ', df.shape)
print('Nulls: ', df.isnull().sum().max())
print('Stats: ', df[['Time', 'Amount', 'Class']].describe())

fraud = df[df['Class'] == 1]
legit = df[df['Class'] == 0]
print(f'legit: {len(legit)} ({round(len(legit) / len(df) * 100, 2)}%)  '
      f'fraud: {len(fraud)} ({round(len(fraud) / len(df) * 100, 2)}%)')

# Analysis plot
fig, axes = plt.subplots(2, 3, figsize=(16, 10))

# Transaction Amount log scaled - time kinda useless, dont care
ax = axes[0, 0]
ax.hist(legit['Amount'], bins=80, alpha=0.6, color='steelblue', label='Legit', log=True)
ax.hist(fraud['Amount'], bins=80, alpha=0.6, color='crimson', label='Fraud', log=True)
ax.set_title('Transaction Amount Distribution (log y-axis)')
ax.set_xlabel('Amount ($)')
ax.set_ylabel('Count (log)')
ax.legend()

# Correlation heatmap
ax = axes[0, 1]
v_cols = [f'V{i}' for i in range(1, 29)]
corr = df[v_cols].corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, ax=ax, cmap='coolwarm', center=0,
            linewidths=0, cbar_kws={'shrink': 0.7}, xticklabels=4, yticklabels=4)
ax.set_title('V1–V28 Correlation (lower triangle)')

# Mean feature value: fraud vs legit
ax = axes[0, 2]
means_legit = legit[v_cols].mean()
means_fraud  = fraud[v_cols].mean()
x = np.arange(len(v_cols))
width = 0.4
ax.bar(x - width/2, means_legit, width, label='Legit', color='steelblue', alpha=0.8)
ax.bar(x + width/2, means_fraud,  width, label='Fraud', color='crimson',  alpha=0.8)
ax.set_xticks(x[::4])
ax.set_xticklabels(v_cols[::4], rotation=45)
ax.set_title('Mean V Feature Value by Class')
ax.set_ylabel('Mean value')
ax.axhline(0, color='black', linewidth=0.5)
ax.legend()

# Excess kurtosis of V1-V28 — Gaussian expectation is 0
ax = axes[1, 1]
kurtosis = df[v_cols].kurtosis()
colors = ['crimson' if abs(k) > 1.0 else 'steelblue' for k in kurtosis]
ax.bar(v_cols, kurtosis, color=colors)
ax.axhline(0, color='black', linewidth=0.8)
ax.axhline(1.0,  color='crimson', linewidth=0.8, linestyle='--', alpha=0.5)
ax.axhline(-1.0, color='crimson', linewidth=0.8, linestyle='--', alpha=0.5)
ax.set_xticks(range(0, len(v_cols), 4))
ax.set_xticklabels(v_cols[::4], rotation=45)
ax.set_title('V1–V28 Excess Kurtosis (|k|>1 flagged red)')
ax.set_ylabel('Excess Kurtosis')

axes[1, 2].set_visible(False)

plt.tight_layout()
plt.savefig('data/analysis.png', dpi=150)
plt.show()
print("\nPlot saved to data/analysis.png")
