import matplotlib.pyplot as plt
import numpy as np

# Enable LaTeX rendering for text
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'Times']
plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'

# Raw data
feature_counts = ["16384", "512", "64", "8"]
mae = [244.19, 240.05, 232.95, 225.89]
rmse = [369.66, 368.88, 353.73, 351.00]
c_index = [0.5647, 0.5798, 0.6285, 0.6764]
spearmanr = [0.1754, 0.2318, 0.4071, 0.4814]

# Normalize data
def normalize(data):
    min_val = min(data)
    max_val = max(data)
    return [(x - min_val) / (max_val - min_val) if max_val != min_val else 0 for x in data]

norm_mae = normalize(mae)
norm_rmse = normalize(rmse)
norm_c_index = normalize(c_index)
norm_spearmanr = normalize(spearmanr)

# Create a single plot
fig, ax = plt.subplots(figsize=(7, 4))

x = np.arange(len(feature_counts))
bar_width = 0.18  # Bar width to prevent overlap

# Define a professional color palette (muted, professional tones)
colors = ['#4c78a8', '#e45756', '#54a24b', '#b279a2']  # Blue, Red, Green, Purple (Tableau-inspired)

# Plot bars for each metric with distinct colors
ax.bar(x - 1.5 * bar_width, norm_mae, bar_width, label='MAE', color=colors[0], edgecolor='black')
ax.bar(x - 0.5 * bar_width, norm_rmse, bar_width, label='RMSE', color=colors[1], edgecolor='black')
ax.bar(x + 0.5 * bar_width, norm_c_index, bar_width, label='C-index', color=colors[2], edgecolor='black')
ax.bar(x + 1.5 * bar_width, norm_spearmanr, bar_width, label='SpearmanR', color=colors[3], edgecolor='black')

# X-axis
ax.set_xticks(x)
ax.set_xticklabels([rf"$F_{{\mathrm{{clinical}}}}$ (4) + $F_{{\mathrm{{image}}}}$ ({n})" for n in feature_counts],
                  fontsize=8, ha="center")  # Horizontal labels

# Y-axis
ax.set_ylabel('Normalized Value', fontsize=12)
ax.set_ylim(-0.05, 1.1)  # Extended lower bound to ensure low values are visible

# Add value labels on top of bars for clarity
for i, v in enumerate([norm_mae, norm_rmse, norm_c_index, norm_spearmanr]):
    for j, val in enumerate(v):
        ax.text(j + (i - 1.5) * bar_width, val + 0.02, f'{val:.3f}',
                ha='center', va='bottom', fontsize=8, rotation=45, color='black')

# Grid
ax.grid(True, which='major', linestyle='--', alpha=0.7)
ax.grid(True, which='minor', linestyle=':', alpha=0.4)
ax.minorticks_on()

# Axis ticks
ax.tick_params(axis='both', which='major', labelsize=10)
ax.tick_params(axis='both', which='minor', length=2)

# Legend
ax.legend(fontsize=10, loc='upper left', bbox_to_anchor=(0.05, 0.95))

# Adjust layout and save
plt.tight_layout()
plt.savefig('merged_performance_plot.pdf', format='pdf', dpi=300, bbox_inches='tight')
plt.show()