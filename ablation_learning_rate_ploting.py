import matplotlib.pyplot as plt
import numpy as np

# Enable LaTeX rendering for text
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'Times']

# Data from the table
#learning_rates = ["$10^{-5}$", "$10^{-4.3}$", "$10^{-4}$", "$10^{-3.7}$", "$10^{-3.4}$"]
learning_rates = [
    "$2.5 \\times 10^{-5}$", "$5 \\times 10^{-5}$", "$1 \\times 10^{-4}$",
    "$2 \\times 10^{-4}$", "$4 \\times 10^{-4}$"
]

mae = [279.45, 279.45, 246.34, 256.89, 247.20]
rmse = [366.18, 366.18, 364.89, 360.18, 364.80]
c_index = [0.4167, 0.4167, 0.5993, 0.4690, 0.5771]
spearmanr = [-0.2457, -0.2457, 0.2829, -0.1089, 0.2308]
spearmanr = [abs(v) for v in spearmanr] # Absolute values


# Normalize data (min-max normalization)
def normalize(data):
    min_val = min(data)
    max_val = max(data)
    return [(x - min_val) / (max_val - min_val) if max_val != min_val else 0
            for x in data]


norm_mae = normalize(mae)
norm_rmse = normalize(rmse)
norm_c_index = normalize(c_index)
norm_spearmanr = normalize(spearmanr)

# Create a figure with 4 subfigures (2x2 grid)
fig, axs = plt.subplots(2, 2, figsize=(7, 3), sharex=True, sharey=True)

# Plot MAE
axs[0, 0].plot(range(len(learning_rates)),
               norm_mae,
               color='#000000',
               marker='o',
               linestyle='-',
               linewidth=1.5)
axs[0, 0].set_title('MAE', fontsize=12)
axs[0, 0].grid(True, which='major', linestyle='--', alpha=0.7)  # Major grid (x and y)
axs[0, 0].grid(True, which='minor', linestyle=':', alpha=0.4)  # Minor grid (x and y)
axs[0, 0].minorticks_on()  # Enable minor ticks
axs[0, 0].tick_params(axis='both', which='major', labelsize=10)
axs[0, 0].tick_params(axis='both', which='minor', length=2)

# Plot RMSE
axs[0, 1].plot(range(len(learning_rates)),
               norm_rmse,
               color='#000000',
               marker='s',
               linestyle='-',
               linewidth=1.5)
axs[0, 1].set_title('RMSE', fontsize=12)
axs[0, 1].grid(True, which='major', linestyle='--', alpha=0.7)  # Major grid (x and y)
axs[0, 1].grid(True, which='minor', linestyle=':', alpha=0.4)  # Minor grid (x and y)
axs[0, 1].minorticks_on()  # Enable minor ticks
axs[0, 1].tick_params(axis='both', which='major', labelsize=10)
axs[0, 1].tick_params(axis='both', which='minor', length=2)

# Plot C-index
axs[1, 0].plot(range(len(learning_rates)),
               norm_c_index,
               color='#000000',
               marker='^',
               linestyle='-',
               linewidth=1.5)
axs[1, 0].set_title('C-index', fontsize=12)
axs[1, 0].grid(True, which='major', linestyle='--', alpha=0.7)  # Major grid (x and y)
axs[1, 0].grid(True, which='minor', linestyle=':', alpha=0.4)  # Minor grid (x and y)
axs[1, 0].minorticks_on()  # Enable minor ticks
axs[1, 0].tick_params(axis='both', which='major', labelsize=10)
axs[1, 0].tick_params(axis='both', which='minor', length=2)
axs[1, 0].set_xlabel('Learning Rate', fontsize=12)

# Plot SpearmanR (absolute)
axs[1, 1].plot(range(len(learning_rates)),
               norm_spearmanr,
               color='#000000',
               marker='d',
               linestyle='-',
               linewidth=1.5)
axs[1, 1].set_title('SpearmanR (Absolute)', fontsize=12)
axs[1, 1].grid(True, which='major', linestyle='--', alpha=0.7)  # Major grid (x and y)
axs[1, 1].grid(True, which='minor', linestyle=':', alpha=0.4)  # Minor grid (x and y)
axs[1, 1].minorticks_on()  # Enable minor ticks
axs[1, 1].tick_params(axis='both', which='major', labelsize=10)
axs[1, 1].tick_params(axis='both', which='minor', length=2)
axs[1, 1].set_xlabel('Learning rate', fontsize=12)

# Set x-axis labels for all subplots
for ax in axs.flat:
    ax.set_xticks(range(len(learning_rates)))
    ax.set_xticklabels(learning_rates, fontsize=10)

# Set common y-axis label
fig.text(0.02, 0.5, 'Normalized Value', va='center', rotation='vertical', fontsize=12)

# Set y-axis limits for all subplots
for ax in axs.flat:
    ax.set_ylim(0, 1.05)

# Add a main title
fig.suptitle('', fontsize=14)

# Adjust layout to prevent overlap
plt.tight_layout()
fig.subplots_adjust(top=0.9, left=0.12)

# Save the figure as a high-resolution image for IEEE paper
plt.savefig('performance_comparison.pdf', format='pdf', dpi=300, bbox_inches='tight')

# Display the plot
plt.show()
