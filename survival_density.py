import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde

# Survival data
survival_days = [
    199, 880, 155, 372, 903, 120, 80, 486, 370, 1731, 510, 1096, 319, 56, 442, 822, 436,
    353, 540, 538, 203, 12, 495, 614, 22, 448, 289, 291, 187, 467, 142, 175, 616, 1076,
    86, 99, 812, 21, 287, 430, 698, 405, 375, 342, 317, 1489, 300, 355, 71, 1282, 394,
    77, 996, 286, 357, 437, 296, 152, 329, 374, 67, 182, 1410, 630, 600, 345, 170, 5,
    262, 630, 387, 74, 409, 77, 425, 1767, 728, 82, 558, 336, 111, 519, 522, 726, 300,
    540, 168, 311, 169, 208, 239, 1283, 515, 495, 1148, 278, 105, 106, 351, 1527, 412,
    355, 127, 244, 616, 1145, 89, 121, 759, 466, 804, 580, 30, 200, 439, 229, 421, 153,
    330, 476, 82, 103, 734, 448, 331, 240, 150, 427, 508, 359, 114, 434, 488, 296, 1020,
    85, 146, 376, 277, 332, 438, 456, 30, 350, 633, 58, 147, 684, 117, 23, 468, 362,
    139, 453, 184, 269, 368, 209, 871, 254, 1155, 473, 84, 1227, 110, 613, 407, 660,
    635, 401, 1561, 210, 232, 346, 692, 334, 621, 524, 260, 544, 828, 688, 33, 180, 213,
    1178, 62, 503, 268, 465, 1458, 634, 78, 424, 208, 50, 597, 503, 327, 191, 610, 318,
    416, 1293, 1592, 385, 737, 1337, 55, 394, 322, 615, 112, 145, 730, 32, 172, 387,
    333, 250, 382, 1278, 626, 747, 336, 464, 197, 946, 82, 788, 265, 576, 104, 579, 448
]

# KDE
data = np.array(survival_days)
kde = gaussian_kde(data)
x_vals = np.linspace(data.min(), data.max(), 1000)
y_vals = kde(x_vals)

# Plot setup
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 12
plt.figure(figsize=(6, 4))
plt.plot(x_vals, y_vals, color='blue', linewidth=2)
plt.fill_between(x_vals, y_vals, color='skyblue', alpha=0.3)

# Labels and grid
#plt.title('Kernel Density Estimate (KDE) of Survival Days', fontsize=14)
plt.xlabel('Survival Days', fontsize=12)
plt.ylabel('Density', fontsize=12)
plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)

# Finalize and save
plt.tight_layout()
plt.savefig("KDE_Survival_Times_Publication.png")
