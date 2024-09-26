import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Data from the table
data = np.array([
    [0.2954, 0.3069, 0.3225, 0.3050, 0.3418, 0.3264, 0.3547, 0.3746],
    [0.2901, 0.3093, 0.3364, 0.3662, 0.3791, 0.4025, 0.4209, 0.4962],
    [0.2460, 0.2722, 0.3011, 0.3418, 0.3596, 0.3972, 0.4462, 0.5838],
    [0.1919, 0.2206, 0.2518, 0.2922, 0.3125, 0.3575, 0.4386, 0.6216],
    [0.1592, 0.1829, 0.2098, 0.2498, 0.2690, 0.3142, 0.4178, 0.6429],
    [0.1344, 0.1552, 0.1781, 0.2147, 0.2355, 0.2752, 0.4002, 0.6525],
    [0.1185, 0.1362, 0.1577, 0.1881, 0.2081, 0.2430, 0.3848, 0.6544],
    [0.1085, 0.1246, 0.1428, 0.1691, 0.1884, 0.2191, 0.3717, 0.6528],
    [0.1021, 0.1165, 0.1321, 0.1552, 0.1727, 0.2013, 0.3605, 0.6488]
])

# Labels for x and y axes
x_labels = ['125', '250', '500', '1,000', '2,000', '4,000', '8,000', '16,000']
y_labels = ['2', '3', '4', '5', '6', '7', '8', '9', '10']

# Create the heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(data, annot=True, fmt=".4f", cmap="winter", vmin=0, vmax=1.0, xticklabels=x_labels, yticklabels=y_labels, cbar=True)

# Title and labels
# plt.title('Heatmap of Values Based on Different N and Classes')
plt.xlabel('Number of JAVA Classes in Train Set')
plt.ylabel('N')
plt.tight_layout()

# Display the heatmap
plt.savefig("heatmap.png", dpi=500)