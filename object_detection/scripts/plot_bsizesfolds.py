import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Load the data
data = pd.read_csv('data/bsizesfolds.csv')

# Calculate the average recall for each configuration (row)
average_recall_per_config = data.mean(axis=1)

print(average_recall_per_config)

# Plotting
plt.figure(figsize=(10, 5))
plt.plot(average_recall_per_config, marker='o', linestyle='-', color='skyblue')
plt.xlabel('Border Size')
plt.ylabel('Average Recall')
plt.title('Average Recall for Border Sizes')
plt.grid(True)
plt.tight_layout()
plt.yticks(np.arange(0.65, 1.0, 0.02))
plt.xticks(range(len(average_recall_per_config)))
plt.show()
