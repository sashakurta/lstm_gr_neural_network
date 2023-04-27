import pickle
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt


with open(Path("optimal_model_result_matrix.pkl"), "rb") as f:
    result = pickle.load(f)
with open(Path("y_train.pkl"), "rb") as f:
    y_train = pickle.load(f)
plt.figure(figsize=(8, 6))
plt.imshow(result, interpolation='nearest', cmap=plt.cm.Blues)
plt.colorbar()


# Set the tick marks and labels
tick_marks = np.arange(len(set(y_train)))
plt.xticks(tick_marks, sorted(set(y_train)))
plt.yticks(tick_marks, sorted(set(y_train)))

# Add labels to the plot
thresh = result.max() / 2.
for i, j in np.ndindex(result.shape):
    plt.text(j, i, f"{result[i, j]:.2f}",
             ha="center", va="center",
             color="white" if result[i, j] > thresh else "black")

# Add axis labels and a title
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')

plt.savefig("visualization_src/conf_matrix1")
