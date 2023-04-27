import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix

# Generate some example data
y_true = np.array([0, 0, 1, 1, 2, 2])
y_pred = np.array([0, 1, 1, 2, 2, 2])

# Calculate the confusion matrix
cm = confusion_matrix(y_true, y_pred)

# Create a plot
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.colorbar()

# Set the tick marks and labels
tick_marks = np.arange(len(set(y_true)))
plt.xticks(tick_marks, sorted(set(y_true)))
plt.yticks(tick_marks, sorted(set(y_true)))

# Add labels to the plot
thresh = cm.max() / 2.
for i, j in np.ndindex(cm.shape):
    plt.text(j, i, cm[i, j],
             ha="center", va="center",
             color="white" if cm[i, j] > thresh else "black")

# Add axis labels and a title
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')

# Show the plot
plt.show()