# Visualize training history
import pickle

import matplotlib.pyplot as plt
import numpy as np

DIR_NAME = 'history/'
with open(DIR_NAME + "model_w_bidir_history.pkl", "rb") as f:
    bi_lstm_model = pickle.load(f)
with open(DIR_NAME + "gru_model_history.pkl", "rb") as f:
    gru_model = pickle.load(f)
with open(DIR_NAME + "lstm_model_history.pkl", "rb") as f:
    lstm_model = pickle.load(f)
models_data = [gru_model, lstm_model, bi_lstm_model]
# list all data in history

# summarize history for accuracy
for model in models_data:
    plt.plot(model.history['accuracy'])
# plt.plot(model.history['val_accuracy'])
plt.title('Model accuracy', fontsize=20, fontweight='bold')
plt.ylabel('accuracy')
plt.xlabel('epoch')
yticks = np.linspace(0, 1, 21)
ytick_labels = [f"{val:.2f}" for val in yticks]
plt.yticks(yticks, ytick_labels)
xticks = np.linspace(0, 250, 11)
xtick_labels = [f"{val:.1f}" for val in xticks]
plt.xticks(xticks, xtick_labels)
plt.legend(["gru", "lstm", "lstm_bidir"], loc='lower right')
plt.grid(True)
plt.savefig("visualization_src/accuracy_rnn")
plt.close()


max_loss = 1
for model in models_data:
    plt.plot(model.history['loss'])
    if max(model.history['loss']) > max_loss:
        max_loss = max(model.history['loss'])
# plt.plot(model.history['val_accuracy'])
plt.title('Model loss', fontsize=20, fontweight='bold')
plt.ylabel('model loss')
plt.xlabel('epoch')
yticks = np.linspace(0, max_loss, 21)
ytick_labels = [f"{val:.2f}" for val in yticks]
plt.yticks(yticks, ytick_labels)
xticks = np.linspace(0, 250, 11)
xtick_labels = [f"{val:.1f}" for val in xticks]
plt.xticks(xticks, xtick_labels)
plt.legend(["gru", "lstm", "lstm_bidir"], loc='upper left')
plt.grid(True)
plt.savefig("visualization_src/loss_rnn")

