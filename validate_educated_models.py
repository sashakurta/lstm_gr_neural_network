import pickle
from pathlib import Path
import keras
from keras.utils import pad_sequences, to_categorical
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

from data_preparation_utils import skip_frame
import numpy as np


DIR_NAME = Path("prepared_validate_data")
with open(DIR_NAME / Path("landmark_damp2.pkl"), "rb") as f:
    landmark_npy_all = pickle.load(f)
with open(DIR_NAME / Path("video_classes_dump2.pkl"), "rb") as f:
    video_class_all = pickle.load(f)


new_lmk_array = skip_frame(landmark_npy_all)
train_x = pad_sequences(new_lmk_array, padding="post", maxlen=50, dtype="float32")

classes = len(set(video_class_all))

train_y = to_categorical([i for i in video_class_all])
print("Training y with shape of: ", train_y.shape)
X_train, X_test, y_train, y_test = train_test_split(train_x, train_y, test_size=0.1)

MODEL_NAME = Path("educated_model_with_bidirectional")
model = keras.models.load_model(MODEL_NAME)

y_prediction = model.predict(X_train)
y_prediction = np.argmax(y_prediction, axis=1)
y_train = np.argmax(y_train, axis=1)
result = confusion_matrix(y_train, y_prediction, normalize='pred')
with open(Path("optimal_model_result_matrix.pkl"), "wb") as f:
    pickle.dump(result, f)
with open(Path("y_train.pkl"), "wb") as f:
    pickle.dump(y_train, f)
