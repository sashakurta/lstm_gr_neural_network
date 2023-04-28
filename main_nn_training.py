import pickle

import mediapipe as mp
from keras import Sequential
from keras.callbacks import LearningRateScheduler
from keras.layers import (GRU, LSTM, Activation, BatchNormalization,
                          Bidirectional, Dense, Dropout)
from keras.utils import pad_sequences, to_categorical
from sklearn.model_selection import train_test_split

from data_preparation_utils import lrSchedule, skip_frame

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
print('Extract prepared data')
with open("prepared_train_data/landmark_damp1.pkl", "rb") as f:
    landmark_npy_all = pickle.load(f)
with open("prepared_train_data/video_classes_dump1.pkl", "rb") as f:
    video_class_all = pickle.load(f)


new_lmk_array = skip_frame(landmark_npy_all)
train_x = pad_sequences(new_lmk_array, padding="post", maxlen=50, dtype="float32")

classes = len(set(video_class_all))
feature_len = 20
max_len = 50

train_y = to_categorical([i for i in video_class_all])
print("Training y with shape of: ", train_y.shape)
X_train, X_test, y_train, y_test = train_test_split(train_x, train_y, test_size=0.2)

model = Sequential()
model.add(GRU(256, return_sequences=True, input_shape=(max_len, feature_len)))
model.add(Dropout(0.25))
model.add(GRU(256, return_sequences=True))
model.add(Dropout(0.25))
model.add(GRU(128, return_sequences=False))
model.add(Dense(64))
model.add(BatchNormalization())
model.add(Activation("relu"))
model.add(Dense(classes, activation="softmax"))
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
model.summary()

LRScheduler = LearningRateScheduler(lrSchedule)
callbacks_list = [LRScheduler]

verbose, epochs, batch_size = 1, 100, 150
model.fit(X_train, y_train,
          validation_data=(X_test, y_test),
          epochs=epochs,
          batch_size=batch_size,
          verbose=verbose,
          shuffle=True,
          callbacks=callbacks_list)
with open("dense_model_history.pkl", "wb") as f:
    pickle.dump(model.history, f)
model.save("dense_educated_model")

