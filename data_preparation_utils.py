import json
import math
from pathlib import Path

import keras
import numpy as np
import pandas as pd
from google.protobuf.json_format import MessageToJson
from keras.utils import pad_sequences


def distance_between(results, p1_loc, p2_loc):
    jsonObj = MessageToJson(results.multi_hand_landmarks[0])
    lmk = json.loads(jsonObj)['landmark']
    p1 = pd.DataFrame(lmk).to_numpy()[p1_loc]
    p2 = pd.DataFrame(lmk).to_numpy()[p2_loc]
    squared_dist = np.sum((p1 - p2) ** 2, axis=0)
    return np.sqrt(squared_dist)


def landmark_to_dist_emb(results):
    jsonObj = MessageToJson(results.multi_hand_landmarks[0])
    lmk = json.loads(jsonObj)['landmark']

    emb = np.array([
        # thumb to finger tip
        distance_between(results, 4, 8),
        distance_between(results, 4, 12),
        distance_between(results, 4, 16),
        distance_between(results, 4, 20),
        # wrist to finger tip
        distance_between(results, 4, 0),
        distance_between(results, 8, 0),
        distance_between(results, 12, 0),
        distance_between(results, 16, 0),
        distance_between(results, 20, 0),
        # tip to tip (specific to this application)
        distance_between(results, 8, 12),
        distance_between(results, 12, 16),
        # within finger joint (detect bending)
        distance_between(results, 1, 4),
        distance_between(results, 8, 5),
        distance_between(results, 12, 9),
        distance_between(results, 16, 13),
        distance_between(results, 20, 17),
        # distance from each tip to thumb joint
        distance_between(results, 2, 8),
        distance_between(results, 2, 12),
        distance_between(results, 2, 16),
        distance_between(results, 2, 20)
    ])
    # use np normalize, as min_max may create confusion that the closest fingers has 0 distance
    emb_norm = emb / np.linalg.norm(emb)
    return emb_norm


def skip_frame(landmark_npy_all, frame=50):
    new_lmk_array = []
    for each in landmark_npy_all:
        if len(each) <= frame:
            new_lmk_array.append(each)
        else:
            to_round = math.ceil(len(each)/frame)
            new_lmk_array.append(each[::to_round])
    return new_lmk_array


def lrSchedule(epoch):
    lr = 0.001
    if epoch > 200:
        lr *= 0.0005
    elif epoch > 120:
        lr *= 0.005
    elif epoch > 50:
        lr *= 0.01
    elif epoch > 30:
        lr *= 0.1

    print('Learning rate: ', lr)
    return lr


def frame_to_timecode(start_frame: int, finish_frame: int) -> str:
    start_seconds = start_frame // 30
    finish_seconds = finish_frame // 30
    start_minute = start_seconds // 60
    finish_minute = finish_seconds // 60
    add_to_start = 1 if start_frame - start_seconds * 30 > 15 else 0
    add_to_finish = 1 if finish_frame - finish_seconds * 30 > 15 else 0
    return (start_minute, start_seconds % 60 + add_to_start), (finish_minute, finish_seconds % 60 + add_to_finish)


def load_model(path="educated_model_with_bidirectional"):
    MODEL_NAME = Path(path)
    return keras.models.load_model(MODEL_NAME)


def predict_gesture(model, landmark_npy_all):
    new_lmk_array = skip_frame([landmark_npy_all])
    train_x = pad_sequences(new_lmk_array, padding="post", maxlen=50, dtype="float32")
    y_prediction = model.predict(train_x)
    return np.argmax(y_prediction, axis=1)

