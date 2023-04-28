import json

import cv2
import mediapipe as mp
import numpy as np
from google.protobuf.json_format import MessageToJson

from data_preparation_utils import load_model, predict_gesture, landmark_to_dist_emb
from train_set_creator import id_to_name

# from train_set_creator import id_to_name

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

model = load_model()
window_name = 'Image'
# font
font = cv2.FONT_HERSHEY_SIMPLEX
# org
org = (50, 50)
# fontScale
fontScale = 1
# Blue color in BGR
color = (255, 0, 0)
# Line thickness of 2 px
thickness = 2
name_of_gesture = 'No_gesture'
landmark_per_50 = []
cap = cv2.VideoCapture(0)

with mp_hands.Hands(max_num_hands=1,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5) as hands:
    while cap.isOpened():
        landmark_npy_single = []
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")

        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = hands.process(image)
        # Draw the hand annotations on the image.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        image = cv2.putText(image, name_of_gesture, org, font,
                            fontScale, color, thickness, cv2.LINE_AA)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                landmark_npy_single.append(landmark_to_dist_emb(results))
                mp_drawing.draw_landmarks(
                    image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            landmark_per_50.append(landmark_npy_single[0])

        if len(landmark_per_50) > 50:
            gest_type = predict_gesture(model, landmark_per_50)[0]
            print(gest_type)
            name_of_gesture = id_to_name(gest_type)

        cv2.imshow('MediaPipe Hands', image)
        if cv2.waitKey(5) & 0xFF == 27:
            break
cap.release()


