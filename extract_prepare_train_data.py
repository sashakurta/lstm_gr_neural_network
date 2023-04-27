import argparse
import os
import pickle

import cv2
import mediapipe as mp

from data_preparation_utils import landmark_to_dist_emb

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands


def extract_video_data(script_args):
    arr = os.listdir(script_args.video_source_dir)
    arr.remove(".DS_Store")
    gestures_classes = []
    hands_model_data = []
    mp_hands_setup = mp_hands.Hands(max_num_hands=1,
                                    min_detection_confidence=0.6,
                                    min_tracking_confidence=0.6)
    for num, video_name in enumerate(arr):
        landmark_npy_single = []
        try:
            video_data = cv2.VideoCapture(script_args.video_source_dir + video_name)
        except Exception as e:
            print("Something went wrong with file {}".format(video_name))
            print("Next exception ocurred {} {}".format(e, type(e)))
        else:
            gestures_classes.append(int(video_name.split("_")[4]))
            while video_data.isOpened():
                success, image = video_data.read()
                if not success:
                    break
                image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
                image.flags.writeable = False
                results = mp_hands_setup.process(image)
                image.flags.writeable = True
                # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                if results.multi_hand_landmarks:
                    for _ in results.multi_hand_landmarks:
                        landmark_npy_single.append(landmark_to_dist_emb(results))
            hands_model_data.append(landmark_npy_single)
            video_data.release()
        if ((num + 1) % 10) == 0:
            print(f"Finished for {(num + 1)} videos")
    print(f"Finished for total {len(arr)} videos. Completed.")
    with open(script_args.hand_model_source_file_name, "wb") as f:
        pickle.dump(hands_model_data, f)
    with open(script_args.gestures_classes_file_name, "wb") as f:
        pickle.dump(gestures_classes, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_source_dir", help="path to videos for preparation", default="edited_videos_validation/")
    parser.add_argument("--hand_model_source_file_name", help="name of file for hand model data",
                        default="landmark_damp2.pkl")
    parser.add_argument("--gestures_classes_file_name", help="name of file for gestures classes data",
                        default="video_classes_dump2.pkl")
    parser.add_argument("--max_num_hands", help="number of hands", default=1)
    args = parser.parse_args()
    extract_video_data(args)
