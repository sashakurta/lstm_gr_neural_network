from moviepy.video.io.VideoFileClip import VideoFileClip

class_dict_to_name = {
        "D0X": "no gesture",
        "B0A": "point one finger",
        "B0B": "point two fingers",
        "G01": "click one finger",
        "G02": "click two fingers",
        "G03": "throw up",
        "G04": "throw down",
        "G05": "throw left",
        "G06": "throw right",
        "G07": "open twice",
        "G08": "double click one",
        "G09": "double click two",
        "G10": "zoom in",
        "G11": "zoom out",
}

class_to_id_dict = {
        "D0X": 0,
        "B0A": 1,
        "B0B": 2,
        "G01": 3,
        "G02": 4,
        "G03": 5,
        "G04": 6,
        "G05": 7,
        "G06": 8,
        "G07": 9,
        "G08": 10,
        "G09": 11,
        "G10": 12,
        "G11": 13,
}

EXPORT_DIR = "/Users/oleksandrkurta/Diploma/external_datasets/mp4/"
annotation_for_validate = "/Users/oleksandrkurta/Diploma/lstm_neural_network_project/annotations/Annot_TestList.txt"
with open(annotation_for_validate, "r") as f:
    validate_len = len(f.readlines())
annotation_for_train = "/Users/oleksandrkurta/Diploma/lstm_neural_network_project/annotations/Annot_TrainList.txt"

with open(annotation_for_train, "r") as f:
    train_len = len(f.readlines())
flag = False
with open(annotation_for_validate, "r") as f:
    for num, clip_info in enumerate(f.readlines()):
        print("{} {}{}".format(
            ((num+1)/validate_len)*100, "x"*int(((num+1)/validate_len)*100), "-"*int(100 - ((num+1)/validate_len)*100)))
        clip_name, label, class_id, t_start, t_end, frames = clip_info.split(",")

        try:
            clip = VideoFileClip(EXPORT_DIR + clip_name + ".mp4")
            print("susses!", EXPORT_DIR + clip_name + ".mp4")
        except OSError as err:
            print(clip_name, "not found")
            continue
        print(t_start, t_end)
        print(frame_to_timecode(int(t_start), int(t_end)))
        try:
            clip1 = clip.subclip(*frame_to_timecode(int(t_start), int(t_end)))
        except Exception:
            print("{} was failed to cut".format(clip_name))
        clip1.write_videofile('edited_videos_validation/base{}_{}_edited_{}_{}.mp4'.format(clip_name, class_to_id_dict[label],
                                                                         class_dict_to_name[label],
                                                                         label),
                              codec='libx264')


