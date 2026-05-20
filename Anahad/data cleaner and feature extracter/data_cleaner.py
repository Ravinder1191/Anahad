import cv2
import mediapipe as mp
import os
import shutil
import numpy as np

INPUT_PATH = "raw data"
GOOD_PATH = "cleaned data"

MIN_HAND_RATIO = 0.42
MIN_POSE_RATIO = 0.55
SAMPLE_FRAMES = 25

mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose

os.makedirs(GOOD_PATH, exist_ok=True)

def is_good_video(video_path):

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames < 10:
        cap.release()
        return False

    frame_indices = np.linspace(0, total_frames - 1, SAMPLE_FRAMES, dtype=int)

    hand_count = 0
    pose_count = 0
    tested_frames = 0

    with mp_hands.Hands(
            max_num_hands=2,
            min_detection_confidence=0.4,
            min_tracking_confidence=0.45
    ) as hands, mp_pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as pose:

        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                continue

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            hand_res = hands.process(rgb)
            pose_res = pose.process(rgb)

            tested_frames += 1

            if hand_res.multi_hand_landmarks:
                hand_count += 1

            if pose_res.pose_landmarks:
                lm = pose_res.pose_landmarks.landmark
                if (lm[mp_pose.PoseLandmark.LEFT_SHOULDER].visibility > 0.4 or
                        lm[mp_pose.PoseLandmark.RIGHT_SHOULDER].visibility > 0.4):
                    pose_count += 1

    cap.release()

    if tested_frames == 0:
        return False

    hand_ratio = hand_count / tested_frames
    pose_ratio = pose_count / tested_frames

    if hand_ratio >= MIN_HAND_RATIO:
        return True
    elif hand_ratio >= 0.35 and pose_ratio >= MIN_POSE_RATIO:
        return True
    else:
        return False


def clean_dataset():
    classes = [d for d in os.listdir(INPUT_PATH)
               if os.path.isdir(os.path.join(INPUT_PATH, d))]

    print(f"Found {len(classes)} classes in your dataset.")

    total_good = 0
    total_videos = 0

    for cls in classes:
        input_folder = os.path.join(INPUT_PATH, cls)
        good_folder = os.path.join(GOOD_PATH, cls)
        os.makedirs(good_folder, exist_ok=True)

        videos = [f for f in os.listdir(input_folder) if f.lower().endswith('.mp4')]

        print(f"\nProcessing class: {cls} ({len(videos)} videos)")

        class_good = 0
        for video in videos:
            total_videos += 1
            video_path = os.path.join(input_folder, video)

            if is_good_video(video_path):
                shutil.copy2(video_path, os.path.join(good_folder, video))
                class_good += 1
                total_good += 1
                print(f"Pass: {video}")

        print(f"  Saved {class_good} good videos from {cls}")

    print(f"Finished! {total_good} good videos copied")

clean_dataset()
