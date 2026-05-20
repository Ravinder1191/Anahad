import cv2 as cv
import mediapipe as mp
import numpy as np
import pandas as pd
import joblib
import time
import threading
import queue
from collections import deque, Counter
from keras.models import load_model
from config.paths import (static_model_path, static_encoder_path, feature_order_path, dynamic_model_path, dynamic_encoder_path, dynamic_scaler_path,
                          angle_csv_path, hand_landmark_path)
from src.features import palm_size
from src.model_utils import compute_thresholds
from src.helper_function import (
    init_mediapipe, add_motion, normalize_length,
    get_hand_points, get_pose_points,
    normalize_hand, normalize_body,
    extract_features as dynamic_extract
)
from src.static_predictor import static_pipeline
from src.word_formation_layer import WordFormation
from src.motion_detector import MotionDetector

s_model   = joblib.load(static_model_path)
s_encoder = joblib.load(static_encoder_path)
f_order   = joblib.load(feature_order_path)
d_model   = load_model(dynamic_model_path, compile=False)
d_encoder = joblib.load(dynamic_encoder_path)
d_scaler  = joblib.load(dynamic_scaler_path)
TH        = compute_thresholds(angle_csv_path, hand_landmark_path)

print("Warming up models...")
d_model.predict(np.zeros((1, 20, 300)), verbose=0)
s_model.predict_proba(pd.DataFrame([np.zeros(len(f_order))], columns=f_order))

holistic = init_mediapipe()
hands = mp.solutions.hands.Hands(max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.4)

cap = cv.VideoCapture(0)
cap.set(cv.CAP_PROP_BUFFERSIZE, 1)
cap.set(cv.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv.CAP_PROP_FRAME_HEIGHT, 360)

word_engine = WordFormation()
motion_detect = MotionDetector(
    window = 20,
    static_threshold = 0.0003,
    dynamic_threshold = 0.0012,
    confirm_frames = 6
)

pts_buf = deque(maxlen=5)
s_prev_label = ""
s_gesture_start = None

seq_buf = deque(maxlen=20)
d_last_valid = None
d_vote_buf = deque(maxlen=6)

d_disp = ""
d_disp_conf = 0.0
s_disp = ""
s_disp_conf = 0.0

MODE = "AUTO"
frame_n = 0
fps = 0
fps_n = 0
fps_t = time.time()

conf_d = 0.75
conf_s = 0.55
dynamic_window = 20
dynamic_min_frames = 8
dynamic_predict_every = 2
dynamic_stable_count = 2
dynamic_margin = 0.05
dynamic_cooldown = 0.9

FRONT_ONLY = {"A", "E", "K", "G", "H", "M", "N", "R", "S", "T", "U", "V", "W", "I", "Y"}

gru_input_q = queue.Queue(maxsize=1)
gru_result_lock = threading.Lock()
gru_result = {
    "label": "",
    "conf": 0.0,
    "final": "",
    "last_label": "",
    "last_time": time.time(),
    "candidate": "",
    "candidate_count": 0
}


def is_palm_facing(hand_landmarks, handedness):
    pts = np.array([[lm.x, lm.y] for lm in hand_landmarks.landmark], dtype=np.float32)
    v1 = pts[5] - pts[0]
    v2 = pts[17] - pts[0]
    cp = v1[0] * v2[1] - v1[1] * v2[0]
    return cp > 0 if handedness == "Right" else cp < 0


def get_dynamic_features(mp_res, last_valid):
    right_pts, left_pts = get_hand_points(mp_res)
    pose_pts = get_pose_points(mp_res)

    if mp_res.right_hand_landmarks or mp_res.left_hand_landmarks:
        right_pts = normalize_hand(right_pts)
        left_pts = normalize_hand(left_pts)

        if pose_pts is not None:
            pose_pts = normalize_body(pose_pts)

        features = dynamic_extract(right_pts, left_pts, pose_pts)
        last_valid = features
        return features, last_valid

    return last_valid, last_valid


def prepare_dynamic_sequence(sequence):
    seq = np.asarray(add_motion(sequence), dtype=np.float32)

    if len(seq) < dynamic_window:
        seq = normalize_length(seq, target_len=dynamic_window)

    return seq.astype(np.float32)


def gru_worker():
    while True:
        seq_scaled, last_label, last_time = gru_input_q.get()

        pred = d_model.predict(seq_scaled, verbose=0)[0]
        d_conf = float(np.max(pred))
        d_label = d_encoder.inverse_transform([np.argmax(pred)])[0]
        top2 = np.sort(np.partition(pred, -2)[-2:])
        d_margin = float(top2[-1] - top2[-2])

        final_label = ""
        current_time = time.time()

        with gru_result_lock:
            candidate = gru_result["candidate"]
            candidate_count = gru_result["candidate_count"]

        if d_conf > conf_d and d_margin >= dynamic_margin:
            if d_label == candidate:
                candidate_count += 1
            else:
                candidate = d_label
                candidate_count = 1

            stable_dynamic = candidate_count >= dynamic_stable_count

            if stable_dynamic and d_label != last_label:
                final_label = d_label
                last_label = d_label
                last_time = current_time
            elif stable_dynamic and current_time - last_time > dynamic_cooldown:
                final_label = d_label
                last_time = current_time
        else:
            candidate = ""
            candidate_count = 0

        with gru_result_lock:
            gru_result["label"] = d_label
            gru_result["conf"] = d_conf
            gru_result["final"] = final_label
            gru_result["last_label"] = last_label
            gru_result["last_time"] = last_time
            gru_result["candidate"] = candidate
            gru_result["candidate_count"] = candidate_count


threading.Thread(target=gru_worker, daemon=True).start()


def reset_state():
    global s_prev_label, s_gesture_start, d_last_valid
    global d_disp, d_disp_conf, s_disp, s_disp_conf

    pts_buf.clear()
    seq_buf.clear()
    d_vote_buf.clear()
    motion_detect._reset()

    s_prev_label = ""
    s_gesture_start = None
    d_last_valid = None

    d_disp = ""
    d_disp_conf = 0.0
    s_disp = ""
    s_disp_conf = 0.0

    with gru_result_lock:
        gru_result.update({
            "label": "",
            "conf": 0.0,
            "final": "",
            "last_label": "",
            "last_time": time.time(),
            "candidate": "",
            "candidate_count": 0
        })


while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv.flip(cv.resize(frame, (1280, 720)), 1)
    h, w = frame.shape[:2]
    frame_n += 1

    try:
        rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        mp_res = holistic.process(rgb)
        rgb.flags.writeable = True

        hand_present = bool(mp_res.left_hand_landmarks or mp_res.right_hand_landmarks)
        motion_mode = motion_detect.update(mp_res) if hand_present else "UNKNOWN"
        pipeline = motion_mode if MODE == "AUTO" else MODE

        s_confirmed = ""
        s_conf = 0.0

        if pipeline == "STATIC" and hand_present:
            hand_res = hands.process(rgb)

            palm_ok = True
            if hand_res and hand_res.multi_hand_landmarks:
                lms = hand_res.multi_hand_landmarks[0]
                palm_side = hand_res.multi_handedness[0].classification[0].label
                palm_ok = is_palm_facing(lms, palm_side)

            stable_label, s_prev_label, s_gesture_start = static_pipeline(
                hand_res, pts_buf, s_prev_label, s_gesture_start,
                s_model, s_encoder, f_order, TH
            )

            if stable_label:
                if not palm_ok and stable_label.upper() in FRONT_ONLY:
                    pass
                else:
                    s_confirmed = stable_label.lower()
                    s_disp = stable_label.lower()

            if hand_res and hand_res.multi_hand_landmarks and len(pts_buf) == 5:
                mean = np.mean(np.stack(list(pts_buf)), axis=0)
                mean /= palm_size(mean)

                feat = {f"x{i}": mean[i, 0] for i in range(21)}
                feat.update({f"y{i}": mean[i, 1] for i in range(21)})

                probs = s_model.predict_proba(pd.DataFrame([feat])[f_order])[0]
                s_conf = float(np.max(probs))
                s_disp_conf = s_conf

        elif pipeline == "DYNAMIC" or not hand_present:
            pts_buf.clear()
            s_prev_label = ""
            s_gesture_start = None
            s_disp = ""
            s_disp_conf = 0.0

        d_confirmed = ""
        d_conf = 0.0

        if pipeline == "DYNAMIC" and hand_present:
            features, d_last_valid = get_dynamic_features(mp_res, d_last_valid)

            if features is not None:
                seq_buf.append(features)

            if len(seq_buf) >= dynamic_min_frames and frame_n % dynamic_predict_every == 0:
                seq = prepare_dynamic_sequence(seq_buf)
                seq_scaled = d_scaler.transform(
                    seq.reshape(-1, seq.shape[1])
                ).reshape(1, dynamic_window, -1)

                with gru_result_lock:
                    ll = gru_result["last_label"]
                    lt = gru_result["last_time"]

                if gru_input_q.full():
                    try:
                        gru_input_q.get_nowait()
                    except queue.Empty:
                        pass

                gru_input_q.put((seq_scaled, ll, lt))

            with gru_result_lock:
                d_label = gru_result["label"]
                d_conf = gru_result["conf"]
                d_confirmed = gru_result["final"]
                gru_result["final"] = ""

            if d_label and d_conf >= conf_d:
                d_vote_buf.append(d_label)
                voted = Counter(d_vote_buf).most_common(1)[0]

                if voted[1] >= 3:
                    d_disp = voted[0]
                    d_disp_conf = d_conf

        elif pipeline == "STATIC" or not hand_present:
            seq_buf.clear()
            d_vote_buf.clear()
            d_disp = ""
            d_disp_conf = 0.0

            with gru_result_lock:
                gru_result["label"] = ""
                gru_result["conf"] = 0.0
                gru_result["final"] = ""
                gru_result["candidate"] = ""
                gru_result["candidate_count"] = 0

        if pipeline == "DYNAMIC" and d_confirmed:
            active = "Dynamic"
        elif pipeline == "STATIC" and s_confirmed:
            active = "Static"
        elif pipeline == "DYNAMIC" and d_conf >= conf_d:
            active = "Dynamic"
        elif pipeline == "STATIC" and s_conf >= conf_s:
            active = "Static"
        else:
            active = "None"

        current_word, sentence = word_engine.update(
            s_confirmed if active == "Static" else "",
            d_confirmed if active == "Dynamic" else "",
            active,
            hand_present
        )

        fps_n += 1
        if time.time() - fps_t >= 1.0:
            fps = fps_n
            fps_n = 0
            fps_t = time.time()

        col = (
            (0, 255, 255) if active == "Static"
            else (100, 180, 255) if active == "Dynamic"
            else (180, 180, 180)
        )

        cv.rectangle(frame, (0, 0), (w, 55), (20, 20, 20), -1)
        cv.putText(
            frame,
            f"FPS:{fps}  Mode:{MODE} ({active})  [M] Switch  [R] Reset  [Q] Quit",
            (15, 38),
            cv.FONT_HERSHEY_SIMPLEX,
            0.82,
            col,
            2
        )

        cv.rectangle(frame, (w - 340, 62), (w - 10, 252), (20, 20, 20), -1)
        cv.putText(frame, "LIVE TRACKER", (w - 330, 88),
                   cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        d_col = (0, 255, 0) if d_disp_conf >= conf_d else (150, 150, 150)
        cv.putText(frame, f"Dynamic: {d_disp or '--'} ({d_disp_conf:.2f})",
                   (w - 330, 125), cv.FONT_HERSHEY_SIMPLEX, 0.65, d_col, 2)

        s_col = (255, 200, 50) if s_disp_conf >= conf_s else (150, 150, 150)
        cv.putText(frame, f"Static : {s_disp or '--'} ({s_disp_conf:.2f})",
                   (w - 330, 162), cv.FONT_HERSHEY_SIMPLEX, 0.65, s_col, 2)

        cv.putText(frame, f"Motion: {motion_mode}",
                   (w - 330, 215), cv.FONT_HERSHEY_SIMPLEX, 0.52, (180, 180, 180), 1)

        cv.putText(frame, f"Spelling : {current_word}",
                   (20, 115), cv.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 255), 2)

        cv.rectangle(frame, (0, h - 98), (w, h - 53), (20, 20, 20), -1)
        cv.putText(frame, f"Sentence : {sentence.strip()}",
                   (15, h - 66), cv.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

        cv.rectangle(frame, (0, h - 46), (w, h), (10, 10, 10), -1)
        cv.putText(
            frame,
            "M: AUTO->STATIC->DYNAMIC  |  ENTER: Add Word  |  BACKSPACE: Delete  |  R: Reset",
            (15, h - 16),
            cv.FONT_HERSHEY_SIMPLEX,
            0.6,
            (160, 160, 160),
            1
        )

    except Exception:
        import traceback
        traceback.print_exc()

    cv.imshow("Anahad (ASL VERSION)", frame)
    key = cv.waitKey(1) & 0xFF

    if key == ord("q"):
        break
    elif key == ord("m"):
        MODE = {"AUTO": "STATIC", "STATIC": "DYNAMIC"}.get(MODE, "AUTO")
        reset_state()
    elif key in (ord("r"), ord("c")):
        word_engine.reset()
        reset_state()
    elif key in (8, 127):
        word_engine.backspace()
    elif key == ord(" "):
        word_engine.manual_space()
    elif key == 13:
        word_engine.commit_word()

cap.release()
cv.destroyAllWindows()